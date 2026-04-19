#include "wavefront.h"
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include "material.h"
#include <cstdio>
#include <iostream>

// ── RNG Init ──────────────────────────────────────────────────────────────────
__global__ void k_initRNG(curandState* rng, int numRays) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRays) return;
    // same seed logic as baseline
    curand_init(1984 + tid, 0, 0, &rng[tid]);
}

#define MAX_DEPTH 50

#define gpuCheck(ans) { gpuAssert_((ans), __FILE__, __LINE__); }
inline void gpuAssert_(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s  %s:%d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

__device__ vec3 randomInUnitSphere(curandState* s) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(s), curand_uniform(s), curand_uniform(s))
            - vec3(1.0f, 1.0f, 1.0f);
    } while (p.squared_length() >= 1.0f);
    return p;
}

// ── Primary ray generation ────────────────────────────────────────────────────

__global__ void k_generateRays(RayState* states, curandState* rng,
                                camera cam,
                                int width, int tileHeight,
                                int fullHeight, int yOffset,
                                int sampleSeed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= tileHeight) return;

    int localIdx = j * width + i;
    int global_j = j + yOffset;

    curand_init(1984ULL
                + (unsigned long long)sampleSeed * (unsigned long long)width * fullHeight
                + (unsigned long long)global_j * width + i,
                0, 0, &rng[localIdx]);

    float u = (float(i)        + curand_uniform(&rng[localIdx])) / float(width);
    float v = (float(global_j) + curand_uniform(&rng[localIdx])) / float(fullHeight);

    states[localIdx].r          = cam.get_ray(u, v, &rng[localIdx]);
    states[localIdx].throughput = vec3(1.0f, 1.0f, 1.0f);
    states[localIdx].radiance   = vec3(0.0f, 0.0f, 0.0f);
    states[localIdx].pixelIndex = localIdx;
    states[localIdx].depth      = 0;
    states[localIdx].alive      = true;
}

__global__ void k_intersect(RayState* states,
                             const int* __restrict__ activeIn, int N,
                             sphere* spheres, int numSpheres,
                             WavefrontQueues Q,
                             GBuffers gb, bool firstBounce)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int       rayIdx = activeIn[tid];
    RayState& s      = states[rayIdx];

    hit_record rec;
    bool hit = false;
    float closest = 1e30f;

    for (int i = 0; i < numSpheres; i++) {
        hit_record temp;
        if (spheres[i].hit(s.r, 0.001f, closest, temp)) {
            hit = true;
            closest = temp.t;
            rec = temp;
        }
    }

    if (firstBounce && hit) {
        MaterialData& md = rec.mat;
        vec3 alb = (md.type == MAT_LAMBERTIAN || md.type == MAT_METAL)
                   ? md.albedo
                   : vec3(1.0f, 1.0f, 1.0f);
        if (gb.albedo) gb.albedo[s.pixelIndex] = alb;
        if (gb.normal) gb.normal[s.pixelIndex] =
            rec.normal * 0.5f + vec3(0.5f, 0.5f, 0.5f);
    }

    if (!hit) {
        Q.miss[atomicAdd(&Q.d_counts[3], 1)] = rayIdx;
        return;
    }

    s.pendingHit = rec;

    MaterialType mt = rec.mat.type;
    if      (mt == MAT_LAMBERTIAN) Q.lambertian[atomicAdd(&Q.d_counts[0], 1)] = rayIdx;
    else if (mt == MAT_DIELECTRIC) Q.dielectric[atomicAdd(&Q.d_counts[1], 1)] = rayIdx;
    else                            Q.metal     [atomicAdd(&Q.d_counts[2], 1)] = rayIdx;
}

// ── Shade: Lambertian ─────────────────────────────────────────────────────────

__global__ void k_shadeLambertian(RayState* states, const int* queue,
                                   int N, curandState* rng)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int       rayIdx = queue[tid];
    RayState& s      = states[rayIdx];
    MaterialData& md = s.pendingHit.mat;

    vec3 target = s.pendingHit.p + s.pendingHit.normal
                + randomInUnitSphere(&rng[rayIdx]);
    s.r          = ray(s.pendingHit.p, target - s.pendingHit.p);
    s.throughput *= md.albedo;
    s.depth++;
    if (s.depth >= MAX_DEPTH || s.throughput.squared_length() < 1e-6f)
        s.alive = false;
}

// ── Shade: Dielectric ─────────────────────────────────────────────────────────

__global__ void k_shadeDielectric(RayState* states, const int* queue,
                                   int N, curandState* rng)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int       rayIdx = queue[tid];
    RayState& s      = states[rayIdx];
    float     refIdx = s.pendingHit.mat.ref_idx;
    vec3      normal = s.pendingHit.normal;
    vec3      p      = s.pendingHit.p;

    vec3  outN;
    float ni_nt, cosine;
    if (dot(s.r.direction(), normal) > 0.0f) {
        outN   = -normal;
        ni_nt  = refIdx;
        cosine = dot(s.r.direction(), normal) / s.r.direction().length();
        cosine = sqrtf(fmaxf(0.0f, 1.0f - refIdx*refIdx*(1.0f - cosine*cosine)));
    } else {
        outN   = normal;
        ni_nt  = 1.0f / refIdx;
        cosine = -dot(s.r.direction(), normal) / s.r.direction().length();
    }

    vec3  reflected = reflect(s.r.direction(), normal);
    vec3  uv        = unit_vector(s.r.direction());
    float dt        = dot(uv, outN);
    float disc      = 1.0f - ni_nt*ni_nt*(1.0f - dt*dt);
    bool  canRef    = disc > 0.0f;

    float r0 = (1.0f - refIdx) / (1.0f + refIdx); r0 *= r0;
    float b  = 1.0f - cosine;
    float rp = canRef ? r0 + (1.0f-r0)*(b*b*b*b*b) : 1.0f;

    vec3 scattered;
    if (curand_uniform(&rng[rayIdx]) < rp) {
        scattered = reflected;
    } else {
        scattered = ni_nt*(uv - outN*dt) - outN*sqrtf(disc);
    }
    s.r = ray(p, scattered);
    s.depth++;
    if (s.depth >= MAX_DEPTH) s.alive = false;
}

// ── Shade: Metal ──────────────────────────────────────────────────────────────

__global__ void k_shadeMetal(RayState* states, const int* queue,
                              int N, curandState* rng)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int       rayIdx = queue[tid];
    RayState& s      = states[rayIdx];
    MaterialData& md = s.pendingHit.mat;
    vec3      normal = s.pendingHit.normal;
    vec3      p      = s.pendingHit.p;

    vec3 refl     = reflect(unit_vector(s.r.direction()), normal);
    vec3 scattered = refl + md.fuzz * randomInUnitSphere(&rng[rayIdx]);

    if (dot(scattered, normal) <= 0.0f) { s.alive = false; return; }
    s.r           = ray(p, scattered);
    s.throughput *= md.albedo;
    s.depth++;
    if (s.depth >= MAX_DEPTH) s.alive = false;
}

// ── Shade: Miss ───────────────────────────────────────────────────────────────

__global__ void k_shadeMiss(RayState* states, const int* queue, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int       rayIdx = queue[tid];
    RayState& s      = states[rayIdx];

    vec3  ud  = unit_vector(s.r.direction());
    float t   = 0.5f * (ud.y() + 1.0f);
    vec3  sky = (1.0f-t)*vec3(1.0f,1.0f,1.0f) + t*vec3(0.5f,0.7f,1.0f);
    s.radiance += s.throughput * sky;
    s.alive     = false;
}

// ── Accumulate ────────────────────────────────────────────────────────────────

__global__ void k_accumulate(RayState* states, vec3* output, int numRays)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRays) return;
    output[states[i].pixelIndex] += states[i].radiance;
}

// ── Stream-compact active list: write alive indices into dst ──────────────────
// Returns new count via d_count[0].

__global__ void k_compact(const RayState* states,
                           const int* src, int* dst,
                           int N, int* d_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int rayIdx = src[tid];
    if (states[rayIdx].alive) {
        int slot = atomicAdd(d_count, 1);
        dst[slot] = rayIdx;
    }
}

// ── Normalize kernel ──────────────────────────────────────────────────────────

__global__ void k_normalize(vec3* buf, int n, float inv_ns)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = vec3(buf[i].x()*inv_ns, buf[i].y()*inv_ns, buf[i].z()*inv_ns);
}

// ── Host wavefront loop ───────────────────────────────────────────────────────

void wavefrontRender(int width, int tileHeight, int yOffset, int fullHeight, int ns,
                     camera& cam, sphere* d_spheres,
                     int numSpheres,
                     GBuffers& gb, vec3* d_output, int gpuId)
{
    int numRays = width * tileHeight;

    // Ray state + RNG
    RayState*    d_states; gpuCheck(cudaMalloc(&d_states, numRays * sizeof(RayState)));
    curandState* d_rng;    gpuCheck(cudaMalloc(&d_rng,    numRays * sizeof(curandState)));

    // Initialize RNG state properly to fix the silent cuda crash!
    k_initRNG<<<(numRays+127)/128, 128>>>(d_rng, numRays);
    gpuCheck(cudaDeviceSynchronize());

    // Material queues — fixed size, never reallocated
    WavefrontQueues Q;
    gpuCheck(cudaMalloc(&Q.lambertian, numRays * sizeof(int)));
    gpuCheck(cudaMalloc(&Q.dielectric, numRays * sizeof(int)));
    gpuCheck(cudaMalloc(&Q.metal,      numRays * sizeof(int)));
    gpuCheck(cudaMalloc(&Q.miss,       numRays * sizeof(int)));
    gpuCheck(cudaMalloc(&Q.d_counts,   4 * sizeof(int)));

    // Active list: two fixed raw buffers, ping-pong between them
    // This avoids ALL thrust device_vector reallocation / iterator invalidation
    int* d_activeA; gpuCheck(cudaMalloc(&d_activeA, numRays * sizeof(int)));
    int* d_activeB; gpuCheck(cudaMalloc(&d_activeB, numRays * sizeof(int)));
    int* d_compactCount; gpuCheck(cudaMalloc(&d_compactCount, sizeof(int)));

    dim3 blk2d(8, 8);
    dim3 grd2d((width+7)/8, (tileHeight+7)/8);

    for (int sample = 0; sample < ns; sample++) {
        // Generate primary rays
        k_generateRays<<<grd2d, blk2d>>>(
            d_states, d_rng, cam,
            width, tileHeight, fullHeight, yOffset, sample);
        gpuCheck(cudaDeviceSynchronize());

        // Fill activeA with [0 .. numRays-1]
        thrust::sequence(thrust::device_pointer_cast(d_activeA),
                         thrust::device_pointer_cast(d_activeA + numRays));

        int*  src = d_activeA;
        int*  dst = d_activeB;
        int   N   = numRays;

        bool firstBounce = true;

        while (N > 0) {
            // Reset queue counters
            gpuCheck(cudaMemset(Q.d_counts, 0, 4 * sizeof(int)));

            // Intersect all active rays, classify into queues
            k_intersect<<<(N+127)/128, 128>>>(
                d_states, src, N, d_spheres, numSpheres, Q, gb, firstBounce);
            gpuCheck(cudaDeviceSynchronize());

            firstBounce = false;

            // Read queue sizes
            int counts[4];
            gpuCheck(cudaMemcpy(counts, Q.d_counts, 4*sizeof(int),
                                cudaMemcpyDeviceToHost));

            // Shade each material queue
            if (counts[0] > 0)
                k_shadeLambertian<<<(counts[0]+127)/128, 128>>>(
                    d_states, Q.lambertian, counts[0], d_rng);
            if (counts[1] > 0)
                k_shadeDielectric<<<(counts[1]+127)/128, 128>>>(
                    d_states, Q.dielectric, counts[1], d_rng);
            if (counts[2] > 0)
                k_shadeMetal<<<(counts[2]+127)/128, 128>>>(
                    d_states, Q.metal, counts[2], d_rng);
            if (counts[3] > 0)
                k_shadeMiss<<<(counts[3]+127)/128, 128>>>(
                    d_states, Q.miss, counts[3]);
            gpuCheck(cudaDeviceSynchronize());

            // Stream compact: write surviving ray indices into dst
            gpuCheck(cudaMemset(d_compactCount, 0, sizeof(int)));
            k_compact<<<(N+127)/128, 128>>>(d_states, src, dst, N, d_compactCount);
            gpuCheck(cudaDeviceSynchronize());

            gpuCheck(cudaMemcpy(&N, d_compactCount, sizeof(int),
                                cudaMemcpyDeviceToHost));

            // Swap buffers
            int* tmp = src; src = dst; dst = tmp;
        }

        // Add this sample's radiance into output
        k_accumulate<<<(numRays+127)/128, 128>>>(d_states, d_output, numRays);
        gpuCheck(cudaDeviceSynchronize());
    }

    // Divide by sample count
    k_normalize<<<(numRays+127)/128, 128>>>(d_output, numRays, 1.0f / float(ns));
    gpuCheck(cudaDeviceSynchronize());

    // Cleanup
    cudaFree(d_states);
    cudaFree(d_rng);
    cudaFree(Q.lambertian); cudaFree(Q.dielectric);
    cudaFree(Q.metal);      cudaFree(Q.miss);
    cudaFree(Q.d_counts);
    cudaFree(d_activeA);
    cudaFree(d_activeB);
    cudaFree(d_compactCount);
}
