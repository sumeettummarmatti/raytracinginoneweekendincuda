#include "wavefront.h"
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include "material.h"
#include <cstdio>

#define MAX_DEPTH 50

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

__device__ vec3 randomInUnitSphere(curandState* s) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(s), curand_uniform(s), curand_uniform(s))
            - vec3(1.0f, 1.0f, 1.0f);
    } while (p.squared_length() >= 1.0f);
    return p;
}

// ── Primary ray generation ────────────────────────────────────────────────────
// FIX: fullHeight is the FULL image height (cfg.height), passed explicitly.
//      yOffset shifts tile rows into global image coordinates.
//      camera::get_ray() requires curandState for DOF — always provide it.

__global__ void k_generateRays(RayState* states, curandState* rng,
                                camera cam,
                                int width, int tileHeight,
                                int fullHeight, int yOffset,
                                int sampleSeed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= tileHeight) return;

    int localIdx  = j * width + i;
    int global_j  = j + yOffset;  // actual row in full image

    // Unique seed per pixel per sample — avoids correlated noise across samples
    curand_init(1984ULL + (unsigned long long)sampleSeed * width * fullHeight
                + (unsigned long long)global_j * width + i,
                0, 0, &rng[localIdx]);

    // FIX: u/v computed against FULL image dimensions
    float u = (float(i)        + curand_uniform(&rng[localIdx])) / float(width);
    float v = (float(global_j) + curand_uniform(&rng[localIdx])) / float(fullHeight);

    // FIX: always pass valid curandState — camera::get_ray needs it for DOF
    states[localIdx].r          = cam.get_ray(u, v, &rng[localIdx]);
    states[localIdx].throughput = vec3(1.0f, 1.0f, 1.0f);
    states[localIdx].radiance   = vec3(0.0f, 0.0f, 0.0f);
    states[localIdx].pixelIndex = localIdx;
    states[localIdx].depth      = 0;
    states[localIdx].alive      = true;
}

// ── Intersect and classify ────────────────────────────────────────────────────

__global__ void k_intersect(RayState* states, int* activeList, int N,
                             LBVH bvh, sphere* spheres,
                             WavefrontQueues Q,
                             GBuffers gb, bool firstBounce)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int      rayIdx = activeList[tid];
    RayState& s     = states[rayIdx];

    hit_record rec;
    // FIX: traverseLBVH is defined inline in lbvh.h — reads rec.mat correctly
    bool hit = traverseLBVH(s.r, bvh, spheres, 0.001f, 1e30f, rec);

    if (firstBounce && hit && gb.albedo && gb.normal) {
        MaterialData& mat = rec.mat;
        vec3 alb;
        if      (mat.type == MAT_LAMBERTIAN) alb = mat.albedo;
        else if (mat.type == MAT_METAL)      alb = mat.albedo;
        else                                  alb = vec3(1.0f, 1.0f, 1.0f); // glass
        gb.albedo[s.pixelIndex] = alb;
        gb.normal[s.pixelIndex] = rec.normal * 0.5f + vec3(0.5f, 0.5f, 0.5f);
    }

    if (!hit) {
        int slot = atomicAdd(&Q.d_counts[3], 1);
        Q.miss[slot] = rayIdx;
        return;
    }

    s.pendingHit = rec;

    // FIX: classify by rec.mat.type (MaterialData), NOT mat_ptr->type
    MaterialType mt = rec.mat.type;
    if (mt == MAT_LAMBERTIAN) {
        Q.lambertian[atomicAdd(&Q.d_counts[0], 1)] = rayIdx;
    } else if (mt == MAT_DIELECTRIC) {
        Q.dielectric[atomicAdd(&Q.d_counts[1], 1)] = rayIdx;
    } else {
        Q.metal[atomicAdd(&Q.d_counts[2], 1)] = rayIdx;
    }
}

// ── Shade: Lambertian ─────────────────────────────────────────────────────────

__global__ void k_shadeLambertian(RayState* states, int* queue,
                                   int N, curandState* rng)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int       rayIdx = queue[tid];
    RayState& s      = states[rayIdx];
    hit_record& rec  = s.pendingHit;
    MaterialData& mat = rec.mat;

    vec3 target = rec.p + rec.normal + randomInUnitSphere(&rng[rayIdx]);
    s.r          = ray(rec.p, target - rec.p);
    s.throughput *= mat.albedo;
    s.depth++;
    if (s.depth >= MAX_DEPTH || s.throughput.squared_length() < 1e-6f)
        s.alive = false;
}

// ── Shade: Dielectric ─────────────────────────────────────────────────────────

__global__ void k_shadeDielectric(RayState* states, int* queue,
                                   int N, curandState* rng)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int       rayIdx = queue[tid];
    RayState& s      = states[rayIdx];
    hit_record& rec  = s.pendingHit;
    float refIdx     = rec.mat.ref_idx;

    vec3  outwardNormal;
    float ni_over_nt, cosine;

    if (dot(s.r.direction(), rec.normal) > 0.0f) {
        outwardNormal = -rec.normal;
        ni_over_nt    = refIdx;
        cosine        = dot(s.r.direction(), rec.normal) / s.r.direction().length();
        cosine        = sqrtf(1.0f - refIdx * refIdx * (1.0f - cosine * cosine));
    } else {
        outwardNormal = rec.normal;
        ni_over_nt    = 1.0f / refIdx;
        cosine        = -dot(s.r.direction(), rec.normal) / s.r.direction().length();
    }

    vec3  reflected = reflect(s.r.direction(), rec.normal);
    vec3  refracted;

    vec3  uv = unit_vector(s.r.direction());
    float dt = dot(uv, outwardNormal);
    float disc = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    bool  canRefract = disc > 0.0f;
    if (canRefract)
        refracted = ni_over_nt * (uv - outwardNormal * dt) - outwardNormal * sqrtf(disc);

    float r0 = (1.0f - refIdx) / (1.0f + refIdx);
    r0 *= r0;
    float base = 1.0f - cosine;
    float reflectProb = canRefract
        ? r0 + (1.0f - r0) * (base * base * base * base * base)
        : 1.0f;

    s.r = (curand_uniform(&rng[rayIdx]) < reflectProb)
        ? ray(rec.p, reflected)
        : ray(rec.p, refracted);

    // glass: throughput unchanged (white attenuation)
    s.depth++;
    if (s.depth >= MAX_DEPTH) s.alive = false;
}

// ── Shade: Metal ──────────────────────────────────────────────────────────────

__global__ void k_shadeMetal(RayState* states, int* queue,
                              int N, curandState* rng)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int       rayIdx = queue[tid];
    RayState& s      = states[rayIdx];
    hit_record& rec  = s.pendingHit;
    MaterialData& mat = rec.mat;

    vec3 reflected = reflect(unit_vector(s.r.direction()), rec.normal);
    vec3 scattered = reflected + mat.fuzz * randomInUnitSphere(&rng[rayIdx]);

    if (dot(scattered, rec.normal) <= 0.0f) { s.alive = false; return; }
    s.r          = ray(rec.p, scattered);
    s.throughput *= mat.albedo;
    s.depth++;
    if (s.depth >= MAX_DEPTH) s.alive = false;
}

// ── Shade: Miss (sky) ─────────────────────────────────────────────────────────

__global__ void k_shadeMiss(RayState* states, int* queue, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int       rayIdx = queue[tid];
    RayState& s      = states[rayIdx];

    vec3  ud  = unit_vector(s.r.direction());
    float t   = 0.5f * (ud.y() + 1.0f);
    vec3  sky = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
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

// ── Thrust helpers ────────────────────────────────────────────────────────────

struct is_dead {
    RayState* states;
    is_dead(RayState* s) : states(s) {}
    __device__ bool operator()(int x) const { return !states[x].alive; }
};

struct normalize_functor {
    float inv_ns;
    normalize_functor(float f) : inv_ns(f) {}
    __device__ vec3 operator()(const vec3& v) const {
        return vec3(v.x() * inv_ns, v.y() * inv_ns, v.z() * inv_ns);
    }
};

// ── Host wavefront loop ───────────────────────────────────────────────────────
// FIX: fullHeight passed explicitly from cfg.height — no hardcoded aspect ratio.

void wavefrontRender(int width, int tileHeight, int yOffset, int fullHeight, int ns,
                     camera& cam, LBVH& bvh, sphere* d_spheres,
                     int numSpheres,
                     GBuffers& gb, vec3* d_output, int gpuId)
{
    int numRays = width * tileHeight;

    RayState*    d_states; cudaMalloc(&d_states, numRays * sizeof(RayState));
    curandState* d_rng;    cudaMalloc(&d_rng,    numRays * sizeof(curandState));

    // allocate queues — worst case all rays in one bucket
    WavefrontQueues Q;
    cudaMalloc(&Q.lambertian, numRays * sizeof(int));
    cudaMalloc(&Q.dielectric, numRays * sizeof(int));
    cudaMalloc(&Q.metal,      numRays * sizeof(int));
    cudaMalloc(&Q.miss,       numRays * sizeof(int));
    cudaMalloc(&Q.d_counts,   4 * sizeof(int));

    dim3 blk2d(8, 8);
    dim3 grd2d((width + 7) / 8, (tileHeight + 7) / 8);

    thrust::device_vector<int> active(numRays);

    // ── per-sample loop ───────────────────────────────────────────────────────
    for (int sample = 0; sample < ns; sample++) {
        std::cerr << "[GPU " << gpuId << "] sample " << sample + 1 << "/" << ns << "\r";
        std::cerr.flush();

        // FIX: pass fullHeight and sampleSeed so UV mapping is correct
        k_generateRays<<<grd2d, blk2d>>>(
            d_states, d_rng, cam,
            width, tileHeight, fullHeight, yOffset,
            sample);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        thrust::sequence(active.begin(), active.end());
        int N = numRays;

        bool firstBounce = true;
        while (N > 0) {
            cudaMemset(Q.d_counts, 0, 4 * sizeof(int));

            k_intersect<<<(N + 127) / 128, 128>>>(
                d_states, active.data().get(), N,
                bvh, d_spheres, Q, gb, firstBounce);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            int counts[4];
            cudaMemcpy(counts, Q.d_counts, 4 * sizeof(int), cudaMemcpyDeviceToHost);

            if (firstBounce) {
                std::cerr << "\n[GPU " << gpuId << "] bounce1 hits="
                          << counts[0] + counts[1] + counts[2]
                          << " miss=" << counts[3] << "\n";
                std::cerr.flush();
            }
            firstBounce = false;

            if (counts[0]) k_shadeLambertian<<<(counts[0] + 127) / 128, 128>>>(
                d_states, Q.lambertian, counts[0], d_rng);
            if (counts[1]) k_shadeDielectric<<<(counts[1] + 127) / 128, 128>>>(
                d_states, Q.dielectric, counts[1], d_rng);
            if (counts[2]) k_shadeMetal<<<(counts[2] + 127) / 128, 128>>>(
                d_states, Q.metal, counts[2], d_rng);
            if (counts[3]) k_shadeMiss<<<(counts[3] + 127) / 128, 128>>>(
                d_states, Q.miss, counts[3]);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            // stream compact — drop dead rays
            auto newEnd = thrust::remove_if(
                active.begin(), active.begin() + N, is_dead(d_states));
            N = (int)(newEnd - active.begin());
        }

        k_accumulate<<<(numRays + 127) / 128, 128>>>(d_states, d_output, numRays);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    std::cerr << "\n[GPU " << gpuId << "] done " << ns << " samples\n";

    // normalize
    thrust::transform(
        thrust::device_pointer_cast(d_output),
        thrust::device_pointer_cast(d_output + numRays),
        thrust::device_pointer_cast(d_output),
        normalize_functor(1.0f / float(ns)));

    cudaFree(d_states); cudaFree(d_rng);
    cudaFree(Q.lambertian); cudaFree(Q.dielectric);
    cudaFree(Q.metal);      cudaFree(Q.miss);
    cudaFree(Q.d_counts);
}
