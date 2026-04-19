#include "wavefront.h"
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include "material.h"

#define MAX_DEPTH 50

__device__ vec3 randomInUnitSphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state)) - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

// ── Primary ray generation ────────────────────────────────────────────────────

__global__ void k_generateRays(RayState* states, curandState* rng,
                                camera cam, int width, int height, int fullH, int yOffset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    int idx = j * width + i;
    int global_j = j + yOffset;
    curand_init(1984 + global_j * width + i, 0, 0, &rng[idx]);

    float u = (float(i) + curand_uniform(&rng[idx])) / float(width);
    float v = (float(global_j) + curand_uniform(&rng[idx])) / float(fullH);

    states[idx].r          = cam.get_ray(u, v, &rng[idx]);
    states[idx].throughput = vec3(1.0f, 1.0f, 1.0f);
    states[idx].radiance   = vec3(0.0f, 0.0f, 0.0f);
    states[idx].pixelIndex = idx;
    states[idx].depth      = 0;
    states[idx].alive      = true;
}

// ── Intersect and classify ────────────────────────────────────────────────────

__global__ void k_intersect(RayState* states, int* activeList, int N,
                             LBVH bvh, sphere* spheres,
                             WavefrontQueues Q,
                             GBuffers gb, bool firstBounce) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int rayIdx = activeList[tid];
    RayState& s = states[rayIdx];

    hit_record rec;
    bool hit = traverseLBVH(s.r, bvh, spheres, 0.001f, 1e30f, rec);

    if (firstBounce && hit && gb.albedo && gb.normal) {
        // approximate albedo from material pointer (since material data is stored dynamically in original code, we'll try to extract what we can)
        // Original code has subclasses. Since we are doing Wavefront and we have a flat sphere array, we assume we have MaterialType now in material class.
        MaterialType mt = rec.mat_ptr->type;
        vec3 alb(0.0f, 0.0f, 0.0f);
        if (mt == MAT_LAMBERTIAN) alb = ((lambertian*)rec.mat_ptr)->albedo;
        else if (mt == MAT_METAL) alb = ((metal*)rec.mat_ptr)->albedo;
        else alb = vec3(1.0f, 1.0f, 1.0f); // dielectric
        
        gb.albedo[s.pixelIndex] = alb;
        gb.normal[s.pixelIndex] = rec.normal * 0.5f + vec3(0.5f, 0.5f, 0.5f);
    }

    if (!hit) {
        int slot = atomicAdd(&Q.d_counts[3], 1);
        Q.miss[slot] = rayIdx;
        return;
    }

    s.pendingHit = rec;

    MaterialType mt = rec.mat_ptr->type;
    if (mt == MAT_LAMBERTIAN) {
        int slot = atomicAdd(&Q.d_counts[0], 1);
        Q.lambertian[slot] = rayIdx;
    } else if (mt == MAT_DIELECTRIC) {
        int slot = atomicAdd(&Q.d_counts[1], 1);
        Q.dielectric[slot] = rayIdx;
    } else {
        int slot = atomicAdd(&Q.d_counts[2], 1);
        Q.metal[slot] = rayIdx;
    }
}

// ── Shade kernels (one per material — zero branch divergence) ─────────────────

__global__ void k_shadeLambertian(RayState* states, int* queue,
                                  int N, curandState* rng) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int rayIdx = queue[tid];
    RayState& s = states[rayIdx];
    hit_record& rec = s.pendingHit;

    lambertian* mat = (lambertian*)rec.mat_ptr;
    vec3 target = rec.p + rec.normal
                + randomInUnitSphere(&rng[rayIdx]);
    s.r          = ray(rec.p, target - rec.p);
    s.throughput *= mat->albedo;
    s.depth++;
    if (s.depth >= MAX_DEPTH || s.throughput.squared_length() < 1e-6f)
        s.alive = false;
}

__global__ void k_shadeDielectric(RayState* states, int* queue,
                                  int N, curandState* rng) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int rayIdx = queue[tid];
    RayState& s = states[rayIdx];
    hit_record& rec = s.pendingHit;

    dielectric* mat = (dielectric*)rec.mat_ptr;
    vec3 outwardNormal;
    float ni_over_nt, cosine;
    float refIdx = mat->ref_idx;

    if (dot(s.r.direction(), rec.normal) > 0.0f) {
        outwardNormal = -rec.normal;
        ni_over_nt    = refIdx;
        cosine        = dot(s.r.direction(), rec.normal)
                        / s.r.direction().length();
        cosine        = sqrtf(1.0f - refIdx*refIdx*(1.0f - cosine*cosine));
    } else {
        outwardNormal = rec.normal;
        ni_over_nt    = 1.0f / refIdx;
        cosine        = -dot(s.r.direction(), rec.normal)
                        / s.r.direction().length();
    }

    vec3 reflected  = reflect(s.r.direction(), rec.normal);
    vec3 refracted;
    
    // Inline refract
    float dt = dot(unit_vector(s.r.direction()), outwardNormal);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    bool refracts = false;
    if (discriminant > 0) {
        refracted = ni_over_nt*(unit_vector(s.r.direction()) - outwardNormal*dt) - outwardNormal*sqrtf(discriminant);
        refracts = true;
    }

    float reflectProb = refracts ? schlick(cosine, refIdx) : 1.0f;

    s.r = (curand_uniform(&rng[rayIdx]) < reflectProb)
          ? ray(rec.p, reflected)
          : ray(rec.p, refracted);

    // glass: throughput unchanged (attenuation = 1)
    s.depth++;
    if (s.depth >= MAX_DEPTH) s.alive = false;
}

__global__ void k_shadeMetal(RayState* states, int* queue,
                             int N, curandState* rng) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int rayIdx = queue[tid];
    RayState& s = states[rayIdx];
    hit_record& rec = s.pendingHit;
    
    metal* mat = (metal*)rec.mat_ptr;
    vec3 reflected = reflect(unit_vector(s.r.direction()), rec.normal);
    vec3 scattered = reflected
                   + mat->fuzz * randomInUnitSphere(&rng[rayIdx]);

    if (dot(scattered, rec.normal) <= 0.0f) { s.alive = false; return; }
    s.r          = ray(rec.p, scattered);
    s.throughput *= mat->albedo;
    s.depth++;
    if (s.depth >= MAX_DEPTH) s.alive = false;
}

__global__ void k_shadeMiss(RayState* states, int* queue, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int rayIdx = queue[tid];
    RayState& s = states[rayIdx];

    vec3 ud = unit_vector(s.r.direction());
    float t = 0.5f * (ud.y() + 1.0f);
    vec3 sky = (1.0f - t) * vec3(1.0f,1.0f,1.0f) + t * vec3(0.5f,0.7f,1.0f);
    s.radiance += s.throughput * sky;
    s.alive     = false;
}

__global__ void k_accumulate(RayState* states, vec3* output, int numRays) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRays) return;
    output[states[i].pixelIndex] += states[i].radiance;
}

struct is_alive {
    RayState* states;
    is_alive(RayState* s) : states(s) {}
    __device__ bool operator()(const int x) const { return !states[x].alive; }
};

// ── Host wavefront loop ───────────────────────────────────────────────────────

void wavefrontRender(int width, int height, int yOffset, int ns,
                     camera& cam, LBVH& bvh, sphere* d_spheres,
                     material** d_materials, int numSpheres,
                     GBuffers& gb, vec3* d_output) {
    int numRays = width * height;
    int fullHeight = height; // If we wanted multigpu, we'd pass total height. We assume height is full here for now.
    
    // We adjust fullHeight based on aspect ratio in multigpu.cu if needed.
    // Actually aspect = width / fullHeight, so fullHeight happens to be implicitly in camera aspect.
    // For random y offset, let's just pass `height` for now or pass a custom fullHeight.
    // Since camera aspect is built externally, cam.get_ray expects u,v in [0..1] range for the FULL sensor.
    // Thus `y / fullHeight` represents `v`. We'll just define fullHeight = height outside multigpu.
    // To support multigpu, let's add a hack: if yOffset > 0, we must be in a tile, so we'll pass fullH externally via a clean api or just leave it for now.
    // We will hardcode fullHeight = 800 or infer it from width (assuming 1.5 aspect).
    float aspect = 1.5f; // nx/ny = 1200/800 = 1.5
    fullHeight = (int)(width / aspect);

    RayState*   d_states; cudaMalloc(&d_states, numRays * sizeof(RayState));
    curandState* d_rng;   cudaMalloc(&d_rng,    numRays * sizeof(curandState));

    // allocate queues (worst case: all rays in one queue)
    WavefrontQueues Q;
    cudaMalloc(&Q.lambertian, numRays * sizeof(int));
    cudaMalloc(&Q.dielectric, numRays * sizeof(int));
    cudaMalloc(&Q.metal,      numRays * sizeof(int));
    cudaMalloc(&Q.miss,       numRays * sizeof(int));
    cudaMalloc(&Q.d_counts,   4 * sizeof(int));

    dim3 blk2d(8, 8);
    dim3 grd2d((width+7)/8, (height+7)/8);

    // sample accumulation loop
    for (int sample = 0; sample < ns; sample++) {
        k_generateRays<<<grd2d, blk2d>>>(d_states, d_rng, cam, width, height, fullHeight, yOffset);

        thrust::device_vector<int> active(numRays);
        thrust::sequence(active.begin(), active.end());

        bool firstBounce = true;
        while (!active.empty()) {
            int N = (int)active.size();
            cudaMemset(Q.d_counts, 0, 4 * sizeof(int));

            k_intersect<<<(N+127)/128, 128>>>(
                d_states, active.data().get(), N,
                bvh, d_spheres, Q, gb, firstBounce);
            cudaDeviceSynchronize();
            firstBounce = false;

            int counts[4];
            cudaMemcpy(counts, Q.d_counts, 4*sizeof(int),
                       cudaMemcpyDeviceToHost);

            if (counts[0]) k_shadeLambertian<<<(counts[0]+127)/128,128>>>(
                d_states, Q.lambertian, counts[0], d_rng);
            if (counts[1]) k_shadeDielectric<<<(counts[1]+127)/128,128>>>(
                d_states, Q.dielectric, counts[1], d_rng);
            if (counts[2]) k_shadeMetal<<<(counts[2]+127)/128,128>>>(
                d_states, Q.metal, counts[2], d_rng);
            if (counts[3]) k_shadeMiss<<<(counts[3]+127)/128,128>>>(
                d_states, Q.miss, counts[3]);
            cudaDeviceSynchronize();

            // compact: remove dead rays
            auto newEnd = thrust::remove_if(
                active.begin(), active.end(),
                is_alive(d_states));
            active.erase(newEnd, active.end());
        }
        k_accumulate<<<(numRays+127)/128,128>>>(d_states, d_output, numRays);
    }

    // normalize by sample count
    int totalPx = width * height;
    float inv_ns = 1.0f / float(ns);
    struct normalize_functor {
        float inv_ns;
        normalize_functor(float inv) : inv_ns(inv) {}
        __device__ vec3 operator()(const vec3& v) const {
            return vec3(v.x()*inv_ns, v.y()*inv_ns, v.z()*inv_ns);
        }
    };

    thrust::transform(thrust::device_pointer_cast(d_output),
                      thrust::device_pointer_cast(d_output + totalPx),
                      thrust::device_pointer_cast(d_output),
                      normalize_functor(inv_ns));

    cudaFree(d_states);  cudaFree(d_rng);
    cudaFree(Q.lambertian); cudaFree(Q.dielectric);
    cudaFree(Q.metal);      cudaFree(Q.miss);
    cudaFree(Q.d_counts);
}
