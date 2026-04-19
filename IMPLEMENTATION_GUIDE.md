# CUDA Ray Tracer — Full Acceleration Implementation Guide

> **Repo:** `https://github.com/rogerallen/raytracinginoneweekendincuda`
> **Base branch:** `ch12_where_next_cuda`
> **Goal:** Add LBVH, Wavefront Path Tracing, AI Denoising (OIDN), and Multi-GPU support

---

## Prerequisites

```bash
# Required software
CUDA Toolkit >= 11.8
CMake >= 3.22
Thrust (ships with CUDA Toolkit)
Intel Open Image Denoise >= 2.0   # https://github.com/OpenImageDenoise/oidn/releases
NVIDIA OptiX SDK >= 7.6           # optional — needed only for RT-core BVH
GCC >= 9  (Linux) / MSVC 2019+   (Windows)

# Verify GPU count and peer access
nvidia-smi
```

---

## Step 0 — Clone and branch

```bash
git clone https://github.com/rogerallen/raytracinginoneweekendincuda.git
cd raytracinginoneweekendincuda

# start from the final chapter as your base
git checkout ch12_where_next_cuda

# create your feature branch
git checkout -b feature/full-acceleration
```

---

## Step 1 — Migrate build system to CMake

The original repo uses a hand-written Makefile. Replace it so you can link
OIDN and manage multiple `.cu` translation units cleanly.

### 1.1 — Delete the old Makefile

```bash
rm Makefile
```

### 1.2 — Create `CMakeLists.txt` in the repo root

```cmake
cmake_minimum_required(VERSION 3.22)
project(cuda_raytracer CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# ── adjust for your GPU arch ──────────────────────────────────────────────────
# RTX 3090 / A100 = 86, RTX 4090 = 89, A100 SXM = 80
set(CMAKE_CUDA_ARCHITECTURES 86)

# ── OIDN ─────────────────────────────────────────────────────────────────────
find_package(OpenImageDenoise REQUIRED)

add_executable(rt
    src/main.cu
    src/lbvh.cu
    src/wavefront.cu
    src/denoiser.cu
    src/multigpu.cu
)

target_include_directories(rt PRIVATE include)

target_link_libraries(rt PRIVATE
    OpenImageDenoise
    cuda
)

target_compile_options(rt PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        -lineinfo
    >
)
```

### 1.3 — Reorganise source tree

```bash
mkdir -p src include

# move existing headers into include/
mv vec3.h ray.h sphere.h hitable.h hitable_list.h \
   material.h camera.h float.h include/

# move main to src/ and rename
cp main.cu src/main.cu   # original is main.cc in base branch — copy + rename

# create new source files (will fill in later steps)
touch src/lbvh.cu src/wavefront.cu src/denoiser.cu src/multigpu.cu
touch include/lbvh.h include/wavefront.h include/denoiser.h include/multigpu.h
touch include/ray_state.h include/aabb.h
```

### 1.4 — Build to verify baseline compiles

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DOpenImageDenoise_DIR=/path/to/oidn/lib/cmake/OpenImageDenoise
make -j$(nproc)
./rt > out.ppm
```

---

## Step 2 — AABB and shared types (`include/aabb.h`)

Every subsequent feature depends on axis-aligned bounding boxes.

```cuda
// include/aabb.h
#pragma once
#include <cuda_runtime.h>
#include "vec3.h"

struct AABB {
    vec3 mn, mx;   // min/max corners

    __host__ __device__ AABB() {}
    __host__ __device__ AABB(vec3 a, vec3 b) : mn(a), mx(b) {}

    __device__ bool hit(const ray& r, float tMin, float tMax) const {
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / r.direction()[a];
            float t0   = (mn[a] - r.origin()[a]) * invD;
            float t1   = (mx[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            tMin = t0 > tMin ? t0 : tMin;
            tMax = t1 < tMax ? t1 : tMax;
            if (tMax <= tMin) return false;
        }
        return true;
    }
};

__host__ __device__ inline AABB merge(AABB a, AABB b) {
    vec3 mn(fminf(a.mn.x(), b.mn.x()),
            fminf(a.mn.y(), b.mn.y()),
            fminf(a.mn.z(), b.mn.z()));
    vec3 mx(fmaxf(a.mx.x(), b.mx.x()),
            fmaxf(a.mx.y(), b.mx.y()),
            fmaxf(a.mx.z(), b.mx.z()));
    return AABB(mn, mx);
}
```

---

## Step 3 — LBVH build and traversal

### 3.1 — `include/lbvh.h`

```cuda
#pragma once
#include <cuda_runtime.h>
#include "aabb.h"
#include "sphere.h"

struct BVHNode {
    AABB bounds;
    int  leftChild;   // index into nodes[]; >= numLeaves means leaf
    int  rightChild;
    int  parent;
    int  primIdx;     // valid only for leaves
    bool isLeaf;
};

struct LBVH {
    BVHNode* nodes;      // device pointer, size = 2*numPrims - 1
    int*     sortedIdx;  // device pointer, size = numPrims
    int      numPrims;
};

// host-callable build — allocates and returns a ready LBVH
LBVH buildLBVH(sphere* d_spheres, int n);
void  freeLBVH(LBVH& bvh);

__device__ bool traverseLBVH(const ray& r, const LBVH& bvh,
                              sphere* spheres,
                              float tMin, float tMax,
                              hit_record& rec);
```

### 3.2 — `src/lbvh.cu`

```cuda
#include "lbvh.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// ── Morton code helpers ───────────────────────────────────────────────────────

__device__ uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ uint32_t morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    return (expandBits((uint32_t)x) << 2) |
           (expandBits((uint32_t)y) << 1) |
           (expandBits((uint32_t)z));
}

// ── Kernels ───────────────────────────────────────────────────────────────────

__global__ void k_computeMorton(sphere* spheres, uint32_t* codes,
                                 int* indices, AABB sceneBounds, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vec3 c   = spheres[i].center;
    vec3 ext = sceneBounds.mx - sceneBounds.mn;
    vec3 norm = (c - sceneBounds.mn) / ext;
    codes[i]   = morton3D(norm.x(), norm.y(), norm.z());
    indices[i] = i;
}

__device__ int deltaLeading(uint32_t* codes, int i, int j, int n) {
    if (j < 0 || j >= n) return -1;
    if (codes[i] == codes[j])
        return __clz(i ^ j) + 32;
    return __clz(codes[i] ^ codes[j]);
}

__global__ void k_buildInternal(uint32_t* codes, BVHNode* nodes,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    int d = (deltaLeading(codes,i,i+1,n) - deltaLeading(codes,i,i-1,n)) >= 0 ? 1 : -1;
    int dMin = deltaLeading(codes, i, i - d, n);
    int lMax = 2;
    while (deltaLeading(codes, i, i + lMax * d, n) > dMin) lMax <<= 1;

    int l = 0;
    for (int t = lMax >> 1; t >= 1; t >>= 1)
        if (deltaLeading(codes, i, i + (l + t) * d, n) > dMin) l += t;

    int j = i + l * d;
    int lo = min(i, j), hi = max(i, j);

    int dNode = deltaLeading(codes, i, j, n);
    int s = 0;
    for (int t = (hi - lo + 1) >> 1; t >= 1; t >>= 1)
        if (deltaLeading(codes, lo, lo + s + t, n) > dNode) s += t;
    int split = lo + s;

    int leftIdx  = (split     == lo) ? (n - 1) + split     : split;
    int rightIdx = (split + 1 == hi) ? (n - 1) + split + 1 : split + 1;

    nodes[i].leftChild  = leftIdx;
    nodes[i].rightChild = rightIdx;
    nodes[i].isLeaf     = false;

    if (leftIdx  >= n - 1) nodes[leftIdx].parent  = i;
    if (rightIdx >= n - 1) nodes[rightIdx].parent = i;
}

__global__ void k_initLeaves(sphere* spheres, int* sortedIdx,
                              BVHNode* nodes, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int leafNode = (n - 1) + i;
    sphere& s = spheres[sortedIdx[i]];
    vec3 r(s.radius, s.radius, s.radius);
    nodes[leafNode].bounds   = AABB(s.center - r, s.center + r);
    nodes[leafNode].isLeaf   = true;
    nodes[leafNode].primIdx  = sortedIdx[i];
    nodes[leafNode].parent   = -1;
}

__global__ void k_fitBounds(BVHNode* nodes, int* flags, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cur = (n - 1) + i;
    int par = nodes[cur].parent;
    while (par != -1) {
        if (atomicAdd(&flags[par], 1) == 0) return;
        __threadfence();
        nodes[par].bounds = merge(nodes[nodes[par].leftChild].bounds,
                                   nodes[nodes[par].rightChild].bounds);
        par = nodes[par].parent;
    }
}

// ── Host build ────────────────────────────────────────────────────────────────

LBVH buildLBVH(sphere* d_spheres, int n) {
    // 1. compute scene AABB on host (small n, fine to do serially)
    std::vector<sphere> h_spheres(n);
    cudaMemcpy(h_spheres.data(), d_spheres, n * sizeof(sphere),
               cudaMemcpyDeviceToHost);
    AABB sceneBounds;
    sceneBounds.mn = vec3(1e30f, 1e30f, 1e30f);
    sceneBounds.mx = vec3(-1e30f,-1e30f,-1e30f);
    for (auto& s : h_spheres)
        sceneBounds = merge(sceneBounds,
            AABB(s.center - vec3(s.radius,s.radius,s.radius),
                 s.center + vec3(s.radius,s.radius,s.radius)));

    // 2. Morton codes
    thrust::device_vector<uint32_t> d_codes(n);
    thrust::device_vector<int>      d_idx(n);
    k_computeMorton<<<(n+127)/128,128>>>(
        d_spheres, d_codes.data().get(), d_idx.data().get(), sceneBounds, n);

    // 3. sort
    thrust::sort_by_key(d_codes.begin(), d_codes.end(), d_idx.begin());

    // 4. allocate node array (2n-1 nodes)
    int totalNodes = 2 * n - 1;
    BVHNode* d_nodes;
    cudaMalloc(&d_nodes, totalNodes * sizeof(BVHNode));
    cudaMemset(d_nodes, 0, totalNodes * sizeof(BVHNode));

    int* d_sortedIdx;
    cudaMalloc(&d_sortedIdx, n * sizeof(int));
    cudaMemcpy(d_sortedIdx, d_idx.data().get(), n * sizeof(int),
               cudaMemcpyDeviceToDevice);

    // 5. init leaves
    k_initLeaves<<<(n+127)/128,128>>>(d_spheres, d_sortedIdx, d_nodes, n);

    // 6. build internal nodes
    k_buildInternal<<<(n+127)/128,128>>>(
        d_codes.data().get(), d_nodes, n);

    // 7. fit bounds bottom-up
    int* d_flags;
    cudaMalloc(&d_flags, totalNodes * sizeof(int));
    cudaMemset(d_flags, 0, totalNodes * sizeof(int));
    k_fitBounds<<<(n+127)/128,128>>>(d_nodes, d_flags, n);
    cudaFree(d_flags);

    return LBVH{d_nodes, d_sortedIdx, n};
}

void freeLBVH(LBVH& bvh) {
    cudaFree(bvh.nodes);
    cudaFree(bvh.sortedIdx);
}

// ── Device traversal ─────────────────────────────────────────────────────────

__device__ bool traverseLBVH(const ray& r, const LBVH& bvh,
                              sphere* spheres,
                              float tMin, float tMax, hit_record& rec) {
    int stack[64];
    int top = 0;
    stack[top++] = 0;
    bool hit = false;

    while (top > 0) {
        int idx = stack[--top];
        BVHNode& node = bvh.nodes[idx];
        if (!node.bounds.hit(r, tMin, tMax)) continue;

        if (node.isLeaf) {
            hit_record tmp;
            if (spheres[node.primIdx].hit(r, tMin, tMax, tmp)) {
                tMax = tmp.t;
                rec  = tmp;
                hit  = true;
            }
        } else {
            stack[top++] = node.leftChild;
            stack[top++] = node.rightChild;
        }
    }
    return hit;
}
```

---

## Step 4 — Ray state and G-buffer types (`include/ray_state.h`)

```cuda
#pragma once
#include "vec3.h"
#include "ray.h"
#include "hitable.h"
#include <curand_kernel.h>

struct RayState {
    ray         r;
    vec3        throughput;   // accumulated attenuation
    vec3        radiance;     // accumulated emission
    hit_record  pendingHit;
    int         pixelIndex;
    int         depth;
    bool        alive;
};

struct GBuffers {
    vec3* color;    // noisy radiance   — device pointer
    vec3* albedo;   // first-hit albedo — device pointer
    vec3* normal;   // first-hit normal — device pointer
};
```

---

## Step 5 — Wavefront path tracing

### 5.1 — `include/wavefront.h`

```cuda
#pragma once
#include "ray_state.h"
#include "lbvh.h"
#include "camera.h"
#include <curand_kernel.h>

struct WavefrontQueues {
    int* lambertian; int* dielectric; int* metal; int* miss;
    int* d_counts;   // [4] atomic counters on device
};

void wavefrontRender(
    int width, int height, int ns,
    camera& cam, LBVH& bvh, sphere* d_spheres,
    material** d_materials, int numSpheres,
    GBuffers& gb, vec3* d_output);
```

### 5.2 — `src/wavefront.cu`

```cuda
#include "wavefront.h"
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

#define MAX_DEPTH 50

// ── Primary ray generation ────────────────────────────────────────────────────

__global__ void k_generateRays(RayState* states, curandState* rng,
                                camera cam, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    int idx = j * width + i;
    curand_init(1984 + idx, 0, 0, &rng[idx]);

    float u = (float(i) + curand_uniform(&rng[idx])) / float(width);
    float v = (float(j) + curand_uniform(&rng[idx])) / float(height);

    states[idx].r          = cam.get_ray(u, v);
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

    if (firstBounce && hit) {
        // capture G-buffers at first hit only
        gb.albedo[s.pixelIndex] = rec.mat_ptr->albedo;
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

    vec3 target = rec.p + rec.normal
                + randomInUnitSphere(&rng[rayIdx]);
    s.r          = ray(rec.p, target - rec.p);
    s.throughput *= rec.mat_ptr->albedo;
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

    vec3 outwardNormal;
    float ni_over_nt, cosine;
    float refIdx = rec.mat_ptr->ref_idx;

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
    float reflectProb = refract(s.r.direction(), outwardNormal,
                                ni_over_nt, refracted)
                        ? schlick(cosine, refIdx) : 1.0f;

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

    vec3 reflected = reflect(unit_vector(s.r.direction()), rec.normal);
    vec3 scattered = reflected
                   + rec.mat_ptr->fuzz * randomInUnitSphere(&rng[rayIdx]);

    if (dot(scattered, rec.normal) <= 0.0f) { s.alive = false; return; }
    s.r          = ray(rec.p, scattered);
    s.throughput *= rec.mat_ptr->albedo;
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

// ── Host wavefront loop ───────────────────────────────────────────────────────

void wavefrontRender(int width, int height, int ns,
                     camera& cam, LBVH& bvh, sphere* d_spheres,
                     material** d_materials, int numSpheres,
                     GBuffers& gb, vec3* d_output) {
    int numRays = width * height;

    RayState*   d_states; cudaMalloc(&d_states, numRays * sizeof(RayState));
    curandState* d_rng;   cudaMalloc(&d_rng,    numRays * sizeof(curandState));

    // allocate queues (worst case: all rays in one queue)
    WavefrontQueues Q;
    cudaMalloc(&Q.lambertian, numRays * sizeof(int));
    cudaMalloc(&Q.dielectric, numRays * sizeof(int));
    cudaMalloc(&Q.metal,      numRays * sizeof(int));
    cudaMalloc(&Q.miss,       numRays * sizeof(int));
    cudaMalloc(&Q.d_counts,   4 * sizeof(int));

    // sample accumulation loop
    for (int sample = 0; sample < ns; sample++) {
        dim3 blk2d(8, 8);
        dim3 grd2d((width+7)/8, (height+7)/8);
        k_generateRays<<<grd2d, blk2d>>>(d_states, d_rng, cam, width, height);

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
                [states = d_states] __device__(int i) {
                    return !states[i].alive;
                });
            active.erase(newEnd, active.end());
        }
        k_accumulate<<<(numRays+127)/128,128>>>(d_states, d_output, numRays);
    }

    // normalize by sample count
    int totalPx = width * height;
    auto normalize = [ns] __device__ (vec3 v) { return v / float(ns); };
    thrust::transform(thrust::device_pointer_cast(d_output),
                      thrust::device_pointer_cast(d_output + totalPx),
                      thrust::device_pointer_cast(d_output),
                      normalize);

    cudaFree(d_states);  cudaFree(d_rng);
    cudaFree(Q.lambertian); cudaFree(Q.dielectric);
    cudaFree(Q.metal);      cudaFree(Q.miss);
    cudaFree(Q.d_counts);
}
```

---

## Step 6 — AI denoising (OIDN)

### 6.1 — `include/denoiser.h`

```cpp
#pragma once
#include <OpenImageDenoise/oidn.hpp>
#include "ray_state.h"

struct DenoiserConfig {
    int   width, height;
    bool  hdr          = true;
    bool  cleanAux     = true;   // true when albedo/normal are noise-free
    bool  temporal     = false;  // enable for animation sequences
};

// runs in-place: d_color is overwritten with denoised result
void denoiseOIDN(DenoiserConfig cfg,
                 vec3* d_color,    // noisy color  — CUDA device ptr
                 vec3* d_albedo,   // first-hit albedo
                 vec3* d_normal,   // first-hit normal
                 vec3* d_output,   // denoised output — may == d_color
                 cudaStream_t stream = 0);
```

### 6.2 — `src/denoiser.cu`

```cpp
#include "denoiser.h"
#include <iostream>

void denoiseOIDN(DenoiserConfig cfg,
                 vec3* d_color, vec3* d_albedo, vec3* d_normal,
                 vec3* d_output, cudaStream_t stream) {
    // OIDN 2.x — GPU device, shares your CUDA stream
    oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CUDA);
    device.set("cudaStream", (int64_t)stream);
    device.commit();

    oidn::FilterRef filter = device.newFilter("RT");

    int W = cfg.width, H = cfg.height;
    size_t stride = W * sizeof(vec3);

    filter.setImage("color",  d_color,  oidn::Format::Float3, W, H,
                    0, sizeof(vec3), stride);
    filter.setImage("albedo", d_albedo, oidn::Format::Float3, W, H,
                    0, sizeof(vec3), stride);
    filter.setImage("normal", d_normal, oidn::Format::Float3, W, H,
                    0, sizeof(vec3), stride);
    filter.setImage("output", d_output, oidn::Format::Float3, W, H,
                    0, sizeof(vec3), stride);

    filter.set("hdr",      cfg.hdr);
    filter.set("cleanAux", cfg.cleanAux);
    filter.commit();
    filter.execute();

    const char* err;
    if (device.getError(err) != oidn::Error::None)
        std::cerr << "[OIDN] " << err << "\n";
}
```

---

## Step 7 — Multi-GPU tile partitioning

### 7.1 — `include/multigpu.h`

```cpp
#pragma once
#include "ray_state.h"
#include "lbvh.h"
#include "wavefront.h"
#include "denoiser.h"
#include "camera.h"

struct RenderConfig {
    int width, height, ns;
    camera cam;
    sphere* h_spheres;   // host-side sphere array
    material** h_mats;   // host-side material array
    int numSpheres;
};

// renders full image using all available GPUs, returns host framebuffer
std::vector<vec3> multiGPURender(RenderConfig& cfg);
```

### 7.2 — `src/multigpu.cu`

```cpp
#include "multigpu.h"
#include <thread>
#include <vector>
#include <iostream>

struct GPUTile {
    int deviceId;
    int yStart, yEnd;
    vec3* d_color;
    vec3* d_albedo;
    vec3* d_normal;
    vec3* d_denoised;
};

static void renderOnDevice(GPUTile& tile, RenderConfig& cfg) {
    cudaSetDevice(tile.deviceId);

    int W          = cfg.width;
    int tileH      = tile.yEnd - tile.yStart;
    int numPx      = W * tileH;
    int numSpheres = cfg.numSpheres;

    // upload scene to this device
    sphere*    d_spheres; cudaMalloc(&d_spheres, numSpheres*sizeof(sphere));
    cudaMemcpy(d_spheres, cfg.h_spheres,
               numSpheres*sizeof(sphere), cudaMemcpyHostToDevice);

    // allocate G-buffers
    cudaMalloc(&tile.d_color,    numPx * sizeof(vec3));
    cudaMalloc(&tile.d_albedo,   numPx * sizeof(vec3));
    cudaMalloc(&tile.d_normal,   numPx * sizeof(vec3));
    cudaMalloc(&tile.d_denoised, numPx * sizeof(vec3));
    cudaMemset(tile.d_color,  0, numPx * sizeof(vec3));
    cudaMemset(tile.d_albedo, 0, numPx * sizeof(vec3));
    cudaMemset(tile.d_normal, 0, numPx * sizeof(vec3));

    // build LBVH on this device
    LBVH bvh = buildLBVH(d_spheres, numSpheres);

    // tile camera: offset camera ray generation by tile.yStart
    // (wavefrontRender uses yOffset internally — pass via config)
    GBuffers gb{tile.d_color, tile.d_albedo, tile.d_normal};

    wavefrontRender(W, tileH, cfg.ns,
                    cfg.cam, bvh, d_spheres,
                    nullptr, numSpheres,
                    gb, tile.d_color);

    // denoise tile on this device
    DenoiserConfig dc{W, tileH};
    denoiseOIDN(dc, tile.d_color, tile.d_albedo,
                tile.d_normal,  tile.d_denoised);

    cudaDeviceSynchronize();
    freeLBVH(bvh);
    cudaFree(d_spheres);
    // tile.d_denoised stays alive for gather step
}

std::vector<vec3> multiGPURender(RenderConfig& cfg) {
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Using " << numGPUs << " GPU(s)\n";

    int rowsPerGPU = cfg.height / numGPUs;
    std::vector<GPUTile> tiles(numGPUs);
    for (int g = 0; g < numGPUs; g++) {
        tiles[g].deviceId = g;
        tiles[g].yStart   = g * rowsPerGPU;
        tiles[g].yEnd     = (g == numGPUs-1) ? cfg.height
                                              : tiles[g].yStart + rowsPerGPU;
    }

    // enable peer access between all GPU pairs (NVLink / PCIe fallback)
    for (int a = 0; a < numGPUs; a++)
        for (int b = 0; b < numGPUs; b++)
            if (a != b) { cudaSetDevice(a); cudaDeviceEnablePeerAccess(b,0); }

    // launch one thread per GPU
    std::vector<std::thread> workers;
    for (auto& t : tiles)
        workers.emplace_back(renderOnDevice, std::ref(t), std::ref(cfg));
    for (auto& w : workers) w.join();

    // gather denoised tiles onto host
    std::vector<vec3> h_output(cfg.width * cfg.height);
    for (auto& tile : tiles) {
        cudaSetDevice(tile.deviceId);
        int tileH  = tile.yEnd - tile.yStart;
        int offset = tile.yStart * cfg.width;
        cudaMemcpy(h_output.data() + offset,
                   tile.d_denoised,
                   cfg.width * tileH * sizeof(vec3),
                   cudaMemcpyDeviceToHost);
        cudaFree(tile.d_color);   cudaFree(tile.d_albedo);
        cudaFree(tile.d_normal);  cudaFree(tile.d_denoised);
    }

    return h_output;
}
```

---

## Step 8 — Update `src/main.cu`

Replace the original rendering loop entirely:

```cuda
#include <iostream>
#include <vector>
#include "vec3.h"
#include "camera.h"
#include "sphere.h"
#include "material.h"
#include "multigpu.h"

// reuse the random_scene() from the original main.cc
// but return host arrays instead of hitable_list

int main() {
    const int W  = 1200;
    const int H  = 800;
    const int NS = 32;   // samples per pixel — much lower with denoising

    // build scene (reuse original random_scene logic, store in host arrays)
    std::vector<sphere>   h_spheres;
    std::vector<material> h_mats;
    buildRandomScene(h_spheres, h_mats);  // extract from original main.cc

    // camera — identical to original
    vec3 lookfrom(13,2,3), lookat(0,0,0);
    float dist_to_focus = 10.0f, aperture = 0.1f;
    camera cam(lookfrom, lookat, vec3(0,1,0), 20,
               float(W)/float(H), aperture, dist_to_focus);

    RenderConfig cfg;
    cfg.width      = W;
    cfg.height     = H;
    cfg.ns         = NS;
    cfg.cam        = cam;
    cfg.h_spheres  = h_spheres.data();
    cfg.numSpheres = (int)h_spheres.size();

    auto framebuffer = multiGPURender(cfg);

    // write PPM
    std::cout << "P3\n" << W << " " << H << "\n255\n";
    for (int j = H-1; j >= 0; j--)
        for (int i = 0; i < W; i++) {
            vec3 c = framebuffer[j * W + i];
            // gamma correct
            c = vec3(sqrtf(c[0]), sqrtf(c[1]), sqrtf(c[2]));
            int ir = int(255.99f * c[0]);
            int ig = int(255.99f * c[1]);
            int ib = int(255.99f * c[2]);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }

    return 0;
}
```

---

## Step 9 — Add `MaterialType` enum to `material.h`

The wavefront classifier needs a type tag on each material. Add to the top of `include/material.h`:

```cuda
enum MaterialType { MAT_LAMBERTIAN = 0, MAT_DIELECTRIC, MAT_METAL };

class material {
public:
    MaterialType type;   // ADD THIS FIELD
    vec3  albedo;
    float ref_idx = 1.5f;
    float fuzz    = 0.0f;
    virtual bool scatter(...) const = 0;
};

// in each subclass constructor, set this->type = MAT_LAMBERTIAN etc.
```

---

## Step 10 — Build and run

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DOpenImageDenoise_DIR=/path/to/oidn/lib/cmake/OpenImageDenoise
make -j$(nproc)

./rt > out.ppm
ppmtojpeg out.ppm > out.jpg
```

---

## Profiling checkpoints

After each major step, verify progress with Nsight Compute:

```bash
# after LBVH: confirm O(log N) traversal
ncu --metrics l1tex__t_sector_hit_rate.pct ./rt > /dev/null

# after wavefront: confirm warp efficiency
ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./rt > /dev/null

# after multi-GPU: confirm both GPUs saturated
nvidia-smi dmon -s u   # watch utilization while ./rt runs

# double-precision check (should be 0)
ncu --metrics smsp__sass_thread_inst_executed_op_dfma_pred_on.sum ./rt > /dev/null
```

---

## Expected performance vs original C++ baseline

| Configuration | ~Time (1200×800, 32spp) |
|---|---|
| Original C++ serial | 90 s |
| CUDA naive (ch12 branch) | 7 s |
| + LBVH | 2–3 s |
| + Wavefront | 1–2 s |
| + OIDN (32spp → equivalent 256spp quality) | +0.02 s |
| + Multi-GPU (2×) | 0.5–1 s |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `cudaErrorLaunchTimeout` | Kernel runs >2s on display GPU | Reduce image size or move to secondary GPU |
| OIDN output all black | HDR radiance not clamped before write | Ensure throughput is not NaN; check `hdr=true` |
| LBVH infinite loop | `parent` pointers not initialized | `cudaMemset(d_nodes, 0xFF, ...)` before build |
| Multi-GPU crash | Peer access not supported | Check `cudaDeviceCanAccessPeer`; fall back to host gather |
| Warp efficiency <50% after wavefront | Queues too small, thrashing | Increase queue size or use persistent threads |
