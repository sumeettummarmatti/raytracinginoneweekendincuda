#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include "vec3.h"
#include "ray.h"
#include "camera.h"

// ── Error checking ────────────────────────────────────────────────────────────
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// ── POD Material & Sphere definitions ─────────────────────────────────────────
#define MAT_LAMBERTIAN 0
#define MAT_DIELECTRIC 1
#define MAT_METAL      2

struct MaterialData {
    int   type;
    vec3  albedo;
    float fuzz;
    float refIdx;

    __host__ __device__ MaterialData() : type(0), fuzz(0.0f), refIdx(1.0f) {}
    __host__ __device__ MaterialData(int t, vec3 a, float f, float r) 
        : type(t), albedo(a), fuzz(f), refIdx(r) {}
};

struct hit_record_pod {
    float t;
    vec3 p;
    vec3 normal;
    MaterialData mat;
};

struct SimpleSphere {
    vec3 center;
    float radius;
    MaterialData mat;

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record_pod& rec) const {
        vec3 oc = r.origin() - center;
        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius*radius;
        float discriminant = b*b - a*c;
        if (discriminant > 0.0f) {
            float temp = (-b - sqrtf(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.point_at_parameter(rec.t);
                rec.normal = (rec.p - center) / radius;
                rec.mat = mat;
                return true;
            }
            temp = (-b + sqrtf(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.point_at_parameter(rec.t);
                rec.normal = (rec.p - center) / radius;
                rec.mat = mat;
                return true;
            }
        }
        return false;
    }
};

// ── Wavefront Structures ──────────────────────────────────────────────────────
struct RayState {
    ray             r;
    vec3            throughput;
    vec3            radiance;
    hit_record_pod  pendingHit;
    int             pixelIndex;
    int             depth;
    bool            alive;
};

struct WavefrontQueues {
    int* lambertian;
    int* dielectric;
    int* metal;
    int* miss;
    int* d_counts;
};

// ── Kernels ───────────────────────────────────────────────────────────────────

#define RND (curand_uniform(&local_rand_state))

__global__ void k_create_world(SimpleSphere* d_list, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;
        d_list[i++] = SimpleSphere{vec3(0.0f, -1000.0f, -1.0f), 1000.0f, MaterialData(MAT_LAMBERTIAN, vec3(0.5f, 0.5f, 0.5f), 0.0f, 0.0f)};
        
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND, 0.2f, b+RND);
                if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                    if(choose_mat < 0.8f) {
                        d_list[i++] = SimpleSphere{center, 0.2f, MaterialData(MAT_LAMBERTIAN, vec3(RND*RND, RND*RND, RND*RND), 0.0f, 0.0f)};
                    } else if(choose_mat < 0.95f) {
                        d_list[i++] = SimpleSphere{center, 0.2f, MaterialData(MAT_METAL, vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND, 0.0f)};
                    } else {
                        d_list[i++] = SimpleSphere{center, 0.2f, MaterialData(MAT_DIELECTRIC, vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.5f)};
                    }
                }
            }
        }
        d_list[i++] = SimpleSphere{vec3(0.0f, 1.0f, 0.0f), 1.0f, MaterialData(MAT_DIELECTRIC, vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.5f)};
        d_list[i++] = SimpleSphere{vec3(-4.0f, 1.0f, 0.0f), 1.0f, MaterialData(MAT_LAMBERTIAN, vec3(0.4f, 0.2f, 0.1f), 0.0f, 0.0f)};
        d_list[i++] = SimpleSphere{vec3(4.0f, 1.0f, 0.0f), 1.0f, MaterialData(MAT_METAL, vec3(0.7f, 0.6f, 0.5f), 0.0f, 0.0f)};
        *rand_state = local_rand_state;
    }
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void k_generateRays(RayState* states, curandState* rng, camera** cam, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;
    int idx = j * width + i;
    
    curandState local_rand_state = rng[idx];
    float u = float(i + curand_uniform(&local_rand_state)) / float(width);
    float v = float(j + curand_uniform(&local_rand_state)) / float(height);
    
    states[idx].r          = (*cam)->get_ray(u, v, &local_rand_state);
    states[idx].throughput = vec3(1.0f, 1.0f, 1.0f);
    states[idx].radiance   = vec3(0.0f, 0.0f, 0.0f);
    states[idx].pixelIndex = idx;
    states[idx].depth      = 0;
    states[idx].alive      = true;
    
    rng[idx] = local_rand_state;
}

__global__ void k_intersect(RayState* states, const int* __restrict__ activeIn, int N, SimpleSphere* spheres, int numSpheres, WavefrontQueues Q) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int rayIdx = activeIn[tid];
    RayState& s = states[rayIdx];

    hit_record_pod rec;
    bool hit = false;
    float closest = 1e30f;

    for (int i = 0; i < numSpheres; i++) {
        hit_record_pod temp;
        if (spheres[i].hit(s.r, 0.001f, closest, temp)) {
            hit = true;
            closest = temp.t;
            rec = temp;
        }
    }

    if (!hit) {
        Q.miss[atomicAdd(&Q.d_counts[3], 1)] = rayIdx;
        return;
    }

    s.pendingHit = rec;
    int mt = rec.mat.type;
    if      (mt == MAT_LAMBERTIAN) Q.lambertian[atomicAdd(&Q.d_counts[0], 1)] = rayIdx;
    else if (mt == MAT_DIELECTRIC) Q.dielectric[atomicAdd(&Q.d_counts[1], 1)] = rayIdx;
    else                            Q.metal     [atomicAdd(&Q.d_counts[2], 1)] = rayIdx;
}

// Helper math
__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - vec3(1.0f,1.0f,1.0f);
    } while (p.squared_length() >= 1.0f);
    return p;
}
__device__ vec3 reflect(const vec3& v, const vec3& n) { return v - 2.0f*dot(v,n)*n; }
__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1.0f-dt*dt);
    if (discriminant > 0.0f) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrtf(discriminant);
        return true;
    }
    return false;
}
__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx)/(1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*powf((1.0f - cosine),5.0f);
}

// ── Shaders ───────────────────────────────────────────────────────────────────

__global__ void k_shadeMiss(RayState* states, const int* queue, int N, vec3* fb) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int rayIdx = queue[tid];
    RayState& s = states[rayIdx];

    vec3 unit_direction = unit_vector(s.r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    vec3 sky = (1.0f - t)*vec3(1.0f, 1.0f, 1.0f) + t*vec3(0.5f, 0.7f, 1.0f);
    
    s.radiance += s.throughput * sky;
    s.alive = false;
    
    // Accumulate to frame buffer directly on miss (ray terminates)
    fb[s.pixelIndex] += s.radiance;
}

__global__ void k_shadeLambertian(RayState* states, const int* queue, int N, curandState* rng) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int rayIdx = queue[tid];
    RayState& s = states[rayIdx];

    if (s.depth >= 50) { s.alive = false; return; }

    curandState local_rng = rng[s.pixelIndex];
    vec3 target = s.pendingHit.p + s.pendingHit.normal + random_in_unit_sphere(&local_rng);
    
    s.throughput *= s.pendingHit.mat.albedo;
    s.r = ray(s.pendingHit.p, target - s.pendingHit.p);
    s.depth++;
    rng[s.pixelIndex] = local_rng;
}

__global__ void k_shadeMetal(RayState* states, const int* queue, int N, curandState* rng) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int rayIdx = queue[tid];
    RayState& s = states[rayIdx];

    if (s.depth >= 50) { s.alive = false; return; }

    curandState local_rng = rng[s.pixelIndex];
    vec3 reflected = reflect(unit_vector(s.r.direction()), s.pendingHit.normal);
    vec3 scatteredDir = reflected + s.pendingHit.mat.fuzz * random_in_unit_sphere(&local_rng);
    
    if (dot(scatteredDir, s.pendingHit.normal) > 0.0f) {
        s.throughput *= s.pendingHit.mat.albedo;
        s.r = ray(s.pendingHit.p, scatteredDir);
        s.depth++;
    } else {
        s.alive = false;
    }
    rng[s.pixelIndex] = local_rng;
}

__global__ void k_shadeDielectric(RayState* states, const int* queue, int N, curandState* rng) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int rayIdx = queue[tid];
    RayState& s = states[rayIdx];

    if (s.depth >= 50) { s.alive = false; return; }

    curandState local_rng = rng[s.pixelIndex];
    vec3 outward_normal;
    vec3 reflected = reflect(s.r.direction(), s.pendingHit.normal);
    float ni_over_nt;
    s.throughput *= vec3(1.0f, 1.0f, 1.0f); // Glass absorbs nothing
    vec3 refracted;
    float reflect_prob;
    float cosine;
    
    float dir_dot_normal = dot(s.r.direction(), s.pendingHit.normal);
    
    if (dir_dot_normal > 0.0f) {
        outward_normal = -s.pendingHit.normal;
        ni_over_nt = s.pendingHit.mat.refIdx;
        cosine = s.pendingHit.mat.refIdx * dir_dot_normal / s.r.direction().length();
    } else {
        outward_normal = s.pendingHit.normal;
        ni_over_nt = 1.0f / s.pendingHit.mat.refIdx;
        cosine = -dir_dot_normal / s.r.direction().length();
    }
    
    if (refract(s.r.direction(), outward_normal, ni_over_nt, refracted)) {
        reflect_prob = schlick(cosine, s.pendingHit.mat.refIdx);
    } else {
        reflect_prob = 1.0f;
    }
    
    if (curand_uniform(&local_rng) < reflect_prob) {
        s.r = ray(s.pendingHit.p, reflected);
    } else {
        s.r = ray(s.pendingHit.p, refracted);
    }
    
    s.depth++;
    rng[s.pixelIndex] = local_rng;
}

__global__ void k_compact(const RayState* states, const int* src, int* dst, int N, int* d_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int rayIdx = src[tid];
    if (states[rayIdx].alive) {
        int slot = atomicAdd(d_count, 1);
        dst[slot] = rayIdx;
    }
}

__global__ void k_normalize(vec3* buf, int n, float inv_ns) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = vec3(buf[i].x()*inv_ns, buf[i].y()*inv_ns, buf[i].z()*inv_ns);
}

// ── Setup & Main ──────────────────────────────────────────────────────────────

__global__ void init_camera(camera **d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        vec3 lookfrom(13.0f, 2.0f, 3.0f);
        vec3 lookat(0.0f, 0.0f, 0.0f);
        float dist_to_focus = 10.0f;
        float aperture = 0.1f;
        *d_camera = new camera(lookfrom, lookat, vec3(0.0f,1.0f,0.0f), 30.0f,
                               float(nx)/float(ny), aperture, dist_to_focus);
    }
}

int main(int argc, char** argv) {
    int nx = 1200;
    int ny = 800;
    int ns = 10;
    int tx = 8;
    int ty = 8;

    std::cerr << "Accelerated render (Wavefront): " << nx << "x" << ny << " image with " << ns << " spp.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);
    
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
    checkCudaErrors(cudaMemset(fb, 0, fb_size));

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaDeviceSynchronize());

    int num_spheres = 486; // Safe upper bound for full scene iteration
    SimpleSphere* d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_spheres * sizeof(SimpleSphere)));
    
    k_create_world<<<1,1>>>(d_list, d_rand_state2);
    checkCudaErrors(cudaDeviceSynchronize());

    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    init_camera<<<1,1>>>(d_camera, nx, ny);
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());

    // Wavefront allocations
    RayState* d_states; checkCudaErrors(cudaMalloc(&d_states, num_pixels * sizeof(RayState)));
    WavefrontQueues Q;
    checkCudaErrors(cudaMalloc(&Q.lambertian, num_pixels * sizeof(int)));
    checkCudaErrors(cudaMalloc(&Q.dielectric, num_pixels * sizeof(int)));
    checkCudaErrors(cudaMalloc(&Q.metal,      num_pixels * sizeof(int)));
    checkCudaErrors(cudaMalloc(&Q.miss,       num_pixels * sizeof(int)));
    checkCudaErrors(cudaMalloc(&Q.d_counts,   4 * sizeof(int)));
    
    int* d_activeA; checkCudaErrors(cudaMalloc(&d_activeA, num_pixels * sizeof(int)));
    int* d_activeB; checkCudaErrors(cudaMalloc(&d_activeB, num_pixels * sizeof(int)));
    int* d_compactCount; checkCudaErrors(cudaMalloc(&d_compactCount, sizeof(int)));

    clock_t start, stop;
    start = clock();

    int BSIZE = 128;

    for (int sample = 0; sample < ns; sample++) {
        k_generateRays<<<blocks, threads>>>(d_states, d_rand_state, d_camera, nx, ny);
        checkCudaErrors(cudaDeviceSynchronize());

        thrust::sequence(thrust::device_pointer_cast(d_activeA), thrust::device_pointer_cast(d_activeA + num_pixels));
        
        int N = num_pixels;
        int* src = d_activeA;
        int* dst = d_activeB;

        while (N > 0) {
            checkCudaErrors(cudaMemset(Q.d_counts, 0, 4 * sizeof(int)));
            
            k_intersect<<<(N+BSIZE-1)/BSIZE, BSIZE>>>(d_states, src, N, d_list, num_spheres, Q);
            checkCudaErrors(cudaDeviceSynchronize());

            int counts[4];
            checkCudaErrors(cudaMemcpy(counts, Q.d_counts, 4*sizeof(int), cudaMemcpyDeviceToHost));

            if (counts[0] > 0) k_shadeLambertian<<<(counts[0]+BSIZE-1)/BSIZE, BSIZE>>>(d_states, Q.lambertian, counts[0], d_rand_state);
            if (counts[1] > 0) k_shadeDielectric<<<(counts[1]+BSIZE-1)/BSIZE, BSIZE>>>(d_states, Q.dielectric, counts[1], d_rand_state);
            if (counts[2] > 0) k_shadeMetal<<<(counts[2]+BSIZE-1)/BSIZE, BSIZE>>>(d_states, Q.metal, counts[2], d_rand_state);
            if (counts[3] > 0) k_shadeMiss<<<(counts[3]+BSIZE-1)/BSIZE, BSIZE>>>(d_states, Q.miss, counts[3], fb);
            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors(cudaMemset(d_compactCount, 0, sizeof(int)));
            k_compact<<<(N+BSIZE-1)/BSIZE, BSIZE>>>(d_states, src, dst, N, d_compactCount);
            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors(cudaMemcpy(&N, d_compactCount, sizeof(int), cudaMemcpyDeviceToHost));
            std::swap(src, dst);
        }
    }

    k_normalize<<<(num_pixels+BSIZE-1)/BSIZE, BSIZE>>>(fb, num_pixels, 1.0f / (float)ns);
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    {
        std::ofstream tf("accel_time.txt");
        tf << "TIMING: total_wall_ms=" << timer_seconds * 1000.0 << "\n";
    }

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            vec3 c = fb[pixel_index];
            float r = std::sqrt(std::max(c.x(), 0.0f));
            float g = std::sqrt(std::max(c.y(), 0.0f));
            float b = std::sqrt(std::max(c.z(), 0.0f));
            std::cout << (int)(255.99f * std::min(r, 1.0f)) << " "
                      << (int)(255.99f * std::min(g, 1.0f)) << " "
                      << (int)(255.99f * std::min(b, 1.0f)) << "\n";
        }
    }

    cudaDeviceReset();
    return 0;
}
