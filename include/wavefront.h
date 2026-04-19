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
    int width, int height, int yOffset, int ns,
    camera& cam, LBVH& bvh, sphere* d_spheres,
    material** d_materials, int numSpheres,
    GBuffers& gb, vec3* d_output, int gpuId);
