#pragma once
#include "ray_state.h"
#include "lbvh.h"
#include "camera.h"
#include <curand_kernel.h>

struct WavefrontQueues {
    int* lambertian;
    int* dielectric;
    int* metal;
    int* miss;
    int* d_counts;   // [4] atomic counters on device
};

// FIX: fullHeight is the FULL image height (cfg.height).
//      tileHeight is how many rows this GPU renders.
//      yOffset is where this tile starts in the full image.
void wavefrontRender(
    int width, int tileHeight, int yOffset, int fullHeight, int ns,
    camera& cam, LBVH& bvh, sphere* d_spheres,
    int numSpheres,
    GBuffers& gb, vec3* d_output, int gpuId);
