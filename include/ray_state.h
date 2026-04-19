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
