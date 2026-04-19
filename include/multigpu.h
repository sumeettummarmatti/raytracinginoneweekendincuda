#pragma once
#include "ray_state.h"
#include "lbvh.h"
#include "wavefront.h"
#include "denoiser.h"
#include "camera.h"
#include <vector>

struct RenderConfig {
    int     width;
    int     height;
    int     ns;
    camera  cam;
    sphere* h_spheres;   // host-side flat sphere array (MaterialData embedded)
    int     numSpheres;
};

// Renders the full image using all available GPUs.
// Returns a host-side framebuffer (linear, top-row-last, linear RGB, not gamma corrected).
std::vector<vec3> multiGPURender(RenderConfig& cfg);
