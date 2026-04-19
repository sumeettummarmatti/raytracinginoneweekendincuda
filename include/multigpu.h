#pragma once
#include "ray_state.h"
#include "lbvh.h"
#include "wavefront.h"
#include "denoiser.h"
#include "camera.h"
#include <vector>

struct RenderConfig {
    int width, height, ns;
    camera cam;
    sphere* h_spheres;   // host-side sphere array
    int numSpheres;
};

// renders full image using all available GPUs, returns host framebuffer
std::vector<vec3> multiGPURender(RenderConfig& cfg);
