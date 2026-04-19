#pragma once
#ifndef NO_OIDN
#include <OpenImageDenoise/oidn.hpp>
#endif
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
