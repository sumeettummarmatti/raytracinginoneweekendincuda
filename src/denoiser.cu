#include "denoiser.h"
#include <iostream>

#ifndef NO_OIDN
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
    
    if (cfg.cleanAux && d_albedo && d_normal) {
        filter.setImage("albedo", d_albedo, oidn::Format::Float3, W, H,
                        0, sizeof(vec3), stride);
        filter.setImage("normal", d_normal, oidn::Format::Float3, W, H,
                        0, sizeof(vec3), stride);
    }
                    
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
#else
void denoiseOIDN(DenoiserConfig cfg,
                 vec3* d_color, vec3* d_albedo, vec3* d_normal,
                 vec3* d_output, cudaStream_t stream) {
    std::cerr << "[OIDN] compiled without OIDN support. Denoising skipped.\n";
    // just copy noisy color to output
    size_t bytes = cfg.width * cfg.height * sizeof(vec3);
    if (d_color != d_output) {
        cudaMemcpyAsync(d_output, d_color, bytes, cudaMemcpyDeviceToDevice, stream);
    }
}
#endif
