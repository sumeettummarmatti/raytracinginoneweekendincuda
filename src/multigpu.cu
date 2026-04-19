#include "multigpu.h"
#include <thread>
#include <vector>
#include <iostream>
#include <functional>

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
    GBuffers gb{tile.d_color, tile.d_albedo, tile.d_normal};

    wavefrontRender(W, tileH, tile.yStart, cfg.ns,
                    cfg.cam, bvh, d_spheres,
                    nullptr, numSpheres,
                    gb, tile.d_color, tile.deviceId);

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

    if (numGPUs == 0) return std::vector<vec3>();

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
