#include "multigpu.h"
#include "material.h"
#include "sphere.h"
#include <thread>
#include <vector>
#include <iostream>

struct GPUTile {
    int   deviceId;
    int   yStart, yEnd;
    vec3* d_color;     // noisy radiance accumulator
    vec3* d_albedo;    // first-hit albedo G-buffer
    vec3* d_normal;    // first-hit normal G-buffer
    vec3* d_denoised;  // OIDN output (separate alloc, never aliased)
};

static void renderOnDevice(GPUTile& tile, RenderConfig& cfg)
{
    cudaSetDevice(tile.deviceId);

    const int W          = cfg.width;
    const int fullH      = cfg.height;       // FIX: use cfg.height, not aspect ratio
    const int tileH      = tile.yEnd - tile.yStart;
    const int numPx      = W * tileH;
    const int numSpheres = cfg.numSpheres;

    // ── upload scene ──────────────────────────────────────────────────────────
    sphere* d_spheres;
    cudaMalloc(&d_spheres, numSpheres * sizeof(sphere));
    cudaMemcpy(d_spheres, cfg.h_spheres,
               numSpheres * sizeof(sphere), cudaMemcpyHostToDevice);

    // diagnostic
    std::cerr << "[GPU " << tile.deviceId << "] uploaded " << numSpheres << " spheres\n";
    {
        sphere h0;
        cudaMemcpy(&h0, d_spheres, sizeof(sphere), cudaMemcpyDeviceToHost);
        std::cerr << "[GPU " << tile.deviceId << "] sphere[0] center=("
                  << h0.center.x() << "," << h0.center.y() << "," << h0.center.z()
                  << ") r=" << h0.radius
                  << " matType=" << (int)h0.mat.type << "\n";
    }

    // ── allocate G-buffers (separate allocs — no aliasing) ───────────────────
    cudaMalloc(&tile.d_color,    numPx * sizeof(vec3));
    cudaMalloc(&tile.d_albedo,   numPx * sizeof(vec3));
    cudaMalloc(&tile.d_normal,   numPx * sizeof(vec3));
    cudaMalloc(&tile.d_denoised, numPx * sizeof(vec3));
    cudaMemset(tile.d_color,    0, numPx * sizeof(vec3));
    cudaMemset(tile.d_albedo,   0, numPx * sizeof(vec3));
    cudaMemset(tile.d_normal,   0, numPx * sizeof(vec3));
    cudaMemset(tile.d_denoised, 0, numPx * sizeof(vec3));

    // ── build LBVH ───────────────────────────────────────────────────────────
    std::cerr << "[GPU " << tile.deviceId << "] building LBVH...\n";
    LBVH bvh = buildLBVH(d_spheres, numSpheres);

    // ── wavefront render ──────────────────────────────────────────────────────
    GBuffers gb{ tile.d_color, tile.d_albedo, tile.d_normal };

    // FIX: pass fullH (cfg.height) so UV mapping covers the full image correctly
    wavefrontRender(W, tileH, tile.yStart, fullH, cfg.ns,
                    cfg.cam, bvh, d_spheres,
                    numSpheres,
                    gb, tile.d_color, tile.deviceId);

    // ── denoise tile on this device ───────────────────────────────────────────
    // FIX: d_color (input) and d_denoised (output) are different allocations
    DenoiserConfig dc{ W, tileH };
    denoiseOIDN(dc,
                tile.d_color,    // noisy input
                tile.d_albedo,
                tile.d_normal,
                tile.d_denoised, // clean output
                0);              // default stream

    // FIX: explicit sync before the caller reads d_denoised
    cudaDeviceSynchronize();

    freeLBVH(bvh);
    cudaFree(d_spheres);
    // d_color/albedo/normal freed in gather step along with d_denoised
}

std::vector<vec3> multiGPURender(RenderConfig& cfg)
{
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    if (numGPUs == 0) {
        std::cerr << "[multiGPU] no CUDA devices found\n";
        return {};
    }
    // Kaggle T4 single-GPU is fine; uncomment to force single:
    // numGPUs = 1;
    std::cerr << "Using " << numGPUs << " GPU(s)\n";

    // ── tile split ────────────────────────────────────────────────────────────
    std::vector<GPUTile> tiles(numGPUs);
    int rowsPerGPU = cfg.height / numGPUs;
    for (int g = 0; g < numGPUs; g++) {
        tiles[g].deviceId = g;
        tiles[g].yStart   = g * rowsPerGPU;
        tiles[g].yEnd     = (g == numGPUs - 1) ? cfg.height
                                                 : tiles[g].yStart + rowsPerGPU;
    }

    // ── peer access (NVLink / PCIe) ───────────────────────────────────────────
    for (int a = 0; a < numGPUs; a++)
        for (int b = 0; b < numGPUs; b++)
            if (a != b) {
                cudaSetDevice(a);
                int canAccess = 0;
                cudaDeviceCanAccessPeer(&canAccess, a, b);
                if (canAccess) cudaDeviceEnablePeerAccess(b, 0);
            }

    // ── one thread per GPU ────────────────────────────────────────────────────
    std::vector<std::thread> workers;
    for (auto& t : tiles)
        workers.emplace_back(renderOnDevice, std::ref(t), std::ref(cfg));
    for (auto& w : workers) w.join();

    // ── gather denoised tiles onto host ───────────────────────────────────────
    std::vector<vec3> h_output(cfg.width * cfg.height, vec3(0.0f, 0.0f, 0.0f));

    for (auto& tile : tiles) {
        cudaSetDevice(tile.deviceId);
        int tileH  = tile.yEnd - tile.yStart;
        int offset = tile.yStart * cfg.width;

        cudaMemcpy(h_output.data() + offset,
                   tile.d_denoised,
                   cfg.width * tileH * sizeof(vec3),
                   cudaMemcpyDeviceToHost);

        cudaFree(tile.d_color);
        cudaFree(tile.d_albedo);
        cudaFree(tile.d_normal);
        cudaFree(tile.d_denoised);
    }

    return h_output;
}
