#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>   // rand, srand
#include <ctime>
#include <cmath>
#include <algorithm>
#include "vec3.h"
#include "camera.h"
#include "sphere.h"
#include "material.h"
#include "multigpu.h"

// Simple host-only RNG for scene construction
#define RND ((float)rand() / (float)RAND_MAX)

// ── Scene builder — flat MaterialData arrays, no host material pointers ───────

void buildSceneArrays(std::vector<sphere>& spheres)
{
    srand(42);  // reproducible scene

    // Ground — large lambertian sphere
    spheres.push_back(sphere(
        vec3(0.0f, -1000.0f, 0.0f), 1000.0f,
        MaterialData{MAT_LAMBERTIAN, vec3(0.5f, 0.5f, 0.5f), 0.0f, 0.0f}));

    // Random small spheres
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose = RND;
            vec3 center(a + 0.9f * RND, 0.2f, b + 0.9f * RND);
            if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                if (choose < 0.8f) {
                    // diffuse
                    spheres.push_back(sphere(center, 0.2f,
                        MaterialData{MAT_LAMBERTIAN,
                            vec3(RND * RND, RND * RND, RND * RND), 0.0f, 0.0f}));
                } else if (choose < 0.95f) {
                    // metal
                    spheres.push_back(sphere(center, 0.2f,
                        MaterialData{MAT_METAL,
                            vec3(0.5f * (1.0f + RND),
                                 0.5f * (1.0f + RND),
                                 0.5f * (1.0f + RND)),
                            0.5f * RND, 0.0f}));
                } else {
                    // glass
                    spheres.push_back(sphere(center, 0.2f,
                        MaterialData{MAT_DIELECTRIC, vec3(0.0f), 0.0f, 1.5f}));
                }
            }
        }
    }

    // Three large feature spheres (glass, diffuse, metal)
    spheres.push_back(sphere(vec3( 0.0f, 1.0f, 0.0f), 1.0f,
        MaterialData{MAT_DIELECTRIC, vec3(0.0f), 0.0f, 1.5f}));
    spheres.push_back(sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f,
        MaterialData{MAT_LAMBERTIAN, vec3(0.4f, 0.2f, 0.1f), 0.0f, 0.0f}));
    spheres.push_back(sphere(vec3( 4.0f, 1.0f, 0.0f), 1.0f,
        MaterialData{MAT_METAL, vec3(0.7f, 0.6f, 0.5f), 0.0f, 0.0f}));
}

// ── CPU hit-test — used for diagnostic only ───────────────────────────────────
static bool cpuHitTest(const ray& r, const std::vector<sphere>& spheres)
{
    hit_record rec;
    bool anyHit = false;
    for (const auto& s : spheres)
        if (s.hit(r, 0.001f, 1e30f, rec)) anyHit = true;
    return anyHit;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    int W  = 1200;
    int H  = 800;
    int NS = 10;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--width"  && i + 1 < argc) W  = std::stoi(argv[++i]);
        if (arg == "--height" && i + 1 < argc) H  = std::stoi(argv[++i]);
        if (arg == "--spp"    && i + 1 < argc) NS = std::stoi(argv[++i]);
    }

    std::cerr << "Accelerated render: " << W << "x" << H
              << "  spp=" << NS << "\n";

    // ── build scene ───────────────────────────────────────────────────────────
    std::vector<sphere> h_spheres;
    buildSceneArrays(h_spheres);
    std::cerr << "Scene: " << h_spheres.size() << " spheres\n";

    // ── camera — match baseline (30° vfov, same lookfrom/lookat) ─────────────
    vec3  lookfrom(13.0f, 2.0f, 3.0f);
    vec3  lookat(0.0f, 0.0f, 0.0f);
    float dist_to_focus = 10.0f;
    float aperture      = 0.1f;

    camera cam(lookfrom, lookat, vec3(0.0f, 1.0f, 0.0f),
               30.0f,                     // vfov — matches original main.cu
               float(W) / float(H),
               aperture, dist_to_focus);

    // ── diagnostic: CPU hit-test for centre pixel ─────────────────────────────
    // FIX: camera::get_ray needs a curandState for DOF — on CPU we skip DOF
    //      by passing aperture=0 in a temp camera
    {
        camera diagCam(lookfrom, lookat, vec3(0.0f, 1.0f, 0.0f),
                       30.0f, float(W) / float(H), 0.0f, dist_to_focus);
        // get_ray with nullptr is safe only when aperture=0 (lens_radius=0, no sampling)
        ray testRay = diagCam.get_ray(0.5f, 0.5f, nullptr);
        bool hit = cpuHitTest(testRay, h_spheres);
        std::cerr << "[DIAG] Centre ray CPU "
                  << (hit ? "HIT geometry" : "MISSED — check scene/camera") << "\n";
    }

    // ── render ────────────────────────────────────────────────────────────────
    RenderConfig cfg;
    cfg.width      = W;
    cfg.height     = H;
    cfg.ns         = NS;
    cfg.cam        = cam;
    cfg.h_spheres  = h_spheres.data();
    cfg.numSpheres = (int)h_spheres.size();

    auto t0 = std::chrono::high_resolution_clock::now();
    auto framebuffer = multiGPURender(cfg);
    auto t1 = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    std::cerr << "TIMING: total_wall_ms=" << ms << "\n";

    // write timing file for benchmark.sh
    {
        std::ofstream tf("accel_time.txt");
        tf << "TIMING: total_wall_ms=" << ms << "\n";
    }

    // ── write PPM ─────────────────────────────────────────────────────────────
    // FIX: gamma correction and clamping done here, not inside GPU kernels
    std::cout << "P3\n" << W << " " << H << "\n255\n";
    for (int j = H - 1; j >= 0; j--) {
        for (int i = 0; i < W; i++) {
            vec3 c = framebuffer[j * W + i];
            // gamma 2 correction
            float r = std::sqrt(std::max(c.x(), 0.0f));
            float g = std::sqrt(std::max(c.y(), 0.0f));
            float b = std::sqrt(std::max(c.z(), 0.0f));
            // clamp to [0,1] before quantising
            r = std::min(r, 1.0f);
            g = std::min(g, 1.0f);
            b = std::min(b, 1.0f);
            std::cout << int(255.99f * r) << ' '
                      << int(255.99f * g) << ' '
                      << int(255.99f * b) << '\n';
        }
    }

    return 0;
}
