#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include "vec3.h"
#include "ray.h"
#include "camera.h"
#include "sphere.h"
#include "material.h"
#include "multigpu.h"

// Host-only RNG for scene construction
#define RND ((float)rand() / (float)RAND_MAX)

// ── Scene builder ─────────────────────────────────────────────────────────────

void buildSceneArrays(std::vector<sphere>& spheres)
{
    srand(42);

    spheres.push_back(sphere(
        vec3(0.0f, -1000.0f, 0.0f), 1000.0f,
        MaterialData{MAT_LAMBERTIAN, vec3(0.5f, 0.5f, 0.5f), 0.0f, 0.0f}));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose = RND;
            vec3 center(a + 0.9f * RND, 0.2f, b + 0.9f * RND);
            if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                if (choose < 0.8f) {
                    spheres.push_back(sphere(center, 0.2f,
                        MaterialData{MAT_LAMBERTIAN,
                            vec3(RND*RND, RND*RND, RND*RND), 0.0f, 0.0f}));
                } else if (choose < 0.95f) {
                    spheres.push_back(sphere(center, 0.2f,
                        MaterialData{MAT_METAL,
                            vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)),
                            0.5f*RND, 0.0f}));
                } else {
                    spheres.push_back(sphere(center, 0.2f,
                        MaterialData{MAT_DIELECTRIC, vec3(0.0f,0.0f,0.0f), 0.0f, 1.5f}));
                }
            }
        }
    }
    spheres.push_back(sphere(vec3( 0.0f,1.0f,0.0f), 1.0f,
        MaterialData{MAT_DIELECTRIC, vec3(0.0f,0.0f,0.0f), 0.0f, 1.5f}));
    spheres.push_back(sphere(vec3(-4.0f,1.0f,0.0f), 1.0f,
        MaterialData{MAT_LAMBERTIAN, vec3(0.4f,0.2f,0.1f), 0.0f, 0.0f}));
    spheres.push_back(sphere(vec3( 4.0f,1.0f,0.0f), 1.0f,
        MaterialData{MAT_METAL, vec3(0.7f,0.6f,0.5f), 0.0f, 0.0f}));
}

// ── Diagnostic: CPU linear hit-test ──────────────────────────────────────────
// FIX: Do NOT call camera::get_ray() on host — random_in_unit_disk() in
//      camera.h dereferences curandState* inside its do-while loop even when
//      lens_radius == 0, because the function is always entered before the
//      multiply. Instead reconstruct the centre ray from the camera's own fields.

static bool diagCentreRayHitsGeometry(
    const camera& cam, const std::vector<sphere>& spheres)
{
    // u=0.5, v=0.5 centre pixel, zero DOF offset
    vec3 direction = cam.lower_left_corner
                   + 0.5f * cam.horizontal
                   + 0.5f * cam.vertical
                   - cam.origin;
    ray testRay(cam.origin, direction);

    hit_record rec;
    for (const auto& s : spheres)
        if (s.hit(testRay, 0.001f, 1e30f, rec)) return true;
    return false;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    int W  = 1200;
    int H  = 800;
    int NS = 10;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--width"  && i+1 < argc) W  = atoi(argv[++i]);
        if (arg == "--height" && i+1 < argc) H  = atoi(argv[++i]);
        if (arg == "--spp"    && i+1 < argc) NS = atoi(argv[++i]);
    }

    std::cerr << "Accelerated render: " << W << "x" << H << "  spp=" << NS << "\n";

    // ── build scene ───────────────────────────────────────────────────────────
    std::vector<sphere> h_spheres;
    buildSceneArrays(h_spheres);
    std::cerr << "Scene: " << h_spheres.size() << " spheres\n";

    // ── camera — 30 deg vfov, matches original main.cu ────────────────────────
    vec3  lookfrom(13.0f, 2.0f, 3.0f);
    vec3  lookat(0.0f, 0.0f, 0.0f);
    float dist_to_focus = 10.0f;
    float aperture      = 0.1f;

    camera cam(lookfrom, lookat, vec3(0.0f,1.0f,0.0f),
               30.0f, float(W)/float(H), aperture, dist_to_focus);

    // ── diagnostic ────────────────────────────────────────────────────────────
    bool centreHit = diagCentreRayHitsGeometry(cam, h_spheres);
    std::cerr << "[DIAG] Centre ray CPU "
              << (centreHit ? "HIT geometry — scene OK"
                            : "MISSED — check camera or scene setup") << "\n";

    // ── render ────────────────────────────────────────────────────────────────
    RenderConfig cfg;
    cfg.width      = W;
    cfg.height     = H;
    cfg.ns         = NS;
    cfg.cam        = cam;
    cfg.h_spheres  = h_spheres.data();
    cfg.numSpheres = (int)h_spheres.size();

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<vec3> framebuffer = multiGPURender(cfg);
    auto t1 = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    std::cerr << "TIMING: total_wall_ms=" << ms << "\n";

    {
        std::ofstream tf("accel_time.txt");
        tf << "TIMING: total_wall_ms=" << ms << "\n";
    }

    // ── write PPM ─────────────────────────────────────────────────────────────
    // FIX: use sqrtf/fmaxf/fminf NOT std::sqrt/std::min/std::max
    //      nvcc rejects std:: math calls in .cu translation units
    std::cout << "P3\n" << W << " " << H << "\n255\n";
    for (int j = H-1; j >= 0; j--) {
        for (int i = 0; i < W; i++) {
            vec3 c = framebuffer[j * W + i];
            float r = sqrtf(c.x() > 0.0f ? c.x() : 0.0f);
            float g = sqrtf(c.y() > 0.0f ? c.y() : 0.0f);
            float b = sqrtf(c.z() > 0.0f ? c.z() : 0.0f);
            r = r < 1.0f ? r : 1.0f;
            g = g < 1.0f ? g : 1.0f;
            b = b < 1.0f ? b : 1.0f;
            std::cout << (int)(255.99f*r) << ' '
                      << (int)(255.99f*g) << ' '
                      << (int)(255.99f*b) << '\n';
        }
    }

    return 0;
}
