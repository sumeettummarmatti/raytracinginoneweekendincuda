#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include "vec3.h"
#include "camera.h"
#include "sphere.h"
#include "material.h"
#include "multigpu.h"

#define RND ((float)rand()/RAND_MAX)

void buildSceneArrays(std::vector<sphere>& spheres) {
    // Ground
    spheres.push_back(sphere(vec3(0,-1000,0), 1000, 
        MaterialData{MAT_LAMBERTIAN, vec3(0.5f, 0.5f, 0.5f), 0, 0}));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose = RND;
            vec3 center(a + 0.9f*RND, 0.2f, b + 0.9f*RND);
            if ((center - vec3(4,0.2f,0)).length() > 0.9f) {
                if (choose < 0.8f)
                    spheres.push_back(sphere(center, 0.2f,
                        MaterialData{MAT_LAMBERTIAN, vec3(RND*RND, RND*RND, RND*RND), 0, 0}));
                else if (choose < 0.95f)
                    spheres.push_back(sphere(center, 0.2f,
                        MaterialData{MAT_METAL, vec3(0.5f*(1+RND), 0.5f*(1+RND), 0.5f*(1+RND)), 0.5f*RND, 0}));
                else
                    spheres.push_back(sphere(center, 0.2f,
                        MaterialData{MAT_DIELECTRIC, vec3(0,0,0), 0, 1.5f}));
            }
        }
    }
    spheres.push_back(sphere(vec3( 0,1,0), 1.0f, MaterialData{MAT_DIELECTRIC, vec3(0,0,0), 0, 1.5f}));
    spheres.push_back(sphere(vec3(-4,1,0), 1.0f,
                     MaterialData{MAT_LAMBERTIAN, vec3(0.4f,0.2f,0.1f), 0, 0}));
    spheres.push_back(sphere(vec3( 4,1,0), 1.0f,
                     MaterialData{MAT_METAL, vec3(0.7f,0.6f,0.5f), 0.0f, 0}));
}

int main(int argc, char** argv) {
    int W  = 1200;
    int H  = 800;
    int NS = 10;

    for (int i=1; i < argc; i++) {
        if (std::string(argv[i]) == "--width" && i+1 < argc) W = std::stoi(argv[++i]);
        if (std::string(argv[i]) == "--height" && i+1 < argc) H = std::stoi(argv[++i]);
        if (std::string(argv[i]) == "--spp" && i+1 < argc) NS = std::stoi(argv[++i]);
    }

    std::cerr << "Accelerated render config: " << W << "x" << H << ", " << NS << " spp\n";

    // build scene
    std::vector<sphere> h_spheres;
    buildSceneArrays(h_spheres);
    std::cerr << "Scene: " << h_spheres.size() << " spheres\n";

    // camera
    vec3 lookfrom(13,2,3), lookat(0,0,0);
    float dist_to_focus = 10.0f, aperture = 0.1f;
    camera cam(lookfrom, lookat, vec3(0,1,0), 30, // Sync with Baseline 30deg vfov
               float(W)/float(H), aperture, dist_to_focus);

    RenderConfig cfg;
    cfg.width      = W;
    cfg.height     = H;
    cfg.ns         = NS;
    cfg.cam        = cam;
    cfg.h_spheres  = h_spheres.data();
    cfg.numSpheres = (int)h_spheres.size();

    // Step 4: Verify a single ray manual test (using a host-side build of LBVH)
    // We'll perform the GPU render, but let's do a quick host-side hit test for the center pixel
    ray testRay = cam.get_ray(0.5f, 0.5f, nullptr); // no rng needed for center of pixel usually
    bool anyHit = false;
    float t_min = 0.001f, t_max = 1e30f;
    for(const auto& s : h_spheres) {
        hit_record rec;
        if(s.hit(testRay, t_min, t_max, rec)) {
            if(!anyHit) printf("[TEST RAY] CPU HIT — center ray hits geometry!\n");
            anyHit = true;
        }
    }
    if(!anyHit) printf("[TEST RAY] CPU NO HIT — center ray misses all geometry. Check camera/scene overlap.\n");

    auto start_time = std::chrono::high_resolution_clock::now();

    auto framebuffer = multiGPURender(cfg);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_time - start_time;
    std::cerr << "TIMING: total_wall_ms=" << duration.count() << "\n";

    // write PPM
    std::cout << "P3\n" << W << " " << H << "\n255\n";
    for (int j = H-1; j >= 0; j--)
        for (int i = 0; i < W; i++) {
            vec3 c = framebuffer[j * W + i];
            // gamma correct
            c = vec3(sqrtf(c.x()), sqrtf(c.y()), sqrtf(c.z()));
            int ir = int(255.99f * c.x());
            int ig = int(255.99f * c.y());
            int ib = int(255.99f * c.z());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }

    return 0;
}
