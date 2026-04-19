#include <iostream>
#include <vector>
#include "vec3.h"
#include "camera.h"
#include "sphere.h"
#include "material.h"
#include "multigpu.h"

#define RND ((float)rand()/RAND_MAX)

void buildRandomScene(std::vector<sphere>& h_spheres, std::vector<material*>& h_mats) {
    h_mats.push_back(new lambertian(vec3(0.5, 0.5, 0.5)));
    h_spheres.push_back(sphere(vec3(0,-1000,0), 1000, h_mats.back()));

    for(int a = -11; a < 11; a++) {
        for(int b = -11; b < 11; b++) {
            float choose_mat = RND;
            vec3 center(a+0.9f*RND,0.2f,b+0.9f*RND);
            if((center-vec3(4.0f,0.2f,0.0f)).length() > 0.9f) {
                if(choose_mat < 0.8f) {
                    h_mats.push_back(new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                    h_spheres.push_back(sphere(center, 0.2f, h_mats.back()));
                }
                else if(choose_mat < 0.95f) {
                    h_mats.push_back(new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                    h_spheres.push_back(sphere(center, 0.2f, h_mats.back()));
                }
                else {
                    h_mats.push_back(new dielectric(1.5f));
                    h_spheres.push_back(sphere(center, 0.2f, h_mats.back()));
                }
            }
        }
    }
    h_mats.push_back(new dielectric(1.5f));
    h_spheres.push_back(sphere(vec3(0, 1, 0), 1.0f, h_mats.back()));
    h_mats.push_back(new lambertian(vec3(0.4f, 0.2f, 0.1f)));
    h_spheres.push_back(sphere(vec3(-4, 1, 0), 1.0f, h_mats.back()));
    h_mats.push_back(new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));
    h_spheres.push_back(sphere(vec3(4, 1, 0), 1.0f, h_mats.back()));
}

int main(int argc, char** argv) {
    int W  = 1200;
    int H  = 800;
    int NS = 32;

    for (int i=1; i < argc; i++) {
        if (std::string(argv[i]) == "--width" && i+1 < argc) W = std::stoi(argv[++i]);
        if (std::string(argv[i]) == "--height" && i+1 < argc) H = std::stoi(argv[++i]);
        if (std::string(argv[i]) == "--spp" && i+1 < argc) NS = std::stoi(argv[++i]);
    }

    std::cerr << "Accelerated render config: " << W << "x" << H << ", " << NS << " spp\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // build scene
    std::vector<sphere>   h_spheres;
    std::vector<material*> h_mats;
    buildRandomScene(h_spheres, h_mats);

    // camera
    vec3 lookfrom(13,2,3), lookat(0,0,0);
    float dist_to_focus = 10.0f, aperture = 0.1f;
    camera cam(lookfrom, lookat, vec3(0,1,0), 20,
               float(W)/float(H), aperture, dist_to_focus);

    RenderConfig cfg;
    cfg.width      = W;
    cfg.height     = H;
    cfg.ns         = NS;
    cfg.cam        = cam;
    cfg.h_spheres  = h_spheres.data();
    cfg.h_mats     = h_mats.data();
    cfg.numSpheres = (int)h_spheres.size();

    auto framebuffer = multiGPURender(cfg);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cerr << "TIMING: total_ms=" << ms << "\n";

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
