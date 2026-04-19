Ray Tracing in One Weekend in CUDA
==================================

This is yet another _Ray Tracing in One Weekend_ clone, but this time using CUDA instead of C++.

By Roger Allen
May, 2018

See the [Master Branch](https://github.com/rogerallen/raytracinginoneweekend) for more information.

Chapter 1
---------

This introduces the basic kernel launch mechanism & host/device memory management.  We are just creating an image on the GPU device and cudaMallocmanaged allows for sharing the framebuffer and automatically copying that buffer to & from the device.

I also added a timer to see how long it takes the GPU to do rendering.

Chapter 2
---------

Because CUDA is compatible with C++ and the vec3.h class will be used on both GPU & CPU, we add `__host__` `__device__` as a prefix to all methods.

Chapter 3
---------

Since the ray class is only used on the GPU, we will just add `__device__` as a prefix to all methods.

The color function just needs a `__device__` added since this is called from the render kernel.

Note, doing a straight translation from the original C++ will mean that any floating-point constants will be doubles and math on the GPU will be forced to be double-precision.  This will hurt our performance unnecessarily.  Special attention to floating point constants must be taken (e.g. 0.5 -> 0.5f).

Use the "profile_metrics" makefile target to count inst_fp_64 and be sure that is 0.

Chapter 4
---------

We only need to add a `__device__` to the hit_sphere() call and use profile_metrics to watch for those floating-point constants.

Chapter 5
---------

Here we have to create our world of spheres on the device and get familiar with how we do memory management for CUDA C++ classes.  Note the cudaMalloc of `d_list` and `d_world` and the `create_world` kernel.

Again, attend to `__device__` and floating-point constants in hitable.h, hitable_list.h and sphere.h.

Chapter 6
---------

In this chapter we need to understand using cuRAND for per-thread random numbers.  See `d_rand_state` and `render_init`.

Note that now using debug flags in compilation makes a big difference in runtime.  Remove those flags for a signficant speedup.

Chapter 7
---------

Matching the C++ code in the color function in main.cu would recurse enough into the color() calls that it was crashing the program by overrunning the stack, so we turn this function into a limited-depth loop instead.  Later code in the book limits to a max depth of 50, so we adapt this a few chapters early on the GPU.

Chapter 8
---------

Just more plumbing for per-thread local random state, mostly.

Chapter 9
---------

Similar to previous modifications.

Chapter 10
----------

Similar to previous modifications.

Chapter 11
----------

Similar to previous modifications.

Chapter 12
----------

And we're done!

---

## Acceleration Upgrades (feature/full-acceleration)

This repository includes a highly accelerated branch (`feature/full-acceleration`) that introduces 4 major GPU-native optimizations to the `ch12` baseline renderer:

### 1. LBVH (Linear Bounding Volume Hierarchy)
Replaces the `O(N)` linear object scan with an **`O(log N)` tree traversal**. 
The BVH is built entirely on the GPU in parallel using Morton codes (Z-order curves), a Thrust-based radix sort, and the Karras 2012 bottom-up tree builder.

### 2. Wavefront Path Tracing 
In the baseline, a single CUDA thread recursively traces one ray through all its bounces. This causes severe **warp divergence** as threads in the same warp hit different materials or bound out into the sky.  
The Wavefront architecture fixes this by decoupling ray states from execution threads. Instead, rays are queued up, sorted by material type (`Lambertian`, `Dielectric`, `Metal`), and executed in highly dense, identical-instruction warps using Thrust stream compaction.

### 3. AI Denoising (Intel OIDN)
We extract auxiliary G-Buffers (Albedo and Surface Normals) during the primary ray intersection. These are fed alongside the noisy output color into **Intel's Open Image Denoise**. This allows us to drastically slash the samples-per-pixel (spp) count (e.g. 32 spp) while reconstructing an image equivalent to 256+ spp.

### 4. Zero FP64 Leaks
Floating-point `double` literals silently wreck GPU efficiency by dropping execution to the FP64 hardware limit (which is only 1/32nd throughput on consumer NVIDIA cards). Entire math files were purged to use strict FP32 (`float` and `sqrtf()`/`tanf()`).

---

### How to Evaluate

To rapidly test the optimizations, switch to a CUDA-equipped Linux or server environment and run the provided benchmark shell script:

```bash
chmod +x benchmark.sh
./benchmark.sh
```

This will automatically build the `ch12` baseline using standard `Make`, build the `Wavefront/LBVH` code using `CMake`, and execute them both. It outputs `.ppm` images and detailed timing statistics per pipeline for direct comparison!
