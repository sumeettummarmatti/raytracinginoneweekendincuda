cmake_minimum_required(VERSION 3.22)
project(cuda_raytracer CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# Auto-detect GPU architecture — works on Kaggle T4, Colab V100/A100, local RTX
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ── OIDN (optional) ───────────────────────────────────────────────────────────
find_package(OpenImageDenoise QUIET)

# ── Sources ───────────────────────────────────────────────────────────────────
# FIX: src/sphere.cu is NOT included here.
#      sphere::hit is __host__ __device__ inline in sphere.h.
#      Including sphere.cu would cause an ODR violation (two definitions).
add_executable(rt_accel
    src/main_accel.cu
    src/lbvh.cu
    src/wavefront.cu
    src/denoiser.cu
    src/multigpu.cu
)

# Include both root (original headers) and include/ (new headers)
target_include_directories(rt_accel PRIVATE . include)

# CUDA stub paths for Kaggle / Colab
link_directories(/usr/local/cuda/lib64/stubs /usr/local/cuda/lib64)

if(OpenImageDenoise_FOUND)
    message(STATUS "OIDN found — denoising enabled")
    target_link_libraries(rt_accel PRIVATE OpenImageDenoise cudart)
else()
    message(STATUS "OIDN NOT found — denoising will be skipped at runtime")
    target_compile_definitions(rt_accel PRIVATE NO_OIDN)
    target_link_libraries(rt_accel PRIVATE cudart)
endif()

target_compile_options(rt_accel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        -lineinfo
        -Wno-deprecated-gpu-targets
    >
)
