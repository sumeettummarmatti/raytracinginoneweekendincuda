cmake_minimum_required(VERSION 3.22)
project(cuda_raytracer CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# Fat Binary: support all common architectures (Pascal, Volta, Turing, Ampere)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)
endif()

# OIDN
find_package(OpenImageDenoise QUIET)

add_executable(rt_accel
    src/main_accel.cu
    src/lbvh.cu
    src/wavefront.cu
    src/denoiser.cu
    src/multigpu.cu
)

# Use both root (for original headers) and include/ (for new headers)
target_include_directories(rt_accel PRIVATE . include)

# Add standard CUDA stub paths for Kaggle/Colab environments
link_directories(/usr/local/cuda/lib64/stubs /usr/local/cuda/lib64)

if(OpenImageDenoise_FOUND)
    target_link_libraries(rt_accel PRIVATE OpenImageDenoise cudart)
else()
    target_compile_definitions(rt_accel PRIVATE "NO_OIDN")
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
