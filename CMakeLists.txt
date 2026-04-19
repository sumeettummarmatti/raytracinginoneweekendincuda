cmake_minimum_required(VERSION 3.22)
project(cuda_raytracer CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# Detect host architecture or fallback to common ones
set(CMAKE_CUDA_ARCHITECTURES native)

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


if(OpenImageDenoise_FOUND)
    target_link_libraries(rt_accel PRIVATE OpenImageDenoise cuda)
else()
    target_compile_definitions(rt_accel PRIVATE "NO_OIDN")
    target_link_libraries(rt_accel PRIVATE cuda)
endif()

target_compile_options(rt_accel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        -lineinfo
        -Wno-deprecated-gpu-targets
    >
)
