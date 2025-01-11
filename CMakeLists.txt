cmake_minimum_required(VERSION 3.29)
project(cuda_learn CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda_learn main.cu
        src/kernel.cu
        src/kernel.cuh)

set_target_properties(cuda_learn PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
