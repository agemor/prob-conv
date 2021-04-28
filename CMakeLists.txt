cmake_minimum_required(VERSION 3.19)
project(fresco CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(fresco main.cu Conv.cu Conv.cuh DeltaConv.cu DeltaConv.cuh Utils.cu Utils.cuh NonZero.cu NonZero.cuh PrefixSum.cu PrefixSum.cuh)

set_target_properties(
        fresco
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(fresco cublas curand)