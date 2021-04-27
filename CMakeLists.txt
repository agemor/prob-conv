cmake_minimum_required(VERSION 3.19)
project(fresco CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(fresco main.cu)

set_target_properties(
        fresco
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(fresco cublas)