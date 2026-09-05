cmake_minimum_required(VERSION 3.24)
project(router_ia_native_memory LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(router_ia_native_memory SHARED router_memory.cu)

target_compile_definitions(router_ia_native_memory PRIVATE ROUTER_IA_NATIVE_MEMORY_EXPORTS)
set_target_properties(router_ia_native_memory PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "router_ia_native_memory"
)

if (MSVC)
    target_compile_options(router_ia_native_memory PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
else()
    target_compile_options(router_ia_native_memory PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

target_link_libraries(router_ia_native_memory PRIVATE cudart)
