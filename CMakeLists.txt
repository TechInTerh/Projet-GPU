cmake_minimum_required(VERSION 3.14)
include_directories(src/ lib/)

project(GPGPU CUDA CXX)

option(PROF "Reference build" NO)
include(FetchContent)


FetchContent_Declare(
        CLI11
        URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
)

FetchContent_Declare(
        spdlog
        URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
)


FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})
find_package(Boost REQUIRED)
find_package(PNG REQUIRED)
find_package(benchmark REQUIRED)
#find_package(OpenCv REQUIRED)



set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


add_library(renderer_gpu)
target_compile_features(renderer_gpu PUBLIC cxx_std_17)
target_link_libraries(renderer_gpu PRIVATE spdlog::spdlog)
target_link_libraries(renderer_gpu PRIVATE ${Boost_LIBRARIES})
target_sources(renderer_gpu PRIVATE
        src/render_gpu/render_gpu.cu
        src/matrix_image/matrix_image.cu
        src/cudaFunctions/cudaFunctions.cu
        src/render_gpu/gray_blur.cu
        src/render_gpu/tools.cu
        src/env.cu
        src/bounding_box.cu
        )
set_target_properties(renderer_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_library(renderer_cpu)
target_compile_features(renderer_cpu PUBLIC cxx_std_17)
target_link_libraries(renderer_cpu PRIVATE spdlog::spdlog)
target_link_libraries(renderer_cpu PRIVATE ${Boost_LIBRARIES})
target_sources(renderer_cpu PRIVATE
        src/render_cpu.cpp src/render_cpu.hpp
        src/matrix_image/matrix_image.cu
        src/env.cu
        src/bounding_box.cu
        )

add_executable(main src/main.cpp)
target_compile_features(main PUBLIC cxx_std_17)
target_link_libraries(main PRIVATE renderer_gpu PNG::PNG CLI11::CLI11 spdlog::spdlog renderer_cpu ${Boost_LIBRARIES})

