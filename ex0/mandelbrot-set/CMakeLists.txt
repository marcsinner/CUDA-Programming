cmake_minimum_required(VERSION 3.5)
project(mset)

set(CMAKE_CXX_STANDARD 11)

set(PRECISION "single" CACHE STRING "type of floating point precision, namely: double/single")
set(PRECISION_OPTIONS single double)
set_property(CACHE PRECISION PROPERTY STRINGS ${PRECISION_OPTIONS})

set(SM "sm_60" CACHE STRING "model of Nvidia Streaming Multiprocessor")
set(SM_OPTIONS "sm_20" "sm_30" "sm_50" "sm_60" "sm_70")
set_property(CACHE SM PROPERTY STRINGS ${SM_OPTIONS})

if ("${PRECISION}" STREQUAL "double")
    set(REAL_SIZE_IN_BYTES 8)
elseif ("${PRECISION}" STREQUAL "single")
    set(REAL_SIZE_IN_BYTES 4)
endif()

find_package(OpenMP REQUIRED)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP C++ has not been found. Please, install")
endif()

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                     -std=c++11;
                     -arch=${SM};
                     -Xptxas -v;
                     -DREAL_SIZE=${REAL_SIZE_IN_BYTES})
cuda_add_library(gpu_kernels STATIC src/kernels/gpu.cu)
target_include_directories(gpu_kernels PRIVATE src ${CUDA_INCLUDE_DIRS})
target_link_libraries(gpu_kernels -lcuda ${CUDA_LIBRARIES})


find_package(Boost COMPONENTS program_options REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JPEG REQUIRED libjpeg)


add_executable(${CMAKE_PROJECT_NAME} src/main.cpp 
                                     src/aux.cpp
                                     src/kernels/serial.cpp
                                     src/kernels/threading.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC src
                                                 src/kernels
                                                 ${Boost_INCLUDE_DIR}
                                                 ${JPEG_INCLUDEDIR})
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC gpu_kernels
                                                   OpenMP::OpenMP_CXX
                                                   ${Boost_LIBRARY_DIRS}/libboost_program_options.a
                                                   ${JPEG_LIBRARIES})

target_compile_definitions(${CMAKE_PROJECT_NAME} PUBLIC "REAL_SIZE=${REAL_SIZE_IN_BYTES}")
