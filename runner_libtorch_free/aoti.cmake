#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)
project(aoti_libtorch_free CUDA CXX)


IF(DEFINED ENV{TORCHCHAT_ROOT})
    set(TORCHCHAT_ROOT $ENV{TORCHCHAT_ROOT})
ELSE()
    set(TORCHCHAT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
ENDIF()

add_executable(aoti_libtorch_free_run
	  runner_libtorch_free/run.cpp
    runner_libtorch_free/include/torch/csrc/inductor/aoti_libtorch_free/package_loader.cpp
    runner_libtorch_free/include/torch/csrc/inductor/aoti_libtorch_free/package_loader_utils.cpp
    runner_libtorch_free/include/torch/csrc/inductor/aoti_libtorch_free/cuda/utils_cuda.cu
    runner_libtorch_free/third-party/miniz-3.0.2/crc.cpp
    runner_libtorch_free/third-party/miniz-3.0.2/miniz.c
)

target_compile_options(aoti_libtorch_free_run PUBLIC -DAOTI_LIBTORCH_FREE -D__AOTI_MODEL__ -DUSE_CUDA -DUSE_EXTERNAL_MZCRC)
target_compile_options(aoti_libtorch_free_run PUBLIC -O3)
#target_compile_options(aoti_libtorch_free_run PUBLIC -O3 -mtune=generic -march=x86-64-v2)
target_include_directories(aoti_libtorch_free_run PRIVATE
	  runner_libtorch_free/include
    runner_libtorch_free/third-party/miniz-3.0.2
)
target_link_libraries(aoti_libtorch_free_run m)

# Find CUDA package
find_package(CUDA REQUIRED)

enable_language(CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)
target_include_directories(aoti_libtorch_free_run PRIVATE ${CUDA_INCLUDE_DIRS})

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
target_link_libraries(aoti_libtorch_free_run ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})

set_property(TARGET aoti_libtorch_free_run PROPERTY
    CXX_STANDARD 17
)
