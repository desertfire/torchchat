#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)
project(aoti_libtorch_free_singe_binary CUDA CXX)


IF(DEFINED ENV{TORCHCHAT_ROOT})
    set(TORCHCHAT_ROOT $ENV{TORCHCHAT_ROOT})
ELSE()
    set(TORCHCHAT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
ENDIF()

add_executable(aoti_libtorch_free_single_binary_run
    runner_libtorch_free_single_binary/run.cpp
    runner_libtorch_free_single_binary/include/torch/csrc/inductor/aoti_libtorch_free/cuda/utils_cuda.cu
)

target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/cnyzukfltdxhtc327mswuv52zgb6xlcf5m74mgv57fovyva3p5gs.wrapper.cpp)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/cr4tcammsbazzjghn4cqndluwq5xvc7brjgs5gyg7p2vjohf4xmo.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/cnvkbngwgvhxwmky27tiknylrvgfd7hrupwczdxythtx2iwmxmnw.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/cefosdeer5eimhrl5y2wqxsmwkolmf64lwwulmzksr4pqsg6baft.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/crhflw4opepsg6lvvv5ga3oknnb4vm4bknuud6vbqayhrl4st654.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/c24pugketupm3cgf7hh4pijiyvg4estagagxw6qi7uhfcqbz3sp6.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/czsemnym6pmqx2kapz52dyhwz6y3qsqv6yz3hmvrmmnytbcyem4x.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/cu6iwiujymsulhca5uo4qwnzntn3x52lori62acz6nlpqetnsayd.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/c546empi4r4rqktgs5mozszzg5e3xoxmu26rq4bnk3oj2frrmm2v.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/c2r6h26gm3hj3wc3o5nlvvwft4lad6ngre76kyjuw63eyciegxmd.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/c3zw26b3if5ad7na6oq4uihubi45imzjonjc5xkcs4kshoa7u3lu.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/cbpmmgpymme5e5deysrtvqkpqxp3ak3jsgwnvqkt6jotvzojb3ci.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/cuw3h6ozeixh5mqp54nt44sgky5dfdrdgknlbhw3dz6xgbmhpdxf.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/csbtmp7akkfebxa4cwwgpf4lnssooptn46my4274afg4g2babm6g.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/c676gwke3qpj2yjj3akc74xndbe6jl7idikdvuyz2phsrwvcftim.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/cmt5ppczxxa4q73xbsur3ao24cq4n5t4ryebe3lupu44jmcidjxq.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/cqqsmtxtmfalmu5pua4v2omod4raauxko4i3rdboccvfadzf2ghp.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/c3loerzgt2ftqga7igfaeulzkuuavytw2v35agpfalklisfzaogh.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/celbgvy2zwlgqwrliqbcc6xai2z6sowb56d4m2mbyhoqm6k2kjne.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/cpevqxptowrcjwtnwjp2y6pwhhxuvu7n6y6wdg5cm5oqrmlqfjfa.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/c2hxe4l2c2xbbly6ykpa5oq7kguqyld2l77xxoi3o2itxp7n7cki.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/c6mocmn67pg5tylothqacas5pqdtsm74zvoviqugdhzc6webgcys.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/c52awljwyyqhwgm5nyw4vwbfpw6xvbzcvozxp454a5xmmy4ysgib.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/c7jjs5qgw52dafckch4n5xkwqlwe62q5lxoaoharlplk6wvfoiwq.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/c2d575tvqkplsnmwjpt56463tqsct66pojbc3z5snggco3yai3oh.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/cymshjw44xdmodbmdst2t3ulmvfbmnrwk77q724ks2an7fkgricj.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/clcipnjoej3edprogdpq22gzne7exqytivisaric2yabqwutvrgh.cubin.o)
target_sources(aoti_libtorch_free_single_binary_run PRIVATE runner_libtorch_free_single_binary/cybdk2w7jmkzulymnppz6xvpxisfet6ed6tzlgpehgqfh6cspxsa.cubin.o)


target_compile_options(aoti_libtorch_free_single_binary_run PUBLIC -DAOTI_LIBTORCH_FREE -D__AOTI_MODEL__ -DUSE_CUDA -DUSE_EXTERNAL_MZCRC -DUSE_MMAP_SELF -DSPEPERATE_WEIGHTS)
target_compile_options(aoti_libtorch_free_single_binary_run PRIVATE  -O3 -DNDEBUG -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-math-errno -fexcess-precision=fast -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -fno-tree-loop-vectorize -march=native -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -fopenmp)
# Backend specific flags
target_compile_options(aoti_libtorch_free_single_binary_run PRIVATE  -D_GLIBCXX_USE_CXX11_ABI=1)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/serialized_weights.bin
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/runner_libtorch_free_single_binary/serialized_weights.bin ${CMAKE_CURRENT_BINARY_DIR}/serialized_weights.bin 
)

add_custom_target(serialized_weights
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/serialized_weights.bin
)

add_dependencies(aoti_libtorch_free_single_binary_run serialized_weights)


target_include_directories(aoti_libtorch_free_single_binary_run PRIVATE
    runner_libtorch_free_single_binary/include
)

target_link_libraries(aoti_libtorch_free_single_binary_run m)

# Find CUDA package
find_package(CUDA REQUIRED)

enable_language(CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)
target_include_directories(aoti_libtorch_free_single_binary_run PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(aoti_libtorch_free_single_binary_run cuda cublas)

set_property(TARGET aoti_libtorch_free_single_binary_run PROPERTY
    CXX_STANDARD 17
)
