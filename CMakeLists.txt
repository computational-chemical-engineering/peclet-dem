cmake_minimum_required(VERSION 3.24)
project(demgpu LANGUAGES CXX)

# Canonical packing-gpu build: the Kokkos + ArborX `demgpu` XPBD DEM module (CUDA / HIP / OpenMP backends,
# selected by the install prefix). Kokkos, ArborX and pybind11 are found via find_package against the
# bootstrapped install prefix (../extern/install/<backend>, built once by ../tools/bootstrap_deps.sh -- a
# HARD build dependency). With nvcc, put it on PATH (export PATH=/usr/local/cuda-13.2/bin:$PATH).
#
# Build (single-rank Python module):
#   cmake -S . -B build \
#     -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda;$(python -m pybind11 --cmakedir)"
#   cmake --build build -j        ->  build/demgpu.*.so
#
# Multi-rank: -DDEMGPU_MPI=ON links MPI + exposes init_mpi/enable_mpi_step/step_mpi in the module; the
# distributed step is also covered by the tests/kokkos_mpi ctests (run under mpirun).

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(DEMGPU_MPI "Build the demgpu module with the distributed (MPI) step exposed" OFF)

find_package(Kokkos CONFIG REQUIRED)
find_package(ArborX CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "demgpu: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES}), ArborX ${ArborX_VERSION}")

# NO_EXTRAS: pybind LTO does not mix with the nvcc launch-compiler objects.
pybind11_add_module(demgpu NO_EXTRAS src/demgpu_bindings.cpp)
target_include_directories(demgpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)
target_link_libraries(demgpu PRIVATE ArborX::ArborX Kokkos::kokkos)

if(DEMGPU_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  set(TPX_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../transport-core/include")
  if(NOT EXISTS "${TPX_INCLUDE}/tpx/halo/particle_halo_kokkos.hpp")
    message(FATAL_ERROR "transport-core not found at ${TPX_INCLUDE} (clone it as a sibling repo)")
  endif()
  target_compile_definitions(demgpu PRIVATE DEMGPU_MPI=1)
  target_include_directories(demgpu PRIVATE ${TPX_INCLUDE})
  target_link_libraries(demgpu PRIVATE MPI::MPI_CXX)
  message(STATUS "demgpu: distributed step ENABLED (MPI)")
endif()
