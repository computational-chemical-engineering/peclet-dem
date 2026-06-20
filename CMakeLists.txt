cmake_minimum_required(VERSION 3.24)
project(dem LANGUAGES CXX)

# Canonical packing-gpu build: the Kokkos + ArborX `dem` XPBD DEM module (CUDA / HIP / OpenMP backends,
# selected by the install prefix). Kokkos, ArborX and pybind11 are found via find_package against the
# bootstrapped install prefix (../extern/install/<backend>, built once by ../tools/bootstrap_deps.sh -- a
# HARD build dependency). With nvcc, put it on PATH (export PATH=/usr/local/cuda-13.2/bin:$PATH).
#
# Build (single-rank Python module):
#   cmake -S . -B build \
#     -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda;$(python -m pybind11 --cmakedir)"
#   cmake --build build -j        ->  build/dem.*.so
#
# Multi-rank: -DDEM_MPI=ON links MPI + exposes init_mpi/enable_mpi_step/step_mpi in the module; the
# distributed step is also covered by the tests/kokkos_mpi ctests (run under mpirun).

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(DEM_MPI "Build the dem module with the distributed (MPI) step exposed" OFF)

find_package(Kokkos CONFIG REQUIRED)
find_package(ArborX CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "dem: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES}), ArborX ${ArborX_VERSION}")

# NO_EXTRAS: pybind LTO does not mix with the nvcc launch-compiler objects.
pybind11_add_module(dem NO_EXTRAS src/dem_bindings.cpp)
target_include_directories(dem PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dem PRIVATE ArborX::ArborX Kokkos::kokkos)

if(DEM_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  set(TPX_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../transport-core/include")
  if(NOT EXISTS "${TPX_INCLUDE}/tpx/halo/particle_halo_kokkos.hpp")
    message(FATAL_ERROR "transport-core not found at ${TPX_INCLUDE} (clone it as a sibling repo)")
  endif()
  target_compile_definitions(dem PRIVATE DEM_MPI=1)
  target_include_directories(dem PRIVATE ${TPX_INCLUDE})
  target_link_libraries(dem PRIVATE MPI::MPI_CXX)
  message(STATUS "dem: distributed step ENABLED (MPI)")
endif()

# --- pip / scikit-build-core install rule -------------------------------------------------------
# `pip install .` (driven by pyproject.toml) builds against a Kokkos+ArborX prefix on
# CMAKE_PREFIX_PATH and installs the extension as a top-level `dem` module. The rule is inert for a
# plain `cmake --build` (no `cmake --install`), so the developer workflow is unchanged. SKBUILD is set
# by scikit-build-core; SKBUILD_PROJECT_NAME-style guards keep it from interfering elsewhere.
if(DEFINED SKBUILD)
  install(TARGETS dem LIBRARY DESTINATION . COMPONENT python)
endif()
