cmake_minimum_required(VERSION 3.24)
project(dem LANGUAGES CXX)

# Canonical packing-gpu build: the Kokkos + ArborX `dem` XPBD DEM module (CUDA / HIP / OpenMP backends,
# selected by the install prefix). Kokkos, ArborX and pybind11 are found via find_package against the
# bootstrapped install prefix (../extern/install/<backend>, built once by ../tools/bootstrap_deps.sh -- a
# HARD build dependency). With nvcc, put it on PATH (export PATH=/usr/local/cuda-13.2/bin:$PATH).
#
# Build (single-rank Python module); nanobind is found via the active interpreter (SuiteNanobind):
#   cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
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
message(STATUS "dem: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES}), ArborX ${ArborX_VERSION}")

# nanobind (shared suite helper) + transport-core's header-only View<->ndarray zero-copy bridge
# (tpx/python/ndarray_interop.hpp), used by the binding regardless of the MPI option.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(SuiteNanobind)
suite_require_nanobind()
set(TPX_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../transport-core/include")
if(NOT EXISTS "${TPX_INCLUDE}/tpx/python/ndarray_interop.hpp")
  message(FATAL_ERROR "transport-core not found at ${TPX_INCLUDE} (clone it as a sibling repo)")
endif()

# NB_STATIC: bundle nanobind's runtime into the module; the Kokkos device path is routed by the
# launch compiler regardless. NOMINSIZE: nanobind's default -Os is rejected by nvcc ("'s': expected a
# number") since Kokkos device sources compile as CXX through the launch compiler.
nanobind_add_module(dem NB_STATIC NOMINSIZE src/dem_bindings.cpp)
# -> peclet.dem._dem (NB_MODULE(_dem)), re-exported by peclet/dem/__init__.py. The __init__ is kept as a
# plain file under packaging/ (OUTSIDE any importable peclet/ dir, so an incomplete source package can
# never shadow the installed one) and staged into <build>/peclet/dem so `PYTHONPATH=<build> python …`
# (`import peclet.dem`) works in the dev loop too.
set_target_properties(dem PROPERTIES OUTPUT_NAME _dem
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/peclet/dem")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/dem_init.py
               ${CMAKE_CURRENT_BINARY_DIR}/peclet/dem/__init__.py COPYONLY)
target_include_directories(dem PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${TPX_INCLUDE})
target_link_libraries(dem PRIVATE ArborX::ArborX Kokkos::kokkos)

if(DEM_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_compile_definitions(dem PRIVATE DEM_MPI=1)
  target_link_libraries(dem PRIVATE MPI::MPI_CXX)
  message(STATUS "dem: distributed step ENABLED (MPI)")
endif()

# --- pip / scikit-build-core install rule -------------------------------------------------------
# `pip install .` (driven by pyproject.toml) builds against a Kokkos+ArborX prefix on
# CMAKE_PREFIX_PATH and installs the extension as a top-level `dem` module. The rule is inert for a
# plain `cmake --build` (no `cmake --install`), so the developer workflow is unchanged. SKBUILD is set
# by scikit-build-core; SKBUILD_PROJECT_NAME-style guards keep it from interfering elsewhere.
if(DEFINED SKBUILD)
  install(TARGETS dem LIBRARY DESTINATION peclet/dem COMPONENT python)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/dem_init.py
          DESTINATION peclet/dem RENAME __init__.py COMPONENT python)
endif()
