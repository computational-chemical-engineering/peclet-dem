# demgpu_kokkos — Kokkos+ArborX pybind11 module (the flip target), additive to the CUDA demgpu.so.
#
# Deliberately separate from the main demgpu build (which FetchContents cuBQL): this needs only
# Kokkos + ArborX + pybind11 (no cuBQL, no thrust). Build:
#   cmake -S kokkos_module -B build_module \
#         -DCMAKE_PREFIX_PATH="<suite>/extern/install/<backend>;$(python -m pybind11 --cmakedir)"
#   cmake --build build_module -j
# Produces demgpu_kokkos.*.so; import it with that build dir on PYTHONPATH.
cmake_minimum_required(VERSION 3.24)
project(demgpu_kokkos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Kokkos CONFIG REQUIRED)
find_package(ArborX CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "demgpu_kokkos: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES}), ArborX ${ArborX_VERSION}")

# NO_EXTRAS disables pybind11's link-time optimization, which does not mix with the nvcc-compiled
# (Kokkos launch-compiler) objects (lto-wrapper link failure).
pybind11_add_module(demgpu_kokkos NO_EXTRAS binding.cpp)
target_include_directories(demgpu_kokkos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda)
target_link_libraries(demgpu_kokkos PRIVATE ArborX::ArborX Kokkos::kokkos)
