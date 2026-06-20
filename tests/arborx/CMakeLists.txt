# Standalone ArborX broad-phase correctness test (Phase 2, milestone 1).
#
# Deliberately isolated from the main demgpu build (which fetches cuBQL and compiles the whole CUDA
# module): this only needs Kokkos + ArborX, consumed via find_package against a cluster module or the
# suite's local install prefix (suite/tools/bootstrap_deps.sh) on CMAKE_PREFIX_PATH. Build:
#   cmake -S tests/arborx -B build_arborx -DCMAKE_PREFIX_PATH=<suite>/extern/install/<backend>
#   cmake --build build_arborx -j && ctest --test-dir build_arborx --output-on-failure
cmake_minimum_required(VERSION 3.24)
project(dem_arborx_broadphase LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Kokkos CONFIG REQUIRED)
find_package(ArborX CONFIG REQUIRED)
message(STATUS "dem/arborx: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES}), ArborX ${ArborX_VERSION}")

enable_testing()
add_executable(test_broadphase_arborx test_broadphase_arborx.cpp)
target_include_directories(test_broadphase_arborx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_broadphase_arborx PRIVATE ArborX::ArborX Kokkos::kokkos)
add_test(NAME broadphase_arborx COMMAND test_broadphase_arborx)

add_executable(test_pipeline test_pipeline.cpp)
target_include_directories(test_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_pipeline PRIVATE ArborX::ArborX Kokkos::kokkos)
add_test(NAME pipeline COMMAND test_pipeline)
