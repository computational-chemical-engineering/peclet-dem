# Standalone portable-kernel unit tests (Phase 2). Kokkos-only (no ArborX, no cuBQL): each ported
# compute kernel is validated in isolation before demgpu is flipped onto Kokkos. Build:
#   cmake -S tests/kokkos -B build_kokkos -DCMAKE_PREFIX_PATH=<suite>/extern/install/<backend>
#   cmake --build build_kokkos -j && ctest --test-dir build_kokkos --output-on-failure
cmake_minimum_required(VERSION 3.24)
project(dem_kokkos_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Kokkos CONFIG REQUIRED)
message(STATUS "dem/kokkos: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES})")

enable_testing()
add_executable(test_contact_preprocessing test_contact_preprocessing.cpp)
target_include_directories(test_contact_preprocessing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda)
target_link_libraries(test_contact_preprocessing PRIVATE Kokkos::kokkos)
add_test(NAME contact_preprocessing COMMAND test_contact_preprocessing)

add_executable(test_narrowphase test_narrowphase.cpp)
target_include_directories(test_narrowphase PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda)
target_link_libraries(test_narrowphase PRIVATE Kokkos::kokkos)
add_test(NAME narrowphase COMMAND test_narrowphase)

add_executable(test_solver_velocity test_solver_velocity.cpp)
target_include_directories(test_solver_velocity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda)
target_link_libraries(test_solver_velocity PRIVATE Kokkos::kokkos)
add_test(NAME solver_velocity COMMAND test_solver_velocity)
