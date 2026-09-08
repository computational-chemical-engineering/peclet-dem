# Kokkos-only kernel unit tests (no ArborX, no MPI): each compute kernel validated in isolation against
# a serial reference. Normally built by the root CMake under -DPECLET_DEM_BUILD_TESTS=ON (one tree per
# backend); the standalone form is kept for a quick kernel-only loop:
#   cmake -S tests/kokkos -B build_kokkos -DCMAKE_PREFIX_PATH=<suite>/extern/install/<backend>
#   cmake --build build_kokkos -j && ctest --test-dir build_kokkos --output-on-failure
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.24)
  project(dem_kokkos_kernels LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  endif()
  find_package(Kokkos CONFIG REQUIRED)
  message(STATUS "dem/kokkos: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES})")
  # The header-only core: dem's SDF layer delegates to peclet::core::geom, so the kernel tests need
  # core's include path too (the sibling checkout by default).
  set(PECLET_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../core" CACHE PATH "core repo")
  set(PECLET_CORE_INCLUDE "${PECLET_CORE_DIR}/include")
  enable_testing()
endif()

set(_dem_src "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
foreach(t contact_preprocessing narrowphase solver_velocity solver_position solver_friction
          integration periodicity thermostat)
  add_executable(test_${t} test_${t}.cpp)
  target_include_directories(test_${t} PRIVATE ${_dem_src} ${PECLET_CORE_INCLUDE})
  target_link_libraries(test_${t} PRIVATE Kokkos::kokkos)
  add_test(NAME ${t} COMMAND test_${t})
  set_tests_properties(${t} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
