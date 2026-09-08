# ArborX broad-phase correctness + the full single-rank pipeline (Kokkos + ArborX, no MPI). Normally
# built by the root CMake under -DPECLET_DEM_BUILD_TESTS=ON; the standalone form is kept:
#   cmake -S tests/arborx -B build_arborx -DCMAKE_PREFIX_PATH=<suite>/extern/install/<backend>
#   cmake --build build_arborx -j && ctest --test-dir build_arborx --output-on-failure
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
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
  # dem_portable.hpp delegates the SDFs to the header-only core (the sibling checkout by default).
  set(PECLET_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../core" CACHE PATH "core repo")
  set(PECLET_CORE_INCLUDE "${PECLET_CORE_DIR}/include")
  enable_testing()
endif()

set(_dem_src "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
foreach(t broadphase_arborx pipeline)
  add_executable(test_${t} test_${t}.cpp)
  target_include_directories(test_${t} PRIVATE ${_dem_src} ${PECLET_CORE_INCLUDE})
  target_link_libraries(test_${t} PRIVATE ArborX::ArborX Kokkos::kokkos)
  add_test(NAME ${t} COMMAND test_${t})
  set_tests_properties(${t} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
