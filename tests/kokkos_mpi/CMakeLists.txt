# packing-gpu — distributed (MPI) Kokkos demStep test (the last suite Kokkos-migration piece).
#
# Standalone find_package project (like tests/kokkos + tests/arborx, and cfd's tests/kokkos_mpi):
# Kokkos + ArborX + MPI + the header-only transport-core + the dem Kokkos headers (src/). The
# distributed demStep is gated behind DEM_MPI (the default dem module never defines
# it -> byte-identical single-GPU). Build:
#   cmake -S tests/kokkos_mpi -B build_kmpi \
#         -DCMAKE_PREFIX_PATH="<suite>/extern/install/<backend>" \
#         -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
#   cmake --build build_kmpi -j && ctest --test-dir build_kmpi --output-on-failure
cmake_minimum_required(VERSION 3.24)
project(dem_mpi_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Kokkos CONFIG REQUIRED)
find_package(ArborX CONFIG REQUIRED)
find_package(MPI REQUIRED)

set(TPX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../transport-core" CACHE PATH "transport-core repo")
if(NOT MPIEXEC_EXECUTABLE)
  set(MPIEXEC_EXECUTABLE /usr/bin/mpirun)
endif()

enable_testing()
foreach(t demstep_mpi)
  add_executable(test_${t} test_${t}.cpp)
  target_include_directories(test_${t} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src ${TPX_DIR}/include)
  target_compile_definitions(test_${t} PRIVATE DEM_MPI)
  target_link_libraries(test_${t} PRIVATE ArborX::ArborX Kokkos::kokkos MPI::MPI_CXX)
  foreach(np 1 2 4)
    # closed (non-periodic) box + fully-periodic lattice (the periodic case exercises the local
    # periodic self-ghosts on undecomposed axes, e.g. z of the np=4 2x2x1 ORB layout).
    add_test(NAME ${t}_closed_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:test_${t}> closed)
    add_test(NAME ${t}_periodic_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:test_${t}> periodic)
  endforeach()
endforeach()
