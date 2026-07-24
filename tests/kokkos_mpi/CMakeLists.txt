# packing-gpu — distributed (MPI) Kokkos demStep test (the last suite Kokkos-migration piece).
#
# Standalone find_package project (like tests/kokkos + tests/arborx, and cfd's tests/kokkos_mpi):
# Kokkos + ArborX + MPI + the header-only core + the dem Kokkos headers (src/). The
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

set(TPX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../core" CACHE PATH "core repo")
if(NOT MPIEXEC_EXECUTABLE)
  set(MPIEXEC_EXECUTABLE /usr/bin/mpirun)
endif()

enable_testing()
# migrateTo onto a shared decomposition (co-rebalance; PECLET_DEM_MPI guards mpi_halo.hpp).
add_executable(test_migrate_mpi test_migrate_mpi.cpp)
target_include_directories(test_migrate_mpi PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src ${TPX_DIR}/include)
target_compile_definitions(test_migrate_mpi PRIVATE PECLET_DEM_MPI)
target_link_libraries(test_migrate_mpi PRIVATE ArborX::ArborX Kokkos::kokkos MPI::MPI_CXX)
foreach(np 1 2 4)
  add_test(NAME migrate_np${np}
           COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:test_migrate_mpi>)
endforeach()

foreach(t demstep_mpi rebalance_mpi)
  add_executable(test_${t} test_${t}.cpp)
  target_include_directories(test_${t} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src ${TPX_DIR}/include)
  target_compile_definitions(test_${t} PRIVATE PECLET_DEM_MPI)
  target_link_libraries(test_${t} PRIVATE ArborX::ArborX Kokkos::kokkos MPI::MPI_CXX)
endforeach()

foreach(np 1 2 4)
  # jacobi_* : legacy count-averaged Jacobi solves (velocityUseGS off) vs the SAME stepMpi on
  #            MPI_COMM_SELF — the exact-redundant baseline (tight tolerances). The periodic case
  #            exercises the local periodic self-ghosts on undecomposed axes.
  # modern   : the full modern solver stack (gravity + per-pair materials + friction + warm-started
  #            PGS + statics) vs the REAL single-rank demStep, tolerance-based.
  # modern_rebal : modern + periodic ownership rebalancing mid-run (persistent-ledger carry).
  # hertz(_rebal): the FORCE-BASED engine (explicit Hertz-Mindlin through the demStepForce driver)
  #            vs the real single-rank step_hertz; _rebal migrates the Mindlin history mid-run.
  foreach(mode jacobi_closed jacobi_periodic modern modern_rebal hertz hertz_rebal)
    add_test(NAME demstep_${mode}_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:test_demstep_mpi> ${mode})
  endforeach()
  add_test(NAME rebalance_np${np}
           COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:test_rebalance_mpi>)
endforeach()
