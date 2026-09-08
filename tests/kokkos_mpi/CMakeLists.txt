# Distributed (MPI) dem tests: migration onto a shared decomposition, the distributed step (both the
# XPBD and the force-based engine, closed + periodic, with mid-run rebalancing) vs the single-rank
# step, and ownership rebalancing -- each at np = 1, 2, 4. Kokkos + ArborX + MPI + the header-only
# core + the dem headers (src/), compiled with PECLET_DEM_MPI (the default module never defines it).
# Normally built by the root CMake under -DPECLET_DEM_BUILD_TESTS=ON -DPECLET_DEM_MPI=ON; the
# standalone form is kept:
#   cmake -S tests/kokkos_mpi -B build_kmpi -DCMAKE_PREFIX_PATH="<suite>/extern/install/<backend>" \
#         -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
#   cmake --build build_kmpi -j && ctest --test-dir build_kmpi --output-on-failure
# MPIEXEC_PREFLAGS (e.g. --oversubscribe on a 4-core runner) is passed through to every launch.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.24)
  project(dem_mpi_tests LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  endif()
  find_package(Kokkos CONFIG REQUIRED)
  find_package(ArborX CONFIG REQUIRED)
  find_package(MPI REQUIRED COMPONENTS CXX)
  set(PECLET_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../core" CACHE PATH "core repo")
  set(PECLET_CORE_INCLUDE "${PECLET_CORE_DIR}/include")
  enable_testing()
endif()
if(NOT MPIEXEC_EXECUTABLE)
  set(MPIEXEC_EXECUTABLE /usr/bin/mpirun)   # FindMPI may otherwise pick a bundled (e.g. ParaView) mpirun
endif()
if(NOT MPIEXEC_NUMPROC_FLAG)
  set(MPIEXEC_NUMPROC_FLAG -np)
endif()
separate_arguments(_mpi_preflags NATIVE_COMMAND "${MPIEXEC_PREFLAGS}")

set(_dem_src "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
foreach(t migrate_mpi demstep_mpi rebalance_mpi)
  add_executable(test_${t} test_${t}.cpp)
  target_include_directories(test_${t} PRIVATE ${_dem_src} ${PECLET_CORE_INCLUDE})
  target_compile_definitions(test_${t} PRIVATE PECLET_DEM_MPI)
  target_link_libraries(test_${t} PRIVATE ArborX::ArborX Kokkos::kokkos MPI::MPI_CXX)
endforeach()

function(_dem_mpi_test name np target)   # ARGN = program arguments
  add_test(NAME ${name}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_mpi_preflags}
                   $<TARGET_FILE:${target}> ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 PROCESSORS ${np} LABELS mpi)
endfunction()

foreach(np 1 2 4)
  # migrateTo onto a shared decomposition (co-rebalance; PECLET_DEM_MPI guards mpi_halo.hpp).
  _dem_mpi_test(migrate_np${np} ${np} test_migrate_mpi)
  # jacobi_* : legacy count-averaged Jacobi solves (velocityUseGS off) vs the SAME stepMpi on
  #            MPI_COMM_SELF — the exact-redundant baseline (tight tolerances). The periodic case
  #            exercises the local periodic self-ghosts on undecomposed axes.
  # modern   : the full modern solver stack (gravity + per-pair materials + friction + warm-started
  #            PGS + statics) vs the REAL single-rank demStep, tolerance-based.
  # modern_rebal : modern + periodic ownership rebalancing mid-run (persistent-ledger carry).
  # hertz(_rebal): the FORCE-BASED engine (explicit Hertz-Mindlin through the demStepForce driver)
  #            vs the real single-rank step_hertz; _rebal migrates the Mindlin history mid-run.
  foreach(mode jacobi_closed jacobi_periodic modern modern_rebal hertz hertz_rebal)
    _dem_mpi_test(demstep_${mode}_np${np} ${np} test_demstep_mpi ${mode})
  endforeach()
  _dem_mpi_test(rebalance_np${np} ${np} test_rebalance_mpi)
endforeach()
