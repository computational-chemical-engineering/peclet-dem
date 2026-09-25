# Distributed (MPI) dem tests: migration onto a shared decomposition, the distributed step (both the
# XPBD and the force-based engine, closed + periodic, with mid-run rebalancing) vs the single-rank
# step, ownership rebalancing, and the collective schedule under rank-divergent layouts -- each at
# np = 1, 2, 4. Kokkos + ArborX + MPI + the header-only core + the dem headers (src/), compiled with
# PECLET_DEM_MPI (the default module never defines it).
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
foreach(t migrate_mpi demstep_mpi rebalance_mpi halo_schedule_mpi ghost_band_mpi align_mpi
          momentum_mpi ownership_mpi)
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
  #            MPI_COMM_SELF — the order-independent baseline (tight tolerances). The periodic case
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
  # halo_schedule: scenes where a rank-local condition differs across ranks (a one-sided halo: a
  # rank that sends ghosts but receives none; Verlet-skin reuse with only some ranks moving; a
  # same-count, slot-reordering migration under Verlet-skin reuse; skin reuse with empty ranks). A
  # desynchronised exchange schedule DEADLOCKS, so the TIMEOUT is the failure signal (the passing
  # runs take ~1 s).
  foreach(mode one_sided_xpbd one_sided_hertz skin_divergent skin_migrate_reorder skin_empty_rank)
    _dem_mpi_test(halo_schedule_${mode}_np${np} ${np} test_halo_schedule_mpi ${mode})
    set_tests_properties(halo_schedule_${mode}_np${np} PROPERTIES TIMEOUT 120)
  endforeach()
  # ghost_band: a cross-face pair just inside the contact reach, vs the step on MPI_COMM_SELF --
  # the band must cover every pair the narrow phase can make, and be rebuilt when it changes.
  # The chain is conservative (one owner per contact, ghost increments reverse-accumulated), so
  # the thread-order flake is gone and these run on the caller's thread pool.
  foreach(mode band_change default_band margin explicit_rcut)
    _dem_mpi_test(ghost_band_${mode}_np${np} ${np} test_ghost_band_mpi ${mode})
  endforeach()
  _dem_mpi_test(align_np${np} ${np} test_align_mpi)
endforeach()
# momentum: a closed, force-free random cluster straddling every rank face and corner; GATES the
# drift of the global linear momentum, centre of mass and angular momentum per step, split into the
# velocity and the position phase (XPBD engine, frictionless / friction / free-fall PGS / position
# phase alone) and for the force-based Hertz engine (per-mode thresholds in the source, §7 G1 of
# docs/mpi_momentum_conservation.md). np = 8 puts the cluster on the common corner of all blocks.
# cluster_sync3 / cluster_norot: the friction cluster at sync_every = 3 / forward_rotation = false;
# cluster_periodic: the cluster on the corner of a fully periodic box (wrap pairs on every axis).
# cluster_jacobi: the legacy count-averaged Jacobi solves (velocityUseGS off), gated at the serial
# drift of per-body averaging (the per-body counts must be the global ones).
foreach(np 1 2 4 8)
  foreach(mode cluster cluster_friction cluster_pgs cluster_posonly hertz cluster_sync3
               cluster_norot cluster_periodic cluster_jacobi)
    _dem_mpi_test(momentum_${mode}_np${np} ${np} test_momentum_mpi ${mode})
  endforeach()
endforeach()
# ownership: the owner-exclusive contact scheme's building blocks (docs/mpi_momentum_conservation.md).
# reverse_*: ghost -> owner reconciliation of ParticleHalo -- exact integer sums onto the owners,
# ghost == owner afterwards, and a second sync with no writes changes nothing (closed; periodic on
# every axis, so periodic self-ghosts appear at np = 1). exactly_once_*: every active contact is
# owned by exactly one rank -- no key owned twice, the owned union equals the serial active set
# (closed; periodic; drift = particles displaced out of their owner's block, pairs one owner sees).
foreach(np 1 2 4 8)
  foreach(mode reverse_closed reverse_periodic exactly_once_closed exactly_once_periodic
               exactly_once_drift)
    _dem_mpi_test(ownership_${mode}_np${np} ${np} test_ownership_mpi ${mode})
  endforeach()
endforeach()
# followup_*: REPORT-ONLY reproductions of the four defects in docs/contact_evidence/FOLLOWUPS.md
# (they print the numbers and never fail; each source has a kFollowupGate / kMissedGate constant
# that turns the report into a gate once the defect is fixed). hub / hub_posonly: a grain of 10 R
# touching ~300 unit grains, far above the colourings' 63-colour cap (same-colour pairs at the hub).
# friction_pair(_pgs): the legacy friction pass's couple -delta n x J_t on one sliding pair (and its
# absence on the PGS cone). missed_drift_pair / missed_drift_lattice: pairs no rank sees once both
# bodies drifted out of their owners' blocks (rebalance_every = 0). missed_periodic: pairs that wrap
# an undecomposed periodic axis while crossing a rank face (one image per particle and rank).
foreach(np 1 2 4)
  foreach(mode hub hub_posonly)
    _dem_mpi_test(followup_${mode}_np${np} ${np} test_momentum_mpi ${mode})
  endforeach()
endforeach()
_dem_mpi_test(followup_friction_pair_np1 1 test_momentum_mpi friction_pair)
_dem_mpi_test(followup_friction_pair_pgs_np1 1 test_momentum_mpi friction_pair_pgs)
foreach(np 1 2 4 8)
  foreach(mode missed_drift_pair missed_periodic)
    _dem_mpi_test(followup_${mode}_np${np} ${np} test_ownership_mpi ${mode})
  endforeach()
endforeach()
_dem_mpi_test(followup_missed_drift_lattice_np4 4 test_ownership_mpi missed_drift_lattice)
# migrate_to_weights(w, align) == flow's aligned partition: also at np = 8, where the budget's
# alignment and the "fewer aligned boxes than ranks" rejection both differ from np <= 4.
_dem_mpi_test(align_np8 8 test_align_mpi)
