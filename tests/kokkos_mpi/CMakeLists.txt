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
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/PecletDemPinMpiexec.cmake)
if(NOT MPIEXEC_EXECUTABLE)
  set(MPIEXEC_EXECUTABLE /usr/bin/mpirun)
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
  # jacobi_* : the mass-split Jacobi solves (velocityUseGS off) vs the SAME stepMpi on
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
  foreach(mode band_change default_band margin explicit_rcut coax_tubes)
    _dem_mpi_test(ghost_band_${mode}_np${np} ${np} test_ghost_band_mpi ${mode})
  endforeach()
  _dem_mpi_test(align_np${np} ${np} test_align_mpi)
endforeach()
# momentum: a closed, force-free random cluster straddling every rank face and corner; GATES the
# drift of the global linear momentum, centre of mass and angular momentum per step, split into the
# velocity and the position phase (XPBD engine, frictionless / friction / free-fall PGS / position
# phase alone) and for the force-based Hertz engine (per-mode thresholds in the source, §7 G1 of
# docs/mpi_momentum_conservation.md). np = 8 puts the cluster on the common corner of all blocks.
# cluster_shear / hertz_shear (WO-7): a shear flow carries bodies several radii out of their owners'
# blocks -- the drift vote migrates them (both engines), conservation must hold through it.
# cluster_sync3 / cluster_norot: the friction cluster at sync_every = 3 / forward_rotation = false;
# cluster_periodic: the cluster on the corner of a fully periodic box (wrap pairs on every axis).
# cluster_jacobi: the mass-split Jacobi solves (velocityUseGS off; docs/contact_solve_framework.md
# §3.1), conservative at every iterate, gated at round-off (the per-body counts must be the global
# ones).
foreach(np 1 2 4 8)
  foreach(mode cluster cluster_friction cluster_pgs cluster_posonly hertz cluster_sync3
               cluster_norot cluster_periodic cluster_jacobi cluster_shear hertz_shear
               hub hub_pgs hub_posonly tri ring_mini cluster_poisson cluster_multilevel
               cluster_escalate cluster_ordered cluster_e09 cluster_e10 ring_collide)
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
# absence on the PGS cone); friction_pair is GATED since the midpoint-arm fix (WO-1, kFrictionPairGate:
# couple ratio <= 1e-2, the float floor). missed_drift_pair / missed_drift_lattice: pairs no rank sees once both
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
# cluster_pgs_e (docs/contact_physics_followups.md §6 G-C1, WO-C1): the dense frictionless cluster,
# one converged PGS step under Moreau's restitution target: the CoM-frame kinetic energy may not
# grow (1e-5). e = 1.0 runs at dt = 1e-4: at the scene's dt = 1e-2 the resting threshold
# (2 dt |g| = 0.2) holds sub-threshold contacts at e = 0, a bounded creation channel (the note's
# R-C3) measured KE1/KE0 = 1 + 1.26e-4 at np 1..8; it scales with dt (1 + 3e-9 at dt = 1e-4), so
# the law itself is gated where the threshold is negligible (session decision 2026-09-26).
foreach(np 1 2 4 8)
  foreach(e 0.5 0.9)
    _dem_mpi_test(momentum_cluster_pgs_e${e}_moreau_np${np} ${np} test_momentum_mpi cluster_pgs_e
                  --e=${e} --rest-target=moreau)
  endforeach()
  _dem_mpi_test(momentum_cluster_pgs_e1.0_moreau_np${np} ${np} test_momentum_mpi cluster_pgs_e
                --e=1.0 --rest-target=moreau --dt=1e-4)
endforeach()
# docs/contact_solve_framework.md WO-0: report-only instrumentation of the contact-solve framework
# (KE per step, CONFLICTS per step, the tri / ring_mini / stabilization scenes; the dynamic
# visibility oracle). None of them fails today; the framework's gates read their lines.
foreach(np 1 2 4)
  foreach(mode cluster_e09 cluster_e10 cluster_poisson cluster_multilevel cluster_escalate
               cluster_ordered cluster_onesided hub_pgs ring_mini)
    _dem_mpi_test(followup_${mode}_np${np} ${np} test_momentum_mpi ${mode})
  endforeach()
endforeach()
# hub_static / hub_ml (docs/contact_solve_framework.md §13.2, §13.6, WO-4b): the overlap projection
# is not over-relaxed (every leaf of a split hub ends in contact) and a multilevel coarse vertex
# carries the mass of its folded copies (momentum at a hub that aggregates). GATED at every np
# (the rank-level mass split, WO-5); hub_ml's positive controls at np 1 and 2 only (§12 S18).
foreach(np 1 2 4)
  foreach(mode hub_static hub_ml)
    _dem_mpi_test(momentum_${mode}_np${np} ${np} test_momentum_mpi ${mode})
  endforeach()
endforeach()
foreach(np 1 2)  # three bodies: A1/A2 and B on two ranks
  foreach(mode tri tri_pgs)
    _dem_mpi_test(followup_${mode}_np${np} ${np} test_momentum_mpi ${mode})
  endforeach()
endforeach()
foreach(np 1 2 4 8)
  foreach(mode oracle_closed oracle_shear oracle_periodic)
    _dem_mpi_test(followup_${mode}_np${np} ${np} test_ownership_mpi ${mode})
  endforeach()
endforeach()
# migrate_to_weights(w, align) == flow's aligned partition: also at np = 8, where the budget's
# alignment and the "fewer aligned boxes than ranks" rejection both differ from np <= 4.
_dem_mpi_test(align_np8 8 test_align_mpi)

# G13 mutation negative controls (docs/contact_solve_framework.md §9 G13, §13.6): each mutant is
# compiled into a dedicated executable with -DPECLET_DEM_TEST_MUTANT=<n> (production never defines
# it) and its ctest PASSES iff the gate FAILS (WILL_FAIL).
function(_dem_mutant_exe target src n)
  add_executable(${target} ${src})
  target_include_directories(${target} PRIVATE ${_dem_src} ${PECLET_CORE_INCLUDE})
  target_compile_definitions(${target} PRIVATE PECLET_DEM_MPI PECLET_DEM_TEST_MUTANT=${n})
  target_link_libraries(${target} PRIVATE ArborX::ArborX Kokkos::kokkos MPI::MPI_CXX)
endfunction()
_dem_mutant_exe(test_ownership_mpi_mut1 test_ownership_mpi.cpp 1)
_dem_mutant_exe(test_momentum_mpi_mut2 test_momentum_mpi.cpp 2)
_dem_mutant_exe(test_momentum_mpi_mut3 test_momentum_mpi.cpp 3)
_dem_mutant_exe(test_momentum_mpi_mut4 test_momentum_mpi.cpp 4)
_dem_mutant_exe(test_ownership_mpi_mut5 test_ownership_mpi.cpp 5)
_dem_mutant_exe(test_momentum_mpi_mut6 test_momentum_mpi.cpp 6)
_dem_mutant_exe(test_momentum_mpi_mut7 test_momentum_mpi.cpp 7)
# G-A4 (docs/contact_physics_followups.md §6): the rigid prolongation drops (Omega - Omega0) x d.
_dem_mutant_exe(test_momentum_mpi_mut8 test_momentum_mpi.cpp 8)
_dem_mpi_test(mutant1_ownership_oracle_closed_np2 2 test_ownership_mpi_mut1 oracle_closed)
_dem_mpi_test(mutant2_momentum_cluster_pgs_np2 2 test_momentum_mpi_mut2 cluster_pgs)
_dem_mpi_test(mutant2_momentum_hub_pgs_np1 1 test_momentum_mpi_mut2 hub_pgs)
_dem_mpi_test(mutant3_momentum_hub_np1 1 test_momentum_mpi_mut3 hub)
_dem_mpi_test(mutant4_momentum_tri_np2 2 test_momentum_mpi_mut4 tri --axis=2)
_dem_mpi_test(mutant5_ownership_missed_drift_lattice_np4 4 test_ownership_mpi_mut5 missed_drift_lattice)
_dem_mpi_test(mutant6_momentum_hub_ml_np1 1 test_momentum_mpi_mut6 hub_ml)
_dem_mpi_test(mutant7_momentum_hub_static_np1 1 test_momentum_mpi_mut7 hub_static)
_dem_mpi_test(mutant8_momentum_hub_ml_np1 1 test_momentum_mpi_mut8 hub_ml)
foreach(t mutant1_ownership_oracle_closed_np2 mutant2_momentum_cluster_pgs_np2
          mutant2_momentum_hub_pgs_np1 mutant3_momentum_hub_np1 mutant4_momentum_tri_np2
          mutant5_ownership_missed_drift_lattice_np4 mutant6_momentum_hub_ml_np1
          mutant7_momentum_hub_static_np1 mutant8_momentum_hub_ml_np1)
  set_tests_properties(${t} PROPERTIES WILL_FAIL TRUE)
endforeach()


# WO-12 (docs/contact_solve_framework.md §13.5): the accumulated position projection's fixed point
# is unique -- the converged overlap-only substep agrees across rank counts to 1e-4 R.
foreach(np 2 4 8)
  add_test(NAME position_agreement_np${np}
           COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/position_agreement.sh ${MPIEXEC_EXECUTABLE}
                   ${_mpi_preflags} -- $<TARGET_FILE:test_momentum_mpi> ${np}
                   ${CMAKE_CURRENT_BINARY_DIR}/position_agreement_np${np})
  set_tests_properties(position_agreement_np${np} PROPERTIES PROCESSORS ${np} LABELS mpi)
endforeach()
# ring_collide (docs/contact_physics_followups.md §3.4, §6 G-B2, WO-B2): the overlap-free tube
# scene. Conservation is gated with the momentum_* modes above; here its feasibility -- with the
# stops on, the position loop converges below a 2000-iteration cap in EVERY step (the
# tunnelled ring_mini runs to the cap), the committed overlap at the default 20 iterations stays
# <= 5e-3 -- and the four-step np agreement of the position-only variant to 1e-4 R.
_dem_mpi_test(ring_collide_converged_np1 1 test_momentum_mpi ring_collide --pos-iters=2000
              --gate-pos-cap)
_dem_mpi_test(ring_collide_overlap_np1 1 test_momentum_mpi ring_collide --gate-overlap=5e-3)
foreach(np 2 4 8)
  add_test(NAME position_agreement_ring_collide_np${np}
           COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/position_agreement.sh ${MPIEXEC_EXECUTABLE}
                   ${_mpi_preflags} -- $<TARGET_FILE:test_momentum_mpi> ${np}
                   ${CMAKE_CURRENT_BINARY_DIR}/position_agreement_ring_collide_np${np}
                   ring_collide_posonly)
  set_tests_properties(position_agreement_ring_collide_np${np} PROPERTIES PROCESSORS ${np}
                       LABELS mpi)
endforeach()
