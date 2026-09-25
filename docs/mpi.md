# dem — distributed (MPI) step

MPI block-parallelism for the DEM/XPBD solver on the shared `core` library (sibling repo
`../../core`), mirroring the approach validated for `flow` (the Eulerian precedent). The Lagrangian counterpart to
the Eulerian grid halo is **particle migration** (reassign particles to their owning rank) + **ghost
particles** (copies within one interaction radius of a block boundary, so each rank's ArborX broad-phase
runs locally), plus periodic **load rebalancing** (weighted-ORB SoA ownership migration).

The distributed step ships **inside the `peclet.dem` Kokkos module** as `step_mpi`, gated behind the
`PECLET_DEM_MPI` build option (the default module never defines it, so the single-rank module stays
byte-identical). This document is the status + how-to-build/run + what-is-validated for that step. The Python
validation scripts it refers to live in `../tests/python/mpi/` (pytest files launched through
`mpirun`, registered as ctests by `-DPECLET_DEM_BUILD_TESTS=ON`); the driver skeleton and the
microbenchmark in `../examples/`.

## The shipped distributed step (`dem`, `-DPECLET_DEM_MPI=ON`)

Build the `peclet.dem` module with MPI exposed, against the bootstrapped Kokkos prefix:
```bash
cd dem && source ../.venv/bin/activate
cmake -S . -B build -DPECLET_DEM_MPI=ON -DCMAKE_PREFIX_PATH="$PWD/../extern/install/<backend>"
cmake --build build -j$(nproc)            # -> build/peclet/dem/_dem.*.so with init_mpi/enable_mpi_step/step_mpi
```

Python surface (gated; present only when built with `-DPECLET_DEM_MPI=ON`):
```python
from peclet import dem
sim = dem.Simulation()
sim.initialize_shape(...); sim.set_positions(...)        # as usual
sim.init_mpi(origin, extent, cells, periodic)            # ORB block decomposition + core particle halo
sim.set_dt(dt)                                           # every stepper uses it; none takes dt
sim.enable_mpi_step(rcut=0.0, sync_every=1,              # band lower bound (0 = the contact reach); refresh cadence
                    forward_rotation=True,               # False = spheres (skips quaternion forward)
                    rebalance_every=0)                   # >0 = re-decompose by particle count every N steps
sim.step_mpi(n)                                          # advance n steps with halo exchange
sim.rebalance()                                          # force a load rebalance now (returns new owned count)
sim.migrate_to_weights(w, align=1)                       # co-rebalance onto the weighted ORB of per-cell weights
                                                         # w (x-fastest over `cells`); align = flow's return value
# diagnostics: sim.rank, sim.num_ghost (properties);
#              sim.diagnostics.mpi_rebuilds / .mpi_gathers (ghost-reuse ratio)
```

### Implementation
- `src/mpi_halo.hpp` (`ParticleHalo`) — a thin wrapper over core's
  `peclet::core::halo::ParticleHaloTopology<3>` (host topology + periodic image shift), `peclet::core::halo::ParticleHalo<3>`
  (on-device gather/scatter + host-staged MPI), `peclet::core::halo::ParticleMigrator`, and the
  weighted-ORB `particle_rebalance` path. Rebuilt each substep from the owned positions.
- `src/step_solve_mpi.hpp` (`demStepMpi`) — the distributed substep. The periodic ghost generation of the
  single-rank step is replaced by a cross-rank gather (ghosts carry **real** mass). The contacts are
  partitioned owned-first (`ContactOwnership`, `partitionContactsKokkos`) and reduced to manifolds
  owned-first, and each rank solves only the contacts it owns; the partner's half of every impulse
  lands in the partner's self-mapped ghost slot and is **reverse-accumulated onto its owner** at the
  next sync, after which the owners refresh their ghost copies (velPred/angVelPred, then
  posPred/quatPred) — every `sync_every` solver iterations and after every phase (see *Contact
  ownership* below). Friction (wall + body-body Coulomb) **is** carried — same kernels as the
  single-rank step.
- `src/solve_driver.hpp` (`demSolveContacts`) — **one driver, two hook policies.** The distributed
  step runs the SAME modern solve sequence as `step(n)` (graph-colored Gauss–Seidel restitution,
  warm-started PGS with persistent contacts, gravity statics / stabilization passes, friction cone,
  colored-GS overlap projection, adaptive stops) in its processor-block Gauss–Seidel form:
  `SoloSolveHooks` compiles the single-GPU sequence, `MpiSolveHooks` adds the owner/ghost
  reconciliations (ghost→owner reverse, then owner→ghost forward) and the `MPI_Allreduce(MAX)` on
  every adaptive-stop residual (a rank-local break would
  desynchronise the collective refreshes and deadlock). Persistent-contact pair keys are built from
  **global ids**, so the ledger survives halo rebuilds and ownership migration, and `MigratePack`
  carries each particle's slice of it across a rebalance. The force-based engine has the same shape
  in `src/solve_driver_force.hpp` (`SoloForceHooks` / `MpiForceHooks`, Mindlin history keyed by gid).
- **Periodicity:** cross-rank ghosts supply the wrap on *decomposed* axes; local periodic self-ghosts
  (the halo built with `includePeriodicSelf`) supply it on *undecomposed* periodic axes.

### Contact ownership, reconciliation and the `sync_every` (M) knob
Every contact is solved by **exactly one** rank: the only owner that sees it, or, when both
owners see it, the owner of its lower-gid body (for a periodic self-image twin: the twin whose
real body has the lower gid). Each rank sweeps only the contacts it owns, writing the partner's
half of every impulse into the partner's ghost slot. At every sync, each ghost's accumulated
change is **added onto its owner** (core `ParticleHalo::reverse`) and the owners then republish
their state (forward). Every impulse and every overlap correction is therefore applied once,
equal and opposite: linear momentum and the centre of mass are conserved to round-off at any np,
thread count and `sync_every`, and the velocity phase conserves angular momentum wherever the
serial model does. `sync_every=M` sets how many sweeps pass between reconciliations: conservation
does not depend on it; larger M means more lag at rank faces and fewer messages.
Interior contacts keep the serial Gauss–Seidel order; a contact across a rank face sees its far
body as of the last reconciliation, so trajectories agree with single-rank statistically, not
bit-for-bit (numbers under *What is validated*). `forward_rotation=False` skips the ghost
angular-velocity and quaternion forwards (spheres).

Under Poisson (event-level) restitution two ranks can draw from one body's orphan account within
one reconciliation interval, each bounded by its own view of the balance; the owner clamps the
delivered balance at 0 (docs/mpi_momentum_conservation.md, R5).

The count-averaged solves divide a body's summed correction by its per-body contact count: the
legacy friction pass, the Jacobi A/B (`set_velocity_solver('jacobi')`) and the colour-saturation
fallbacks of the Gauss–Seidel loops. Each rank counts only the contacts it owns, so before every
such divide the ghosts' partial counts are summed onto their owners and the totals forwarded back
(`syncFrictionCounts`, `syncContactCounts`): every copy of a body divides by the serial count.
The saturation fallback is decided per rank, so its activation is voted in the same
`MPI_Allreduce` as the loop's stop residual (no extra message); the count exchange then runs on
every rank, only in iterations where some rank has uncoloured leftovers.

### What is validated
- `tests/kokkos_mpi/` — the distributed Kokkos `demStep`/`rebalance` ctests, run under `mpirun` at
  **np=1,2,4**, in both a closed (non-periodic) box and a fully-periodic lattice (the periodic case
  exercises the local periodic self-ghosts on undecomposed axes). Build/run (from the root tree):
  ```bash
  cmake -S . -B build_dev -DCMAKE_PREFIX_PATH="<suite>/extern/install/<backend>" \
        -DPECLET_DEM_MPI=ON -DPECLET_DEM_BUILD_TESTS=ON -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
  cmake --build build_dev -j && OMP_NUM_THREADS=1 ctest --test-dir build_dev -L mpi --output-on-failure
  ```
  (`tests/kokkos_mpi` also still configures standalone.)
- `tests/python/mpi/` — the Python drivers on core's `peclet.core.mpi` + mpi4py, registered as the
  `python_mpi_*_np{1,2,4}` ctests (exit 77 = SKIP when that stack is missing): `test_validate_exact`
  (per-particle vs serial), `test_validate_periodic` (wrap through the split axes: 2-body, corner,
  N-body with resting straddlers), `test_verify_distributed` (elastic energy + settling-pack
  observables) and `test_verify_rotating_drum_mpi` (moving SDF wall + rebalancing).
- np=1 agrees with the single-rank step to float noise (max 8.5e-5 over 15 steps). At np=2/4 the
  modern stack (processor-block Gauss–Seidel with rank-local colouring, one owner per contact)
  sweeps the finite-iteration PGS in a different order than single-rank, and a contact across a
  rank face sees its far body as of the last reconciliation, so per-particle agreement on a stiff,
  randomly overlapping IC is *statistical* (measured 2026-09-25, N=200, 15 steps, OMP 1: np=2 mean
  3.9e-3, 95 % quantile 3.2e-2, max 6.6e-2; np=4 mean 4.0e-3, 95 % quantile 3.2e-2, max 7.2e-2 --
  the redundant two-owner scheme gave 5.3e-3 / 3.6e-2 / 0.10 and 5.0e-3 / 3.5e-2 / 0.11 in the same
  session); resting wrap contacts agree exactly and the aggregate observables (energy, overlap,
  pile geometry) match to their tolerances. The C++ `demstep_*` ctests are the tolerance-based
  statement of the same thing.
- `momentum_*` gates conservation (dP ≤ 1e-6, CoM ≤ 1e-5 R, velocity-phase dL ≤ 1e-6 frictionless,
  ≤ 1e-4 with the legacy friction pass) at np 1/2/4/8, and `ownership_*` gates that every contact
  is owned exactly once.

### Validation lessons carried into the Kokkos step
- **Full per-particle state must travel through migration** (quaternion + angular velocity, not just
  velocity/id) — otherwise off-centre collisions' spin is discarded and rotational KE leaks.
- **The predicted position must be *forwarded*, not copied from the committed position** at gather time:
  `predict_velocity` already advances `posPred = pos + v·dt`, so copying would leave ghosts one
  predict-step stale and systematically dissipate energy at the boundary.
- **A periodic axis only works distributed if it is split across ≥2 ranks** (a rank never ghosts to
  itself for the cross-rank wrap; undecomposed periodic axes use the local self-ghosts instead).
- **The MPI call sequence must be identical on every rank — never gate an exchange on a rank-local
  condition.** The halo is not symmetric: a rank can owe ghosts to a neighbour that owes it none.
  Skipping the forwards on `numGhost == 0` deadlocked the step whenever such a rank existed
  (coupling's moving suspension at np=4, 2026-09-24); a forward is now skipped only when the rank
  neither sends nor receives. Likewise the Verlet-skin rebuild decision is Allreduce-OR'd (the
  rebuild is an NBX collective and one rank's particles can cross the skin while its neighbours'
  rest). `tests/kokkos_mpi/test_halo_schedule_mpi.cpp` (`one_sided_xpbd`, `one_sided_hertz`,
  `skin_divergent`, ctest TIMEOUT 120 s) pins both.
- **The ghost band must cover the contact reach, over the GLOBAL maximum radius** (2026-09-24).
  The XPBD narrow phase reports a pair while the gap is below the margin 0.1 R_max, so a partner
  across a block face can sit up to 2 R_max + 0.1 R_max from it. The step uses
  `max(rcut, 2.1 R_max_global)` (Allreduce MAX, re-evaluated every step, growth included) and the
  margin is 0.1 R_max_global on every rank; the ghost lists are rebuilt whenever the band changes.
  Before, the default band was one rank-local radius, an explicit `rcut = 2r` missed the margin,
  a polydisperse run's ranks disagreed on the margin, and a larger band or skin reused lists built
  with the old one -- each resolved a cross-face pair on one side only.
  `tests/kokkos_mpi/test_ghost_band_mpi.cpp` (`band_change`, `default_band`, `margin`,
  `explicit_rcut`) pins all four against `MPI_COMM_SELF`.

### Load rebalancing
With `rebalance_every=N` (or an explicit `sim.rebalance()`), the decomposition is recomputed by
particle count (weighted ORB) and SoA ownership is migrated — keeping per-rank work balanced as the
packing evolves. See `core`'s `BlockDecomposer::init(..., weights)` /
`DistributedOctree::rebalanceByParticleCount` and `../../core/CLAUDE.md`.

**Co-rebalancing with the flow solver** (`peclet.coupling`'s `CfdDem.rebalance()`) goes through
`migrate_to_weights(w, align)` instead: both codes build the partition from the same replicated
weight field `w` (fluid work + gamma · particle count per ORB cell), so nothing but the split
positions is shared. flow's `diagnostics.rebalance_by_weights(w)` picks an alignment `2^a` for its
pressure multigrid — every split plane on a multiple of `2^a` cells, the largest `a` whose weight
imbalance stays within 1.05 — and returns it; `align=` makes dem build exactly that partition
(core's coarse-first aligned weighted ORB). The default `align=1` is the plain weighted ORB.
`tests/kokkos_mpi/test_align_mpi.cpp` checks dem's partition against flow's call, cell for cell,
at np = 1, 2, 4, 8.

The original host-C++ bring-up harness (particle migration + the three ghost-exchange schemes A/B/C
matched cell-for-cell to a serial reference) validated this machinery before it was wired into the
module; it was retired at 1.0.0 (it targeted core's pre-`peclet::core` headers) and lives in git
history before dem `b43040c`. Since 2026-09 the shipped `step_mpi` uses reverse reduction: owner-exclusive
contacts plus a ghost→owner accumulation at every sync. The earlier "EXACT" variant solved each
cross-rank contact on both owners, which is exact only for Jacobi-type deltas and broke momentum
conservation under the Gauss–Seidel stack.

See [multi_gpu_testing.md](multi_gpu_testing.md) for the multi-GPU profiling/scaling playbook, `../../docs/ROADMAP.md`
(Phase 4 / Phase 7) and the "MPI / flow" section of `../../flow/CLAUDE.md` for the Eulerian precedent.
