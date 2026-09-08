# dem — distributed (MPI) step

MPI block-parallelism for the DEM/XPBD solver on the shared `core` library (sibling repo
`../../core`), mirroring the approach validated for `sdflow`. The Lagrangian counterpart to
the Eulerian grid halo is **particle migration** (reassign particles to their owning rank) + **ghost
particles** (copies within one interaction radius of a block boundary, so each rank's ArborX broad-phase
runs locally), plus periodic **load rebalancing** (weighted-ORB SoA ownership migration).

The distributed step now ships **inside the `dem` Kokkos module** as `step_mpi`, gated behind the
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
sim.enable_mpi_step(rcut, sync_every=1,                  # ghost cutoff; owner->ghost refresh cadence
                    forward_rotation=True,               # False = spheres (skips quaternion forward)
                    rebalance_every=0)                   # >0 = re-decompose by particle count every N steps
sim.step_mpi(n)                                          # advance n steps with halo exchange
sim.rebalance()                                          # force a load rebalance now (returns new owned count)
# diagnostics: sim.rank(), sim.num_ghost()
```

### Implementation
- `src/mpi_halo.hpp` (`ParticleHalo`) — a thin wrapper over core's
  `peclet::core::halo::ParticleHaloTopology<3>` (host topology + periodic image shift), `peclet::core::halo::ParticleHalo<3>`
  (on-device gather/scatter + host-staged MPI), `peclet::core::halo::ParticleMigrator`, and the
  weighted-ORB `particle_rebalance` path. Rebuilt each substep from the owned positions.
- `src/sim.hpp` (`demStepMpi`) — the distributed substep. The periodic ghost generation of the
  single-rank step is replaced by a cross-rank gather (ghosts carry **real** mass), and the
  owners refresh their ghost copies (velPred/angVelPred, then posPred/quatPred) every `sync_every`
  solver iterations (and the last). Each owned particle therefore sees all its neighbours — owned or
  ghost — and computes its **full XPBD delta locally**; the ghost deltas land on self-mapped
  slots and are discarded. Friction (wall + body-body Coulomb) **is** carried — same kernels as the
  single-rank step. **Solver parity gap (open):** the velocity/position solves here are still the
  older count-averaged **Jacobi** kernels; the newer single-rank stack (graph-colored Gauss–Seidel,
  warm-started PGS with persistent contacts, gravity statics / grounded shock propagation, adaptive
  stop) is not yet wired into the MPI path, and `MigratePack` does not carry persistent-contact
  state across a rebalance.
- **Periodicity:** cross-rank ghosts supply the wrap on *decomposed* axes; local periodic self-ghosts
  (the halo built with `includePeriodicSelf`) supply it on *undecomposed* periodic axes.

### The EXACT scheme + the `sync_every` (M) knob
`sync_every=1` is **EXACT**: every owned particle has all its neighbours refreshed every iteration, so
it reproduces the serial XPBD delta (bit-exact at np=1; np=2/4 differ only by Jacobi atomic-ordering
float noise at the block split, not physics). `sync_every=M>1` is an approximation — boundary error
grows with M in exchange for fewer halo exchanges per step. `forward_rotation=False` skips the ghost
quaternion forward and is **exact for spheres**.

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
- np=1 agrees with the single-rank step to float noise (max 1e-4 over 15 steps). At np=2/4 the
  modern stack (processor-block Gauss–Seidel with rank-local colouring) sweeps the finite-iteration
  PGS in a different order than single-rank, so per-particle agreement on a stiff, randomly
  overlapping IC is *statistical* (measured 2026-09-08, N=200, 15 steps: mean 5e-3, 95 % quantile
  4e-2, max 0.11 = a quarter diameter); resting wrap contacts agree exactly and the aggregate
  observables (energy, overlap, pile geometry) match to their tolerances. The C++ `demstep_*`
  ctests are the tolerance-based statement of the same thing.

### Validation lessons carried into the Kokkos step
- **Full per-particle state must travel through migration** (quaternion + angular velocity, not just
  velocity/id) — otherwise off-centre collisions' spin is discarded and rotational KE leaks.
- **The predicted position must be *forwarded*, not copied from the committed position** at gather time:
  `predict_velocity` already advances `posPred = pos + v·dt`, so copying would leave ghosts one
  predict-step stale and systematically dissipate energy at the boundary.
- **A periodic axis only works distributed if it is split across ≥2 ranks** (a rank never ghosts to
  itself for the cross-rank wrap; undecomposed periodic axes use the local self-ghosts instead).

### Load rebalancing
With `rebalance_every=N` (or an explicit `sim.rebalance()`), the decomposition is recomputed by
particle count (weighted ORB) and SoA ownership is migrated — keeping per-rank work balanced as the
packing evolves. See the suite memory note on dynamic load balancing and `core`'s
`particle_rebalance` / `rebalanceByParticleCount`.

The original host-C++ bring-up harness (particle migration + the three ghost-exchange schemes A/B/C
matched cell-for-cell to a serial reference) validated this machinery before it was wired into the
module; it was retired at 1.0.0 (it targeted core's pre-`peclet::core` headers) and lives in git
history before dem `b43040c`. The shipped `step_mpi` follows the **EXACT** variant (full owner→ghost
state refresh, every owned particle computes its complete serial delta locally) rather than the
reverse-reduction schemes B/C.

See [multi_gpu_testing.md](multi_gpu_testing.md) for the multi-GPU profiling/scaling playbook, `../../docs/ROADMAP.md`
(Phase 4 / Phase 7) and the "MPI / sdflow" section of `../../flow/CLAUDE.md` for the Eulerian precedent.
