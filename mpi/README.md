# dem — distributed (MPI) step

MPI block-parallelism for the DEM/XPBD solver on the shared `transport-core` library (sibling repo
`../../transport-core`), mirroring the approach validated for `sdflow`. The Lagrangian counterpart to
the Eulerian grid halo is **particle migration** (reassign particles to their owning rank) + **ghost
particles** (copies within one interaction radius of a block boundary, so each rank's ArborX broad-phase
runs locally), plus periodic **load rebalancing** (weighted-ORB SoA ownership migration).

The distributed step now ships **inside the `dem` Kokkos module** as `step_mpi`, gated behind the
`DEM_MPI` build option (the default module never defines it, so the single-rank module stays
byte-identical). This document is the status + how-to-build/run + what-is-validated for that step, plus
the standalone transport-core bring-up harness that still lives in this directory.

## The shipped distributed step (`dem`, `-DDEM_MPI=ON`)

Build the `dem` module with MPI exposed, against the bootstrapped Kokkos prefix:
```bash
cd dem && source .venv/bin/activate
cmake -S . -B build -DDEM_MPI=ON -DCMAKE_PREFIX_PATH="$PWD/../extern/install/<backend>"
cmake --build build -j$(nproc)            # -> build/dem.*.so with init_mpi/enable_mpi_step/step_mpi
```

Python surface (gated; present only when built with `-DDEM_MPI=ON`):
```python
import dem
sim = dem.Simulation()
sim.initialize_shape(...); sim.set_positions(...)        # as usual
sim.init_mpi(origin, size, gsize, periodic)              # ORB block decomposition + tpx particle halo
sim.enable_mpi_step(rcut, sync_every=1,                  # ghost cutoff; owner->ghost refresh cadence
                    forward_rotation=True,               # False = spheres (skips quaternion forward)
                    rebalance_every=0)                   # >0 = re-decompose by particle count every N steps
sim.step_mpi(nsteps)                                     # advance with halo exchange
sim.rebalance()                                          # force a load rebalance now (returns new owned count)
# diagnostics: sim.rank(), sim.num_ghost()
```

### Implementation
- `src/mpi_halo.hpp` (`KokkosParticleHalo`) — a thin wrapper over transport-core's
  `tpx::halo::ParticleHaloTopology<3>` (host topology + periodic image shift), `tpx::halo::ParticleHalo<3>`
  (on-device gather/scatter + host-staged MPI), `tpx::halo::ParticleMigrator`, and the
  weighted-ORB `particle_rebalance` path. Rebuilt each substep from the owned positions.
- `src/sim.hpp` (`demStepMpi`) — the distributed substep. Identical to the single-rank `demStep` except
  the periodic ghost generation is replaced by a cross-rank gather (ghosts carry **real** mass), and the
  owners refresh their ghost copies (velPred/angVelPred, then posPred/quatPred) every `sync_every`
  solver iterations (and the last). Each owned particle therefore sees all its neighbours — owned or
  ghost — and computes its **full serial XPBD delta locally**; the ghost deltas land on self-mapped
  slots and are discarded. Like the single-rank step's reference, `step_mpi` carries **no body-body
  friction passes** — its velocity solve is pure normal restitution.
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
  exercises the local periodic self-ghosts on undecomposed axes). Build/run:
  ```bash
  cmake -S tests/kokkos_mpi -B build_kmpi \
        -DCMAKE_PREFIX_PATH="<suite>/extern/install/<backend>" \
        -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
  cmake --build build_kmpi -j && ctest --test-dir build_kmpi --output-on-failure
  ```
- np=1 is bit-exact to the single-rank step; np=2/4 agree to atomic-ordering float noise.

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
packing evolves. See the suite memory note on dynamic load balancing and `transport-core`'s
`particle_rebalance` / `rebalanceByParticleCount`.

## Standalone bring-up harness (host C++ + transport-core)

This directory also holds the original bring-up tests — host C++ + MPI + header-only transport-core,
no Kokkos/ArborX/Python needed — that validated the migration + ghost-exchange machinery before it was
wired into the module. They still build and run:
```bash
cmake -S mpi -B mpi/build -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
cmake --build mpi/build -j
ctest --test-dir mpi/build --output-on-failure        # *_np{1,2,4}
```
Force `-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun` (FindMPI may otherwise pick ParaView's bundled mpiexec,
which launches OpenMPI binaries as singletons).

- **`test_particle_migration.cpp`** — decomposes the periodic domain (`tpx::decomp::BlockDecomposer`),
  builds a `tpx::halo::DomainMap`, and migrates particles with `tpx::halo::ParticleMigrator`. Each
  particle's full SoA record is the opaque payload; its position drives ownership. Then calls
  `gatherGhosts(rcut)` to collect copies within one interaction radius of the block boundary (periodic
  images handled) — the input to a local broad-phase. Validated: count conserved, every particle on its
  owning rank, id multiset preserved, np=1,2,4.
- **The three ghost-exchange schemes** (built on transport-core's persistent `ParticleHalo` —
  `build` + field-agnostic `forward` / `reverse(sum)`), each matched to a serial reference cell-for-cell
  at np=1,2,4 with a toy soft-sphere force in place of XPBD so they build with just MPI + transport-core:
  - **A — replicate / frozen ghosts** (`test_dem_step.cpp`): import ghost state, compute boundary pairs
    twice, integrate owned.
  - **B — Newton-on** (`test_dem_scheme_b.cpp`): import ghost state, compute each pair once,
    `reverse(force, sum)` the ghost reactions to owners, integrate owned.
  - **C — force-accumulation, local ghost integration** (`test_dem_scheme_c.cpp`): compute each pair
    once, `reverse(force)` to owners, `forward(totalForce)` to ghosts, integrate owned + ghosts locally.

The shipped `step_mpi` follows the **EXACT** variant (full owner→ghost state refresh, every owned
particle computes its complete serial delta locally) rather than the reverse-reduction schemes B/C.

See [multi_gpu_testing.md](multi_gpu_testing.md) for the multi-GPU profiling/scaling playbook, `../../docs/ROADMAP.md`
(Phase 4 / Phase 7) and the "MPI / sdflow" section of `../../sdflow/CLAUDE.md` for the Eulerian precedent.
