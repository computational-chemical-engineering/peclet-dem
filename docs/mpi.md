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
body as of the last reconciliation. The fixed point every projection-form phase converges to is
the coupled (serial) solution for any consistent copy count `k` (`docs/contact_solve_framework.md`
§1.3 P4): run to a forced, generous iteration count with every adaptive stop disabled, np 2/4/8
converge to np 1 to the float floor. At the default, finite adaptive-stop budget, trajectories
agree with single-rank statistically, not bit-for-bit (numbers under *What is validated*).
`forward_rotation=False` skips the ghost angular-velocity and quaternion forwards (spheres).

Under Poisson (event-level) restitution two ranks can draw from one body's orphan account within
one reconciliation interval; each active copy holds its *share* `B/k` of the owner's balance under
mass splitting (*Bodies updated in several places*, below), the owner reduces the shares back to a
single balance at every sync, and the clamp at 0 still applies (docs/mpi_momentum_conservation.md,
R5; `docs/contact_solve_framework.md` §13.3).

The legacy friction pass stays a **raw, count-averaged Jacobi sync**, unaffected by the copies
framework below because it runs after the phase's own reconciliation, on settled state: each rank
counts only the contacts it owns, the ghosts' partial counts are summed onto their owners and
forwarded back (`syncFrictionCounts`), and every copy of a body divides by the serial count. The
**`'jacobi'` diagnostic (`set_velocity_solver('jacobi')`) is mass-split, not count-averaged**:
every contact is its own copy, solved against masses `m/count` (velocity and position both), and
the true-mass deltas are summed with factor 1 — conservative like every other projection-form
phase, unlike the `min(1, 2/count)` / `1/count` count-averaged forms it replaced. The colouring's
old count-averaged "leftover" fallback is gone with it: a colouring with edges still uncoloured
after every vertex above 32 active edges has been split into local mass-split copies
(`docs/contact_solve_framework.md` §4) throws an invariant violation on every rank, never silent
fallback work.

### Bodies updated in several places

A body written from more than one place during a substep — several ranks holding it, or (locally,
even at np 1) more than 32 contact edges landing on it in one colouring — is represented by
**copies**, and every contact's impulse is owned once, by exactly one copy
(`docs/contact_solve_framework.md` §0–§5). Every **projection-form** update — warm-started PGS with
cone and Poisson release, the overlap projection, every stabilization mode, the multilevel cycle,
and the mass-split `'jacobi'` diagnostic — **mass-splits** its copies (`m/k`, `I/k`, the exact
active count `k`) and reconciles them by the mean at every sync: conservative at every iterate,
with the coupled (serial) solution as its fixed point for any consistent `k`. The **event-form**
`g = 0` one-shot restitution sweep gets the opposite treatment, **exclusive holding**: a greedy
rank colouring (`C <= 64`) picks exactly one holder rank per shared body each sync interval, only
the holder's contacts fire, and the run is then a legal serial Gauss–Seidel order that inherits
every serial property exactly, including conservation and each firing's non-increasing energy —
mass-splitting the one-shot instead compounds restitution (measured: 14–18 % kinetic energy lost
per substep on a dense cluster at `e = 0.9`/`1.0`) and is rejected. The Gauss–Seidel colourings
never exceed 64 colours: a vertex above 32 active edges gets local mass-split copies, round-robin
over its edges, and the phase is recoloured — the palette lemma then guarantees every colouring
completes, so the old count-averaged fallback for an uncolourable leftover is gone. Visibility —
every pair within contact reach seen by **both** bodies' owners — needs a ghost band of
`reach + S + P` (`S = 0.25 R_max` drift slack, `P` the Verlet skin or else the substep's predicted
displacement), a migration to current blocks whenever a body drifts more than `S` from its owner's
block (voted inside the existing radius `Allreduce`, so no extra message), and **every** periodic
image within that band, not just one.

| phase | form | policy | notes |
|---|---|---|---|
| Warm start | known impulses | applied on true masses before copies are seeded | none |
| One-shot restitution, `g = 0` (colored GS) | event | **X** — one holder rank per sync interval; local hub copies mass-split (M) | reproduces serial to 0.03 % |
| PGS normal + cone + Poisson release, `g != 0` | projection | **M**, `ω_vel = 1` | |
| Stabilization (one-sided, escalate, ordered) | projection | **M** | |
| Multilevel: fine sweep + coarse cycle | projection + coarse accelerator | **M**; a folded hub's copies are one coarse vertex of mass `a·m/k` | |
| Legacy friction (`g = 0`, no gravity, or Jacobi A/B) | explicit Jacobi pass, raw sync | count-averaged (unaffected by copies) | single application point (midpoint); world-frame inverse inertia |
| Overlap projection (position phase) | accumulated, retractable projection (projected SOR on each contact's net push, `ω_pos = 1.5`; unique fixed point) | **M** | multi-point pairs solved as one unit |
| Poisson bank / orphan scatter | bookkeeping | true masses; credit only by the pair's owner | |
| `'jacobi'` diagnostic | projection, mass-split | every contact its own copy, `k` = global contact count | |
| Hertz–Mindlin | explicit, redundant on both owners | unchanged; needs symmetric visibility + canonical pair orientation | |

**Validated** (`docs/contact_evidence/IMPL_A.md`; gates G1 conservation, G2 energy, G5 visibility
oracle, G6 race-freedom):
- conservation: `dP` ~1e-9 to 1e-8 at np 1–8 across the gated scenes;
- the review's 3-body scene: kinetic energy at np 2–8 equals np 1 exactly (0.2459 at `e = 0.8`);
- the visibility oracles: `missing = dup = 0` on closed, sheared and periodic scenes, at np 1–8;
- np 1 is byte-identical to before, except the named changes (non-spherical position sweeps, any
  colouring that used to fail, the `'jacobi'` diagnostic, `g = 0` runs with body–body friction, and
  periodic runs).

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
- **The MPI call sequence must be identical on every rank — never gate an exchange on a rank-local
  condition.** The halo is not symmetric: a rank can owe ghosts to a neighbour that owes it none.
  Skipping the forwards on `numGhost == 0` deadlocked the step whenever such a rank existed
  (coupling's moving suspension at np=4, 2026-09-24); a forward is now skipped only when the rank
  neither sends nor receives. Likewise the Verlet-skin rebuild decision is Allreduce-OR'd (the
  rebuild is an NBX collective and one rank's particles can cross the skin while its neighbours'
  rest). `tests/kokkos_mpi/test_halo_schedule_mpi.cpp` (`one_sided_xpbd`, `one_sided_hertz`,
  `skin_divergent`, ctest TIMEOUT 120 s) pins both.
- **The ghost band must cover contact reach plus drift slack plus prediction, over the GLOBAL
  maximum radius**: `band = max(rcut, reach + S + P)`, `reach = 2.1 R_max`, `S = 0.25 R_max` drift
  slack, `P` = the Verlet skin or else the substep's predicted displacement
  (`docs/contact_solve_framework.md` §5.1; Allreduce MAX, re-evaluated every step, growth
  included). A body drifting more than `S` outside its owner's block triggers `migrateToBlocks`
  before the next gather — a vote folded into the same `Allreduce` as the band, so it costs no
  extra message. Before the drift slack existed, the default band was `2.1 R_max` with no margin
  for motion between gathers, so a fast or infrequently-regathered body's partner could fall
  outside it; an explicit `rcut = 2r` missed the margin entirely, and a polydisperse run's ranks
  used to disagree on it. `tests/kokkos_mpi/test_ghost_band_mpi.cpp` (`band_change`,
  `default_band`, `margin`, `explicit_rcut`) and `test_missed_drift_*` pin the band and the vote
  against `MPI_COMM_SELF` and an O(N^2) oracle.
- **Ghost selection and the drift vote measure distance on the domain-clamped ownership
  coordinate on non-periodic axes, not the raw position** (S20, 2026-09-26): a body outside an
  unwalled, non-periodic boundary is still projected onto the domain box before its distance to a
  block is computed, in both the halo topology build and the vote, or a partner across a block
  face is silently never sent (measured: 121–122 missing pairs at np 4/8 in a sheared, unwalled
  domain before the fix; `oracle_shear` now gates `missing = dup = 0`).
- **Every periodic axis sends every image within the band, not just one** (`allImages`, core's
  `ParticleHaloTopology::build`, opt-in and additive — default off, byte-identical, for every
  other consumer). A periodic self-image pair is owned by whichever twin's real body has the lower
  gid when both twins exist on the rank, and outright by whichever twin is present when the drift
  slack has moved a body's owner across the periodic face and only one twin survives. A periodic
  axis still only ghosts across the wrap when it is split across ≥2 ranks (a rank never ghosts to
  itself for the cross-rank wrap; an undecomposed periodic axis uses the local self-ghosts
  instead), but a decomposed axis now sees every image a body needs: `oracle_periodic`'s
  missing/dup count is 0 at every np, against 13–27 missing before.

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
