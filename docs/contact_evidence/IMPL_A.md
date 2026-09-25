# Implementation record A: contact-solve framework (`docs/contact_solve_framework.md`)

Acceptance numbers per work order. Raw outputs: `impl_a/`; baselines: `baseline/`.

## WO-0

Instrumentation and baselines. Branch `contacts`, base `0e5a9eb` (src and tests = `e90caff`).
Host: 48 cores, 1-min load 56-74 during the runs; `OMP_PROC_BIND=false OMP_WAIT_POLICY=passive`,
`mpirun --bind-to none`. GPU: RTX 5080 (shared, 86 % busy at the CUDA runs).

**Files.** `tests/kokkos_mpi/test_momentum_mpi.cpp`, `tests/kokkos_mpi/test_ownership_mpi.cpp`,
`tests/kokkos_mpi/CMakeLists.txt`; the capture hook (coordinator decision 1):
`git diff e90caff --stat -- src` = `mpi_halo.hpp +18, particles.hpp +18, sim.hpp +5,
step_solve_mpi.hpp +42` (83 insertions, 0 deletions).
- `Particles::debugCapture` (flag) + `DebugContactCapture debugCaptured` (host vectors).
- `ParticleHalo::debugGhostShiftsBySlot()`: the periodic shift of every ghost slot (host copy of
  `shiftDev_` through `ghostSlot_`). The image triple is `lround(shift / box)` per axis: exact, no
  position arithmetic.
- `debugCaptureOwnedContacts(P, halo, ncOwned)` in `demStepMpi`, right after
  `partitionContactsKokkos`, only when the flag is set: owned contacts `[0, ncOwned)` (gidA, gidB,
  image of each slot, dist), owned gid / posPred / rad.
- `Simulation::debugCaptureContacts(bool)`, `debugCapturedContacts()` (C++ only, MPI only).

**Inertness proof** (`impl_a/wo0_inertness.txt`, 127 IDENT, 0 DIFF; OMP 1):

| comparison | cases | result |
|---|---|---|
| pristine `e90caff` test binary vs WO-0 tests on pristine src: existing modes' `--dump` + MOMENTUM/HUB/FRIC/FRICPAIR lines (np 1; np 4 `cluster`, `cluster_pgs`; `--solo` `cluster_periodic`, `hub`) | 15 dumps + 17 line sets | identical |
| WO-0 tests, pristine src vs + hook (capture off): every WO-0 dump mode, np 1 (+ `--solo` variants, np 4 `cluster`, `cluster_pgs`) | 29 dumps + 29 line sets (MOMENTUM, KE, CONFLICTS, HUB, FRIC) | identical |
| `oracle_{closed,shear,periodic}` capture on vs `--capture=0`, final-state dump, np 1, 2, 4, 8 | 12 | identical |
| `test_ownership_mpi` existing modes (reverse_*, exactly_once_*, missed_*) pristine vs WO-0 binary, stdout, np 1, 2, 4, 8 | 29 | identical |
| `build_base` baseline dumps vs the pristine-src dumps | 25 | identical |

**Acceptance.**

| check | result |
|---|---|
| src changed only by the capture hook, byte-inert | yes (table above) |
| targeted ctests `momentum|ownership|followup` (build_ct, OMP 2) | 116 / 116 pass (43 new) |
| full build_ct ctest (OMP 2, `--bind-to none`) | 195 / 195 pass (12 `python_mpi_*` SKIP: no core Python on `PYTHONPATH`) |
| oracle reproduces D4: `oracle_shear` missing > 0 at np >= 4, `oracle_periodic` at np 2 and 4 | yes (table below) |
| `ring_mini` per-point position-graph degree > 64 | 1427 at np 1 |

**Oracle baseline** (`baseline/oracle.txt`; build_base, OMP 1, 50 steps; max over steps, first
step > 0). `dup = extra = 0` everywhere.

| mode | np 1 | np 2 | np 4 | np 8 |
|---|---|---|---|---|
| `oracle_closed` missing | 0 | 0 | 0 | 0 |
| `oracle_shear` missing (first step) | 0 | 592 (8) | **718 (2)** | **737 (2)** |
| `oracle_periodic` missing (first step) | 0 | **12 (1)** | **30 (1)** | 20 (2) |
| required pairs (max over steps), shear / periodic | 9701 / 10481 | 9682 / 10493 | 9674 / 10510 | 9681 / 10485 |

First missing pairs: periodic np 2 gid 56 -- 4031 image (-1, 0, -1) (wraps x decomposed + z
undecomposed: D4b); shear np 4 gid 56 -- 1024 image (0, 0, 0) at step 2.

**CONFLICTS at baseline** (`baseline/lines.txt`; max over steps and ranks; identical at OMP 1 and 8):

| mode | np | vel | pos | degVel | degPos | colVel | colPos | leftPos |
|---|---|---|---|---|---|---|---|---|
| `hub` | 1 / 2 / 4 / 8 | 179 / 64 / 64 / 34 | 237 / 94 / 94 / 54 | 242 / 127 / 127 / 97 | 300 / 157 / 157 / 117 | 63 | 63 | 0 |
| `hub_pgs` | 1 / 2 / 4 / 8 | 179 / 64 / 64 / 34 | 237 / 94 / 94 / 54 | same as hub | | 63 | 63 | 0 |
| `hub_posonly` | 1 / 2 / 4 / 8 | 237 / 94 / 94 / 54 | 237 / 94 / 94 / 54 | 300 / 157 / 157 / 117 | same | 63 | 63 | 0 |
| `ring_mini` | 1 | 0 | 0 | 15 | **1427** | 15 | **24** (25 at OMP 8) | **24052** |
| `ring_mini` | 2 / 4 / 8 | 0 | 0 | 11 / 11 / 9 | 1148 / 1143 / 1043 | 12 / 11 / 9 | 19-27 | 13612 / 11340 / 6688 |
| every cluster / tri mode | 1-8 | 0 | 0 | <= 6 | <= 9 | <= 7 | <= 10 | 0 |

`ring_mini` np 1 OMP 1: MOMENTUM dP 2.65e-8, dXpos **3.5e-2**, dLvel 1.55e-3; KE 52.43 -> 46.13 over
10 steps.

**CUDA** (`baseline/cuda_lines.txt`, build_ct_cuda, 3 runs each, np 1): `hub` dP 1.03e-2..1.48e-2
(step_mpi) / 1.03e-2..1.21e-2 (`--solo`), vel/pos 179/237; `hub_posonly` dP 0, dXpos 7.7e-4..1.75e-3,
237/237; `ring_mini` dP 1.4e-8..2.2e-8, dXpos 5.4e-2..6.7e-2 (step_mpi) / 3.5e-2..4.0e-2 (`--solo`),
vel/pos 0/0, leftPos 23984..24048.

**tri check** (`impl_a/wo0_tri_axes.txt`): np 1 KE_s3 = 0.1379 (e 0.5) / 0.2459 (e 0.8) on every
axis (the note's targets). np 2 splits z, so only `--axis=2` crosses the rank face there: 0.2907 /
0.5546 (the review's 0.291 / 0.555); axes 0 and 1 equal np 1. At np 4 and 8, `--axis=0` gives 0.2907.

**Scene choices (the note does not fix them; decision 4).**
- `ring_mini`: `initializeShape(HOLLOW_CYLINDER, radius 0.5, height 1.5, wall 0.18)` + `setPositions`
  (unit mass, the shape's inverse inertia) + `setQuaternions`; scale 1. Lattice 3 x 3 x 3 at 0.9
  about the cluster centre (0.3, -0.2, 0.1), jitter 0.05 uniform per axis. Velocities: drift
  (0.7, -0.4, 0.3) + 1.5 N(0,1) - r / r_max, r = lattice offset, r_max = 0.9 sqrt(3); no spins.
  Positions and velocities from `makeCluster`'s stream `mt19937(20260925)`; orientations from a
  separate `mt19937(11)` (normalised 4-D Gaussians). e = 0.5 (the file's `--e` default), tangential
  restitution 0, friction 0.02, pos/vel 20/8, dt 1e-2, g 0, 10 steps.
  - The survey scripts pass `height=0.75`; the RingBed packer passes `height = D * aspect = 1.5`
    and the shell/SDF treat `height` as the full height. I followed the note (H = 1.5).
- `tri_pgs`: g = (0, 0, -10), frictionless, stabilization 'off' (the file's gravity rule).
- `cluster_{multilevel,escalate,ordered,onesided}`, `cluster_poisson`: `cluster_pgs` (mu 0.4, spins,
  g) with that stabilization mode / the Poisson model.
- `hub_pgs`: `hub` + g = (0, 0, -10) (stabilization 'off').
- Oracle: `makeScene`'s lattice (16^3, radii 1 +- 0.1 R, jitter 0.04; periodic 0.3), owners at the
  original positions, dt 1e-2, g 0, e 0.5, frictionless, pos/vel 20/8, the API's unit masses.
  `oracle_shear`: v_x = 0.2 (z - 8) / dt (z measured from the box centre). `oracle_closed` and
  `oracle_periodic` start at rest (the exactly_once scenes carry no velocities).
- `tri` / `tri_pgs` ctests at np 1 and 2 only (three bodies).
- `--relabel`: `std::shuffle` of the whole body list with `mt19937(seed)` before `ownedOf`.

**Deviations and findings.**
1. CONFLICTS is tallied on the host by the test (`colorDiag`), not by
   `Simulation::debugColoringConflicts`. Same definition (sum over (slot, colour) of count - 1,
   colours >= 0; velocity graph through `realIndices`). Reason: under MPI that function sizes its
   mask by `max(numParticles, numReal) + 1`, and after `demStepMpi` `numParticles == numReal`, while
   contacts reference ghost slots: an out-of-bounds atomic write. The test fills both colour arrays
   with -3 before each step (the existing hub `markColors`), so stale non-owned entries are skipped;
   proven inert above.
2. Added `leftVel` / `leftPos` to CONFLICTS (items left at colour -1, i.e. the count-averaged
   fallback's set).
3. **`ring_mini` reaches the "dead" count-averaged position fallback at np 1.** `colorContactsKokkos`
   runs at most `numBodies + 2` arbitration rounds (29 for 27 bodies), and a round colours at most
   one contact per body. With ~24000 contacts, 24052 stay at -1, `posLeftover > 0`, and the
   Jacobi fallback runs every substep; `pos` conflicts read 0 because the cap stops before colour
   62 saturates. This is the source of dXpos 3.5e-2 at np 1, OMP 1. FOLLOWUPS §1 ("leftover is
   always 0") and the note's "dead fallbacks" (§0, §4.2) assume no round cap. This bears on WO-3 /
   WO-4 (deleting the fallbacks changes np 1 ring results; the recolour-with-copies round bound).
   Not acted on.
4. The note says new modes are report-only with "`kGate` stays false". `kGate` is `true` in the file
   (the G1 gate of the pre-existing modes) and is left `true`. The new modes are report-only
   through `tolOf` (-1) and `Mode::reportOnly`.
5. Hook location: the flag and buffers are on `Particles` as decided. The per-slot shift accessor
   had to go on `ParticleHalo`, which owns the private shift and slot views.

**Commands.**
```bash
docs/contact_evidence/baseline/run_baseline.sh build_base docs/contact_evidence/baseline        # dumps, lines.txt, oracle.txt
docs/contact_evidence/baseline/run_baseline.sh cuda build_ct_cuda docs/contact_evidence/baseline # cuda_lines.txt
docs/contact_evidence/impl_a/wo0_inert_test.sh <pristine test_momentum_mpi> <new> <outdir>
docs/contact_evidence/impl_a/wo0_dumps.sh <test_momentum_mpi> <outdir>       # then cmp *.dump
mpirun --bind-to none -np N build_ct/tests/kokkos_mpi/test_ownership_mpi oracle_shear [--capture=0 --dump=f]
```
`build_base`: configured from the WO-0 tree with build_ct's flags; the 8 kokkos_mpi targets built.
Frozen from here on.

## WO-1 (07114fb)

Cherry-picked from the parked `95c5637`; gate restated per §12 S1. Raw: `impl_a/wo1_reverify.txt`
(and the parked `wo1_cf.txt`, `wo1_fricpair.txt`). Host load 56-68, OMP 1 unless stated.

| check | gate | result |
|---|---|---|
| `friction_pair` couple ratio \|dL\|/\|dist n x J_t\|, delta/R 0.001 / 0.01 / 0.05 / 0.1 / 0.2 (dt 1e-2) | <= 1e-2 (S1) | 1.3e-3 / 1.0e-4 / 7.8e-6 / 1.6e-5 / 7.7e-6 (was 1.000) |
| same, dt 0.02 / 0.005 / 0.0025 at delta 0.05 | <= 1e-2 | 2.1e-5 / 1.2e-5 / 8.7e-6 |
| `cluster_friction` dLvel, np 1 / 2 / 4 / 8 x OMP 1, 8 | <= 1e-7 (S1) | max 6.9e-9 (was 1.9e-5) |
| frictionless np 1 OMP 1 dumps vs WO-0 baseline | identical | 23 / 23 identical; `cluster_friction`, `ring_mini`(`_solo`) change (friction 0.02) |

Test changes: `kFrictionPairGate` (friction_pair only; friction_pair_pgs stays report-only),
`tolOf(cluster_friction).dLvel = 1e-7`.

## WO-2 (5c91df9)

Cherry-picked from the parked `b47fae0`; §12 S2 applied (`test_cooling_slope_vs_enskog` keeps the
Enskog band, prints r_j for information). `tolOf(cluster_jacobi)` tightened to the G1 row. Raw:
`impl_a/wo2_reverify.txt`.

| check | gate | result |
|---|---|---|
| `cluster_jacobi` np 1 / 2 / 4 / 8 x OMP 1, 8 | dP <= 1e-6, dXpos <= 1e-5 | dP <= 1.07e-8, dXpos <= 2.6e-7, dLvel <= 7.3e-9 (was dP 9.0e-3) |
| `demstep_jacobi_{closed,periodic}_np{1,2,4}` | pass | 6 / 6 |
| every other np 1 OMP 1 dump + np 4 `cluster`, `cluster_pgs` vs WO-1 | identical | 28 / 28 identical (only `cluster_jacobi` changes) |
| Enskog cooling (info) | 0.5 < r_gs < 2.5 | GS 1.804, mass-split Jacobi 2.141 x Enskog |

**Battery at WO-2** (`impl_a/wo2_battery.txt`; build_ct, `-j1`, OMP 2, `--bind-to none`,
`PYTHONPATH=core/build_rel_py`, which imports `peclet.core.mpi`): **195 / 195 pass** = 182 C++
(kokkos, arborx, kokkos_mpi) + `python_tests` + **12 `python_mpi_*` RUN and pass** (none skipped).

## WO-3: STOPPED before implementation (a stated premise of §4.3 is false)

§4.3 defines a unit as "the set of contacts of one body pair (walls: one body and one wall)" and
says "the unit list reuses the manifold reduction's sort". For walls the sort cannot give that:
- `pairKey` (`src/contact_preprocessing.hpp:396`) keys every boundary contact as
  `(bodyA << 32) | 0xFFFFFFFF`, so all planes AND all SDF walls of one body fall in ONE segment
  (the velocity manifold is "merged per body" by design, `:94`).
- `ContactC` carries no wall identity (`bodyB = -1` for every plane and SDF wall;
  `narrowphase.hpp:393, 461, 484`), so "one body and one wall" cannot be derived afterwards either.

Consequence for each reading (none is fixed by the note):
- (a) unit = the sort's segment (all walls of a body): a sphere touching 2-3 planes (box edges and
  corners, drums, packings) changes at np 1 (its wall contacts become one edge applied in index
  order instead of 2-3 separately coloured edges), contradicting §0 "spheres, analytic walls
  byte-identical". The momentum-test modes have no walls, so the WO-3 dump acceptance would not
  see it; the python tests with boxes would.
- (b) one unit per wall contact: spheres stay bitwise, but a ring or SDF particle on a wall keeps
  one edge per contact point at its vertex (wall degree stays per point; ring beds in containers
  would lean on WO-4 hub copies).
- (c) add a wall id to `ContactC` (narrow phase) and key units by (body, wall): matches the
  definition, spheres bitwise, but changes a data structure in a file the WO-3 list does not name.
