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
Resolved by §12 S6 (7d88a64): option (c).

## WO-3 (13158a6 on branch `contacts-wo3-parked`, NOT on `contacts`): implemented, two stops

Position units per §4.3 with the S6 wall id (`bodyB = -1 - wallIndex`; planes first, SDF walls from
`numPlanes`). Every wall test in `src` was already `bodyB < 0`; no other negative `bodyB` existed.
`pairKey` and the manifold segments are unchanged. Units come from their own two key sorts (by
`unitKey`, then by (leader, contact index)). Both position colourings take the units (empty views =
the old per-contact code, which `test_coloring_overflow` still calls), and both colourings now run
at the top of `demSolveContacts`. Raw: `impl_a/wo3_*.txt`. Host load 60-75, OMP 1 unless stated.

| check | gate | result |
|---|---|---|
| sphere dumps np 1 OMP 1 (+ np 4 `cluster`, `cluster_pgs`) vs WO-2 | identical | 27 / 27 identical; only `ring_mini`(`_solo`) change |
| S6 wall scenes, np 1 OMP 1 (`impl_a/s6_wall_dumps.py`: box corner at g != 0 incr/full colouring and g = 0; rotating SDF drum + 2 cap planes, incr/full) | identical | 5 / 5 identical. Control: `unitKey = pairKey` (option a) changes all 5 |
| `ring_mini` np 1, CONFLICTS pos (host OMP 1, OMP 8, CUDA x 3, `--solo` too) | 0 | 0 everywhere |
| `ring_mini` np 1 dXpos | <= 1e-5 | 2.8e-6 (OMP 1), 4.7e-6 (OMP 8), 1.5e-6..4.0e-6 (CUDA, step_mpi and `--solo`). Before: 7.7e-2 (WO-1/2), 3.5e-2 (WO-0), CUDA 3.5e-2..6.7e-2 |
| `ring_mini` position colours / unit degree / per-point degree | report | 19 / 19 / 1427 (was 24 capped + 24052 uncoloured contacts) |
| `tests/kokkos`, `tests/arborx`, `python_mpi` | pass | pass (battery 194 / 195) |
| `tests/python` | pass, "a failure is a stop" | **1 FAIL**: `test_hollow_cylinder_overlap` (below) |
| §12 S3: round cap not exhausted on `ring_mini` and the survey scenes | leftover 0 | **ring_mini np 1 OMP 8: leftPos 2-6 units in 4 of 6 runs** (step_mpi); 1 of 3 (`--solo`); OMP 1 and CUDA: 0. Survey (`impl_a/survey_leftovers.py`, OMP 8): bi4 (0, 0), bi6 (0, 0), ring N = 80 1500 steps (0, 0) |

**Stop 1, `test_hollow_cylinder_overlap`.** It asserts `max_overlap > 0` after step 0 as proof that
the solver saw the penetration. `max_overlap` is the position loop's last-iteration residual; the 30
contact points of the one pair are now one unit swept in order, so the loop resolves the overlap
inside the substep and reads 0.0. The bodies still separate (dx 0.202 after step 0, 0.544 after 99;
the test's own separation assert would pass). Changing the assertion is the note author's call.

**Stop 2, §12 S3.** `ring_mini` has 27 bodies, so the cap is `numBodies + 2 = 29` rounds against a
unit degree of 19. Thread-order contact numbering changes the leaders and so the arbitration keys;
at OMP 8 some substeps need more than 29 rounds. The survey scenes have caps of 82 (ring) and
2002 / 4002 (bi4, bi6) and never exhaust them. Per S3, WO-4 is not started.

**Finding, not a gate: `ring_mini` at np 4 and 8 now diverges.** OMP 1: np 4 dXpos 2.7e-3,
ovl 6.9e4; np 8 dXpos 0.14, ovl 3.6e6 (np 2: 1.9e-6, fine). Before WO-3: np 4 ovl 0.22, np 8 ovl 0.26
(OMP 8 18.4). My reading: full pair-sequential projections on each rank are raw-summed across ranks
by c771e07's reverse (Jacobi across ranks, effective omega ~ (k+1)/2 > 2 for corner bodies, §1.3 P5).
The capped per-point colouring's count-averaged fallback used to damp this. WO-5's M-consensus is
the designed fix. Sphere modes are unaffected (bitwise). No ctest gates `ring_mini` at np >= 2
(report-only modes).

**Also seen:** bi6 (OMP 8) had velocity conflicts max 1 over the run (the forced colour 62 at
degree 115; baseline survey 0). The velocity colouring code is unchanged, so this is order-dependent
and WO-4's.

**Additions not named by the note** (instrumentation, results unchanged): `diagnostics.coloring_leftovers()`
(tier 2; the S3 measurement), the test's CONFLICTS line counts units (`degPos`, `leftPos`) and
prints `degPosPt` (the per-point degree), and `debugColoringConflicts` counts units.

## WO-3 (785d984): committed with §12 S7 / S8

The parked 13158a6 cherry-picked; S7: every fine colouring's round cap is its vertex count + 2
(`numManifolds + 2`, `numUnits + 2`). `colorKey(idx)` carries the index in its low word (a unit's
key is its leader's), so keys are unique and no tie-break change was needed. S8:
`test_hollow_cylinder_overlap` asserts detection (`num_contacts > 0`, `num_manifolds >= 1`; the
scene has two bodies, no walls, no periodic axis) and keeps the separation assert. Raw:
`impl_a/wo3_s7_reverify.txt`. Host load 20-60, OMP 1 unless stated.

| check | gate | result |
|---|---|---|
| sphere dumps np 1 OMP 1 (+ np 4 `cluster`, `cluster_pgs`) vs WO-2 | identical | 27 / 27 identical; only `ring_mini`(`_solo`) change |
| S6 wall scenes (corner x 3, drum x 2) vs WO-2 | identical | 5 / 5 |
| `ring_mini` CONFLICTS pos, leftPos: OMP 1; OMP 8 step_mpi x 8, `--solo` x 6; CUDA x 3 x 2 | 0 | 0 everywhere (the parked build: leftPos 2-6 in 4 / 6 runs) |
| `ring_mini` dXpos | <= 1e-5 | 2.8e-6 (OMP 1), <= 5.9e-6 (OMP 8), <= 4.7e-6 (CUDA); 19 position colours |
| S3 survey, OMP 8: bi4, bi6, ring N = 80 x 1500 steps | 0 leftovers | (0, 0) each |
| `tests/python` | pass | 47 passed (hollow cylinder included) |

## WO-4 (ca32026): complete colourings, hub copies, local folds, demStep images

Raw: `impl_a/wo4_*.txt`; scripts `impl_a/mlpile.py`, `impl_a/cmpnpz.py`. The copy machinery is
`src/solve_copies.hpp` (no ArborX, so `tests/kokkos` links it).

| check | gate | result |
|---|---|---|
| `test_coloring_overflow` star D = 32..300 (host OMP 8, CUDA), `kGate = true` | 0 conflicts, leftover 0 after copies | 0 / 0 at every D; D = 65 / 100 / 300: 2 / 3 / 9 copies, 22 / 25 / 30 colours |
| `hub`, `hub_posonly`, `hub_pgs` np 1, OMP 1 and 8 x 3, step_mpi and `--solo` | dP <= 1e-6, dXpos <= 3e-5, CONFLICTS 0 | dP <= 6.1e-7, dXpos <= 3.5e-7, CONFLICTS 0; copies vel 7 (9 posonly), pos 9 |
| same on CUDA x 3 | same | dP <= 5.7e-7 (baseline 1.0e-2..1.5e-2), dXpos <= 3.1e-7, CONFLICTS 0 |
| every non-hub closed dump np 1 OMP 1 (+ np 4) vs WO-3 | identical | 22 / 22; the 5 S6 wall scenes identical; a 150-step multilevel settling pile, 4 stabilization modes, `np.array_equal` identical |
| `cluster_periodic --solo` position-phase CoM drift (minimum image, new test metric) | <= 1e-5 R | 1.8e-6 R (before, WO-3 src: 2.3e-3 R); max overlap over the run 4.3e-2 (before 4.4e-3) |
| `ring_mini` np 1 (host, CUDA) | unchanged vs WO-3 | identical dump; CUDA dXpos <= 5.3e-6 |
| battery (`build_ct`, `-j1`, OMP 2, `--bind-to none`, core Python on PYTHONPATH, `--timeout 300`) | all pass | **192 / 195**: python_tests (1 case), python_mpi_validate_periodic_np2 (timeout), _np4 -- see Stop A |

**Stop A (a statistical-agreement test fails): `kSplitOmegaPosition = 1.5` on demStep's periodic
image copies.** An isolated wrap pair is one contact seen by two slots of each body (k = 2), so the
split, relaxed projection separates it by 1.5 x the overlap and the inequality never pulls the gap
back. Three tests fail on the single-rank reference (`impl_a/wo4_omega_pos.txt`):
`test_wrap_pair_matches_in_box_pair` (0.150 vs 0.100), `validate_periodic` np 2 (serial 0.900 vs
distributed 0.800; rank 0 asserts, rank 1 hangs) and np 4 (corner 0.846 vs 0.800, straddlers).
With `kSplitOmegaPosition = 1.0` (an experiment, not committed) all three pass (0.800 / 0.800,
straddlers 4.8e-7). The same overshoot will apply to every rank-split pair once WO-5 splits
ghosts. Decision needed: omega_pos for split slots (1.0; 1.5 only at hubs; or accept and change
the tests). WO-5 not started.

**Finding B (implemented as written; no gate exercises it):** §4.4 / §3 "fold before the coarse
cycle, re-seed after" with group masses from the solve views (§4.5) is not momentum-conservative
at a hub that aggregates: the coarse cycle moves the hub's base by dV computed with mass m/s, the
re-seed hands dV to all s copies and the next fold keeps it whole, so the body gains m dV instead
of (m/s) dV. A fold (not a re-seed) after the coarse cycle, or true masses in the coarse cycle,
would conserve; the second breaks the rank-level M for ghost copies (§3 says they enter with solve
masses). `hub_pgs --stab=multilevel --vel-iters=1` (the pass runs; dP 2.9e-7) does not aggregate
the hub, so no number shows it.

**Interim under MPI (WO-4 only, superseded by WO-5):** hub copies at np >= 2 are rank-local
(k = s, true masses across ranks, raw reverse); sigma is re-marked after every rank sync and the
orphan shares are summed back to the balance before each sync. Hub numbers at np >= 2 were not
measured.

**Open points found for WO-5 (the note leaves them open):**
1. `m_vel = 1 << col(rank)` needs the rank colouring, which is WO-6's; the opening payload's
   velMask has no value to carry in WO-5.
2. §4.6 step 5 seeds ghost orphan shares from "B the forwarded balance", but neither
   `OpeningState` nor the later `VelocityState` forwards carry the orphan balance; after a sync the
   ghost cannot re-seed B_new / k.
3. With local hubs under rank-level M the local copies must take the base's share B / k (WO-4 divides
   by s, the single-rank rule); the phase-end owner balance is seed + a(own)(share - seed / k).
4. a_vel / a_pos need the per-slot degree every substep under MPI (WO-4 computes it only on a
   failed colouring), and the solve views need the per-slot global k (WO-4 uses the group's k).

## WO-4b (9cbd29f on branch `contacts-wo4b-parked`, NOT on `contacts`): implemented, two stops

§13.1 A (`kSplitOmegaPosition` and the position relaxation branch deleted, the position
`SlotOverride` carries only its overrides) and §13.2 B (hierarchy build and coarse cycle take
`invMassCoarse = P.invMass`) as written; `split_stats` gains `mlHubAggregated` (split velocity
vertices, `splitSlot`, in a level-1 group of >= 2 members; max over the step call's substeps),
`velItersUsed`, `posItersUsed`; `test_momentum_mpi` gains `hub_static`, `hub_ml` (hub last,
`makeHubLast`; hub_ml's speed uses each leaf's own makeHub gauss draw), `--pos-iters`, the ITERS /
HUBGAP / MLCTRL lines; 6 ctests. Raw: `impl_a/wo4b_*.txt`; scripts `wo4b_refdumps.sh`,
`wo4b_hub.sh`. Instrumentation proved inert first (ca32026 src + counters + new test vs ca32026:
29 / 29 dumps, 5 / 5 S6, 4 / 4 pile identical). Host load ~60.

| check | gate | result |
|---|---|---|
| 1. Stop-A tests | pass | `test_wrap_pair_matches_in_box_pair` pass; `validate_periodic` np 2 / np 4: 0.8000 / 0.8000, straddlers 4.8e-7 |
| 2. `hub_static` np 1, gap | <= 0.05 delta | **FAIL (stop 2)**: OMP 1 0.034; OMP 8 x 3 0.047 / 0.061 / 0.059 (step_mpi), 0.051 / 0.052 / 0.059 (`--solo`); CUDA x 3 0.041 / 0.051 / 0.053, 0.038 / 0.039 / 0.043. ctest (OMP 2) passes |
| 2. `hub_static` residual, dXpos, \|P\| | 1e-4 R, 3e-5 R, 0 | <= 4.6e-6 R, <= 5.6e-8, 0 exactly (host, CUDA) |
| 2. discrimination on ca32026 | gap >= 0.3 delta | 0.584 delta |
| 3. `hub_ml` np 1 | dP <= 5e-6, controls | dP <= 6.9e-8 (host), <= 1.2e-7 (CUDA); **controls FAIL (stop 1)**: velCopies 3, mlLevels 0, mlHubAggregated 0 in every run |
| 3. discrimination on ca32026 | dP > 1e-4 | 2.9e-7 (no level ever built) |
| 4. `hub`, `hub_posonly`, `hub_pgs` np 1 (host OMP 1, 8 x 3; CUDA x 3; step_mpi and `--solo`) | dP <= 1e-6, dXpos <= 3e-5, CONFLICTS 0 | dP <= 3.9e-7, dXpos <= 9.4e-7, CONFLICTS 0 |
| 5. `cluster_periodic --solo` CoM drift | <= 1e-5 R | 1.9e-6 R (OMP 1, 8) |
| 6. np 1 OMP 1 dumps vs ca32026 | identical except named | 22 / 29 identical; DIFF exactly `hub*` (6) and `cluster_periodic_solo`; `ring_mini`(`_solo`), np 4 `cluster` / `cluster_pgs` identical; S6 5 / 5, pile 4 / 4 identical |
| 7. battery (`build_ct`, OMP 2, `-j1`, core Python on PYTHONPATH) | all pass | **200 / 201**: only `momentum_hub_ml_np1` (stop 1); the 12 `python_mpi_*` RUN and pass |

**Stop 1 (a stated fact is false): `hub_ml` as specified never builds a multilevel level.** The pass
runs in all 10 steps (post-loop residual 0.41-0.55 > vRestS 0.2), but the shell leaves
(makeHub's `ns = 2.5 (rs/R)^2`, spacing ~1.2 D) do not touch each other, so the eligible contact graph
is a star: the pairwise matching merges the hub with one leaf, `ngNew = 150 > 0.9 ngPrev = 151`, and
the stall rule rejects the level. So the positive controls fail and B cannot be reproduced (2.9e-7 on
ca32026). R-F8's retries (speed 0.5, hub density 1/8) do not change the graph topology. Evidence for
the note's author (test-only experiment, not committed): a denser shell `ns = 3.0 (rs/R)^2` (N = 181,
leaves touching) builds 1 level, aggregates the hub, and gives **dP 1.5e-2 on ca32026 -> 1.7e-7 to
2.1e-7 after WO-4b** (np 1, OMP 1 and 8, step_mpi and `--solo`); `3.6` gives 2 levels, dP 1.8e-2 on
ca32026. Needed: the scene (shell density or another way to give the hub aggregable neighbours).

**Stop 2 (a stated estimate is false): the `hub_static` gap bound 0.05 delta does not hold at OMP 8
or on CUDA.** §13.6 estimates the coupling gaps at ~ k m_leaf/m_hub delta <= 0.02 delta. Measured
0.034 delta (OMP 1) to 0.061 delta (OMP 8), thread-order dependent (the colouring order). My reading:
the estimate omits the Gauss-Seidel partial sums inside one copy -- a copy of mass m/k takes ~32
sequential pushes of delta m_leaf/(m/k) = 0.01 delta each before the fold, a random walk of
~sqrt(32) x 0.01 = 0.06 delta, which the non-retractable projection keeps as gap. omega 1.5 on
ca32026 gives 0.584 delta, so a bound of about 0.1-0.2 delta would still discriminate by 3-6x.
Needed: the bound (or the scene).

WO-5 not started (it depends on WO-4b, and its acceptance reuses both modes).

## WO-4b (8b4a5f3): committed with §12 S10 / S11 / S12

The parked 9cbd29f cherry-picked and amended: S10 (`hub_ml` = the dense shell `ns = 3.0 (rs/R)^2`,
N = 181), S11 (`hub_static` gap bound 0.15 delta), S12 (a fused main velocity / position loop writes
its iteration count to a device scalar, `FusedLoopSpec::iters`; read back only with the C++ switch
`Simulation::debugIterationCounters`, which the test sets; off = no store, no copy; test flag
`--fused=auto|on|off`), clang-format over the touched files. Raw: `impl_a/wo4b2_*.txt`. Host load
80-95.

| check | gate | result |
|---|---|---|
| 2. `hub_static` np 1 (OMP 1, OMP 8 x 3; step_mpi, `--solo`) | gap <= 0.15 delta, residual <= 1e-4 R, dXpos <= 3e-5, \|P\| = 0 | gap <= 0.057 delta, residual <= 4.3e-6 R, dXpos <= 5.7e-8, \|P\| 0 |
| same, CUDA x 3 | same | gap <= 0.056 delta, residual <= 4.0e-6 R, dXpos <= 6.6e-8, \|P\| 0 |
| 2. discrimination (ca32026 numerics + the WO-4b tests) | gap >= 0.3 delta | 0.584 delta |
| 3. `hub_ml` np 1 (host OMP 1, 8 x 3; CUDA x 3; step_mpi, `--solo`) | dP <= 5e-6, controls | dP <= 2.7e-7 (host), <= 3.4e-7 (CUDA); velCopies 3, levels 1, mlHubAggregated 1 in every run |
| 3. discrimination (ca32026 numerics) | dP > 1e-4, controls | dP 1.50e-2; levels 1, aggregated 1 |
| 4. `hub`, `hub_posonly`, `hub_pgs` (host, CUDA) | dP <= 1e-6, dXpos <= 3e-5, CONFLICTS 0 | dP <= 3.7e-7, dXpos <= 9.4e-7, CONFLICTS 0 |
| 5. `cluster_periodic --solo` drift | <= 1e-5 R | 1.1e-6 R (host), 1.7e-6 R (CUDA) |
| S12: CUDA `--solo --fused=on` vs `off`, `--steps=1` | counts reported | `cluster_posonly --pos-iters=400` 97 / 97 (fused) vs 95-96 (launch; CUDA run-to-run); `cluster_pgs --vel-iters=400` 116 vs 116 |
| 7. battery (`build_ct`, OMP 2, `-j1`, core Python on PYTHONPATH) | all pass | **201 / 201**, the 12 `python_mpi_*` run and pass |

Item 6 (np 1 byte-identity to ca32026) stands from the parked run; the S12 change is host-inert by
construction (the counter lives in the CUDA-only fused kernel and a branch on a device-loop flag),
and the WO-5 G3 run below re-proves every np 1 dump against a build of 8b4a5f3.

## WO-5 (2dfdee4 on branch `contacts-wo5-parked`, NOT on `contacts`): implemented, six stops

§13.3 as written (activity pass; PGS opening `openVelocityPhase` 40 B / 28 B; `g = 0` counts
opening; owner seeds; a-weighted pack; k-weighted apply, raw for k <= 1; orphan shares, the balance
forwarded under Poisson, the phase-end restore; `invMassCoarse`; solve views from the per-slot k);
§6.2 (`ghostCanon_`, the slot map in `demStepMpi`); hooks; debug checks; `split_stats.orphanClamps`.
An MPI rank with `!exchanges()` runs WO-4's single-rank path (step 0), bit for bit.

**Finding (acted on, flagged): the MPI twin-dedup switch has to cover every dedup site.** §4.2
item 2 names the velocity colourings only, but the warm gather, `computeVn0`, the warm start, the
side flags, the Poisson bank, the Jacobi count / solve and the legacy friction normal pass all skip
`realIdx(A) > realIdx(B)`. With the §6.2 map a periodic self image of a lower slot makes the OWNED
twin satisfy that test, so they would drop it (no warm start, no bank). All take
`dedupTwins = !distributed` (np 1 Solo unchanged; np 1 step_mpi closed has no images).

Raw: `impl_a/wo5_*.txt` (summaries `wo5_matrix_summary.txt`, `wo5_g7_tables.txt`); scripts
`wo5_{dumps,matrix,matrix_cuda,g4,g7,perf}.sh`, `wo5_g7.py`, `wo5_msumm.py`. Host load 80-95.

| check | gate | result |
|---|---|---|
| 1. G1 matrix np 1/2/4/8 x (OMP 1, OMP 8 x 3), 11 modes | §9 + §13.6 rows | dP <= 8.1e-7 (PGS family, <= 5e-6), 0 (posonly); dX, dXpos <= 6.6e-7 (clusters), <= 3.5e-6 (hubs), <= 4.7e-6 (ring); CONFLICTS 0 everywhere. **Misses:** `cluster_multilevel` dLvel 3.2e-5 (np 4), 8.7e-5 (np 8) vs 1e-6 (stop 6); `ring_mini` dLvel 1.5e-3-2.3e-3 vs 1e-5 at EVERY np, np 1 included (pre-existing: 1.55e-3 at the WO-0 baseline) |
| 1. same, CUDA np 1 x 3 (step_mpi, `--solo`) | same | dP <= 7.7e-7, dXpos <= 7.8e-7, CONFLICTS 0; hub_static gap 0.057 delta; hub_ml controls L1 A1 |
| 2. `ring_mini` np 2 / 4 / 8 (S9 closed) | dP <= 1e-6, dXpos <= 1e-5 R, CONFLICTS 0 | dP <= 3.5e-8, dXpos <= 4.7e-6, CONFLICTS 0 (was np 8: dXpos 0.14, ovl 3.6e6) |
| 2. `ring_mini` ovl | <= 10 x np 1 and <= 0.1 R | np 1 0.250 (= 0.50 R); ratio np 2 / 4 / 8 = 0.91 / 0.89 / 1.16. **The 0.1 R bound fails at np 1 itself (stop 1)** |
| 3. `hub_static` np 2 / 4 / 8 | gap <= 0.15 delta, residual <= 1e-4 R | gap 0.025 / 0.014 / 0.011 delta, residual 4.7e-5 / 6.5e-5 / 7.7e-5 R |
| 3. `hub_ml` np 2 / 4 / 8 | controls, dP <= 5e-6 | dP 4.1e-7 / 2.5e-7 / 2.4e-7; controls np 2 L1 A5; **np 4, 8: no level built (stop 2)** |
| 4. G3 np 1 vs a build of 8b4a5f3, OMP 1: 24 modes x (step_mpi, `--solo`), 5 S6 scenes, 4 piles | identical but named | **55 / 56 identical**; DIFF only `cluster_periodic` np 1 step_mpi (named) |
| 4. named change `cluster_periodic` np 1 step_mpi | report | dP 2.71e-9 -> 5.08e-9; periodic CoM drift 9.3e-7 -> 8.5e-7 R; ovl 2.90e-2 -> 6.24e-2; KE s1 2322.8 -> 2330.5, s10 1156.1 -> 1150.5 |
| 5. G4 np 4, 8, OMP 1, 5 runs | identical | 12 / 12 IDENT (`cluster_pgs`, `cluster_poisson`, `cluster_posonly`, `hub_pgs`, `hub_ml`, `ring_mini`) |
| 6. G7a `tri_pgs --axis=2` np 2 vs np 1, it 4/8/16/32/64 | non-increasing, <= 1e-3 at 16, <= 1e-5 at 64, np2 <= np1 | 3.5e-2 / 3.9e-3 / 1.57e-3 / 1.57e-3 / 1.57e-3; KE_np2 <= KE_np1 always. **Misses (stop 3)** (default axis 0 does not cross the np 2 face: 0 at every count) |
| 6. G7c `cluster_pgs --steps=1`, RMS \|v_np - v_np1\| / \|v\| at 8/32/128 | non-increasing, <= 1e-3 at 128 | np 2: 1.3e-1 / 4.0e-2 / 6.8e-4; np 4: 1.7e-1 / 6.3e-2 / **3.2e-3**; np 8: 2.2e-1 / 8.0e-2 / **3.5e-3** (stop 4) |
| 6. G7e ovl, ratio to np 1 (np 1: 2.70e-2 / 2.19e-2 / 2.06e-2) | hard <= 10 x np 1 and <= 0.1 R (0.05); report ratio | `cluster` 1.66 / 1.68 / 1.94x (np 8: 0.0523 > 0.05); `cluster_pgs` 1.93 / 2.26 / 2.26x; `cluster_friction` 2.03 / 2.53 / 2.54x (np 4, 8: 0.0521, 0.0523 > 0.05). c771e07: 0.024-0.029 at np 2-8 (stop 5; ratio > 1.5 = R-U4 evidence per R-F6) |
| 6. G7f ITERS ratio (np 1: pos 96, vel 116) | np 2 <= 3.5, np 4 / 8 <= 5; report > 2.5 | pos 1.64 / 1.64 / 2.43x; vel 1.29 / 1.74 / 1.75x |
| 7. orphanClamps, `cluster_poisson`, every np and OMP, CUDA | 0 | 0 |
| 8. Stop-A tests, `validate_periodic` np 2 / 4 | pass | pass (battery) |
| debug-build checks (`-O2` without NDEBUG), 14 mode x np runs | never fire | none fired |
| 9. performance | median <= 1.10 | NOT RUN (`impl_a/wo5_perf.sh build_base/tests/kokkos_mpi/test_momentum_mpi <WO-5 build>/tests/kokkos_mpi/test_momentum_mpi 5`) |
| battery (`build_ct` built from 2dfdee4, OMP 2, `-j1`) | all pass | **200 / 201**: only `momentum_hub_ml_np4` (stop 2); the 12 `python_mpi_*` run and pass |

**Stop 1: `ring_mini`'s "<= 0.1 R at every step" is unreachable at np 1.** ovl (the position loop's
last residual, absolute) is 0.20-0.27 at np 1 in every build since WO-0 (0.4-0.54 R, R = 0.5): the
jammed ring lattice does not converge in 20 position iterations. The relative guard holds (<= 1.16 x
np 1) and nothing diverges.

**Stop 2: `hub_ml` builds no multilevel level at np 4 and 8** (the WO-4b build: the same, so it is
the scene, not WO-5). The whole 181-body scene lies inside every rank's ghost band, so each rank's
vertex count is 181 while it owns only ~1/np of the eligible edges: the pairwise matching merges a
few pairs and the 90 % stall rule rejects the level (instrumented: np 4 levels 0 in every one
of the 36 rank-substeps the pass ran, eligible manifolds 0-11 per rank). Conservation holds (dP <= 2.5e-7).

**Stop 3: G7a's 1e-3 at 16 and 1e-5 at 64 sit below the PGS loop's adaptive stop.** The np 2
iterates reproduce Appendix A.1's M-PGS model (0.1006 at 4, 0.1039 at 8); the loop then stops at
`maxApproach <= 0.02 vRest` near 16 iterations with 0.10410 against 0.10427, so 32 and 64 are the
same run. The gate assumed the cap is the iteration count.

**Stop 4: G7c misses at np 4 / 8 and R-F2 does not rescue it.** At 128 iterations np 1 has stopped
at 116 (tolerance) while np 4 / 8 need 202 / 203 (G7f). R-F2's default was applied as an
experiment, not committed (`impl_a/wo5_rf2_experiment.txt`, `kSplitOmegaVelocity = 1.5`): G7c at
128 = 9.8e-4 / 1.15e-3 / 1.35e-3 (np 2 / 4 / 8) -- still > 1e-3 at np 4, 8, so R-F2 says stop;
vel iterations 178 / 180 / 185 (np 2 worse than omega 1's 150); G7a 6.0e-4 from 8 on; and it
changes the np 1 PGS hub dumps (`hub_pgs`, `hub_ml`, step_mpi and `--solo`) against G3, because
`splitSlot` also marks local hub copies. Needed: whether R-F2 is scoped to rank-split slots, and
the G7c bound.

**Stop 5: G7e's hard 0.1 R is exceeded by 4-5 %** (`cluster` np 8 0.0523; `cluster_friction` np 4, 8
0.0521, 0.0523; limit 0.05). At a fixed budget of 20 position iterations the rank faces leave 1.7-2.5x
np 1's residual, which §13.4 predicts (1.5-9x); c771e07's raw sum left 0.024-0.029. Per R-F6 the
ratios (> 1.5) go to the user with R-U4.

**Stop 6: `cluster_multilevel` dLvel 3.2e-5 (np 4) / 8.7e-5 (np 8) against G1's 1e-6.** In one of
24 (np 4) / 56 (np 8) rank-substeps the pass now builds a coarse level (c771e07 and WO-4b: none,
dLvel 1.5e-8), and the coarse cycle is translation-only by design (§13.6 reports dLvel for
`hub_ml` for that reason). An experiment with WO-4b's coarse masses (`invMass` for every vertex)
builds the same level and gives 2.6e-4 / 8.5e-5, so it is not the `k / a` rule. dP stays 7.9e-7.

**Also not in the note:** `ring_mini`'s G1 dLvel 1e-5 fails at every np and since WO-0 (1.5e-3 at
np 1); WO-5 does not change it. `contacts` stays at 8b4a5f3; `build_ct` / `build_ct_cuda` hold the
2dfdee4 binaries.

## WO-5: committed with §12 S13 / S14 / S16 / S18

The parked 2dfdee4 cherry-picked onto `contacts` (8b4a5f3 + note e58d7f1), then:
- **S14 (the stop residual includes the consensus correction).** A device scalar
  `Particles::maxConsensus` receives, by atomic max, the largest correction any consensus applies to
  an ACTIVE copy: the local fold (`foldCopiesKokkos`, `|T/k - (x_q - sigma - shift)|`), the owner row
  of an M sync (`|mean - own|` in increment form) and every active ghost copy after the forward
  (`haloGhostConsensus`, `|(x_new - base) - inc/a|`). Velocity phase: `|dv|` of `velPred`
  (the note's `|v_avg - v_copy|`, absolute like the phase's residual); position phase: `|dx|`.
  Every loop of a phase with copies (main velocity, one-sided, multilevel, ordered, escalate,
  position) folds it into its existing stop vote (`max(res, cons)`, the SAME Allreduce-MAX) and zeroes
  it after the read; a sync's correction therefore enters the next iteration's vote. A phase without
  copies neither reads nor writes it (np 1 without copies: byte-identical, below).
- **S13 (G7 with the stops off).** C++-only `Simulation::debugNoAdaptiveStop(bool)`
  (`Particles::noAdaptiveStop`; test flag `--no-stop`): every adaptive stop, the fused device loops'
  tolerances included, is skipped; the votes still run (collective, colouring invariant). Not bound
  to Python; no environment variable.
- **S16.** `test_momentum_mpi`'s dLvel baseline is the predicted omega (`predictedOmega`: the
  predict's gyroscopic Euler term replayed in double; isotropic bodies keep w exactly).
- **S18.** `hub_ml`'s positive controls are required at np 1 and 2 only; ring_mini's and G7e's
  overlap bounds are `<= 3 x np 1` of the same build (`wo5_msumm.py`, `wo5_g7_s13.py`).

Raw: `impl_a/wo5f_*.txt`; scripts `wo5_matrix.sh`, `wo5_matrix_cuda.sh`, `wo5_g4.sh`,
`wo5_dumps.sh`, `wo5_g7_s13.sh` / `.py` (G7a with `--axis=2`, which crosses the np 2 face). Host
load 60-75 (48 cores).

| check | gate | result |
|---|---|---|
| 1. G1 matrix np 1/2/4/8 x (OMP 1, OMP 8 x 3), 11 modes | §9 + §13.6 rows | dP <= 8.1e-7 (PGS family), 0 (position-only); dX, dXpos <= 6.7e-7 (clusters), <= 3.4e-6 (hubs), <= 5.5e-6 (ring); CONFLICTS 0; orphanClamps 0. Reported, not gated (S17): `cluster_multilevel` dLvel 1.6e-4 (np 4) / 6.8e-5 (np 8), `hub_ml` 3.0e-3 / 3.7e-3 (np 1 / 2). `ring_mini` dLvel 6.4e-4-9.2e-4: legacy friction's inertia, fixed by WO-5b |
| 1. same, CUDA np 1 x 3 (step_mpi, `--solo`) | same | dP <= 7.8e-7, dXpos <= 8.4e-7, CONFLICTS 0; hub_static gap 0.058 delta; hub_ml controls L1 A1 |
| 2. `ring_mini` np 2 / 4 / 8 | dP <= 1e-6, dXpos <= 1e-5 R, CONFLICTS 0, ovl <= 3 x np 1 (S18) | dP <= 3.5e-8, dXpos <= 5.5e-6, CONFLICTS 0; ovl ratio 0.99 / 0.95 / 1.16 |
| 3. `hub_static` np 2 / 4 / 8 | gap <= 0.15 delta, residual <= 1e-4 R | gap 0.032 / 0.013 / 0.012 delta; residual 4.5e-5 / 6.2e-5 / 7.7e-5 R |
| 3. `hub_ml` np 2 / 4 / 8 | controls at np <= 2 (S18), dP <= 5e-6 | np 2 L1 A5, dP 4.3e-7; np 4 / 8 (no level, conservation only) dP 2.5e-7 / 1.8e-7 |
| 4. G3 np 1 OMP 1 vs a build of 8b4a5f3: 24 modes x (step_mpi, `--solo`), 5 S6 scenes, 4 piles | identical but named | **55 / 56 identical**; DIFF only `cluster_periodic` np 1 step_mpi (named): dP 2.71e-9 -> 5.08e-9, CoM drift 9.3e-7 -> 8.5e-7 R, ovl 2.90e-2 -> 6.24e-2, KE s1 2322.8 -> 2330.5, s50 664.7 -> 666.0 |
| 5. G4 np 4 / 8, OMP 1, 5 runs | identical | 12 / 12 IDENT |
| 6. G7a `tri_pgs --axis=2 --no-stop`, it 4/8/16/32/64 | non-increasing; <= 1e-3 at 16, <= 1e-5 at 64; KE_np2 <= KE_np1 | 3.5e-2 / 3.64e-3 / 6.29e-5 / 1.41e-7 / 1.41e-7; KE_np2 <= KE_np1 always. Pass |
| 6. G7c `cluster_pgs --steps=1 --no-stop`, RMS vs np 1 same N, it 8/32/128/256 | non-increasing, <= 1e-3 at 256 (S13) | np 2: 1.3e-1 / 4.0e-2 / 7.8e-4 / 4.6e-5; np 4: 1.7e-1 / 6.3e-2 / 3.2e-3 / 1.0e-4; np 8: 2.2e-1 / 8.0e-2 / 3.6e-3 / 1.2e-4. Pass |
| 6. G7e ovl, ratio to np 1 (2.70e-2 / 2.19e-2 / 2.06e-2) | <= 3 x np 1 (S18); ratio to the user with R-U4 | `cluster` 1.66 / 1.68 / 1.94; `cluster_pgs` 1.93 / 2.26 / 2.26; `cluster_friction` 2.03 / 2.53 / 2.54. Pass |
| 6. G7f ITERS ratio, stops on with S14 (np 1: pos 96, vel 116) | np 2 <= 3.5, np 4 / 8 <= 5; report > 2.5 | pos 1.64 / 1.64 / 2.43; vel 1.29 / 1.74 / 1.75 (unchanged by S14, below) |
| 7. orphanClamps, `cluster_poisson`, every np / OMP, CUDA | 0 | 0 |
| 8. Stop-A tests, `validate_periodic` np 2 / 4 | pass | pass (battery) |
| battery (`build_ct`, OMP 2, `-j1`, core Python on PYTHONPATH) | all pass | **201 / 201**; the 12 `python_mpi_*` ran |
| 9. performance | median <= 1.10 | measured after WO-6 against 8b4a5f3 (brief), below |

**S14 is inert in every gated scene (measured, not assumed).** With a throwaway print in the vote
(a scratch build, not committed): `cluster_pgs` np 4 `--vel-iters=400` folds a non-zero consensus
correction into 888 votes, always 0.3-0.6 x the sweep residual; at the stop (iteration 202) the
residual is 4.16e-3 against the tolerance 0.02 vRest = 4e-3 and the consensus 1.25e-3. `hub` np 1:
11 of 32 votes carry a fold correction, none above the residual. The consensus correction is
dominated by the sweep's own corrections, so the stop iteration, G7f and every np 1 dump are
unchanged (the note expected more iterations at interfaces). The mechanism is in place and
collective-safe; its effect is nil where measured.

## WO-5b: world-frame inverse inertia in legacy friction (§12 S15), verified 2026-09-26

The implementing agent was interrupted by a usage limit, so the session verified the change itself.
Host load 0.3. Reference: a build of 0ea32ba (WO-5) in `../dem-ref5`. Raw outputs are in the
session scratchpad, `wo5b/`.

- **np 1 byte identity against WO-5 (`wo5_dumps.sh`):** 45 of 47 mode dumps are identical. The 2
  that differ are `ring_mini` and `ring_mini_solo`, i.e. non-spherical, the named change. The S6
  wall scenes and the multilevel pile, 9 of 9, are identical.
- **`ring_mini` dLvel, np 1 / 2 / 4 / 8, OMP 1:**
  - before (WO-5): 5.3e-4 / 8.6e-4 / 4.3e-4 / 3.4e-4
  - after: 3.6e-8 / 3.8e-8 / 2.0e-8 / 3.4e-8

  This matches INVESTIGATION_WO5 (2–4e-8). dP ≤ 2.3e-8. The overlap is 0.247 / 0.226 / 0.212 /
  0.291, against 0.250 / 0.215 / 0.220 / 0.289 before.
- **The hand-worked ring–ring impulse test** (tests/kokkos/test_solver_friction.cpp) passes. All
  7 friction ctests pass.
- **Position path** (`solver_position.hpp`): the world-frame form is applied for consistency.
  `deltaQuat` is never committed (`applyUpdatesKokkos` commits `deltaPos` only), so it changes no
  result today.

## WO-6: rank-level X for the g = 0 one-shot (implemented and verified by the session, 2026-09-26)

The Opus subagents had hit their weekly usage limit, so the session implemented WO-6 itself.

**Code:**
- `ParticleHalo::ensureRankColoring`: greedy colouring of "blocks within 2 band" (§12 S19),
  computed from the replicated decomposition with no communication. `rankColor()` and
  `numRankColors()` expose the result.
- The g = 0 opening (`openPositionCounts(P, velocityMask)`) carries `PositionCountsMask`
  {mask, a_pos}: 16 B per ghost each way, in the same round.
- `P.velMask`, `P.xGate`, and `P.solveEpoch` (incremented per `demSolveContacts`).
- `computeXGateKokkos` and `xHolder` (§1.4). The one-shot kernel takes `gate`: a gated contact
  records its approach and writes nothing.
- `RankK` gained `color`, `numColors` and `syncInterval`.

**Results** (OMP 1 unless stated; raw output in the session scratchpad `wo6/`):

| gate | result |
|---|---|
| review 3-body `tri --axis=2`, KE at step 3, np 1/2/4/8 | e = 0.5: 0.1379044264 at every np (WO-5 np 2: 0.2907, energy created). e = 0.8: 0.2459279908 at every np, the review's np 1 value. Axes 0 and 1 unchanged, np 2 = np 1. |
| KE non-increasing, np 8 OMP 8 | `cluster`, `cluster_e09`, `hub` decrease every step. `cluster_e10` (elastic) is flat to +2e-8 relative (float round-off). |
| inertness vs WO-5 (ref build 0ea32ba), np 1/2/4/8 | 44 / 44 IDENT. The modes are `cluster_pgs`, `_poisson`, `_multilevel`, `_escalate`, `_ordered`, `_onesided`, `_posonly`, `hub_pgs`, `hub_posonly`, `hub_static`, `hub_ml`. |
| G4 run-to-run, np 4/8, 3 runs | 6 / 6 IDENT (`cluster`, `tri`, `hub`). |
| G1 conservation, np 1/2/4/8 × OMP 1/8 | 64 / 64 OK. Modes: `cluster`, `_friction`, `_sync3`, `_norot`, `_e09`, `_e10`, `hub`, `tri`. Max dP 7.7e-9 (hub 3.1e-7), dX ≤ 5.9e-7 R, dLvel ≤ 2.2e-8. |
| battery (OMP 2, python_mpi run) | 201 / 201 |

**Not yet measured:** `unfiredSplitContacts` in `perf_gas`. The counter exists under
`iterCounters`, but `perf_gas` does not print it; that goes into WO-11's evidence. Also not yet
measured: ms/step in the pinned protocol (WO-11).
