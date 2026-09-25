# dem contact solve: four follow-up defects, reproduced and measured

Measured 2026-09-25 on branch `contacts` (worktree `suite/dem-contacts`, base dem `c771e07`, the
momentum-conserving distributed step of `docs/mpi_momentum_conservation.md`). **Nothing in `src/`
is changed.** This note reproduces and measures each defect. It does not choose a fix.

**Builds.**
- Host: `build_ct`, prefix `host-openmp`, Release, `PECLET_DEM_BUILD_TESTS=ON`, `PECLET_DEM_MPI=ON`.
- GPU: `build_ct_cuda`, prefix `nvidia-cuda`, RTX 5080. Only the three test targets were built.

**Host conditions.** 48 cores, 1-min load 50–60, shared with other sessions. Every run used
`OMP_PROC_BIND=false OMP_WAIT_POLICY=passive`.

**Reproducing.** The report-only tests are listed in the table below. The raw outputs are in this
directory, and every number quoted here comes from one of them.

| defect | test (mode) | ctest | raw output / script |
|---|---|---|---|
| 1 colour overflow | `tests/kokkos/test_coloring_overflow.cpp` | `coloring_overflow` | stdout |
| 1 | `test_momentum_mpi hub`, `hub_posonly` (`--solo`, `--hub=S`) | `followup_hub*_np{1,2,4}` | `hub.txt`, `hub_cuda.txt`, `cuda_control.txt` (`run_hub.sh`) |
| 1, survey | `degree_survey.py` | — (minutes per case) | `survey.txt` |
| 2 friction couple | `test_momentum_mpi friction_pair`, `friction_pair_pgs` (`--delta`, `--dt`); `cluster_friction --dt --posit` | `followup_friction_pair*_np1` | `friction.txt` (`run_friction.sh`) |
| 3 per-body Jacobi | `test_momentum_mpi cluster_jacobi` (existing gate) | `momentum_cluster_jacobi_np*` | `jacobi.txt` |
| 4 missed pairs | `test_ownership_mpi missed_drift_pair`, `missed_drift_lattice`, `missed_periodic` | `followup_missed_*` | `missed.txt` (`run_missed.sh`) |

**The report-only tests never fail.** Each source has a gate constant, `false` today:
- `kGate` in `test_coloring_overflow.cpp`;
- `kFollowupGate` in `test_momentum_mpi.cpp`;
- `kMissedGate` in `test_ownership_mpi.cpp`.

Its comment states the condition it will gate once the defect is fixed. The 18 new ctests pass in
11 s: `ctest -R "followup|coloring_overflow"`.

**The test edits are inert.** The pre-existing modes were run against binaries built from HEAD's
test sources (`run_inert.sh`, `inert.txt`), and 38 of 38 matched:
- 9 `test_momentum_mpi` modes at np 1 and 4, OMP 1: final-state `--dump` and the MOMENTUM line
  byte-identical;
- 5 `test_ownership_mpi` modes at np 1, 2, 4 and 8: printed reports identical.

---

## 1. Colour overflow: same-colour contacts at a body with more than 63 contacts

### Mechanism
Each Gauss–Seidel sweep runs one colour class at a time, with the contacts of a class in
parallel. Each contact reads and writes its two bodies' velocity or position in place, with no
atomics. That is race-free only if no two contacts in one class share a body.

The colourings give each contact the lowest colour that is free at both of its bodies. The search
stops at 62, and colour 62 is handed out whether or not it is already taken. That leaves 63
colours, 0..62. So a body with D > 63 contacts carries at least D − 63 contacts of the same colour,
and pigeonhole makes this certain. Colour 62 can be forced on a contact (A, B) only when colours
0..61 are all taken across the two masks, which needs deg(A) + deg(B) ≥ 64.

Inside that class, several threads then read-modify-write the same body, and updates are lost. A
lost update on one side of an impulse pair is a momentum error. In the position phase it moves the
centre of mass.

The code intends something else: leave such contacts uncoloured (−1) and apply them with the
count-averaged fallback. That branch is never reached:
- every arbitration round colours at least one contact (the one holding the maximum key at both of
  its bodies);
- so `rem == prevRemaining` never holds and the stall break never fires;
- so `leftover` is always 0.

### Code
Five colourings share the cap:

| colouring | file:line | used by |
|---|---|---|
| `colorManifoldsKokkos` | `src/solver_velocity.hpp:343` | velocity sweeps (every configuration) |
| `colorManifoldsIncrementalKokkos` | `src/solver_velocity.hpp:495` | velocity, single-GPU, g ≠ 0 |
| `colorContactsKokkos` | `src/solver_position.hpp:206` | position sweeps (every configuration, **defaults included**) |
| `colorContactsIncrementalKokkos` | `src/solver_position.hpp:360` | position, single-GPU, g ≠ 0 |
| multilevel (aggregate groups) | `src/solver_multilevel.hpp:328` (`kMlSlotSkip - 1` = 62) | gravity statics, ordered sweeps |

`src/solver_velocity.hpp:339-354`:
```cpp
          std::uint64_t forbidden = bodyMask(ea);
          if (eb >= 0)
            forbidden |= bodyMask(eb);
          int c = 0;
          while (c < 62 && (forbidden & (std::uint64_t(1) << c)))
            ++c;  // lowest free colour (cap 63; dense sphere degree ~12, far below)
          mColor(idx) = c;
          const std::uint64_t bit = std::uint64_t(1) << c;
          bodyMask(ea) |= bit;
          if (eb >= 0)
            bodyMask(eb) |= bit;
        },
        rem);
    space.fence();
    if (rem == prevRemaining)
      break;  // colour-mask saturation (degree > 62): leftovers stay -1, Jacobi fallback applies
```
`src/solver_position.hpp:205-220` is the same loop (`col < 62`), followed by the stall comment
"The stuck contacts stay -1 and are handled by the Jacobi fallback in the solve."

The sweeps assume independent classes:
- `src/solve_driver.hpp:316`: "colour classes are body-disjoint";
- the in-place writes, for example `src/solver_velocity.hpp:1702-1704`
  (`velPred(realA, 0) += Jlin.x * invMassA;`), `src/solver_velocity.hpp:1131` (the PGS sweep) and
  `src/solver_position.hpp:488-495` (`posPred(idA, 0) += n.x * dLambda * invMassA;`).

The unreachable fallback is `src/solve_driver.hpp:496-507` (`bool velFallback = velLeftover > 0;`)
and `:917-927` (`posFallback`).

### Repro
```bash
OMP_NUM_THREADS=4 build_ct/tests/kokkos/test_coloring_overflow           # star graph, D = 32..300
OMP_NUM_THREADS=8 build_ct/tests/kokkos_mpi/test_momentum_mpi hub --solo  # single-rank demStep
mpirun --bind-to none -np 4 build_ct/tests/kokkos_mpi/test_momentum_mpi hub_posonly   # step_mpi
docs/contact_evidence/run_hub.sh build_ct       # the matrix; build_ct_cuda for the GPU
```
**The hub scene.** A grain of scale S = 10 (radius 10 R) sits at the cluster centre. About
2.5 (S + 1)² ≈ 300 unit grains (scale 1 ± 0.1) lie on a Fibonacci shell around it, each
overlapping the hub by 0.02–0.07 and moving inward at about 1. There is no gravity and no friction.
`hub` has the velocity solve on (8 iterations); `hub_posonly` has it off, the dem default.

### Numbers
**Direct check of the colouring** (`test_coloring_overflow`; host and CUDA identical). One hub
touches D leaves:

| D | colours | leftover | same-colour pairs at the hub | contacts with colour 62 |
|---|---|---|---|---|
| 32, 62 | 32, 62 | 0 | 0 | 0 |
| 63 | 63 | 0 | 0 | 1 |
| 64 | 63 | 0 | **1** | 2 |
| 65 | 63 | 0 | 2 | 3 |
| 100 | 63 | 0 | 37 | 38 |
| 300 | 63 | 0 | **237** | 238 |

The contact graph and the manifold graph give identical rows. `leftover` is 0 in every case.

**Hub scene** (20 steps). `posConf` / `velConf` count same-colour pairs per body in the rank-local
colourings, taken as the maximum over steps and summed over ranks.

At np 1:
- the hub has 300 contacts and 242 active manifolds;
- posConf = 237 and velConf = 179 in every run;
- the counts are identical at OMP 1 and 8 and on CUDA (the colouring is deterministic).

At np 2 and 4, each rank colours only its owned share of the hub's contacts. That share is at most
157 contacts per rank. The conflict counts are then 174 / 116 (np 2) and 111 / 64 (np 4).

| backend | mode | np | OMP | dP (3 runs) | dXpos / R (3 runs) |
|---|---|---|---|---|---|
| host | hub | 1 | 1 | 1.6e-7 (=) | 2.7e-7 (=) |
| host | hub | 1 | 8 | 2.5e-7 .. **6.7e-3** | 2.3e-7 .. 1.2e-5 |
| host | hub | 2 | 8 | 3.3e-7 .. **1.1e-3** | 2.3e-7 .. 2.7e-7 |
| host | hub | 4 | 8 | 1.5e-7 .. 2.3e-7 | 5.1e-7 .. 6.1e-7 |
| host | hub_posonly | 1 | 1 | 0 | 1.1e-5 (=) |
| host | hub_posonly | 1 | 8 | 0 | **6.4e-5 .. 2.1e-4** |
| host | hub_posonly | 2 / 4 | 8 | 0 | 3.8e-6 .. 8.4e-5 |
| CUDA | hub | 1 | – | **1.0e-2 .. 1.6e-2** | 2.4e-4 .. 1.0e-3 |
| CUDA | hub | 2 | – | 6.1e-3 .. 7.2e-3 | 1.9e-4 .. 8.2e-4 |
| CUDA | hub | 4 | – | **4.0e-2 .. 4.1e-2** | 1.7e-3 .. 2.2e-3 |
| CUDA | hub_posonly | 1 / 2 | – | 0 | **9.4e-4 .. 2.2e-3** |
| CUDA | hub_posonly | 4 | – | 0 | **1.3e-2 .. 1.4e-2** |

The single-rank `demStep` (`--solo`) and `step_mpi` at np 1 behave the same:
- host OMP 8, one run in three: dP 2.9e-3 and 6.7e-3;
- CUDA: dP 1.0e-2..1.6e-2 and 1.1e-2..1.5e-2.

(dP is normalised by Σ m|v|, dXpos by R. `(=)` means all three runs agreed. For CUDA, the host
thread count does not matter, and all 36 GPU runs are inside the ranges shown.)

The 1.1e-5 floor of `hub_posonly` at OMP 1 does not change from run to run. It is most likely
float rounding of the heavy hub's position (the hub is 1000 times the mass of a leaf, at a
position of order 0.3). That was not investigated further.

**Control on CUDA** (`cuda_control.txt`), the same scene with a smaller hub:

| hub scale | hub degree | same-colour pairs | dP | dXpos / R |
|---|---|---|---|---|
| 3 | 39 | 0 | 9.3e-8 | 1.7e-7 |
| 4 | 61 | 0 | 9.5e-8 | 1.6e-7 |
| 5 | 88 | 25 / 11 | **1.8e-2** | 1.6e-3 |
| 6 | 121 | 58 / 40 | 2.6e-2 | 1.7e-3 |

The `cluster` control on CUDA gives dP 3.1e-9. The step is round-off conservative on the GPU at hub
degree 39 and 61, and it jumps by five orders of magnitude as soon as the colouring is invalid. On the
host at 8 threads the race is intermittent, because the window is small and the pool is
time-shared at load 55:
- velocity phase: 3 of 12 runs (one each at np 1 `--solo`, np 1 and np 2; none at np 4);
- position phase (`hub_posonly`): above the OMP 1 level in 3 of 3 runs at np 1 and 2 of 3 at
  np 2 and at np 4.

### Does production reach it? (`survey.txt`, `degree_survey.py`)
The survey used growth-packed, fully periodic, jammed beds (φ = 0.62 for spheres). The degree
counts pairs within the narrow-phase reach r_i + r_j + 0.1 R_max. That is the position colouring's
graph for spheres.

| bed | size ratio | narrow-phase contacts per particle (2 n_c / N) | max degree | same-colour pairs, velocity / position graph (max over the run) |
|---|---|---|---|---|
| monodisperse (`pack.py`, foxberry / porous-scaling beds) | 1 | 8.5 | 11 | 0 / 0 |
| ± 10 % (the momentum cluster) | 1.22 | 8.6 | 12 | 0 / 0 |
| bidisperse, 50 % volume each | 2 | 8.2 | 20 | 0 / 0 |
| bidisperse | 3 | 8.7 | 34 | 0 / 0 |
| bidisperse | 4 | 9.3 | **62** | 0 / 0 |
| bidisperse (N 4000) | 6 | 10.7 | **114** | 0 / **952** (690 in the last substep) |
| hollow cylinders, RingBed config (D 1, H/D 1.5, wall 0.18, N 80; `ring_progress.txt`), φ 0.23 / 0.36 / 0.37 | – | **79 / 163 / 613** contact points (manifolds 5.4 / 9.2 / 9.8) | – | 0 / **1240**, 0 / **3869**, 0 / **3190** (worst 6439) |

**Reading.**
- **Spheres.** dem's own examples, the benchmark beds and the momentum cluster sit at degree 11–12,
  five times below the cap. The cap is reached at a size ratio of about 4 in a dense bidisperse bed
  (degree 62 measured) and exceeded beyond it.
- **Coupling.** Coupling's polydisperse test uses radii 0.5 / 0.8 (ratio 1.6), far below the cap.
- **Non-spherical particles.** The position colouring works on individual contact **points**, not on
  pairs. Each probe point of an SDF shell that penetrates a neighbour is its own contact. A
  RingBed-style ring bed therefore carries **79 contact points per particle on average already at
  φ = 0.23**. The count rises to 613 by φ = 0.37, while there are only about 10 pair manifolds per
  particle. That gives 1240–6439 same-colour pairs in the position colouring at every sampled substep.
  **Production hits this**:
  - the RingBed ring packing has same-colour contacts in its position phase (and, by the same
    mechanism but not measured, `examples/verify_packing_hollow_cylinders.py`), which is a race on every multi-thread and
    GPU run, and `relax()` runs the same phase;
  - the velocity (manifold) graph stays at 0 there.

  The same holds for spheres with a size ratio ≳ 5 (ratio 6: degree 114, up to 952 pairs). There
  the position graph counts the speculative neighbours inside the margin, while the velocity graph
  of active contacts stays below the cap (0 velocity conflicts).

### What a fix has to decide
- **What becomes of a contact that no colour 0..62 can take.** The options:
  - leave it −1 for the count-averaged fallback (the stated intent). That makes defect 3 reachable
    from defaults in exactly these regions, because that fallback is per-body and non-conserving;
  - widen the palette (more mask words). A body of degree D needs at least D colours, so this means
    D sequential phases;
  - handle high-degree bodies separately: a serial or atomic pass for their contacts, or a
    different graph for the position solve, such as pairs instead of probe points for SDF shapes.
- **Whether the fix covers all five colourings.** Either one shared kernel or five changes.
- **What happens to the MPI vote.** It is already in place (`allMaxAny`) and would start firing.
- **The gate.** `kGate` / `kFollowupGate`: same-colour pairs = 0 at every np and thread count, and
  `hub` dP at the OMP 1 level on CUDA.

---

## 2. The legacy friction pass applies a couple δ n × J_t per contact

### Mechanism
The legacy friction pass applies the tangential impulse +J_t at A's surface point p_A and −J_t at
B's surface point p_B. The narrow phase places p_B = p_A − dist·n, where dist is the signed gap:
negative for an overlap, positive for a speculative contact inside the 0.1 R_max margin. So the
two arms differ by dist·n, and every friction impulse also applies a couple

  ΔL = (p_A − p_B) × J_t = dist · n × J_t.

This is not round-off. It is first order in the contact's gap or overlap and does not vanish as
dt → 0 at a fixed gap.

The manifold path (the normal solve and the PGS friction cone) moves both arms to the common
midpoint (p_A − dist n/2 = p_B + dist n/2), so it has no couple.

### Code
`src/narrowphase.hpp:278-281`: the two arms.
```cpp
          const F3 pSurfA = sub3(pWorld, scale3(nWorld, pointRadius));
          const F3 rA = sub3(pSurfA, posA);
          const F3 rB = sub3(sub3(pSurfA, scale3(nWorld, effDist)), posB);
```
`src/solver_friction.hpp:165, 183, 205-217`: the legacy pass applies ±J_t with `rA × t` and
`rB × t`.
```cpp
        const F3 rA{c.rA.x, c.rA.y, c.rA.z}, rB{c.rB.x, c.rB.y, c.rB.z};
        ...
        const F3 rnA = cross3v(rA, t), rnB = cross3v(rB, t);
        ...
        Kokkos::atomic_add(&deltaAngVel(realA, 0), rnA.x * invIA.x * lt);
        ...
          Kokkos::atomic_add(&deltaAngVel(realB, 0), -rnB.x * invIB.x * lt);
```
`src/contact_preprocessing.hpp:434-437`: the manifold path uses the common midpoint.
```cpp
  const F4 shift{c.normal.x * c.dist * 0.5f, c.normal.y * c.dist * 0.5f, c.normal.z * c.dist * 0.5f,
                 0.0f};
  const F4 rA_mid{c.rA.x - shift.x, c.rA.y - shift.y, c.rA.z - shift.z, 0.0f};
  const F4 rB_mid{c.rB.x + shift.x, c.rB.y + shift.y, c.rB.z + shift.z, 0.0f};
```
`src/solve_driver.hpp:259, 277, 806-815`: which path runs.
- `friction = frictionDynamic > 0 || wallFrictionMax > 0`.
- `legacyFriction = friction && !(g != 0 && velocityUseGS)`.
- So with g = 0 and any friction, the legacy pass runs once per substep after the velocity loop.
- With g ≠ 0 and Gauss–Seidel, the PGS cone runs instead.

### Repro
```bash
OMP_NUM_THREADS=1 build_ct/tests/kokkos_mpi/test_momentum_mpi friction_pair --delta=0.05 [--dt=..]
OMP_NUM_THREADS=1 build_ct/tests/kokkos_mpi/test_momentum_mpi friction_pair_pgs --delta=0.05
docs/contact_evidence/run_friction.sh build_ct
```
**`friction_pair`.** Two unit spheres approach along x at v_n = 0.2 and slide along y at v_t = 1,
with μ = 0.4, e = 0.5 and one step. They are placed to overlap by δ at the predicted positions
x + v dt, where the narrow phase and the velocity phase work. The test measures the velocity
phase's angular impulse ΔL about the origin, with lever arms at those positions, and compares it
with dist · n × J_t, where J_t is the tangential part of m_A Δv_A.

### Numbers
**One sliding pair.** The measured couple equals the prediction to 4 digits at every overlap and
dt, so it is linear in δ. On the PGS cone path it is round-off.

| δ / R | dt | \|J_t\| | legacy \|ΔL\| | predicted \|dist n × J_t\| | ratio | PGS cone \|ΔL\| |
|---|---|---|---|---|---|---|
| 0.001 | 0.01 | 0.0380 | 1.708e-5 | 1.710e-5 | 0.999 | 8.6e-9 |
| 0.01 | 0.01 | 0.0380 | 1.881e-4 | 1.880e-4 | 1.000 | 3.7e-8 |
| 0.05 | 0.01 | 0.0379 | 9.467e-4 | 9.467e-4 | 1.000 | 2.3e-8 |
| 0.1 | 0.01 | 0.0379 | 1.893e-3 | 1.893e-3 | 1.000 | 7.3e-9 |
| 0.2 | 0.01 | 0.0378 | 3.775e-3 | 3.775e-3 | 1.000 | 3.8e-8 |
| 0.05 | 0.02 / 0.005 / 0.0025 | 0.036–0.039 | 8.90e-4 / 9.74e-4 / 9.87e-4 | same | 1.000 | – |

At fixed δ the per-contact couple does not depend on dt. J_t is set by μ times the normal impulse,
which does not scale with dt here.

**The `cluster_friction` floor.** This is dLvel at np 1, OMP 1, over the same physical time
(0.5). It shows no trend with dt (8× range) or with position iterations (10×), because the gaps do
not shrink:

| position iterations | dt 0.02 | 0.01 | 0.005 | 0.0025 | mean \|dist\| / R of friction contacts | gap fraction |
|---|---|---|---|---|---|---|
| 20 | 3.6e-5 | 1.9e-5 | 4.4e-5 | 3.6e-5 | 0.033 → 0.021 | 0.66 → 0.79 |
| 5 | 4.8e-5 | 3.0e-5 | 3.4e-5 | 3.7e-5 | 0.033 → 0.022 | 0.64 → 0.79 |
| 2 | 4.3e-5 | 2.4e-5 | 3.5e-5 | 3.0e-5 | 0.034 → 0.023 | 0.62 → 0.79 |

(The position phase's residual overlap `ovl` grows from about 0.02 at 20 iterations to 0.18 at 2
iterations, in length units with R = 0.5, without moving dLvel.)

62–79 % of the friction-active body–body contacts are **speculative**: dist > 0, inside the
broad-phase margin 0.1 R_max, and approaching, so λ_n > 0. Their arm mismatch is the gap, bounded
by the margin and not by dt or by the position solve. The floor is therefore set by the margin.
The PGS cone path (`cluster_pgs`) gives dLvel 1.9e-8 (dt 0.01) and 5.6e-8 (dt 0.0025).

### Does production reach it?
Not from defaults: body–body friction defaults to 0. It is reached by every **g = 0 run with
friction and velocity iterations**:
- growth packings with friction;
- HCS with friction;
- the RingBed packing (`configs/packing_hollow_example.json`: friction 0.02, g = 0, 26 iterations).

With g ≠ 0 (drums, silos, settling beds) the PGS cone runs, and that path is clean. Without
velocity iterations, body–body friction has no normal load (λ_n accumulates in the velocity loop)
and does nothing.

### What a fix has to decide
- **The application point.** The midpoint, as the manifold path already uses, gives ΔL = 0 but
  changes np = 1 results of every g = 0 friction run, the RingBed packing among them. The
  alternative is to retire the legacy pass and run the manifold/PGS cone at g = 0 as well, a
  larger behaviour change.
- **Friction on speculative (dist > 0) contacts.** Whether it should act at all is a modelling
  question, raised by the gap fraction above. It is separate from the couple.
- **The gate.** `kFollowupGate`: \|ΔL\| ≤ 1e-6 \|prediction\| on `friction_pair`. Then the
  `cluster_friction` dLvel threshold of 1e-4 (today set to the floor) can drop to the frictionless
  1e-6.

---

## 3. Count-averaged Jacobi scales the two halves of a contact differently

### Mechanism
In the count-averaged solves each body sums its contacts' corrections and then scales the sum by
a factor from **its own** contact count:
- velocity: min(1, 2/count);
- position: 1/count.

The two bodies of one contact generally have different counts, so the equal and opposite halves
±J are scaled differently. For example, with A in 3 contacts and B in 1, A receives −(2/3)J and B
receives +J, so momentum grows by J/3. In the position phase the mismatch moves the centre of
mass.

This is the serial model, and the same at every np. Since WO-3b the counts are the global ones, so
np ≥ 2 reproduces np = 1 (`jacobi.txt`):

| np | dP | dX / R | dXpos / R | dLvel |
|---|---|---|---|---|
| 1 | 8.96e-3 | 2.14e-2 | 8.19e-4 | 3.16e-3 |
| 2 | 9.02e-3 | 2.13e-2 | 8.20e-4 | 3.16e-3 |
| 4 | 9.02e-3 | 2.12e-2 | 8.21e-4 | 3.16e-3 |

(`cluster_jacobi`, OMP 1. The GS modes are at 1e-9.)

### Code
`src/solver_velocity.hpp:244`, `applyVelocityDeltasAveragedKokkos`:
```cpp
        const float f = Kokkos::fmin(1.0f, 2.0f / static_cast<float>(count));
        for (int c = 0; c < 3; ++c) {
          velPred(i, c) += deltaVel(i, c) * f;
          angVelPred(i, c) += deltaAngVel(i, c) * f;
```
`src/integration.hpp:180-185`, `applyUpdatesKokkos`:
```cpp
        const int count = constraintCounts(i);
        if (count <= 0)
          return;
        const float f = 1.0f / static_cast<float>(count);
        detail::st3(posPred, i, add3(ldF3(posPred, i), scale3(ldF3(deltaPos, i), f)));
        detail::st3(velPred, i, add3(ldF3(velPred, i), scale3(ldF3(deltaVel, i), f)));
```

### Every place a per-body factor is applied
| # | where | trigger | reachable from defaults? |
|---|---|---|---|
| 1 | `solve_driver.hpp:520-525`: velocity Jacobi (`solveVelocityKokkos` → `applyVelocityDeltasAveragedKokkos`) | `velocityUseGS = false`, i.e. `diagnostics.set_velocity_solver('jacobi')` (`dem_bindings.cpp:161`), with velocity iterations > 0 | **No.** Diagnostic A/B only. |
| 2 | `solve_driver.hpp:933-937`: position Jacobi (`solvePositionKokkos` → `applyUpdatesKokkos`) | the same switch (`velocityUseGS` also selects the position solver) | **No.** |
| 3 | `solve_driver.hpp:496-507`: GS velocity fallback | `velLeftover > 0` | **No. Unreachable:** defect 1, `leftover` is always 0. |
| 4 | `solve_driver.hpp:917-927`: GS position fallback | `posLeftover > 0` | **No. Unreachable,** likewise. |

**The friction count-averaging is not per body.** `solver_friction.hpp:196` scales one contact's
impulse by `1/max(count_A, count_B)`, a single factor for both halves, so it conserves linear
momentum. Its only non-conservation is the couple of defect 2.

**The `'jacobi'` diagnostic** is row 1 plus row 2: one switch, both solves.

### What a fix has to decide
- **Whether the averaging factor should be per contact**, symmetric in the two bodies (for example
  the friction pass's max of the two counts), or stay per body. The register holds
  "**The velocity solve uses the over-relaxed `min(1, 2/count)` average**, not a raw Jacobi sum"
  (dem CLAUDE.md, settled decisions). A symmetric variant keeps the over-relaxation, but it
  changes the legacy A/B's numbers and needs a recorded decision.
- **The coupling with defect 1.** The fallbacks (rows 3 and 4) become live the moment the colour
  overflow is fixed by leaving contacts uncoloured. Fixing defect 1 that way without fixing this
  one trades a race for a deterministic momentum leak in the same crushed or high-degree regions.
- **The gate.** `momentum_cluster_jacobi` today holds the np = 1 drift (dP ≤ 1.5e-2). After a fix
  it would drop to round-off.

---

## 4. Pairs that no rank sees

`test_ownership_mpi` runs the distributed step's own contact detection on every rank:
- the halo gather at the XPBD band 2.1 R_max;
- the broad and narrow phase;
- no solve.

It runs the same detection on the whole set on `MPI_COMM_SELF`, and counts the serial active pairs
(dist ≤ 0) that are **not visible on any rank**. No ownership rule can solve such a pair; both
bodies pass through each other. Owners are fixed per particle, as with `rebalance_every = 0`.
16³ box, R = 0.5, scales 1 ± 0.1.

### 4a. Drift out of the owner's block (`rebalance_every = 0`)

**Mechanism.** A rank holds its own particles, wherever they are, plus the other ranks' particles
within `band` of its **block** (block-based, not particle-based). The band equals the contact reach
2.1 R_max, with no slack. `verlet_skin` > 0 widens the band by the skin, but its default is 0.

A pair (A owned by r, B owned by s ≠ r) is visible on r iff B is within `band` of block r, and on
s iff A is within `band` of block s. It is lost iff **both** hold:
- dist(B, block r) > band;
- dist(A, block s) > band.

This has two consequences:
- **Both bodies must have left their own blocks.** If B were still inside block s, then
  dist(A, block s) ≤ |AB| ≤ band.
- **Three blocks are needed.** The pair must sit near a block other than the two owners'. So a
  two-rank slab decomposition of a closed box never loses a pair: every pair within reach has one
  body within band of the other's slab.

In the symmetric geometry (touching bodies at F ∓ d/2 across the r|s face, both a distance y past
the blocks' common edge), both distances are √(d²/4 + y²). So the pair is lost once

  **y > √(band² − d²/4)**
- = 1.857 R_max for touching equal grains (d = 2R, band = 2.1 R_max);
- up to 2.1 R_max for small grains (d ≪ band).

Other geometries follow from the two inequalities. The triangle inequality gives only a weak
general lower bound: each body lies at least band − |AB| (≥ 0.1 R_max) outside its own block.

**Numbers.** `missed_drift_pair`: two grains, owners r and s, face at 8, both moved y past 8 on the
second axis (`missed.txt`). The loss appears between y = 1.85 R and 1.86 R, against the predicted
1.857 R:

| np | layout | y/R: 0 | 1.0 | 1.5 | 1.8 | 1.85 | 1.86 | 1.9 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1, 2 | slab (no third block) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 2 × 2 × 1 | 0 | 0 | 0 | 0 | 0 | **1** | 1 | 1 | 1 |
| 8 | 2 × 2 × 2 | 0 | 0 | 0 | 0 | 0 | **1** | 1 | 1 | 1 |

(At y = 1.85 R: dist = 1.0468 < band 1.05. At y = 1.86 R: 1.0512 > 1.05.)

`missed_drift_lattice`: the closed jittered lattice, all particles advected by D along x with the
owners fixed. Pairs lost out of about 5000 serial pairs:

| D / R | 0.5–3.0 | 4.0 | 6.0 |
|---|---|---|---|
| np 1, 2 | 0 | 0 | 0 |
| np 4 | 0 | **10** | **13** |
| np 8 | 0 | **21** | **33** |

A missed pair at np 4, D = 4 R: gid 105 (owner 0) at (9.52, 7.49, 0.51) and gid 119 (owner 1) at
(9.49, 8.50, 0.47).
- Owner 0's block is [0,8)×[0,8), owner 1's is [0,8)×[8,16).
- Both grains have been advected into x > 8, the blocks of ranks 2 and 3.
- dist(a, block 1) = 1.60 and dist(b, block 0) = 1.57, both above the band 1.155.
- Lattice grains start at x = i + 0.5, so the grains nearest the face (x = 7.5) are lost once
  7.5 + D > 8 + √(band² − d²/4) ≈ 9.04 (d ≈ 1, band 1.155), i.e. D > 3.1 R. This fits 0 lost at
  D = 3 R and losses at 4 R.

**Reached by production?**
- **`step_mpi` alone, defaults: yes.** `enable_mpi_step(rebalance_every=0)` is the default
  (`dem_bindings.cpp:787`), and nothing else migrates unless the driver calls `rebalance()` or
  `migrate_to_weights`. Any flowing bed (silo discharge, a drum
  without `rebalance_every`, a sedimenting suspension) accumulates the drift without bound. Pairs
  are lost once grains of two owners meet more than about 1.86 R_max past their blocks near a
  third block. The MPI drum test sets `rebalance_every=15`, which bounds the drift to 15 steps of
  motion.
- **Coupling: bounded.** `CfdDem.step` calls `migrate_to_weights` before every fluid step
  (`coupling/python/peclet_coupling/driver.py:731-736`). The drift is `dem_substeps` substeps of
  motion. The driver's comment asserts that this stays below a ghost band; the code does not
  check it.
- §1.2 of the design note measured about 1.5 R in 50 steps for the momentum cluster, just below
  the threshold.

**What a fix has to decide.**
- **The mechanism**, one of:
  - per-step (or drift-triggered) migration of particles that left their block, which design-note
    R4 recommends;
  - a band slack equal to the maximum drift since the last migration (Allreduced, like the Verlet
    skin; cheaper in messages, larger halos);
  - particle-based rather than block-based ghost selection.
- **Whether a detection guard is wanted in the meantime**: the global maximum of the distance
  outside the own block against √(band² − R_max²), a warning or an exception.
- **The gate.** `kMissedGate` in `test_ownership_mpi.cpp`: 0 missed pairs in
  `missed_drift_pair` and `missed_drift_lattice`.

### 4b. Periodic wrap across an undecomposed axis (strong jitter)

**Mechanism.** Core's cross-rank halo sends each particle to each other rank **at most once**, at
the single periodic image nearest that rank's block, axis by axis
(`ParticleMigrator::withinRcutOfBlock`, `core/include/peclet/core/halo/particle_migrator.hpp:163-193`).

On a periodic axis that the ORB leaves **undecomposed**, the destination block spans the whole
box. The unshifted image then always wins, because its gap on that axis is 0. The wrapped image
is never sent, even when it is within the band.

A pair that crosses a rank face **and** wraps that undecomposed periodic axis therefore needs a
second image of the partner that neither owner receives, so the pair is invisible on both.

The local self-ghosts (`includePeriodicSelf`) do enumerate every image
(`imagesWithinRcutOfBlock`, `particle_migrator.hpp:202-243`). That is why np = 1 is correct: only
the cross-rank path is affected.

`core/include/peclet/core/halo/particle_halo_topology.hpp:62-67`:
```cpp
    for (std::size_t i = 0; i < pos.size(); ++i) {
      for (int r = 0; r < nranks; ++r) {
        if (r == me)
          continue;
        if (!mig_->withinRcutOfBlock(pos[i], r, rcut, img))
          continue;
```
`particle_migrator.hpp:179-190`: per axis, keep only the smallest-gap candidate of {x, x − L, x + L}.

**The cause is the image enumeration, not the band width.** For every missed pair, the test asks
core which images of the partner lie within the band of each owner's block. In all 5 of 5 missed
pairs, and on both sides of each, the needed image is **inside** the band ("2 images qualify"),
and the one sent is the unshifted image. The band is wide enough; the second image is dropped.

**Numbers.** `missed_periodic`: the lattice with jitter 0.3 in a box periodic on every axis, 5754
serial pairs (`missed.txt`).

| np | layout | undecomposed periodic axes | missed |
|---|---|---|---|
| 1 | 1 × 1 × 1 | x, y, z (self-ghosts, all images) | 0 |
| 2 | 2 × 1 × 1 | y, z | **2** |
| 4 | 2 × 2 × 1 | z | **3** |
| 8 | 2 × 2 × 2 | none (block 8 ≥ 2 band) | 0 |

The pairs (all from `missed.txt`):
- np 2: gid 151 (owner 0, (7.57, 9.76, 0.21)) and gid 3992 (owner 1, (8.41, 9.74, 15.70)).
  |sep| = 0.978, wrapping z.
  - Rank 0 receives gid 3992 at image (0, 0, 0) and needs (0, 0, −1).
  - Rank 1 receives gid 151 at (0, 0, 0) and needs (0, 0, +1).
- np 2: gid 3592 / 3831, wrapping y.
- np 4: gid 136 / 3960 and 151 / 3992, wrapping z.
- np 4: gid 240 / 3840, wrapping y (a decomposed axis) **and** z (undecomposed).
  - Rank 1 receives the y-wrapped image (0, +1, 0) and needs (0, +1, −1). Only the z component is
    wrong.

AFTER.md calls these "corner-wrap pairs". The measurement is more general: **any** pair that
crosses a rank face while wrapping an undecomposed periodic axis, at an edge of the box and not
only at a corner. With the weak jitter (0.04, `exactly_once_periodic`) no contact spans the
diagonal needed, so the committed gate passes.

**Reached by production? Yes, whenever a periodic axis is not decomposed.**
- On a cube that means np = 2 and 4 of the ORB; np 3, 5, 6, 7 depend on the split sequence.
- Slab decompositions of periodic beds are affected in general, for example a tall column split
  only along its axis with periodic sides.
- In a dense bed the affected pairs lie along the lines where a rank face meets a periodic face,
  so the count scales with the face area × band. Here that is 2–3 pairs out of 5754.
- A lost pair interpenetrates for as long as the geometry holds.

The companion assumption R8 (a decomposed periodic axis needs a block extent ≥ 2 · band, so that
one image per particle and rank suffices) is a different failure of the same one-image rule, and
it is also unchecked.

**What a fix has to decide.**
- **Where the fix lives.**
  - **Core:** send every qualifying image per (particle, rank), as the self-ghost path already
    does. This is a core release. It changes ghost counts and the per-rank slot layout, and dem's
    `ContactOwnership` (`copyRanks`) and the ghost-to-owner reverse would have to be checked
    with two copies of one gid on one rank (not examined here).
  - **dem-side:** a supplementary exchange.
  - **Decomposition:** always decompose every periodic axis, and enforce P1.
- **Whether R8 / P1 is fixed by the same change**, since multiple images per rank also cover a
  block narrower than 2 · band.
- **The gate.** `kMissedGate`: `missed_periodic` 0 at np 2 and 4. Afterwards the strong jitter
  could move into `exactly_once_periodic`.

---

## Open questions (none blocks this note)
- **Defect 1, SDF shapes.** The ring bed was stopped at φ = 0.37 (step 1500 of 6000; the host
  growth run slows to about 5 min per 250 steps). The jammed state (φ 0.55) was not reached. The
  contact-point count grows with φ (79 → 613 per particle), so the conclusion does not depend on
  it.
- **Defect 1, the symptom in ring beds.** The same-colour pairs were counted, but their effect on
  a ring bed's centre of mass or overlap was not measured. The hub scene shows the size of the
  effect for spheres.
- **Defect 2, speculative contacts.** 62–79 % of the friction-active contacts are speculative
  (dist > 0). Whether legacy friction should act on them is a modelling question, not part of the
  couple.
- **Performance.** Not measured here; nothing was changed.
