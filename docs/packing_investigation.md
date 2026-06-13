# Packing investigation — why periodic sphere packing doesn't reach random close packing

Systematic investigation of why periodic monodisperse sphere packing fails to reach the expected random
close packing density (φ ≈ 0.64, coordination Z ≈ 6). Reference benchmark: frictionless monodisperse RCP
φ ≈ 0.64, isostatic Z ≈ 6, residual overlap ≪ 1 %.

Artifacts: `pack_meter.py` (trustworthy meter), `phase0_sphere_repro.py` (reproduction sweep).

## Phase 0 — trustworthy measurement + reproduction

### Finding 0.1 — the shipped meter is wrong
`verify_packing_spheres.py` computes `phi = N·(4/3)π·radius³ / domain³` with `radius` = the *base* 0.5 and
a domain sized for `phi_ref = 0.6`, **ignoring the grown per-particle `scale`**. It therefore reports the
design φ (0.6) by construction, regardless of the actual packing. `pack_meter.py` replaces it: effective
radii `r_i = base·scale_i·global_scale`, true periodic box volume, overlap-corrected solid fraction,
coordination number Z, rattler count, max overlap, and g(r), all with a periodic KD-tree.

### Finding 0.2 — the protocol never searches for jamming
The growth model grows every particle to a *preset* target size (`growth_factor` 0.05 → 1.0) that fills the
box to the chosen design φ, then stops (`growth_rate` is zeroed at completion). There is no gravity and no
continued drive, so nothing compacts the particles toward jamming. Reproduction (N=2000, frictionless,
periodic, single-phase growth + settle):

| design φ | φ_corrected | max overlap | Z | rattlers |
|---|---|---|---|---|
| 0.55 | 0.550 | 3.7e-4 | 0.00 | 2000 |
| 0.60 | 0.600 | 9.8e-5 | 0.00 | 2000 |
| 0.62 | 0.620 | 9.1e-5 | 0.00 | 2000 |
| 0.64 | 0.640 | **9.5e-2** | 3.42 | 210 |
| 0.66 | 0.658 | **3.6e-1** | 5.60 | 20 |
| 0.68 | 0.675 | **3.8e-1** | 6.41 | 6 |

Below ≈0.62 the particles form **zero contacts** (a gapped, contact-free arrangement that merely fills the
box); at/above 0.64 the box is too small and they **interpenetrate grossly** (9–38 %). No design φ produces
the signature of a real jammed packing (Z ≈ 6 with overlap ≪ 1 %).

### Finding 0.3 — the engine does not resolve contacts (primary defect)
A *gentle* approach (slow growth `rate=0.3`, 100 solver iterations, 4000 settle steps) at φ=0.64, run to
**full rest** (KE ≈ 4e-8):

```
φ_corrected = 0.640   max_overlap = 8.8 %   Z = 2.08   rattlers = 997 / 2000
```

8.8 % interpenetration **and** 50 % rattlers **at zero kinetic energy** is physically impossible for a real
packing — deeply overlapped pairs stay stuck while other particles never engage, and neither more
iterations nor more settling time fixes it. The XPBD **position solver is not driving contacts to
resolution**. This is an engine-level defect, and it is the primary cause: even a correct jamming-search
protocol cannot produce a valid packing on top of a solver that leaves ~9 % overlaps unresolved.

## Conclusions and next steps

Two distinct problems, the second dominant:
1. **Protocol:** grows to a preset φ instead of searching for the jamming density (needs an adaptive
   growth / overlap-criterion search, as in the RingBed reference protocol).
2. **Engine (primary):** the position solver leaves large persistent overlaps with simultaneous rattlers
   even at full rest — contacts are not being resolved.

Planned next (Phase 1): (a) localize the engine defect by diffing/comparing against the pinned older engine
`RingBed-CFD-Surrogate/extern/packing-gpu@324e4fa` the user reports worked; (b) read the XPBD position
solver (`solver_position.cu`, `xpbd_solver.cu`, `contact_preprocessing.cu`, periodic min-image in the
narrow phase) to find why overlaps are not resolved (Jacobi relaxation / mass weighting / manifold or
periodic-image bug). Then fix the engine, validate a clean RCP (Z ≈ 6, overlap ≪ 1 %, isotropic g(r)),
and only then layer the adaptive jamming-search protocol and extend to rings.

## Phase 1 — localize the engine defect

Artifact: `phase1_protocol_spheres.py` (the RingBed annealing protocol ported to spheres).

### Finding 1.1 — not a regression
The solver translation units (`solver_position.cu`, `solver_velocity.cu`, `xpbd_solver.cu`,
`narrowphase.cu`, `contact_preprocessing.cu`) are **byte-identical** to the pinned older engine
`extern/packing-gpu@324e4fa` the user reports worked; only the MPI halo code differs. The older engine
reproduces the same broken sphere result (φ=0.64 gentle → 13 % overlap, Z=2.1, 990 rattlers). The defect
is intrinsic and long-standing, not introduced by recent edits.

### Finding 1.2 — the pipeline is velocity-solve + position post-stabilization (docs were stale)
The shipped `docs/solver_details.md` is out of date. The real `step()` runs: predict (gravity) → ghosts →
broad/narrow phase → **Phase A velocity solve** (restitution / friction / growth-separation) → predict
position from the resolved velocity → **Phase B position solve** (Jacobi post-stabilization to remove
penetration) → commit (`d_pos = d_pos_pred`, velocity NOT re-derived from the position change). So the
contact *dynamics* live in the velocity solve, which does nothing at rest with no relative velocity — the
scheme is designed around **agitation** (the thermostat) + elastic collisions during growth, then cooling.
Phase 0's naive test (no thermostat, restitution 0) therefore froze; that was a missing-protocol artifact,
not proof of an engine bug.

### Finding 1.3 — the overlap query defeats the protocol (the engine defect)
Running the full RingBed annealing protocol (MB seed, elastic + thermostat during adaptive growth, then
dissipative cooling) the engine grows **unchecked** to a collapsed state — φ_protocol=0.72, but
φ_corrected=0.465, max_overlap=90 %, Z=5.7 — because **`get_max_overlap()` reads ≈0 throughout growth**, so
the overlap-criterion feedback never backs the growth off. Direct comparison on a φ=0.68 grown state:

```
engine get_max_overlap() = 0.035   <- the protocol's control signal (under-reports ~10-30x)
engine compute_overlaps() = 0.95   <- a CORRECT signal exists, sees the deep overlap
meter: max_overlap 0.41,  phi_corr 0.674,  Z 6.17,  rattlers 4/1000
```

`get_max_overlap()` returns the **position solver's mid-iteration residual** (`solver_position.cu:240`,
zeroed every iteration), not the committed overlap — it under-reports by ~10-30×. `compute_overlaps()`
re-runs detection and reports the real value. The packing at φ=0.68 actually has a healthy network
(Z≈6, 4 rattlers) but a few **pathologically deep stuck overlaps** the position solver never resolves, and
the control signal is blind to them.

### Phase 1 conclusion
The engine defect is twofold and actionable:
1. **`get_max_overlap()` is not a valid overlap measure** (position-solver residual, under-reports ~10-30×).
   Any protocol that controls growth off it steers blind → unchecked inflation → collapse. A correct
   signal (`compute_overlaps()`) already exists.
2. **A few contacts are left deeply unresolved** (stuck deep overlaps) even when the bulk network is fine —
   to be traced to the once-per-step stale contact list, the periodic min-image in the per-contact `C`, or
   Jacobi under-relaxation in `solver_position.cu`.

Planned Phase 2: (a) fix the overlap measurement so the criterion is meaningful (report the committed
detection-time overlap; have the protocol use it), (b) trace and fix the stuck deep-overlap pairs, then
(c) re-run the adaptive protocol and validate a clean RCP (Z≈6, overlap ≪ 1 %, isotropic g(r)).

## Phase 2 — the engine is sound; the fix is a correct signal + a tuned protocol

Artifact: `phase2_protocol_fixed.py` (the annealing protocol driven by a correct Python overlap signal).

### Finding 2.1 — the position solver is correct
Two spheres at 50 % overlap snap to exactly touching (`d=1.0000`) in a single step. The solver resolves
isolated overlaps exactly; it does not "leave overlaps unresolved" as Phase 0 suggested.

### Finding 2.2 — both engine overlap *queries* are wrong; the solve is fine
Ground truth (brute-force min-image, confirmed by `pack_meter`) on a dense φ=0.68 state: true max overlap
**0.389**. The engine reports `get_max_overlap()` = **0.035** (under, position-solver residual on the stale
once-per-step contact list) and `compute_overlaps()` = **0.963** (over, a periodic-ghost artifact). Periodic
vs walls packings are otherwise identical (φ_corr 0.658, Z 4.6 both), so the bad ghost overlaps corrupt the
*measurement*, not the *solve*. The deep "stuck overlaps" of Phase 0/1 were **over-jamming defects created
when the protocol overshoots past RCP** — driven by the under-reporting control signal — not a solver bug.

### Finding 2.3 — with a correct signal + tuned protocol the engine reaches a genuine RCP
Driving the annealing protocol (MB seed, elastic + thermostat growth, adaptive growth keyed to a
brute-force-correct Python overlap, dissipative cooling + a final quench) with **gentle growth** (so the
`gf³` overlap rise near jamming is slow enough for the feedback to engage) yields, for periodic monodisperse
spheres:

```
phi = 0.632 (~RCP)   max_overlap 0.3%   KE 2e-6 (settled)   g(r) contact peak r/D = 0.992
coordination Z vs contact gap tolerance (RCP contacts are ~touching, so a compression-only test undercounts):
  gap +0.000: Z = 5.34       gap +0.002: Z = 6.03 (isostatic)      gap +0.005: Z = 6.28, 0 rattlers
```

φ ≈ 0.64, isostatic Z ≈ 6, contact-touching g(r), ~0 rattlers — a **textbook random close packing**.

### Phase 2 conclusion
The packing engine is fundamentally **sound**. "Periodic packing doesn't reach max density" was caused by
(1) the broken meter (reported the design φ), (2) the broken control signal (`get_max_overlap` under-reports
→ the adaptive growth overshoots past RCP into a collapsed/over-jammed state), and (3) growth too fast for
the feedback to engage near jamming — not a defect in the XPBD solver. With a correct overlap signal and a
gentle-growth + quench protocol the engine produces a genuine RCP.

### Finding 2.4 — engine fix: `compute_overlaps()` now correct (ghost regeneration)

`compute_overlaps()` over-reported because it copied the committed positions into the `*_pred` buffers for
the REAL particles only and never regenerated the periodic ghosts — so the narrow phase compared committed
real particles against **stale ghost positions** from the previous step. Fixed by regenerating the ghosts
from the committed state (`update_ghosts()`) before detection (`src/simulation.cpp`). It now matches the
brute-force ground truth exactly (0.458 vs 0.458, was 0.96) and does not corrupt the sim state (stepping
continues normally afterwards; the periodicity regression test passes).

Driving the annealing protocol with the **native fixed** `compute_overlaps()` (no Python crutch) reaches a
genuine RCP: φ=0.635, Z=6.27 at the contact gap (0 rattlers), g(r) contact peak r/D=0.990. So the engine
now exposes a correct overlap signal natively.

`get_max_overlap()` is left as-is: it is the position solver's last-iteration residual (a solve-convergence
metric), NOT the committed overlap — `compute_overlaps()` is the query protocols should use for the
overlap criterion. (Making `get_max_overlap()` report the committed value would require a per-step
re-detection; not worth perturbing the validated step() pipeline.)

Remaining work: package the tuned protocol as a reusable entry point (replacing the broken
`verify_packing_spheres.py`), run the friction sweep (φ(μ): 0.64 → 0.55), and re-validate the ring
(hollow-cylinder) packing against the RingBed reference.

## Phase 3 — packaging, friction, rings

### Finding 3.1 — reusable protocol (`pack.py`)
`pack.py` packages the validated annealing protocol (`pack_spheres` + `analyze_spheres`), driven by the
fixed native `compute_overlaps()`. Default frictionless run: φ=0.633, Z=6.01 at the contact gap, g(r) peak
r/D=0.991 — PASS. This replaces the broken `verify_packing_spheres.py`.

### Finding 3.2 — engine bug fixed: friction velocity solve produced NaN -> crash
With friction > 0 at dense jamming the run crashed (CUDA illegal access in the BVH build): a single
position blew up to ~1e8 while velocities stayed ~0 — a runaway position-solver correction. Cause: the
friction velocity solve divided by the tangential generalized mass `w_total_t` **without a guard** (the
position solver guards `w_total < 1e-6`; the friction path did not), so a degenerate tangential DOF (or a
huge normal impulse from a deep overlap) produced an enormous angular impulse -> garbage quaternion ->
garbage lever arm -> the runaway position correction -> NaN/Inf positions -> illegal access in the BVH.
Fixed in `solver_velocity.cu` (guard `w_total_t`, clamp `max_f` to the non-negative normal impulse).
Frictional jamming now completes.

### Finding 3.3 — friction is otherwise non-functional (deeper, separate defect)
Direct tests (two spheres, sliding and glancing collisions, `restitution_tangent` 0 and 1) show **no
tangential damping and no induced spin** for any friction coefficient — the friction impulse is computed
but has no effect on the dynamics. So the φ(μ) jamming sweep cannot be demonstrated: the annealing
protocol reaches the same RCP (φ≈0.63, Z≈6) for µ = 0, 0.3, 0.5. Two things are needed and are a scoped
follow-up: (a) repair the friction impulse in the velocity solve (the computed tangential impulse does not
move the bodies), and (b) note that even when repaired this is a *collisional* (velocity-impulse) friction
— it acts only on contacts with a normal approach velocity, so a *quasi-static* packing (persistent
contacts, ~0 normal velocity) needs a position/force-based persistent friction to show the random-loose
trend. The frictionless RCP (the main result) is unaffected.

### Finding 3.4 — rings (hollow cylinders) validated
`pack.py::pack_rings` runs the same annealing protocol with orientations and the SDF-based native overlap
(`compute_overlaps()`, which the fix makes correct for SDF shapes too). Squat-thick rings (aspect 1.0, wall
thickness ratio 0.30) on the fixed engine, with the achieved density tracking the overlap criterion:

```
criterion 2e-3 -> phi = 0.281 (clean)    criterion 8e-3 -> phi = 0.318    criterion 2e-2 -> phi = 0.346
```

At a comparable overlap tolerance (criterion 8e-3) the packing reaches φ = 0.318, squarely in the RingBed
reference range (0.30-0.33) — and rings pack far looser than spheres (0.64), as expected. The density is
now *controllable* via a trustworthy criterion (the reference's 0.30-0.33 was obtained with the broken
under-reporting signal, so it carried unmeasured interpenetration; the fixed signal gives an honest,
tunable density). No crash, no friction needed (frictionless).

## Phase 4 — friction model analysis (how it works, why it is inert, how to fix it carefully)

### How contact response is modelled
Each step runs two solves over the once-per-step **manifolds** (a manifold = the group of contact points
between two bodies A,B, reduced to AGGREGATE geometric sums `normal_sum`, `torque_armA/B_sum`, ...):

* **Velocity solve** (`solver_velocity.cu`) — manifold-aggregate impulses for **restitution** and
  **collisional friction**:
  * normal relative velocity `vn_agg = (vA-vB)·N_sum (+ growth)`; **approaching ⇔ vn_agg > 0**;
  * **normal restitution**: target `vn_target = -e_n · vn_agg`, normal impulse `λ = (vn_target - vn_agg)/w`
    `= -(1+e_n)·vn_agg / w`  → **λ < 0 for an approaching contact**;
  * **tangential restitution + Coulomb friction**: at the center-of-pressure, tangential rel. velocity
    `vt`; target `dv_t = (e_t - 1)·|vt|` (a *tangential-restitution* target: `e_t=0` ⇒ stop sliding,
    `e_t=1` ⇒ no tangential change), impulse `λ_t = dv_t/w_t`, **Coulomb-clamped** to `|λ_t| ≤ μ·(normal
    impulse)`.
* **Position solve** (`solver_position.cu`) — post-stabilization that removes penetration. For a *resting*
  contact the actual normal contact FORCE lives here as the constraint's Lagrange multiplier `Δλ_n`, NOT in
  the velocity solve.

### Why friction is inert — two distinct issues
1. **Coulomb-limit sign (and a regression I introduced).** The normal impulse `λ` is **negative** for
   approaching contacts. The original `max_f = μ·λ` is therefore negative and the clamp is malformed. My
   Phase-3 NaN-guard changed it to `μ·max(λ,0)`, which is **0 for every approaching contact** — i.e. it
   *disabled* collisional friction entirely (that is why even oblique collisions now show no tangential
   damping / no spin). The correct Coulomb limit is **`μ·|λ|`**.
2. **No static/persistent friction (fundamental).** Even with `μ·|λ|`, the velocity-solve normal impulse
   `λ ≈ 0` for a *resting* contact (`vn_agg ≈ 0` — there is no approach velocity to cancel). So `μ·|λ| ≈ 0`
   ⇒ no friction. The velocity-solver friction is **collisional only**; it acts during impacts, never on
   the persistent contacts of a packing or a pile. The normal force that should bound friction in a resting
   contact is the *position* solver's `Δλ_n`, which the friction code never sees.

### Restitution is separate and must be preserved
Normal restitution (`e_n`, the normal impulse) and tangential restitution (`e_t`, the `dv_t` target) are
velocity-solver mechanisms, orthogonal to the Coulomb clamp, and matter for collisional regimes (fluidized
beds). The friction repair must change only the Coulomb *limit* and add a new persistent-friction
mechanism — it must not alter the restitution targets or the normal/tangential impulses themselves.

### Proposed fix — staged and careful
* **Fix A (small, safe): collisional friction.** Set the Coulomb limit to `μ·|λ|` (replacing the
  `max(λ,0)` regression). Restores impact friction (oblique-collision tangential damping, backspin) with
  zero effect on restitution. Validate with a clean oblique-bounce / backspin test.
* **Fix B (the real one for packing): positional (static) friction in the position solver.** The standard
  XPBD dynamic-friction step (Müller et al. 2020, *Detailed Rigid Body Simulation with XPBD*): after the
  normal position constraint yields `Δλ_n`, apply a tangential positional correction limited by `μ·Δλ_n`
  that removes the tangential relative *displacement* accumulated since the contact formed (anchor = the
  start-of-step contact point, since contacts are detected once per step). This bounds friction by the
  actual resting contact force, so it works for persistent contacts (packing, piles), and is fully
  decoupled from the velocity-solver restitution → fluidized-bed behaviour is untouched.

### Risks / things to be careful about
* **Manifold aggregation:** friction acts at the center-of-pressure (`rA_cp/rB_cp` already computed);
  positional friction must use the same aggregate point, and per-point vs aggregate matters for multi-point
  manifolds (rings).
* **Anchors:** positional friction needs a per-contact tangential anchor and the accumulated `Δλ_n`; with
  once-per-step detection the anchor is the contact's start-of-step position.
* **Static vs dynamic μ:** the API exposes only `friction_dynamic`; using it as the static limit is the
  common approximation.
* **Growth term:** the growth separation velocity is folded into `vn_agg`/`v_rel_point`; positional
  friction must not double-count it.

Recommended order: Fix A first (one line, restores+validates collisional friction and undoes my
regression), then Fix B (the substantive change that makes frictional *packing* — φ(μ) — possible).

### Implementation + results
**Fix A (done, `solver_velocity.cu`):** Coulomb limit `μ·|λ|`. Validated — an oblique collision now damps
the tangential velocity (2.79→1.97) and induces spin (0→4.1); **normal restitution is exactly preserved**
(`e_n` 0/0.5/1.0 → rebound restitution 0/0.5/1.0); frictionless is unchanged.

**Fix B (done, `solver_position.cu`):** XPBD positional friction — a Coulomb-limited (`μ·|dLambda|`)
tangential position correction that removes the tangential relative displacement of the contact points
since the contact formed (anchor = `d_pos` + the stored anchor lever arms), with the same generalized-mass
weighting and sign convention as the normal constraint, gated on `friction>0`. Validated:
* **frictionless RCP byte-identical** (φ=0.639, Z=6.5 — the friction block is skipped at μ=0);
* **gravity sedimentation φ(μ) shows the random-loose trend**: μ = 0.0/0.2/0.4/0.6 → φ = 0.654/0.638/0.619/
  0.619 (saturating), i.e. friction now demonstrably loosens the packing — the goal.

**Known limitation (a real Fix C):** the engine never feeds the position correction back into velocity
(it commits `v = v_pred` from the velocity solve only — see Finding 1.2). So positional friction holds a
*bulk* packing (the collective network + dissipation reach a quasi-static state) but cannot arrest a
*single* particle under sustained load: on a 20° incline a frictional sphere still slides (μ above the
friction angle does not hold it; friction only slows it, 4.30→3.07). Completing friction for sustained
single-contact static cases (piles holding an angle of repose, an isolated grain on a slope) needs a
velocity-coupling step — derive a velocity contribution from the friction position correction (Δv = Δx_t/dt)
without disturbing the Phase-A restitution. That is a more invasive, restitution-sensitive change and is
left as a scoped follow-up; the packing goal (φ(μ)) is met without it.

## Summary

"Periodic packing doesn't reach max random packing" was **not a fundamental engine defect**. Root causes,
all now resolved: (1) the verify meter reported the design φ by construction; (2) the protocol's control
signal — the engine's `compute_overlaps()` — over-reported via stale periodic ghosts, so the adaptive
growth steered blind and overshot into collapse; (3) the growth schedule was too fast for the feedback.
With the meter replaced, `compute_overlaps()` fixed (ghost regeneration), and a gentle-growth annealing
protocol (`pack.py`), the engine produces a genuine random close packing of spheres (φ≈0.63, isostatic
Z≈6, g(r) contact peak) and a correct, tunable ring packing (φ≈0.28-0.35). Along the way a friction-path
NaN crash was fixed; the deeper friction-model repair (the impulse is inert; quasi-static persistent
friction) is a scoped follow-up.
