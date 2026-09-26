# Architect brief 3: three physics-consistency follow-ups of the contact-solve framework

Written 2026-09-26. The framework is `docs/contact_solve_framework.md`: §1–§11, the session decisions
§12 S1–S24, and the §13 amendment, WO-12. It is DONE and on dem main `0520b21`; `AFTER.md` holds
the evidence. This brief does **not** reopen the framework.

## 1. The questions (one design note, three parts)

- **Q-A (S17): the multilevel coarse cycle.** It conserves linear momentum, but not angular
  momentum. Design the rigid-body-aggregate coarse cycle (6-DOF coarse bodies), or show that a
  cheaper construction is exact.
- **Q-B (`computeW` / the position phase is translation-only).** The overlap projection's effective
  mass includes rotational terms, but only translation is applied. Decide which is right:
  - make the phase consistent translation-only;
  - apply the rotation;
  - something else.

  Say what that does to (i) the fixed point, (ii) the friction cone's normal load, and (iii) the
  `ring_mini` stall, which new evidence below says is probably NOT caused by `computeW`.
- **Q-C (R-U1): the PGS restitution target.** `0` for pre-separating contacts creates kinetic
  energy at np 1. Give the contact-law options, each with its physics, its cost and what it breaks,
  plus a recommendation. **The user chooses the contact law**, so this part ends in options, not a
  decision.

For each part, return a design (or options), work orders, and verification gates. It must be
executable by an Opus implementer that is forbidden to decide anything the note leaves open.

## 2. Why it needs the architect

- **Q-A** is a new discretization of the coarse space: a Galerkin-style restriction and
  prolongation with rotational DOFs, under mass splitting and MPI.
- **Q-B** would reverse a settled register entry, "Position solve is translation-only;
  overlap removal stays decoupled from velocity" (2026-07-10, "rejected: none stated"). Reversing
  one takes a new recorded decision.
- **Q-C** is a contact-law change, which moves every `g ≠ 0` result.

## 3. Current state (excerpts)

### 3.1 Q-A: `src/solver_multilevel.hpp`
Multilevel is stabilization mode 2, which is **opt-in**. The default is mode 1, `onesided`
(`src/particles.hpp:227`). The coarse sweep, `MlCoarseSweep::solveOne` (`:433-481`), is a PGS
normal update of an aggregate manifold. It uses **translation only**, and its effective mass is
|N|²(1/M_gA + 1/M_gB):

```cpp
const int gA = grp(realIdx(m.bodyA));  const int gB = (m.bodyB >= 0) ? grp(realIdx(m.bodyB)) : -1;
const F3 Nsum{m.normal_sum...};  const F3 rAavg = rA_sum/num_points, rBavg = rB_sum/num_points;
const F3 diffCenters = (gB < 0) ? rAavg : sub3(rAavg, rBavg);
vA = velG(off+gA); vB = gB>=0 ? velG(off+gB) : wallVel_sum/num_points;
const float vn = dot3(sub3(vA, vB), Nsum);
const float sgn = (dot3(Nsum, diffCenters) > 0) ? 1 : -1;
const float w = dot3(Nsum, Nsum) * (invMassG(off+gA) + invMassG(off+gB));
const float dp = sgn*vn / w;            // e = 0: target 0 (pure inelastic support)
pNew = max(0, lambdaAcc(idx) + dp); d = pNew - pOld; lambdaAcc(idx) = pNew;
const F3 J = Nsum * (-sgn*d);
velG(off+gA) += J*invMA;  velG(off+gB) -= J*invMB;
```

The prolongation (`:832-838`) gives every member its aggregate's ΔV, which is the
mass-proportional impulse:

```cpp
velPred(i) += velG(off + grp(i)) - velG0(off + grp(i));
```

- J therefore acts at the aggregates' centres of mass, not at the contact point:
  ΔL = (X_gA − X_gB) × J.
- Measured: `hub_ml` (np 1, 181 bodies, 10 steps) gives dP 1.9e-7, dLvel **3.0e-3**, overlap
  3.3e-5.
- The aggregate groups are built per level from the contact graph. Under MPI the coarse mass is
  a·m/k (S-decision "coarse mass"), since the members are mass-split copies. The coarse vertex's
  inverse mass is `invMassCoarse` (`particles.hpp:417`).
- `lambdaAcc` is a per-coarse-manifold accumulator, and a Poisson release in flight skips the
  coarse transport (`restRel`).
- The stop criterion is `maxApproach`.

### 3.2 Q-B: `src/solver_position.hpp`

```cpp
KOKKOS_INLINE_FUNCTION float computeW(F3 r, F3 dir, float invM, F3 invI) {
  const F3 rn = cross3v(r, dir);
  return invM + rn.x*rn.x*invI.x + rn.y*rn.y*invI.y + rn.z*rn.z*invI.z;   // body-frame components
}
```

`PositionContactSweep` (`:575-665`), the production coloured GS overlap projection after WO-12:

```cpp
C = dot3((pA + rA) - (pB + rB), n);      // rA, rB, n delta-rotated by the PREDICTED orientation
if (C >= 0 && lam <= 0) return;
const float wTotal = computeW(rA,n,invMassA,invIA) + computeW(rB,n,invMassB,invIB);   // :625
lamNew = fmax(0, lam - omega*C/wTotal);  dLambda = lamNew - lam;   // omega_pos = 1.5
posLambdaAcc(idx) = lamNew;  atomic_max(posResidual, |dLambda|*wTotal);  // stop: < 1e-4 R
// Translation-only correction, in place (rotation discarded to match applyUpdatesKokkos).
posPred(idA) += n*dLambda*invMassA;   posPred(idB) -= n*dLambda*invMassB;
```

- `posLambdaAcc` is the contact's net position impulse. The caller converts it to impulse units
  and carries it into the next substep's **Coulomb bound**, so the friction cone sees the total
  normal load. Changing w̃ changes that number.
- The Jacobi twin `solvePositionKokkos` computes a world-frame dθ (S15) and scatters `deltaQuat`,
  but `applyUpdatesKokkos` commits `deltaPos` only.
- For spheres, rA ∥ n, so the rotational term is 0: **spheres are unaffected by every Q-B option.**
- Mass splitting divides w̃ per copy (k·w, S-decisions); the rank-level M consensus averages
  positions. Any rotation applied would need an orientation consensus too.

**New evidence (2026-09-26, this session): the `ring_mini` stall is NOT a convergence rate.**
- The scene: 27 hollow cylinders, outer diameter 1, height 1.5, wall 0.18, spacing **0.9** (less
  than the diameter), random orientations. Many start interpenetrated.
- `ovl` is the largest linearized violation in the last position sweep. At np 1 and 10 steps:

| position iterations (stops off) | 20 | 200 | 2000 |
|---|---|---|---|
| ovl | 0.290 | 0.325 | 0.275 |

- A wrong diagonal (w̃ > A_ii) only under-relaxes GS; it cannot change a fixed point. So either:
  - the translation-only LCP is **infeasible** for interlocked or wall-crossing ring pairs (e.g.
    opposing normals within one body pair); or
  - the linearization, frozen within the substep, never closes the true overlap.

  Please state which, and whether applying rotation would make the scene feasible at all.
- A test scene that starts un-interpenetrated may be the right gate instead.

### 3.3 Q-C: `src/solver_velocity.hpp:1191-1202` (`PGSManifoldSweep`, the production `g ≠ 0` path)

```cpp
const float v0til = sgn * vn0(idx);          // PRE-solve approach, measured before warm start
if (fabs(vn0(idx)) < restVelThreshold*lenN) restitution = 0;
const float target = (v0til > 0) ? -restitution*v0til : 0;   // 0 for pre-separating contacts
float dp = (vtil - target) / wTotal;  pNew = max(0, pOld + dp);   // accumulated, clamp >= 0
```

- There are more channels on the same accumulator: Poisson release, the Walton tangential target
  `−β·vt0` and the event peak `vPeak`. See `:1240-1340` and `:1395-1401`. Poisson mode
  deliberately keeps per-substep Newton restitution alive; its comment records that forcing e = 0 on
  persistent contacts measured worse.
- The `g = 0` path is a different law: the event-form one-shot `(1+e)·approach` while approaching,
  exclusive holding across ranks (policy X). It is energy-correct, so **Q-C is PGS only.**

Measured (framework Appendix A.2: throwaway Python model, validated against the kernels to 4
digits). N = 343 jittered 7³ lattice, radii ±10 %, 536 overlapping contacts, Gaussian velocities, one
converged substep, KE₀ = 527.2:

| e | serial one-shot | dem-PGS np 1 | Moreau-PGS np 1 (target −e·γ⁻ on every closed contact) |
|---|---|---|---|
| 0.0 | 341.36 | 341.27 | 341.27 |
| 0.5 | 368.6–369.2 | 405.9 | 387.7 |
| 0.9 | 475.9–476.6 | **560.9** (+6.4 % over KE₀) | 491.9 |
| 1.0 | 527.2 | **614.1** (+16 %) | 527.2 |

Mechanism, as far as known: a contact separating before the solve (target 0) that is driven into
approach by its neighbours' impulses is held at zero relative velocity. Its neighbours' restitution
targets were computed from pre-solve approaches that assumed it would not push back. Moreau's
simultaneous law was suggested as a follow-up (framework §10.1, R-U1), "gated g ≠ 0 first".

## 4. Constraints and invariants

- **The framework invariant** is v = v0 + M⁻¹Jᵀλ with one owned λ per contact, and a single
  application point per contact. Mass splitting (m/k, I/k) with mass-weighted consensus applies to
  PGS and the overlap projection; exclusive holding (policy X) applies to the `g = 0` one-shot.
  Complete colouring (64 colours, hub copies above 32 edges).
- **Conservation gates** must keep holding: dP ≤ 1e-8 (8e-7 under gravity), and dLvel ≤ ~4e-8 on
  everything except the multilevel scenes, which are REPORTED today (S17). That is np 1/2/4/8,
  OMP 1/8, and CUDA.
- **np 4/8 bitwise run-to-run at 1 thread.** The np ≥ 2 results equal np 1 to the float floor.
- **Spheres must stay byte-identical** under Q-B unless a named change says otherwise. S15's
  isotropy guard is the pattern.
- **Everything runs on device** (Kokkos: CUDA/HIP/OpenMP) and is MPI-distributable. Host paths are
  oracles only.
- **Single precision** state (float). Tolerances are stated at the float floor (S1).
- **Performance.** We are already +9–19 % over the pre-framework baseline; see `AFTER.md` §7. A
  design must not add a pass per iteration to the default path (`onesided` stabilization, PGS).
  Q-A's cost matters only in the opt-in multilevel mode.

## 5. Already decided — not open

- The whole framework (§12 S1–S24, §13 / WO-12: Λ' = max(0, Λ − ωC/w̃), ω_pos = 1.5, stop at
  1e-4 R).
- The Almgren/Moreau-style *position* retractable projection is settled.
- The **velocity/position split**: the velocity solve owns dissipation; the position solve only
  removes overlap and **never writes velocity**. This part is NOT under question in Q-B. Only
  whether the position phase may rotate is.
- `g = 0` uses the event one-shot plus policy X (settled, energy-exact).
- Multilevel stays opt-in.
- **Open:** Q-A in full; Q-B's choice (and whether the register entry is reversed); Q-C's options
  (user decides).

## 6. Tried and rejected, with evidence

- **Mass-splitting the `g = 0` one-shot** compounds restitution: −14 % KE at np 8, e = 0.9.
  Rejected; use X.
- **Folding `g = 0` into dem-PGS** creates energy at np 1 (the Q-C table). It is rejected for
  `g = 0` regardless of Q-C.
- **ω_pos = 1.5 on a non-retractable projection** left permanent gaps; superseded by WO-12.
- **For `ring_mini`, more position iterations do not help** (the table in §3.2).
- **ω 1.7** is worse on hubs; 1.5 is kept.

## 7. Verification that exists

- ctests in `tests/kokkos_mpi/`: `test_momentum_mpi` modes (`cluster*`, `hub*`, `hub_ml`,
  `cluster_multilevel`, `ring_mini`, `tri`, `tri_pgs`, `periodic`, `shear*`, …) at np 1/2/4/8;
  `test_ownership_mpi` oracles; `position_agreement_np{2,4,8}` (1e-4 R); 7 G13 mutants
  (WILL_FAIL). Battery: 264 host, 59 CUDA subset.
- Test binary options: `--omega`, `--pos-iters`, `--no-stop`, `--perf-g`, `--dump`. Output lines:
  KE per step, MOMENTUM (dP dX dL dLvel ovl), ITERS, CONFLICTS.
- Physics gates elsewhere: `tests/python` (Enskog cooling slope, restitution / Walton rebound,
  drum, silo), Dosta benchmark (`memory: dem-dosta-benchmark`: 25k rebound 0.170 against 0.30).
  Q-C must say which of these it moves and what it expects.

## 8. Deliverable

A design note at **`docs/contact_physics_followups.md`** (dem worktree
`/home/frankp/Codes/suite/dem-contacts`, branch `contacts`), with these sections:

1. The Q-A design: the coarse DOFs, restriction and prolongation, the effective mass, MPI and
   mass-split treatment, cost, and how exact ΔL = 0 is proved.
2. The Q-B analysis and decision (or a recommendation, if it turns on the user), including the
   diagnosis of the `ring_mini` stall and a replacement gate scene.
3. The Q-C options table plus a recommendation (for the user).
4. Work orders, each with files, gates and the bitwise or named-change statement.
5. Register entries to add or supersede.

No production code. Measure where you need to: the build trees `build_ct` (host OpenMP) and
`build_ct_cuda` exist. Bound threads: `OMP_NUM_THREADS=2 OMP_PROC_BIND=false`. For MPI ctests
use `--bind-to none`. Cores 8–23 are pinned by another session.

## 9. Out of scope

- Performance of the default path (being worked in parallel, in-session).
- The coupling tests.
- The `g = 0` one-shot law.
- Reopening mass splitting or X.
- Poisson banking internals, beyond saying whether Q-C options interact with them.
