# Contact physics follow-ups: rigid multilevel aggregates, the position-phase diagonal, the PGS restitution law

Design note, 2026-09-26. Brief: `docs/contact_evidence/ARCHITECT_BRIEF_3.md`. Base: dem main `0520b21`
(branch `contacts`). The framework `docs/contact_solve_framework.md` (§1–§13, S1–S24, WO-12) is
settled and is not reopened here. Evidence numbers quoted below come from runs of `build_ct`
(host OpenMP, OMP 2, cores 24–31) and from throwaway Python models. Appendix A describes the
models precisely enough to rebuild them.

## 0. Decisions at a glance

| # | Question | Decision | Changes results of |
|---|---|---|---|
| **A** | S17: multilevel angular momentum | Coarse bodies become **rigid 6-DOF aggregates**. The coarse state is the mass/inertia projection of the fine state onto rigid motions (the current *projection form*, extended). Coarse impulses act at the contact point. Prolongation is the rigid motion, spins included. ΔL = 0 exactly (to float). No cheaper construction is exact and sound (§1.7). | opt-in `multilevel` runs only |
| **B** | `computeW` / translation-only position phase | **Keep the phase translation-only** (the register entry stands), and make it **consistent**: the diagonal becomes the translational effective mass `w̃ = invM_A + invM_B` (solve views). Neither the fixed point nor ω_pos = 1.5 changes. | non-spherical bodies and spinning spheres (finite-iteration position results) |
| **B-diag** | The `ring_mini` stall | It is **not a solver defect**. The scene starts *tunnelled*: 85 of 106 contacting pairs have a linearised overlap problem with **no solution**. With linearised rotation up to 3 rad, 84 still have none. Rotation cannot rescue it. | — |
| **F1** | Found while answering B | Analytic `hollow_cylinder` and `box` shapes use their *radius* as the broad-phase and band radius, not their circumscribed radius. Their end and corner contacts are **never detected**: coaxial tubes overlapping by 0.3 give 0 contacts. **Fix: WO-B0.** | every tube and box run |
| **F2** | Found while answering B | The narrow phase tests only the lower-index body's shell against the other body's SDF. A penetration of 0.0235 (a quarter of the wall) goes unseen while 0.001 is reported. **Open: user decision R-B2.** | (if adopted) every non-sphere run |
| **B-gate** | Replacement gate | `ring_collide`: 27 tubes that start overlap-free (rejection-sampled), after WO-B0. Gates: convergence, np agreement at 1e-4 R, conservation. `ring_mini` stays as a conservation-only scene. | tests only |
| **C** | R-U1: the PGS restitution target | **Options for the user (§3).** Recommendation: **Moreau's law**, target `−e·u⁻` on every closed contact. It is energy-consistent by theorem, costs nothing, and differs from today only on kinetically pre-separating contacts. Until the user chooses: the default stays Newton, and WO-C1 adds Moreau as a diagnostics A/B and gathers the numbers. | nothing until the user chooses |

## 1. Constraints and invariants (binding on every work order)

- **Framework invariant.** Velocity phase: v = v0 + M⁻¹Jᵀλ, with one owned λ per contact and one
  application point. Mass splitting: m/k and I/k per copy, mean reconciliation. Policy X applies to
  the g = 0 one-shot. Complete colouring. None of this changes.
- **Row convention of a velocity manifold** (`contact_preprocessing.hpp:474-500`,
  `solver_velocity.hpp:1155-1170`): `N = normal_sum`, `TauA = Σ rA_mid × n`, `TauB = −Σ rB_mid × n`.
  - The approach velocity is `vn = vA·N + ωA·TauA − vB·N + ωB·TauB`.
  - The impulse `λ' = −sgn·d` gives `ΔvA = N λ' invM_A`, `ΔωA = R I_A⁻¹ Rᵀ (TauA λ')`,
    `ΔvB = −N λ' invM_B`, `ΔωB = R I_B⁻¹ Rᵀ (TauB λ')`.
  - `rA_mid` and `rB_mid` meet at one point (the midpoint), which makes the fine ΔL exactly 0.
- **Quaternions** are (x, y, z, w), active rotation (`dem_portable.hpp:63`). World inertia is
  `R diag(1/invI) Rᵀ`.
- **Precision.** State is float. Gates sit at the float floor (S1). The new coarse geometry
  (centre of mass, inertia) is accumulated in **double** (§2.3).
- **Conservation gates.**
  - dP ≤ 1e-6 (5e-6 under gravity).
  - dLvel ≤ 1e-6 on every mode. After WO-A2 this **includes** `hub_ml` and `cluster_multilevel`.
  - Configurations: np 1/2/4/8, OMP 1/8, and CUDA (np 1 fused + np 2).
- **Reproducibility.**
  - np 4/8 are bitwise run-to-run at OMP 1.
  - np 1 stays byte-identical for everything not named as changed in the work order. Proof by the
    G3 dumps (`--dump`, OMP 1).
- **Device and MPI.** Everything runs on device and must be MPI-distributable. Host code is an
  oracle only.
- **Performance.**
  - Nothing is added to the default path (onesided + PGS, spheres).
  - Q-A costs only in the opt-in multilevel mode.
  - WO-B1 removes work.
  - WO-B0 adds broad-phase work only for tube and box runs.

## 2. Q-A: rigid-body aggregates in the multilevel coarse cycle

### 2.1 Why the current cycle violates L

`MlCoarseSweep::solveOne` (`solver_multilevel.hpp:433-481`) changes an aggregate's translation only.
Prolongation (`:832-838`) adds the same ΔV to every member. A crossing impulse ±J therefore acts at
the aggregates' centres of mass, and ΔL = (X_gA − X_gB) × J. Measured today: `hub_ml` np 1 dLvel
2.98e-3 (dP 1.9e-7), `cluster_multilevel` 1.4e-8. Linear momentum is exact.

### 2.2 The coarse space and why this form

- **Coarse DOF per aggregate g of level ℓ:** (V_g, Ω_g).
- **Prolongation P (rigid motion, spins included):** member q gets
  `δv_q = δV + δΩ × d_q` and `δω_q = δΩ`, where d_q is its offset from the group centre of mass.
- **Coarse mass matrix** `PᵀMP = diag(M_g 𝟙, I_g)`. The cross terms vanish because d is measured
  from the centre of mass. Here
  `I_g = Σ_q [ J_q + μ_q (|d_q|² 𝟙 − d_q d_qᵀ) ]`, where J_q is the member's world-frame spin
  inertia and μ_q its coarse mass (§2.5).
- **Coarse row of a crossing manifold:** `J_c P`. With `Nrow = +N` for A and `−N` for B, the coarse
  angular arm is `T = Tau + d × Nrow`:

      T_A = TauA + d_A × N          T_B = TauB − d_B × N

  Here d_A (d_B) is body A's (B's) offset from its own group's centre of mass.
- **Evaluation: the projection form, kept from today.** The coarse state at the start of a level is
  the M-orthogonal projection of the fine state onto rigid motions:

      V_g = Σ μ v / M_g        Ω_g = I_g⁻¹ Σ_q [ μ_q d_q × v_q + J_q ω_q ]

  The coarse PGS runs on (V, Ω). Afterwards the delta (V − V0, Ω − Ω0) is prolonged. Translation-only
  (today) is the limit I_g⁻¹ → 0 of this construction, so the cycle's semantics are unchanged apart
  from rotation.

**Why the projection form and not the Galerkin/residual form** (evaluate `J_c(v + Pδ)`):
- The projection form is KE-non-increasing at every coarse step. The fine KE change of a
  prolonged δ is `u_cᵀ(PᵀMP)δ + ½δᵀ(PᵀMP)δ`, with `u_c = (PᵀMP)⁻¹PᵀMv` the restricted state. That
  is exactly the coarse KE change, and a target-0 projected coordinate step never raises the
  coarse KE.
- The residual form uses Δλ = −J_c v/w but the KE change is `Δλ·J_cΠv + ½wΔλ²`. It **creates
  energy** whenever the aggregate's rigid approach is less than half the contact's own approach
  (J_cΠv > ½J_c v, approach negative), i.e. on deforming aggregates.
- It would also change the tuned semantics of the opt-in mode: the slip gate at 8 g dt was
  calibrated on silo discharge.

**Why spins are included.**
- The inertia always has a well-defined inverse: I_g ≥ Σ J_q > 0.
- Its conditioning is bounded by about 1 + 2.5 (R_g/r)² for spheres.
- For frictional aggregates, co-rotation is the no-slip rigid mode.
- Excluding sphere spins is the main alternative (§2.7). It makes every level-1 pair of spheres
  singular.
- Named consequence: frictionless sphere spins are coupled to aggregate rotation in multilevel mode.
  Energy never increases and L is exact (open item R-A1).

### 2.3 Build: the per-level geometry (once per hierarchy build)

In `buildContactHierarchyKokkos`, per level, after the existing group-mass pass. `massG` and
`invMassG` stay **exactly** today's float arithmetic.

1. **Origin.** `O_g = fl( Σ μ_q x_q / Σ μ_q )`, sums in double (`Kokkos::atomic_add` on double).
   - x_q = `posPred(q)`, and μ_q = `effMass(invMassCoarse(q))` (today's function).
2. **Centre offset.** `e_q = fl(x_q − O_g)` (float). Then `c_g = fl( Σ μ_q e_q / Σ μ_q )`, double
   accumulation, stored as float.
   - The member offset used **everywhere**, recomputed on the fly and never stored, is
     `d_q := fl(e_q − c_g)`. This makes Σ μ_q d_q vanish to second order, so dP does not grow with
     |x|/R_g.
3. **Spin inertia.** `J_q = R_q diag(1/max(invIc_q.x, 1e-30), …) R_qᵀ`.
   - R_q comes from `P.quat(q)`, the same array the fine PGS uses.
   - `invIc_q` is §2.5's coarse inverse inertia.
   - Isotropic invIc (spheres): take `diag` directly, with no rotation (the S15 guard pattern).
4. **Group inertia.** `I_g = Σ_q [ J_q + μ_q (|d_q|² 𝟙 − d_q d_qᵀ) ]`, 6 components, double atomics.
5. **Inverse.** `invI_g = I_g⁻¹` via the double adjugate and determinant. Store 6 floats
   (xx, yy, zz, xy, xz, yz).
   - A non-positive or non-finite determinant is a logic error: throw in debug; in release set
     invI_g = 0, which makes that group translation-only. **Precision fallback** in §6 G-A1.

New per-group pools, allocated **lazily** on the first multilevel pass with the same growth rule
as `mlInvMassG`:
- `mlOriginG`, `mlComOffG`, `mlAngG`, `mlAngG0` (3 floats each), and `mlInvIG` (6 floats);
- per slot: `invInertiaCoarse` (3 floats).

The default path allocates nothing.

**Aggregation never crosses a periodic wrap.** The matching's contend/commit (`:171-219`) also
requires, for both endpoints, `posPred(slot) == posPred(realIdx(slot))` bitwise.
- An image slot has a shifted position, and its aggregate would have no consistent geometry.
- Crossing manifolds may still be wraps. Their arms use the canonical positions
  `posPred(realIdx(·))`.
- Named change: periodic runs in multilevel mode.

### 2.4 The cycle (per level, per stabilization iteration), with the colouring unchanged

**Restrict**, in `multilevelCoarseCycleKokkos` and in the fused device cycle:
- V_g as today.
- New: accumulate `L_g = Σ_q [ μ_q (d_q × v_q) + J_q ω_q ]` (3 float atomics per member).
- Then per group `Ω_g = invI_g · L_g`, and snapshot `Ω0_g = Ω_g`.

**Coarse update** (`MlCoarseSweep::solveOne`; keep today's statements **in order** and append the
rotational terms). `T_A`, `T_B` as in §2.2, with `d_b = fl(fl(posPred(realIdx(body_b)) − O_{g_b}) − c_{g_b})`:

    vnLin = dot(sub(V_A, V_B), N)                 // today's expression (wall: V_B = wall velocity)
    vn    = vnLin + dot(T_A, Ω_A) + dot(T_B, Ω_B)  // wall: no B term
    wLin  = dot(N, N) * (invM_A + invM_B)          // today's expression
    w     = wLin + T_A·(invI_A T_A) + T_B·(invI_B T_B)
    dp = sgn*vn / w;  pNew = max(0, lambdaAcc + dp);  d = pNew − pOld;  λ' = −sgn*d
    V_A += N λ' invM_A;   Ω_A += invI_A (T_A λ')
    V_B −= N λ' invM_B;   Ω_B += invI_B (T_B λ')   // (wall: nothing)
    atomic_max(<the sweep's residual view, as today>, |d| w / lenN)

The following are unchanged: `sgn`, the `restRel` skip, the `lenN` guard, and λ on the shared
`lambdaAcc`.

**Prolong**, per member q with g = grp(q):

    v_q += (V_g − V0_g) + (Ω_g − Ω0_g) × d_q
    ω_q += (Ω_g − Ω0_g)

**The coarse colouring already makes groups disjoint within a colour,** so Ω needs no atomics
(exactly as V). The sequence fine sweep → fold → coarse cycle → re-seed is unchanged.
`reseedCopiesKokkos` already re-seeds `angVelPred`.

### 2.5 Mass splitting and MPI

The coarse vertex q (§13.2) carries `μ_q = a m/k`. Its spin inertia is the same fraction:
`(a/k) I_q`. Build it exactly as `buildInvMassCoarseKokkos` does
(`solve_copies.hpp:699`), with the ratio first:

    ratio = float(k(q)) / float(max(1, a(q)));   invIc(q) = invInertia(q) * ratio   // per component

- At np 1 without copies, `invIc = P.invInertia` bitwise.
- The sleep swap touches only `invMass` (`step_solve.hpp:255`), so the coarse inertia source is
  `P.invInertia`, as in the fine PGS.
- Sleepers never aggregate (`excludeImmovable`). As singleton crossing endpoints their coarse row
  equals the fine row (d = 0 ⇒ T = Tau).

Under MPI, the hierarchy stays rank-local over owned + ghost bases. The existing syncs reconcile
v and ω (reverse, then forward, mean for k > 1).

### 2.6 Proof that ΔL = 0 (and ΔP = 0) exactly

1. **One coarse update is a contact-point impulse pair.**
   - Aggregate A receives linear impulse Nλ' and angular impulse T_Aλ' about X_A = O_A + c_A.
     Aggregate B receives −Nλ' and T_Bλ' about X_B.
   - The total angular impulse about the origin is
     `X_A×Nλ' + (TauA + d_A×N)λ' − X_B×Nλ' + (TauB − d_B×N)λ' = [p_A×N + TauA − p_B×N + TauB]λ'`.
   - This is the fine manifold's own angular impulse,
     `Σ_k ((p_A + rA_mid,k) − (p_B + rB_mid,k)) × n_k λ'`, which is 0 because the midpoints meet.
   - Walls are external: they exchange with the world.
2. **The prolongation delivers exactly the coarse impulses.**
   - `Σ_q μ_q δv_q = M_g δV + δΩ × Σ μ_q d_q = M_g δV`.
   - `Σ_q [μ_q x_q × δv_q + J_q δω_q] = X_g × M_g δV + I_g δΩ`, using Σ μ d = 0 and the
     definition of I_g.
   - The coarse state moved by exactly δV = ΣNλ'/M_g and δΩ = I_g⁻¹ΣTλ'.
3. **The split system maps to the true system exactly** (§1.3 P1).
   - The re-seed writes δ to all a local copies.
   - The fold and the rank reconciliation map split-system P and L to true P and L, because
     copies share x and orientation: `Σ_copies (m/k)x×v + (I/k)ω = m x×v̄ + I ω̄`.
   - Hence ΔP_total and ΔL_total equal the wall impulses.
4. **KE never increases in a coarse step** (the §2.2 identity).
5. **Float residue.**
   - Σμd ≠ 0 is second order after the c_g recentring.
   - Storing invI_g as float leaves ~ε·cond(I_g) on the angular part.
   - Measured by G-A1, with a pre-decided fallback.

A singleton group reproduces the fine body's row (d = 0 ⇒ T = Tau, invI_g = world I⁻¹), to float
rounding.

### 2.7 Rejected cheaper constructions

- **Coarse impulse redirected along the line of centres (X_A − X_B).** ΔL = 0 trivially, but:
  - the impulse is no longer along the constraint normal;
  - the ledger λ stops being the manifold's normal impulse;
  - it injects tangential impulse into frictionless contacts;
  - the operator is non-symmetric, so the KE monotonicity is lost.
- **Torque into member spins only** (δv uniform, δω = (ΣJ)⁻¹τ). Exact, but the rotational inertia
  becomes ~(r/R_g)² of the true one. That gives huge spurious spins, stored permanently in
  frictionless spheres.
- **Orbital-only rigid motion (no spins).** Exact for spheres in exact arithmetic, but:
  - every collinear aggregate (all level-1 sphere pairs, columns) has a singular I_g;
  - a pseudo-inverse threshold η drops axial torques of relative size ~√η, which is not the float
    floor;
  - non-sphere spins enter the normal rows, so they must be in the space anyway.
  - It is kept as the documented alternative for R-A1.
- **Residual/Galerkin evaluation.** Can create energy (§2.2); rejected.
- **Skipping coarse updates with torque.** Disables nearly all transport.

### 2.8 Cost (opt-in multilevel only)

- **Per coarse update:** about 42 floats loaded instead of 8. That is Ω, invI, O, c for two groups
  and two positions, plus about 100 flops (two crosses, two symmetric matvecs).
- **Per member, level and cycle:** +3 atomics in restrict, one world-inertia product (two
  quaternion rotations; skipped for isotropic bodies), and +3 writes and a cross in prolong.
- **Build:** two extra passes with double atomics, and a 3×3 inverse per group.
- **Memory:** +18 floats per pool slot, allocated lazily.
- **Expectation, to measure (G-A6):** the coarse cycle takes 1.5–2.5× today's time. The multilevel
  step takes +10–30%.

## 3. Q-B: the position phase, `computeW`, and the ring_mini stall

### 3.1 Facts

- **The fixed point does not depend on w̃.** The accumulated projection
  `Λ' = max(0, Λ − ωC/w̃)` stops only when every contact has Λ > 0 ∧ C = 0, or Λ = 0 ∧ C ≥ 0. Those
  conditions involve positions only, and the applied translation per Λ is `invM n` whatever w̃ is.
  w̃ is a per-row relaxation, `ω_eff = ω·(invM_A+invM_B)/w̃`.
- **Today's rotational term is meaningless, and it is not frame-indifferent.**
  - It refers to a rotation that is never applied.
  - `computeW` (`solver_position.hpp:23`) multiplies a world-frame r×n by **body-frame** invI
    components. For anisotropic bodies (tubes: invI = (3.63, 5.67, 3.63)), rotating the whole scene
    changes w̃, and so the results.
  - It also makes the stop metric |dΛ|·w̃ overstate the actual position change by w̃/(invM_A+invM_B)
    (tube rows: typically ~2.4, up to ~5.6).
- **Spheres.** For a non-spinning sphere the term is below half an ulp of invM at the test
  coordinates, so it is bitwise inert. For a *spinning* sphere it is not: the position phase
  delta-rotates rA and n by the predicted spin (`:596-606`), so |r×n| ~ R·ω dt. The brief's
  "spheres are unaffected by every option" holds only for non-spinning spheres.

### 3.2 ring_mini: the diagnosis

The stall is **infeasibility of the linearised overlap problem**, not a convergence rate.

**Measurement.** Model (Appendix A.1) on the `ring_mini` state after one step: the pair contact set
exactly as dem detects it (lower-index shell against the higher-index SDF, margin 0.05).

| | pairs in contact | LP-infeasible |
|---|---|---|
| translation only | 106 | **85** |
| linearised 6-DOF, \|dθ\|∞ ≤ 0.05 | 106 | **85** |
| linearised 6-DOF, \|dθ\|∞ ≤ 3 rad | 106 | **84** |

Details of the measurement:
- The deepest SDF penetration in the scene is **0.0900 = t/2**. That is the most a thin wall's SDF
  can express.
- Almost every infeasible pair contains exactly opposing normals: min n·n′ = −1.000.
- The max–min slack of each infeasible pair (−0.07…−0.086) is close to its deepest penetration:
  motion barely improves the worst row.

**Mechanism.**
- Spacing 0.9 with random orientations puts tube walls *through* each other at t = 0.
- A shell point past the other wall's mid-surface reads the SDF of the *opposite* face, so its
  normal flips.
- Opposing rows whose lever-arm differences are (nearly) parallel to n form a force- *and*
  torque-balanced Farkas certificate. No translation removes both. No linearised rotation does
  either. Mutually threaded walls cannot be separated by any rigid motion without passing through.
- A converging projection on an infeasible problem trades violation between rows. That is why
  `ovl` sits at 0.27–0.33 for 20/200/2000 iterations, above the largest SDF depth of 0.09.
- A wrong w̃ cannot cause this; the linearisation is not the cause either.

**Answers.**
- Hypothesis 1 (infeasible) holds.
- **Applying rotation does not make the scene feasible.**
- The scene is an initial-condition error for a hard-contact solver.

**Two detection defects make an overlap-free start tunnel as well.** Both were found while testing a
replacement scene.
- **F1, bounding radius.**
  - `initializeShape` (`shape_registry.hpp:39-101`) registers `radius` as the shape's
    `baseRadius`. For a tube the bounding radius is √(R²+(H/2)²) = 0.901; for a box it is √3·R.
    `baseRadiusHost_` is documented as the "per-shape canonical bounding radius" (`:372`), and the
    SDF and scene shapes pass a true bounding radius.
  - The broad-phase box is `rad + margin = 0.55` (`broadphase_arborx.hpp:47`). Coaxial tubes whose
    centres are 1.2/1.4/1.45 apart (overlaps 0.3/0.1/0.05) give **0 contacts** and do not move.
  - In the rejection-sampled scene (spacing 1.5, overlap-free start), dem reports 0 contacts for
    steps 1–6 while the model sees 16–21 pairs within the margin, penetrating by up to
    0.013–0.090. By step 7 the scene is tunnelled: depth 0.0900, 11–15 infeasible pairs.
  - The same scene with a correct bounding radius (tube as an SDF grid with bounding radius 0.9014)
    detects contacts from step 1. Its deepest two-way penetration stays ≤ 0.024 in every step
    checked (1–10, then every third step to 40), with **0 LP-infeasible pairs** (both directions'
    rows, translation only) in each.
- **F2, one-way shell detection.**
  - The broad phase emits i < j (`broadphase_arborx.hpp:90`), and `detectContactsKokkos` tests only
    i's shell against j's SDF (`narrowphase.hpp:241-265`).
  - At step 6 of the corrected scene: `compute_overlaps()` = 1.04e-3, while j's shell sits 0.0235
    inside i (pair 22–23) with i's side reading 0.0000.
  - It did not tunnel within 40 steps there, but it is a visibility gap of the class G5 was built
    to exclude.

### 3.3 The decision (option a: consistent translation-only)

**The position phase stays translation-only** (register entry of 2026-07-10 kept, amended in §5).

**The diagonal becomes the translational effective mass of the solve views:**

    wTotal = invMassA + invMassB           // wall: invMassA; invMass* are the solve views (k·invM)

This applies in `PositionContactSweep` (`:625`) and in the Jacobi twin (`solvePositionKokkos`,
`:127-128`).
- The twin's world-frame dθ and `deltaQuat` scatter (`:135-172`) are dead work, since
  `applyUpdatesKokkos` commits `deltaPos` only. Delete them.
- `computeW` stays for its other caller: the legacy-friction load estimate in `solver_friction.hpp`,
  which is out of scope (R-B7).
- ω_pos stays 1.5. The stop metric `|dΛ|·w̃` becomes the true relative position change, as WO-12
  intended.

**Why this and not rotation (b).**
- (b) rescues nothing: §3.2 shows 84/106 pairs stay infeasible.
- It is not needed for feasibility in non-tunnelled scenes: 0 infeasible pairs over 40 steps.
- It would cost:
  - orientation writes in the sweep;
  - per-iteration lever-arm refresh (C must see the accumulated δθ);
  - an orientation consensus under M, which adds 3 floats to every position sync;
  - folds of rotations for hub copies;
  - a new L defect (rotating I^w at fixed ω).
- Its effect on the three quantities, for the record:
  - (i) A different fixed point: the least-(m, I)-metric correction, unique in (Δx, Δθ). The
    translation shrinks at off-centre contacts.
  - (ii) Converged Λ smaller by invM/(invM + rotational compliance), so less Coulomb capacity.
  - (iii) ring_mini unchanged: 84/106 pairs infeasible.
- It stays the physically better *metric* (least action) and is recorded as R-B4 with a trigger.

**Effect on the three quantities.**
- **(i) The fixed point is unchanged** (for feasible problems) and unique in translations (WO-12).
- **(ii) The friction cone's normal load.**
  - Converged Λ: unchanged.
  - At a finite iteration count, non-sphere rows push typically ~2.4× (up to ~5.6×) more per
    iteration, so Λ is closer to its converged value.
  - Consequence: slightly more Coulomb capacity carried into the next substep for tubes and boxes.
  - On infeasible (tunnelled) pairs, Λ grows with the iteration count, as today but faster (R-B6).
- **(iii) ring_mini is unchanged:** infeasible either way. It keeps its conservation gates; its
  `ovl` is documented as not a convergence metric.

**Convergence rate: measured in the model (A.3), not assumed.**
- Iterations to |dΛ|w̃ < 5e-5, 27 tubes:
  - today (ω 1.5): 19, 9, 9, 5, 1, 4
  - consistent (ω 1.5): 27, 11, 10, 9, 1, 6
- 64 tubes:
  - today: 22, 46, 6, 7
  - consistent: 13, 33, 9, 9
- Consistent with ω 1.0 is fastest on the redundant multi-point units (6, 5, 4, 3 / 18, 42, 2, 3).
- Neither form dominates; the spread is ±50%.

ω_pos is not re-tuned here: it is a settled constant, and the evidence is one small model. R-B3
names the measurement that would reopen it.

### 3.4 Replacement gate scene `ring_collide` (after WO-B0)

- **Scene.** 27 hollow cylinders (D 1, H 1.5, wall 0.18, unit mass).
  - Placement: jittered 3×3×3 lattice at spacing **1.5** about the cluster centre (jitter
    0.05·uni(−1,1) per axis).
  - Orientation by **rejection sampling** from `std::mt19937(26u)`: a normalised 4-D Gaussian
    quaternion per body, up to 2000 draws.
  - Acceptance: the host oracle finds no point of the new body's shell inside any placed body's
    SDF, and no placed body's shell point inside the new body's SDF (analytic `HollowCylinder`,
    `genCylinderShell` spacing 0.09, dist < 0 rejects). If 2000 draws fail, abort the setup with a
    message; do not fall back.
  - Velocities by `makeRingMini`'s recipe (drift + 1.5 N(0,1) − r/r_max).
  - Physics: g = 0, friction 0.02, e 0.5, dt 1e-2, iterations 20/8, 10 steps.
  - The model scene (seeded differently) forms contacts from step 1, with 250–500 contact points.
- **Modes.**
  - `ring_collide` gates conservation.
  - `ring_collide_posonly` (velocity iterations 0, 4 steps) is the np-agreement gate. The four-step
    agreement is a new claim: WO-12 proved it for one step. Its Lipschitz argument is in R-B8.
- **Numbers** are in §6 G-B2.

## 4. Q-C: the PGS restitution target (the user chooses)

### 4.1 The energy identity (frictionless, one substep)

Here u is the normal separation velocity (u > 0 separates), and u = u⁻ + Wλ with W the Delassus
operator. Then `ΔT = ½ λᵀ(u⁺ + u⁻)`. For a law with complementarity `λ_c (u⁺_c + e_c u⁻_c) = 0`:

- **dem-PGS today** (target 0 for pre-separating contacts, i.e. e_c = 0 where u⁻ > 0):

      ΔT = [ e·S₊ − ½(1−e) λᵀWλ ] / (1+e),   S₊ = Σ_{u⁻_c>0} λ_c u⁻_c ≥ 0

  Every *pre-separating contact that ends up loaded* creates e·λ_c u⁻_c/(1+e). This is exactly the
  mechanism the brief describes.
- **Moreau** (e_c = e on every closed contact): `ΔT = −½ (1−e)/(1+e) λᵀWλ ≤ 0`, and = 0 at e = 1.
  The guarantee needs a uniform e among the loaded contacts. The resting threshold (e = 0 below
  2 g dt) and per-pair materials leave a creation channel bounded by ½ Σ λ_c u⁻_c over those
  contacts. None was observed in the model (A.4).

### 4.2 Options

Values are from model A.4: 343 spheres, 670 closed contacts of which 338 pre-separating, one
converged substep, KE₀ = 527.18.

| Option | Law in the PGS (g ≠ 0 path) | KE at e = 0.5 / 0.9 / 1.0 | Physics | Cost | What it breaks or moves |
|---|---|---|---|---|---|
| **N** (today) | target `−e·v0til` if approaching pre-solve, else 0 | 391.7 / **634.6** / **719.3** | Newton per contact. Creates up to +36% KE in dense kinetic states (brief A.2: +6.4% / +16%). | 0 | — |
| **M** (Moreau) | target `−e·v0til` on **every** manifold with \|vn0\| ≥ vRest; e = 0 below | 349.8 / 482.2 / **527.18** | Energy-consistent (theorem). A pre-separating pair driven into approach may approach up to e·\|u⁻\| before it resists, so it collides in the next substep with its own restitution. Simultaneous impact, like N: a column hit from above rebounds as if rigid (+e in the model), and the cradle spreads (−0.6, 0.4, 0.4, 0.4, 0.4). | 0 (one expression) | Changes only kinetically pre-separating closed contacts (\|u⁻\| > 2 g dt). Unchanged: single contacts, resting beds, one-sided contacts (e = 0), the multilevel coarse leg (e = 0), g = 0 (the one-shot). Moves: dense collisional flows (drum flowing layer, silo orifice), impacts into *moving* beds. |
| **S** (split) | event one-shot `(1+e)·approach` (colour order, policy X under MPI), then PGS with e = 0 | 317.1 / 451.4 / 527.18 | Sequential binary collisions, energy-consistent. Reproduces the cradle (0, 0, 0, 0, 1) and wave transmission (column rebound 0.005 at e 0.5, 0.655 at 0.9). Order-dependent: ±0.07% over 6 orders in brief A.2. | +5–17 one-shot sweeps per substep (A.2), plus X's masks on the PGS path | One-shot impulses must enter the Coulomb bound and warm-start ledger (new bookkeeping). Couples the g ≠ 0 path to X. A large implementation, and it moves every g ≠ 0 collision. |
| P2 (Poisson two-LCP) | compression LCP (e = 0), then restitution LCP with impulses e·λ_comp | – | Energy consistency not guaranteed in multi-contact | 2× velocity phase | Overlaps the existing `'poisson'` bank model. Not recommended. |
| X (energy clip) | N, then rescale targets if ΔT > 0 and re-solve | – | Non-local hack | 2× | Not recommended. |

**Interaction with Poisson banking.** The release channel fires exactly on kinetically separating
contacts (`v0til < −vRest`), the set where M changes the main target. Recommended default: **M
applies to the `'newton'` restitution model only**; `'poisson'` keeps its Newton per-substep targets
(R-C4).

**Recommendation to the user: M.** It removes the energy creation at zero cost with a one-line law
change. It leaves every single-contact, resting and one-sided result bitwise unchanged. The event
fidelity that S would add (sequential transmission in one substep) is a larger, separate project.
Until the user decides, the default stays N (WO-C1 adds M as a diagnostics A/B only).

**Which physics gates move under M.**
- Unchanged, by construction:
  - `test_binary_exactness` and the Walton/bounce/restitution tests (single contacts);
  - the statics battery (resting, below vRest);
  - `cluster_e09/e10`, Enskog cooling, `tri` (g = 0 one-shot path);
  - `cluster_poisson`.
- Expected to move, and reported: the rotating drum (tests/python and Dosta bimodal drum), the silo
  (Dosta), and the 25k steel-ball impact (Dosta harness `~/Codes/dem-bench/peclet/`).
  - The bed ahead of the impact is at rest, so M ≈ N there.
  - Expect < 2% change on the 0.170 rebound. This is an expectation, and G-C3 measures it.

## 5. Register entries (the caller writes them into `../docs/decisions/dem.md`)

When each lands, also add entries 1, 3, 4 and 6 to the settled list in `CLAUDE.md` (dem).

1. **New. Multilevel coarse bodies are rigid 6-DOF aggregates (projection form, spins included).**
   - Decided: 2026-09-26 (this note §2; lands with WO-A2).
   - Rejected:
     - translation-only aggregates (ΔL = (X_A−X_B)×J; `hub_ml` dLvel 3.0e-3);
     - a coarse impulse redirected through the centres (non-associated, breaks the ledger and KE);
     - a spin-only torque (tiny rotational inertia, spurious spins);
     - orbital-only (singular for every level-1 sphere pair);
     - residual/Galerkin evaluation (can create KE).
   - Why: ΔL exact, KE non-increasing per coarse step, translation recovered as I_g⁻¹ → 0.
   - Supersedes the "report dLvel only" clause of S17.
2. **Amended.** "Position solve is translation-only; overlap removal stays decoupled from velocity"
   (2026-07-10).
   - Add, rejected: rotation in the position phase (2026-09-26). It does not rescue tunnelled states
     (ring_mini: 84/106 pairs infeasible even with linearised rotation ≤ 3 rad). Non-tunnelled tube
     scenes are translation-feasible (0 infeasible pairs over 40 steps). It costs an orientation
     consensus, folds, lever-arm refresh and an L defect.
   - Revisit trigger: R-B4.
3. **New. The overlap projection's diagonal is the translational effective mass `invM_A + invM_B`
   of the solve views** (2026-09-26, WO-B1).
   - Rejected: `computeW`'s rotational term (a rotation never applied; world arm × body-frame
     inertia, not frame-indifferent; overstated the stop metric).
   - The fixed point and ω_pos = 1.5 are unchanged. Named change: non-spheres and spinning spheres.
4. **New. An analytic non-spherical shape's `baseRadius` is its circumscribed radius** (tube
   √(R²+(H/2)²), box √3·R) (WO-B0).
   - Rejected: the geometric radius. End and corner contacts were invisible to the broad phase and
     the band: coaxial tubes overlapping by 0.3 gave 0 contacts.
   - Inertia formulas keep the geometric radius.
5. **New, open defect. The narrow phase tests one shell direction per pair** (F2, R-B2): a measured
   0.0235 penetration was unseen while 0.001 was reported. Status: open; user decision.
6. **New. `ring_mini` is a conservation scene, not a convergence scene.** It starts tunnelled (85/106
   pairs have an infeasible linearised overlap problem), so its `ovl` cannot converge under any
   solver. The convergence gate is `ring_collide`.
   - Rejected: chasing ring_mini's `ovl` with solver changes (computeW, rotation, more iterations).
7. **Pending the user (Q-C).** "The PGS restitution target is <N | M | S>", with the §4.1 identity as
   the why and the other options as rejected.
   - Update R-U1 in the framework §11 to point here.

## 6. Verification gates

**G-A1 conservation (WO-A2).**
- `hub_ml` and `cluster_multilevel`: dLvel ≤ **1e-6** (today 2.98e-3 and 1.4e-8).
  - dP ≤ 5e-6; dX and dXpos keep their thresholds; the positive controls are unchanged (np 1, 2).
  - Configurations: np 1/2/4/8 × OMP 1/8, CUDA `--solo --fused=on` np 1, and CUDA step_mpi np 1/2.
- Pre-decided fallback: if any dLvel > 1e-6, store `invI_g` in double (6 doubles per group) and
  re-measure. If it is still above, **stop and report** the conditioning of the offending groups.

**G-A2 oracle (new `tests/kokkos/test_multilevel_rigid.cpp`).**
- Scene: 40 bodies (spheres + tubes, random inertia frames), 2 hand-built levels, one wall, random
  v and ω. One coarse cycle (device) is compared against a host double reference of §2.3–§2.4:
  - per-group V, Ω and member velocities within 1e-5 relative;
  - whole-system |ΔP| ≤ 1e-6·Σ|J|;
  - whole-system |ΔL| ≤ 1e-6·Σ|J|·R_g,max (wall impulses subtracted);
  - KE_after ≤ KE_before·(1+1e-6).
- A singleton group's w and vn equal the fine PGS's within 1e-6 relative.

**G-A3 inert refactor (WO-A1).** With rotation compiled out, np 1 at OMP 1 is **byte-identical to
0520b21** for `hub_ml`, `cluster_multilevel` and the `test_colored_gs` multilevel pile (dumps).
CUDA atomics are nondeterministic, so the check is host-only.

**G-A4 mutant 8** (`PECLET_DEM_TEST_MUTANT == 8`: the prolongation drops `(Ω−Ω0) × d`):
`mutant8_momentum_hub_ml_np1` WILL_FAIL.

**G-A5 byte-identity.** Every non-multilevel mode is byte-identical at np 1 (G3 dumps), and the
battery passes (264 host, 59 CUDA + the new tests).

**G-A6 performance and physics, reported.**
- ms/step for `hub_ml`, `cluster_multilevel` and a 25k multilevel pile, before and after.
- Silo discharge rate and drum angle in multilevel mode, before and after. A change > 10% goes to the
  user; it is not a stop.
- Frictionless-sphere spin KE in `hub_ml` over 10 steps (R-A1).

**G-B0 detection (WO-B0).**
- A new `tests/kokkos/test_narrowphase.cpp` case covers three shallow scenes, all with penetration
  0.03 < t/2. In each the pair must be detected (contacts > 0), and after one step (20 position
  iterations, velocity solve off) the committed overlap (`compute_overlaps`) must be ≤ 1e-3:
  - coaxial tubes at centre distance 1.47;
  - a T-junction: B's axis ⊥ A's axis, and B's end cap presses 0.03 into A's outer wall;
  - two unit cubes: A oriented so that one body diagonal points at B's centre, B axis-aligned, and
    A's corner 0.03·R inside B's face.
- Detection only, no overlap bound (these are tunnelled states: §3.2): coaxial tubes at centre
  distance 1.4 and 1.2 give contacts > 0 (today 0).
- MPI: a `test_ghost_band_mpi` mode with the coaxial scene straddling the rank face ⇒ both owners
  see the pair (np 2).
- Reported: the RingBed-style growth packing (`docs/contact_evidence/ring_progress.py`, 500 steps):
  contacts per particle, φ and ms/step, before and after.

**G-B1 diagonal (WO-B1).**
- The rule, checkable from the dumps: a sphere mode whose bodies all have ω = 0 throughout the run
  is **byte-identical** at np 1, OMP 1. Examples: `cluster`, `cluster_posonly`, `hub`,
  `hub_static`, `tri`, `cluster_periodic`.
- Sphere modes with ω ≠ 0 are the named change. Examples: `cluster_friction`, `cluster_pgs`,
  `cluster_sync3`, `cluster_norot`, and `hub_ml` / `cluster_multilevel` once WO-A2 has landed,
  since rigid aggregates spin spheres. All tube modes are also the named change.
- Report max |Δx|/R after the run; expected ≤ 1e-4.
- All conservation gates and `position_agreement_np{2,4,8}` pass.
- Reported: position iterations (median over steps) on `ring_collide` and the ring packing, today
  vs WO-B1. A ratio > 1.5 goes to the caller (R-B3); it is not a stop.

**G-B2 `ring_collide` (WO-B2).**
- Conservation, np 1/2/4/8 × OMP 1/8: dP ≤ 1e-6, dXpos ≤ 3e-5, dLvel ≤ 1e-6.
- Feasibility as convergence, np 1: with stops on and `--pos-iters=2000`, `posItersUsed` < 2000
  in **every** step. The harness records the max over steps. This is the discriminator:
  ring_mini runs to the cap.
- Committed overlap at the default 20 iterations ≤ 5e-3 (the model gives ≤ 2.3e-3 on the grid-SDF
  variant); report the value.
- `ring_collide_posonly`, `--no-stop --pos-iters=2000`, 4 steps: np 2/4/8 positions agree with np 1
  to **1e-4 R** (R = 0.5, via `position_agreement.sh` generalised with a mode argument).
  If this fails, **stop**: the uniqueness premise is false for multi-point units.

**G-C1 energy (WO-C1).**
- New harness mode `cluster_pgs_e`: cluster_pgs with μ = 0, no spins, stabilization off,
  `--e=<e>`, `--rest-target=<newton|moreau>`, one step, `--no-stop --vel-iters=2000`.
- Moreau at np 1/2/4/8:
  - KE_cm after ≤ KE_cm before·(1 + 1e-5) at e = 0.5/0.9/1.0;
  - at e = 1, KE_cm after ≥ before·(1 − 2e-3).
- Newton: the KE ratio is reported (expected > 1 at e ≥ 0.9).
- New pytest `test_restitution_target_energy` (diagnostics A/B): the same bound at np 1.

**G-C2 inertness.**
- Default (Newton): everything byte-identical.
- Moreau: `test_binary_exactness`, restitution, bounce, Walton and the statics battery are
  byte-identical (no pre-separating kinetic contacts). If one differs, report which contact
  classified differently.

**G-C3 report to the user**, Newton vs Moreau: drum surface velocity and angle, silo discharge rate,
Dosta 25k rebound displacement, and a dense sheared or poured bed's granular temperature.

## 7. Work orders

Work in a worktree `suite/dem-<topic>`, as a sibling of `dem`. Stage named paths. One commit per
work order. Nothing is pushed between work orders that carry a known regression. Dependencies are
A1 → A2; B0 → B2; B1 is independent; C1 → (user) → C2.

**WO-A1: rigid-aggregate plumbing, inert. Byte-identical (G-A3).**

Files:
- `src/solver_multilevel.hpp`: `MlScratch`, `buildContactHierarchyKokkos`, `MlCoarseSweep`,
  `multilevelCoarseCycleKokkos`, `demMlCoarseCycleDevice`, `demFusedCoarseCycleK`,
  `demFusedMlLoopK`, `demLaunchFusedMlLoop`;
- `src/solve_copies.hpp`: `buildInvInertiaCoarseKokkos`;
- `src/particles.hpp`: the lazy pools and `invInertiaCoarse`;
- `src/solve_driver.hpp`: pass `posPred`, `angVelPred`, `P.quat`, `invInertiaCoarse`;
- `tests/kokkos/test_multilevel_rigid.cpp` + CMake.

Steps:
1. Implement §2.3–§2.5. The rotational statements go under `template <bool Rot>` on `MlCoarseSweep`
   and the cycle. Production instantiates `Rot = false`; the geometry build runs in both.
   Use `if constexpr`, not multiply-by-zero: sign-of-zero would break byte identity.
2. The oracle test instantiates both. `Rot = true` meets G-A2. `Rot = false` reproduces
   ΔL = (X_A−X_B)×J, which proves the test discriminates.

Acceptance: G-A3, G-A2, battery green.

**WO-A2: rotation on. Named change: multilevel runs; periodic multilevel aggregation.**

Files: the same, plus `tests/kokkos_mpi/test_momentum_mpi.cpp` (tolerances `hub_ml` and
`cluster_multilevel` dLvel 1e-6; delete the S17 comments), `tests/kokkos_mpi/CMakeLists.txt`
(mutant 8), `docs/solver_details.md` (multilevel paragraph).

Steps:
1. Remove the template parameter (rotation always on).
2. Add the wrap exclusion to matching (§2.3).
3. Add mutant 8.

Acceptance: G-A1, G-A4, G-A5, and the G-A6 report. Then write register entry 1.

**WO-B0: bounding radius (F1). Named change: every analytic tube and box run.**

Files: `src/shape_registry.hpp` (`initializeShape` passes the circumscribed radius to
`appendShape`; the inertia formulas use the `radius` argument, never `baseRadius_`), plus a reader
audit of `baseRadius_`/`P.baseRadius`/`rad` (every use is a reach use; list them in the commit
message), tests.

Acceptance: G-B0. Then write register entry 4.

**WO-B1: consistent translational diagonal. Named change: non-spheres and spinning spheres.**

Files: `src/solver_position.hpp` (`:625`, `:127-128`; delete `:135-172`'s dθ/`deltaQuat` scatter;
`computeW` stays), `docs/solver_details.md`.

Acceptance: G-B1. Then write register entries 2 and 3.

**WO-B2: `ring_collide` gate. Tests only.** Depends on WO-B0.

Files: `tests/kokkos_mpi/test_momentum_mpi.cpp` (scene, modes, `posItersUsed` max over steps,
`ring_mini` comment: conservation only), `tests/kokkos_mpi/position_agreement.sh` (mode argument),
`CMakeLists.txt`.

Acceptance: G-B2. Then write register entry 6.

**WO-C1: Moreau as a diagnostics A/B. Default byte-identical.**

Files:
- `src/solver_velocity.hpp`: `PGSManifoldSweep` gains `int restTarget`. In `'newton'` model with
  moreau:

      target = (fabs(vn0) >= vRest·lenN) ? −restitution·v0til : 0

  `restitution` is already 0 for side-flagged and sub-threshold contacts. Also the fused device
  loop, which shares the struct.
- `src/sim.hpp`: `Simulation::setRestitutionTarget`.
- `src/dem_bindings.cpp`: `sim.diagnostics.set_restitution_target('newton'|'moreau')`, default
  `'newton'`. Setting `'moreau'` while the restitution model is `'poisson'` raises
  `ValueError`, and so does switching the model to `'poisson'` while the target is `'moreau'`
  (R-C4).
- The harness flag `--rest-target`, mode `cluster_pgs_e`, and a pytest.

Acceptance: G-C1, G-C2. Deliver the G-C3 report to the user.

**WO-C2 (after the user's choice).** Make the chosen law the only one and delete the A/B. Re-baseline
the named change. Write register entry 7.

## 8. Risks and open questions (each has a default; work proceeds on it)

| # | Item | Needs | Default |
|---|---|---|---|
| R-A1 | Spins in the coarse space mean frictionless sphere spins change in multilevel mode (KE non-increasing, L exact). The alternative is orbital-only for SPHERE members with an eigen-projection of collinear groups; its axial-torque loss is ~√η. | **Preference**, informed by G-A6's spin-KE number | Spins included. Revisit if the measured frictionless spin KE exceeds 1% of the step's KE loss. |
| R-A2 | Float `invI_g` on chain-like aggregates (cond ~ (R_g/r)²) | Fact (G-A1) | The pre-decided fallback: double inverse, then stop. |
| R-A3 | Rotation shifts silo discharge and drum angle in multilevel mode | Fact (G-A6) | Report. A change > 10% goes to the user; not a stop. |
| R-A4 | Pre-existing, unchanged: a coarse correction is not in range(M⁻¹Jᵀ). Internal "glue" impulses of an aggregate are recorded in no λ, so the ledger under-counts internal loads, and tension can occur across quasi-static separating contacts. | Fact / preference | No action here. Recorded. |
| R-B1 | WO-B0's named change: margins, bands and tolerances scale by 1.80 (tubes H/D = 1.5) or 1.73 (boxes). Output "radius" fields report the bounding radius. | **Preference** (scope of a correctness fix) | Do it (a missed contact is a correctness defect). |
| R-B2 | F2: two-way shell detection. The XPBD engine only: Hertz would double-count stiffness. Always use the sphere probe for sphere–shell pairs. The contact buffer grows 2× for shell–shell pairs. | **Preference** (cost) + fact (cost, G-B0 packing) | Not in this package. Recommended next. |
| R-B3 | Iteration rate of the consistent diagonal (±50% in the model, both directions) | Fact (G-B1) | Accept. Ratio > 1.5 goes to the caller. Tuning ω per unit size (the model favours ω = 1 for multi-point units) needs its own decision. |
| R-B4 | Rotation in the position phase (option b) | **Preference** | Not now. Trigger: a non-tunnelled packing with translation-infeasible pairs (LP audit, A.1), or a measured density deficit against a 6-DOF reference. |
| R-B5 | The position phase delta-rotates *sphere* contact geometry by the predicted spin (C error O(R(ω dt)²)) | Fact | Leave; noted. |
| R-B6 | On tunnelled pairs Λ, and so the next Coulomb bound, grows with the iteration count | Fact | Accept: after WO-B0 tunnelling is an initial-condition error. No guard is designed. |
| R-B7 | `solver_friction.hpp` load estimates use `computeW` (the same frame mix) | Fact | Out of scope; check in the next friction package. |
| R-B8 | `ring_collide_posonly` agreement over 4 steps relies on per-step uniqueness plus Lipschitz propagation | Fact | Gate at 1e-4 R. On failure, stop (see G-B2). |
| R-C1 | The restitution law | **Preference** | N stays the default until the user chooses. WO-C1 proceeds. |
| R-C2 | The A/B's public name | **Preference** | `diagnostics.set_restitution_target('newton'\|'moreau')`. |
| R-C3 | Moreau's guarantee needs a uniform e among loaded contacts; the resting threshold and per-pair materials leave a bounded channel | Fact | Gate on uniform-e scenes; report mixtures. |
| R-C4 | Moreau in `'poisson'` mode | **Preference** | No: Poisson keeps its Newton targets. |

## Appendix A. The measurements (throwaway models, not committed; rebuild from this description)

**A.1 Pair LP audit.**
- Input: a `--dump` state of `test_momentum_mpi` (records `{int32 gid; float pos[3], vel[3], w[3], quat[4]}`).
- Geometry:
  - the tube shell per `genCylinderShell(0.5, 1.5, 0.18, 0.09)` (1218 points);
  - the analytic SDF and gradient of core's `HollowCylinder` (axis y);
  - world transforms by the (x, y, z, w) rotation matrix.
- Contacts per pair i < j: the points of i's shell with SDF_j < 0.05. Rows: n = R_j ∇SDF_j/|·|,
  d = SDF_j, rA = p − x_i, rB = p − n d − x_j.
- LP (scipy HiGHS): maximise s subject to `d_k + n_k·(D + θ×r_k) ≥ s`, with |D|∞ ≤ 5 and
  |θ|∞ ≤ {0, 0.05, 3}. Infeasible ⇔ s* < 0.
- Results in §3.2. On `ring_mini` after one step (np 1), 106 pairs are in contact.

**A.2 Detection probes.**
- Probe 1: two coaxial tubes (axis x, via a −90° rotation about z) at centre distance
  1.2/1.4/1.45 through `peclet.dem` (build_ct). `num_contacts` is 0 each time, and positions are
  unchanged after `step()`.
- Probe 2: the rejection-sampled 27-tube scene, run twice:
  - with `initialize_shape('hollow_cylinder', …)`;
  - with `set_sdf_shape` (grid h = 0.01 of the analytic SDF, the same shell, bounding radius
    0.9014).

**A.3 Diagonal-rate model.**
- On states of the grid-SDF scene: predicted positions x + v dt, frozen one-way rows.
- Translation-only accumulated PSOR in pair units, list order, until max|dΛ|·w̃ < 5e-5.
  - w̃ = 2 (consistent), or w̃ = 2 + Σ(r×n)²·invI_body (today's `computeW`).
- Numbers in §3.3. The 64-tube scene: 4³ lattice, spacing 1.5, velocity scale 1.5.

**A.4 Restitution model.**
- Scene: frictionless spheres, 7³ lattice at spacing 0.95 + N(0, 0.04) jitter, radii 0.5(1 ± 0.1),
  m ∝ r³, V ~ N(0,1), `numpy.random.default_rng(1)`.
- Contacts: every overlapping pair (670).
- PGS as `PGSManifoldSweep`'s normal update, converged to 1e-12.
- One-shot: list order, `(1+e)·approach` while approaching, sweeps until none approach.
- Newton's cradle: five unit spheres at spacing 0.999, v₀ = (1, 0, 0, 0, 0).
- Column: four unit spheres on an infinite-mass base at spacing 0.999, top ball at −1.
- Per-pair e ~ U[0.3, 0.9] and U[0, 1], and vRest ∈ {0.1, 0.3}: Moreau ≤ KE₀ in every case (e.g.
  526.74 at e = 1, vRest = 0.3). Newton stays > KE₀ at e ≥ 0.9.
