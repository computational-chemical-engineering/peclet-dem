# One conservative contact-solve framework for dem: bodies updated in several places

Design note, 2026-09-25. Branch `contacts` (worktree `suite/dem-contacts`), base dem `e90caff`
(= `c771e07` + evidence and report-only tests). Brief: `docs/contact_evidence/ARCHITECT_BRIEF.md`.
Evidence: `docs/contact_evidence/FOLLOWUPS.md`, `docs/contact_evidence/review/review_momentum.md`,
`docs/mpi_momentum_conservation.md` (the c771e07 design). This note stands on its own; the
implementer does not need the brief. It writes no production code.

---

## 0. The decision

**The principle.** A body that is updated in more than one place during a sweep interval (on
several ranks, or through more than 32 edges inside one colouring) is represented by **copies**.
Every contact acts on exactly one copy of each of its bodies, and each contact impulse lives in
exactly one place (its owner's `λ_c`). Copies are reconciled by one of two policies, chosen **by
the form of the update rule**, not by where the copies live:

- **Policy M, mass splitting** (projection-form updates: warm-started PGS with cone and Poisson
  release, the overlap projection, every stabilization mode, the multilevel cycle, the Jacobi
  diagnostic). Each of the `k` copies carries `m/k` and `I/k`. All copies update concurrently. At a
  reconciliation the body takes the mean of its copies. The fixed point is the coupled solution,
  convergence is guaranteed, and momentum is conserved exactly at every iterate.
- **Policy X, exclusive holding** (the event-form update: the `g = 0` one-shot restitution sweep).
  In each sync interval exactly one rank *holds* each shared body, and only the holder's contacts
  write it. Copies carry the true mass. Reconciliation is the plain c771e07 reverse. The parallel
  run is then a legal serial Gauss–Seidel order, so it inherits every serial property exactly:
  conservation, and each update being an exact, dissipative binary collision.

**Verdict on the brief's §4 hypothesis: amended.**
- Mass splitting with consensus is right for every projection-form phase and for local hubs.
- It is **wrong for the one-shot restitution sweep**. There it compounds restitution: an
  under-relaxed collision leaves a residual approach, which is restituted again. Measured on a
  343-body dense cluster at np 8, e = 0.9, it removes 14 % of the kinetic energy in one substep
  against the serial run, and at e = 1 it removes 18 % where serial conserves exactly. It also never
  meets the one-shot stop, so it always runs to the iteration cap (§1.5, Appendix A).
- Exclusive holding replaces it there.
- Folding the one-shot into dem's PGS form is rejected too. dem's PGS target is 0 for a contact
  that was separating before the solve, and that **creates** energy even at np 1: +6.4 % at e = 0.9
  and +16 % at e = 1.0 in one converged substep of the same cluster. It would also cost about 5× the
  iterations.

**The other invariants.**
- **Single application point.** Every impulse acts at one point. This fixes the legacy friction
  couple (D2). It changes np 1 whenever body–body friction runs at `g = 0`. That is accepted: it is
  a physics bug.
- **Visibility.** Every pair within contact reach is visible to **both** of its bodies' owners and
  solved by exactly one of them. This needs three things:
  - a ghost band of `reach + S + P`, where `S = 0.25 R_max` is the drift slack and `P` is the Verlet
    skin or the prediction displacement;
  - a migration to the current blocks whenever any body's predicted position is more than `S`
    outside its owner's block. The vote is folded into an existing Allreduce, so it costs no new
    message;
  - every periodic image within the band sent, from a new opt-in in core's halo topology. This is a
    core minor release.
- **Race-freedom.**
  - The 64-colour greedy never forces a colour. When a body has more than 32 edges, it gets local
    mass-split copies so that no colouring vertex does, and then the greedy provably succeeds.
  - The overlap projection is coloured and swept per contact **pair** (all points of a pair
    sequentially in one work item). Ring beds then drop from 79–613 edges per body to about 10–20.
  - Coarse multilevel edges with no free colour are skipped at that level.
  - The dead count-averaged fallbacks are deleted.

**np 1 results that change (all named, all intended):**
1. **D1:**
   - non-spherical particles, whose position sweep becomes pair-sequential;
   - any body whose greedy colouring fails today (degree > 62 at a vertex pair).
2. **D2:** `g = 0` runs with body–body friction.
3. **D3:** the `'jacobi'` diagnostic, which becomes mass-split Jacobi.
4. **Periodic `demStep` and periodic `step_mpi` at np 1:**
   - the position-phase images become reconciled copies;
   - under MPI the velocity phase maps images to one slot.

Everything else at np 1 on a closed domain is byte-identical: spheres, analytic walls, every
colouring that is valid today.

**Expected cost.** For spheres, the round count per substep is unchanged on the `g ≠ 0` path and
+2 on the `g = 0` path. Ghosts grow by about 12 % because of the drift slack. The expected total is
+2–6 % ms/step at np 8 × 2, within the 10 % gate. On the single-GPU sphere path nothing new runs.
Ring beds should get *faster* than today's racy baseline, because the position phase drops from 63
colours (capped) to about 20–30 per sweep.

---

## 1. Setting, invariants and proofs

### 1.1 Notation and the two update forms
- Bodies `i` carry a generalized mass `M_i = diag(m_i 1, I_i)`, with `I_i` in the world frame and
  the orientation frozen during the velocity phase.
- A contact `c` between bodies `a` and `b` has a Jacobian row block `J_c`:
  - normal: the linear part `n` on `a` and `−n` on `b`, with angular parts `r_a × n` and
    `−r_b × n`;
  - tangential rows likewise.
- `λ_c` is its accumulated impulse (normal, tangential, Poisson release, and the position-channel
  `Λ_c`).
- The phase's state (velocity or predicted position) is
  `v = v⁰ + M⁻¹ Σ_c J_cᵀ λ_c`. This is the **single-ledger invariant**: one global λ vector, each
  entry stored once (by the contact's owner, c771e07).

Every per-contact update in dem has one of two forms:
- **Projection form.** `λ_c ← Π_c(λ_c + ω (t_c − J_c v) / w_c)`, where the **target `t_c` is fixed
  during the phase** and `Π_c` projects onto a set that depends only on `λ` (λ ≥ 0, the Coulomb disc
  `|λ_T| ≤ μ(λ_n + carry)`, or the Poisson budget `[0, owed]`). Members:
  - PGS normal (`target = −e·max(vn0, 0)`);
  - the PGS cone;
  - the Poisson release;
  - the overlap projection (`t = 0` on the gap; a projection onto the half-space `C ≥ 0`, applied
    only when violated);
  - the one-sided, escalate, ordered and multilevel passes, which are the same kernels with side
    flags;
  - the Jacobi solves.
- **Event form.** `Δλ_c = (1 + e) · approach_c(v_current) / w_c`, applied only while approaching,
  never retracted. The target depends on the **current** iterate. This is the `g = 0` one-shot
  sweep, `solveVelocityColoredGSKokkos` (`src/solver_velocity.hpp:1582-1732`), called at
  `src/solve_driver.hpp:484`. Each application is an exact binary collision of the two *whole*
  bodies along `J_c`:
  `ΔKE = ½ (e² − 1) approach² / w_c ≤ 0`.
  Its result depends on the order. At np 1 the ordering spread is small (0.2 % of KE over six
  colour orderings of a dense cluster, Appendix A).

### 1.2 Copies
- A **copy** of body `i` is a state slot through which contacts update `i` during a phase:
  - the owner row;
  - a ghost slot on another rank;
  - a periodic-image slot;
  - a **hub copy slot**, an extra slot appended to the SoA (§4).
- Each copy `q` has a **seed** `σ_q`, its value at the last reconciliation. For positions the seed
  includes the copy's periodic shift.
- A copy is **active** in a phase if at least one edge of that phase's colouring graph (§4) writes
  it.
- `k_i` is the number of **active** copies of `i`, taken over all ranks and all copy kinds. It is
  computed exactly once per phase per substep (§4.6). `k_i ≥ 1`; a body with no active copy uses
  `k_i = 1`.

Two bookkeeping facts make both policies exact:
- a copy is written only by contact updates of the phase (the partner half of an impulse pair), by
  the warm start, or by multilevel prolongation;
- every such write is linear in `λ`.

### 1.3 Policy M (mass splitting): definition and proofs

**Definition.**
- During the phase, every copy `q` of `i` is solved with inverse mass `k_i/m_i` and inverse inertia
  `k_i I_i⁻¹`, through *solve views* (§4.5).
- Contacts update copies exactly as they update bodies today.
- Reconciliation (at every rank sync, and every iteration for local copies, §4.4):

  ```
  v_i  ←  v̄_i + (1/k_i) Σ_{active copies q} (ṽ_q − σ_q)          (M-consensus)
  ```

  Here `v̄_i` is the owner's seed, and after reconciliation every copy is re-seeded to `v_i`, plus
  its shift for positions. With `k_i = 1` this is exactly c771e07's `owner += Σ ghost increments`,
  and is computed in that form (bitwise).

**P1: linear momentum, at every iterate.**
- A contact update gives equal and opposite impulses `±J` to one copy of each body. On the copy,
  `Δṽ_q = (k_i/m_i) J`.
- M-consensus gives `m_i Δv_i = (m_i/k_i) Σ_q (k_i/m_i) J_q = Σ_q J_q`: the body receives exactly the
  sum of impulses applied to its copies.
- Summed over bodies, every internal impulse cancels. The only change is from walls (external).
- This holds for **any** `k_i` used *consistently*, i.e. the same `k_i` in every copy's solve mass
  and in the consensus. Consistency is the only requirement for conservation. Exactness of `k` only
  matters for convergence (§1.5).
- Positions: `m_a Δx_a + m_b Δx_b = n dλ − n dλ = 0` per contact. The consensus is the same
  algebra, so the **CoM is conserved**.

**P2: angular momentum, velocity phase.**
- Positions are fixed during the phase.
- If `J_c` acts at a single point `x_c` on both bodies (§1.7), the contact's total change is
  `(x_a × J + (x_c − x_a) × J) − (x_b × J + (x_c − x_b) × J) = 0`.
- Copies of one body share its orientation (quaternions are forwarded and not written in the
  velocity phase), so `I/k` is the same matrix on each copy.
- The M-consensus on `ω` then delivers `I Δω = Σ_q (x_c − x_i) × J_q`.
- A periodic domain does not conserve `L` in serial either.

**P3: energy (Jensen).**
- At a reconciliation, the split-system kinetic energy `Σ_q ½ (m/k) |ṽ_q|²` is at least the
  reconciled `½ m |v̄'|²`, because the mean minimizes the mass-weighted spread.
- So **reconciliation never adds energy**. Any energy change comes from the local updates alone.
- Consequence for PGS: the distributed iteration adds no energy beyond what PGS updates add in
  serial. dem's serial PGS itself does add energy at unconverged iterates. Measured on the tri
  scene at e = 0.8, from 0.333 to 0.342 at iteration 1 (Appendix A). With `ω = 1`, M's iterates on
  the tri scene approach the converged value **from below**.

**P4: fixed point = coupled solution.**
- A projection-form update leaves `λ_c` unchanged iff its residual (with clamps) is zero. The
  residual's zero set does not depend on the positive scalar `w_c`: a mass-split copy only rescales
  `w_c`.
- At a fixed point of the M iteration, every copy equals the consensus state. So every contact's
  residual, evaluated on whole bodies, is zero.
- That is the coupled (serial) fixed point. For PGS normal plus Poisson release it is unique in `v`.
  For the cone it is PGS's own fixed-point set. For the overlap projection it is the feasible set
  (POCS).

**P5: convergence.**
- Write the split problem: copies with masses `m/k`, the contacts acting on copies, plus the
  equality constraints "all copies of `i` equal". Its primal is exactly the original problem.
- The M iteration is **block coordinate descent on the dual** of this split problem:
  - contacts owned by different ranks act on **disjoint copies**, so their simultaneous update *is*
    a sequential one;
  - the consensus is the exact minimization over the equality multipliers (a projection in the
    `M̃` metric).
- Projected block SOR on a convex quadratic dual with a PSD Hessian converges for every block
  relaxation `ω ∈ (0, 2)`. The consensus block uses `ω = 1`; the contact blocks use `ω_phase`
  (§4.5).
- **Scope superseded by §13.1:** this argument holds only for accumulated multipliers with a
  retractable clamp (the PGS family). The overlap projection is non-accumulated POCS: it converges
  to a feasible point, but only `ω = 1` avoids a permanent overshoot.
- c771e07's raw sum has no such guarantee. It is Jacobi across copies with an effective
  `ω ≈ (k+1)/2`, which exceeds 2 for k ≥ 4 (edges and corners).

**Requirement.** P4 and P5 need projection form. Event-form updates do not satisfy them (§1.5).

### 1.4 Policy X (exclusive holding): definition and proofs

**Definition** (the one-shot phase only).
- Rank colouring:
  - ranks are coloured `col(r) ∈ [0, C)`, `C ≤ 64`, by greedy colouring in rank order of the graph
    "blocks within `band + S` of each other, periodic images included";
  - it is computed on every rank from the global block geometry, with no communication;
  - it is recomputed when the decomposition or the band changes.
- Holder: for a rank-split body `b` (active on ≥ 2 ranks), let `mask_b` be the OR of `2^col(r)`
  over its active ranks (§4.6). Let `t` be the sync-interval index, the same on all ranks (§4.6).

  ```
  holder(b, t) = (mask_b has bit (t mod C)) ? (t mod C)
                 : the ((t mod popcount(mask_b)))-th set bit of mask_b     (lowest bit = 0th)
  ```
- A contact on rank `r` **fires** in interval `t` iff every rank-split endpoint `b` has
  `holder(b, t) == col(r)`.
- Non-firing contacts still compute and record their approach in the stop residual. They do not
  write.
- Copies carry the true mass. Reconciliation is c771e07's raw reverse, then forward.

**X1: a legal serial order.**
- In interval `t`, a rank-split body is written only by its holder rank.
- A body active on one rank only is written only there.
- Hence the ranks' writes in interval `t` touch disjoint bodies.
- Within a rank the colouring makes the sweep a serial GS.
- After the sync every copy equals the unique written copy.
- So the distributed run equals a serial GS sweep in the order: interval by interval, and inside an
  interval the ranks' sequences concatenated in any order.

**X2: conservation.**
- This is c771e07's proof: each contact is applied once, equal and opposite, and the reverse
  delivers it exactly.
- Because at most one copy per body changes per interval, the raw sum is exact.

**X3: energy.**
- Each firing is the serial kernel applied to whole bodies at a consistent state: an exact binary
  collision with `ΔKE ≤ 0` for `e ≤ 1`.
- So KE is non-increasing at every update, at every np.

**X4: no starvation.**
- At `t ≡ col(r) (mod C)`, every rank-split body active on `r` has bit `col(r)` set, so its holder
  is `r`.
- Every contact therefore fires at least once in `C` consecutive intervals. For two-rank face
  bodies it fires every other interval.
- The loop's stop test includes non-firing contacts' residuals, so it cannot end while a gated
  contact still approaches. The iteration cap can.

### 1.5 Why the one-shot needs X and the projection phases get M (measured, Appendix A)

**The one-shot under M.**
- On a copy the collision is exact, but after consensus the pair still approaches (the other
  copies did not move). The next iteration restitutes that residual again.
- The effective restitution is therefore below `e`, and at `e = 1` the pair can end at rest.
- Measured:

  | scene | serial | M, np 2 | M, np 8 |
  |---|---|---|---|
  | tri, e = 0.8 | 0.2459 | 0.0715 | – |
  | cluster, e = 0.9 | 476 | 449 | 409 |
  | cluster, e = 1.0 | 527 | 489 | 434 |

- The "no approach" stop is never met, so M always runs to the cap.

**The one-shot under X.** It reproduces serial to 0.03 % (476.0 at np 8), in 13–17 iterations
against serial's 5.

**PGS under X.** Iterations to converge rise at small blocks (per-rank 43 bodies: 74 against 25;
per-rank 216: 171 against 49). **PGS under M:**
- with `ω = 1`: 54 and 93;
- with `ω = 1.5`: 33 and 57;
- at per-rank 729 bodies: 108 (`ω = 1`) and 97 (`ω = 1.5`) against 93 serial.

M converges much better where the rank faces dominate (strong scaling), and needs no schedule.

### 1.6 Race-freedom (palette lemma)
- An edge coloured by the greedy "lowest colour free at both endpoints" sees at most
  `(deg(u) − 1) + (deg(v) − 1)` taken colours.
- With 64 colours (bits 0..63 of the existing `uint64_t` mask), a free colour exists whenever
  `deg(u) + deg(v) ≤ 64`, and in particular when every vertex has degree ≤ 32.
- Round-based arbitration commits, every round, the maximum-key edge at both its endpoints, and that
  edge finds a free colour. So the colouring completes with every edge validly coloured.
- A vertex is a *state slot*, i.e. a copy. Hub copies (§4) bring every vertex to ≤ 32 edges, so
  every colour class writes each copy at most once.
- The multilevel coarse colouring's vertices are aggregates. An edge with no free colour is left
  uncoloured and skipped at that level. That is sound because the coarse cycle only accelerates:
  the fine sweep defines the fixed point, and coarse sweeps are symmetric on groups.

### 1.7 Single application point
- The manifold path already uses the common midpoint
  (`transformContact`, `src/contact_preprocessing.hpp:421-450`: `rA_mid = rA − ½ dist n`,
  `rB_mid = rB + ½ dist n`, so `x_A + rA_mid = x_B + rB_mid`).
- The legacy friction pass uses the two surface points. That gives `ΔL = dist · n × J_t`, measured
  ratio 1.000 (FOLLOWUPS §2).
- Decision: every per-contact velocity computation of the legacy pass uses the midpoint arms, in
  `accumulateNormalImpulseKokkos`, `solveContactFrictionKokkos` and every other arm use in
  `src/solver_friction.hpp`.
- Wall contacts keep their current arm (`src/solver_friction.hpp` wall branch and
  `computePlaneLoadKokkos`). A wall is external, so any point is conservative, and wall-only runs
  stay bitwise.

### 1.8 Visibility theorem
**Setting.**
- `reach = 2 R_max + margin`, with `margin = 0.1 R_max` and `R_max` the global maximum radius.
- The narrow phase reports a pair iff the predicted centres are closer than
  `r_a + r_b + margin ≤ reach`.
- The halo sends owned `j` to rank `r` iff an image of its build-time position `x_j^b` lies within
  `band` of block `r` (all images, §5.3).
- Two bounds hold for every owned `i`:
  - `E_i := dist(x_i, block_owner(i)) + d_i ≤ S`, where `d_i` bounds this substep's predicted
    displacement (§5.1);
  - `|x_j − x_j^b| + d_j ≤ P`.
- Then for a reportable pair (`i` on `r`, `j` on `s`):
  `dist(x_j^b, block_r) ≤ (|x_j^b − x_j| + d_j) + reach + (dist(x_i, block_r) + d_i) ≤ P + reach + S`.

**Conclusion.** With `band ≥ reach + S + P`, `r` holds `j` and, symmetrically, `s` holds `i`: every
reportable pair is visible to **both** owners. The ownership rule's gid tie-break then assigns it to
exactly one of them (c771e07 §2.1). The Hertz engine uses the same argument with its list radius in
place of `reach` (§5.5).

---

## 2. Rejected alternatives

| Alternative | Why rejected (evidence: Appendix A unless cited) |
|---|---|
| Keep c771e07's raw sum for the one-shot | Creates energy: tri e = 0.8 gives 0.555 at np 2 against 0.333 initial (review). Sign-indefinite in clusters. |
| Mass splitting for the one-shot (the brief's hypothesis, unamended) | Compounding restitution. Cluster np 8 loses 14 % KE per substep at e = 0.9 and 18 % at e = 1. Never meets the stop, so always runs to the cap. |
| Fold `g = 0` into dem's current PGS form | Creates energy **at np 1**: cluster +6.4 % (e = 0.9), +16 % (e = 1.0) in one converged substep. The cause is target 0 for pre-separating contacts. About 5× the one-shot's iterations (25 against 5). Changes every `g = 0` np 1 run. |
| Fold `g = 0` into Moreau-form PGS (target `−e·γ⁻` on every contact) | Dissipative at convergence, but changes the np 1 model (cluster e = 0.5: 387.7 against 368.7) and costs about 5× the iterations at np 1. A contact-law change is out of scope. |
| Exclusive holding for every phase | Exact, but PGS needs 3–3.5× serial iterations at small blocks (74 against 25; 171 against 49). M needs 1.2–2.1× (`ω = 1`) or about 1.2× (`ω = 1.5`). |
| `k` = number of ranks holding the body (a free upper bound) | Idle held copies under-relax. np 8: 181 iterations against 54 with the exact active count. |
| SOR `ω = 1.5` on the PGS normal by default | Near-serial convergence, but iteration 1 on tri reaches 0.362 against 0.333 initial. Kept as a measured lever (R-F2), default `ω_vel = 1`. |
| A distributed colouring with a sync per interface colour | Violates the directive: no per-colour messages. |
| Unbounded colouring (multi-word masks, colour count = max degree) | Colours grow to the maximum degree for the **whole** system: 114 for ratio-6 beds and 600 for per-point ring beds, i.e. launches or grid barriers per sweep. Bounded greedy plus hub copies keeps ≤ 64 colours and pays only at the hub. |
| Hubs by time-slicing (virtual colouring vertices fired every `s`-th iteration) | Exact, but heavy hubs converge `s`× slower: star hub, s = 4 needs 9 iterations against 3 with mass-split copies. Every realistic hub is heavy (§4.1). |
| Per-point position colouring with hub copies for rings | 20+ copies per ring, and convergence depends on the stride assignment. Pair units are exact and cut colours from 63 to about 20. |
| Leave uncoloured edges to the count-averaged fallback (the dead code's intent) | The fallback is the per-body Jacobi of D3, which is non-conservative, and it would make D3 reachable from defaults. |
| Per-step migration of all drifted particles | A host migrate round every substep. The vote-triggered migration at slack `S` gives the same guarantee and migrates about every 20 substeps in the perf scene (§7). |
| dem-side image completion (receivers derive the missing images) | Possible without a core release, but it builds derived ghost slots, a second forward and reverse, and geometric partner tests. The halo's correctness belongs in core: an opt-in flag, byte-identical for other consumers. |
| Owner-view `k` (the owner infers every rank's activity from its own visible list) | Saves the reverse but cannot see remote hub copies, and is off by one at round-off-edge contacts. The exact count rides on an existing sync on the `g ≠ 0` path (§4.6). |

---

## 3. How each existing phase maps onto the framework

| Phase (code) | Form | Rank-split bodies | Local hub copies | np 1 change |
|---|---|---|---|---|
| Warm start (`warmStartApplyKokkos`, PGS path) | known impulses | applied on true masses **before** copies are seeded. The opening sync reconciles it raw (exact for known λ) | not yet seeded | none |
| One-shot restitution, `g = 0` GS (`solveVelocityColoredGSKokkos`) | event | **X** (gate, true masses, raw reverse) | M with `k_local = s` (§4.1) | only where today's colouring fails |
| PGS normal + cone + Poisson release, `g ≠ 0` (`PGSManifoldSweep::solveOne`) | projection | **M**, `ω_vel = 1` (normal only) | M | only where the colouring fails |
| Stabilization: one-sided (smode 1), escalate (3), ordered (4) | projection (side flags hold one side: a sink by design, unchanged) | M | M | same |
| Multilevel (smode 2): fine sweep + coarse cycle | projection + coarse accelerator | M (ghost copies enter aggregates with solve masses via `effMass(invMassSolve)`; **superseded by §13.2**: a coarse vertex has mass `a·m/k`) | fold before the coarse cycle, re-seed after (coarse vertex mass: §13.2) | coarse colouring: skip instead of forcing colour 62 |
| Legacy friction, `g = 0` (count-averaged Jacobi, `src/solve_driver.hpp:806-815`) | explicit Jacobi pass | raw (unchanged: `syncFrictionCounts`, then the pass, then `syncVelocities`) | runs after the final fold | **D2**: midpoint arms |
| Overlap projection, all runs (`PositionContactSweep`) | projection (POCS) | **M**, `ω_pos = 1` (**superseded by §13.1**; was 1.5) | M | **D1**: pair units for multi-point pairs, bitwise for single-point |
| Poisson bank update, orphan scatter | bookkeeping | true masses; credit only by the pair's owner (§6.3) | after the final fold | none |
| Jacobi diagnostic, velocity + position (`velocityUseGS = false`) | projection / event, Jacobi | "every contact its own copy": `k = global contact count`, raw reverse (§3.1) | no colouring, no hubs | **D3** |
| Hertz–Mindlin (`demStepHertzMpi`) | explicit, redundant on both owners | unchanged; needs symmetric visibility (§5.5) | none | none |

### 3.1 Mass-split Jacobi (D3)
- Every contact is its own copy.
- The count pass runs first: a new kernel counts, per body, the active contacts the Jacobi solve
  will touch (velocity: manifolds with `num_points > 0`; position: contacts). Under MPI,
  `syncContactCounts` (c771e07 WO-3b, kept) then makes the counts global.
- The solve computes each contact's impulse against copy masses:
  `λ_c = r_c / (n_a w_a + n_b w_b)`, where `w` is the true generalized inverse mass along `J_c` and
  `r_c` is today's numerator. The velocity numerator is the one-shot `(1+e)·approach`; the position
  numerator is `−C`.
- It accumulates **true-mass** deltas `J_c λ_c / m`.
- The apply is a plain add, factor 1. This replaces `min(1, 2/count_i)` in
  `applyVelocityDeltasAveragedKokkos` and `1/count_i` in `applyUpdatesKokkos`.
- Properties: conservative (P1 with `k = n`), KE non-increasing (per-copy exact collisions plus
  Jensen), binary collisions exact (`n = 1`).
- It is more strongly damped than the legacy ω = 2 average at multi-contact bodies. It is a
  diagnostic A/B path, and the change is recorded (§10).

---

## 4. Colouring, hub copies and position units

### 4.1 Hubs in practice
Measured (FOLLOWUPS §1): **every** realistic hub is a heavy body.
- Ratio-6 bidisperse spheres: position-graph degree 114 (speculative neighbours in the margin).
  The large sphere has 216× the mass.
- The synthetic hub scene: one large grain with 242 active manifolds, radius 10 R.

Ring beds reach 79–613 per-*point* edges, but only 5–20 per *pair*, and §4.3 removes them from the
hub class. For a heavy hub split into `s` copies, the reduced mass of a collision with a light
partner changes by the relative factor `1 − (s − 1) m/M`: 1.4 % for M/m = 216 and s = 4. The
one-shot's compounding at a heavy hub is therefore negligible. For a light hub (> 32 edges among
equal masses), which is geometrically impossible for spheres in the velocity graph (kissing number
12), it is not. That is recorded as risk R-P3, with a diagnostic counter.

### 4.2 The colouring procedure (all four fine colourings)
The four fine colourings are:
- `colorManifoldsKokkos` (`src/solver_velocity.hpp:277`);
- `colorManifoldsIncrementalKokkos` (`:395`);
- `colorContactsKokkos` (`src/solver_position.hpp:154`), which becomes `colorUnitsKokkos` (§4.3);
- `colorContactsIncrementalKokkos` (`:253`), which becomes the unit version.

Changes:
1. **No forced colour.** The free-colour search runs over bits 0..63:
   `c = 0; while (c < 64 && (forbidden >> c) & 1) ++c;`.
   - If `c == 64`, the edge is marked `−3` (uncolourable). It counts neither as colourable nor as
     remaining, so the round loop terminates.
   - `leftover` = the number of `−3` edges.
   - The `rem == prevRemaining` stall break is kept as a safety bound only.
   - For every colouring that is valid today, this is byte-identical: the first free colour is the
     same, and colour 63 is only ever reached where today forces 62.
2. **Periodic twin dedup** (velocity colourings, `realIdx(bodyA) > realIdx(bodyB) → −2`):
   - it stays on the single-rank path;
   - under MPI it is **disabled** by a new argument `dedupTwins = !Hooks::distributed`, because
     ownership already excludes the non-owned twin from `[0, nmOwned)` (c771e07) and the §6.2 slot
     map would otherwise make the two rules disagree.
3. **Both phases are coloured at the top of `demSolveContacts`,** before any other work. Today the
   position colouring and its buckets run after the velocity phase (`src/solve_driver.hpp:822-880`).
   - Its inputs (contacts, sleep masks, the previous-substep ledgers) do not change in between, so
     moving it is byte-identical.
   - It is required so that both phases' copy counts are known before the opening sync (§4.6) and
     before any capacity growth.
4. **Attempt, then split.** Colour each phase graph once.
   - If `leftover == 0`, done. This is always the case when no vertex pair has degree sum > 64, so
     the common path pays nothing new.
   - Otherwise build hub copies for that phase (§4.4) and recolour that phase once from scratch
     (full, not incremental) on the copy vertices. The palette lemma guarantees `leftover == 0`.
   - If it is still > 0, it is a bug. Under MPI the non-zero flag is voted inside the first
     existing stop Allreduce (`allMaxAny`, kept) and every rank throws `std::runtime_error`
     ("colouring invariant violated: N uncolourable edges after hub splitting"). A single rank
     throws directly.
   - The incremental colourings (single-GPU, `g ≠ 0`) on failure fall back to this full path with
     copies.

### 4.3 Position units (the ring-bed fix)
- A **unit** is the set of contacts of one body pair (walls: one body and one wall).
- The unit list reuses the manifold reduction's sort. `reduceContactsToManifoldsKokkos`
  (`src/contact_preprocessing.hpp`) sorts contacts by pair key and builds segments.
- Extend it (optional outputs, default off, so np 1 callers without units are byte-identical) to
  emit:
  - `unitStart` (`int`, `nu + 1`);
  - `unitContacts` (`int`, `nc`): the original contact indices of each unit, **sorted ascending by
    original contact index** inside the unit;
  - `unitLeader(u)` = the smallest original contact index in `u`.
- Owned units come first, like owned manifolds: a unit's ownership is its contacts' ownership,
  which is the pair's.
- **Colouring:** one edge per unit, endpoints `(bodyA, bodyB)` of its contacts,
  `key = colorKey(unitLeader(u))`.
  - For single-point units this is exactly today's edge, key and graph, so the colouring is
    identical.
  - Sleep: a unit is inactive iff all its contacts are sleep-masked.
  - The incremental unit colouring carries colours by the leader contact's existing contact key
    (`P.contactKeys`). This is identical for single-point units. Multi-point units usually
    re-arbitrate, which is accepted (rings run `g = 0`, where incremental colouring is off anyway).
- **Sweep:** `PositionContactSweep::solveOne(u)` loops `unitContacts[unitStart(u), unitStart(u+1))`
  in that order and applies today's per-contact body to each.
  - For single-point units it is today's arithmetic on today's contact, so spheres stay bitwise.
  - Buckets list unit indices.
  - The fused sweep calls the same `solveOne(u)`.
- `posLambdaContact` stays per contact.

### 4.4 Hub copies (per phase)
**Construction.** Only after attempt 1 fails, per phase:
1. **Degree.** For every colouring vertex `v`, `d_v` = the number of the phase's active edges at
   `v`.
   - Velocity: manifolds with initial colour `−1`, i.e. not `−2`.
   - Position: units not sleep-masked.
   - Vertex = `realIdx(body)` in the velocity phase and the raw slot in the position phase, as
     today.
   - Wall edges have one endpoint.
2. **Hubs.** `d_v > 32` gives `s_v = ⌈d_v / 32⌉` copies: the base slot plus `s_v − 1` new slots.
3. **Deterministic, balanced assignment.**
   - List the hub-incident edges as `(hubCompactId, edgeIndex)`. Edge index = manifold index or
     unit index; both orders are canonical because they are sorted by pair key.
   - `sort_by_key` on `key = (uint64(hubCompactId) << 32) | edgeIndex`.
   - The `j`-th edge of hub `v` in that order goes to copy `j mod s_v`, with copy 0 the base slot.
   - Each copy gets `⌊d/s⌋` or `⌈d/s⌉ ≤ 32` edges. An edge between two hubs is assigned
     independently at each end.
4. **Slots.**
   - Copy slots are appended at `[numParticles, numParticles + nCopies)`.
   - `nCopies` is known before any view is captured: colouring is the first work in
     `demSolveContacts`.
   - If `numParticles + nCopies > capacity`, call `P.ensureCapacity(numParticles + nCopies + 64)`.
     `ensureCapacity` resizes every per-body view with its content, including `invMassEff`
     (`src/particles.hpp:475-530`), so it is safe here.
   - The velocity and position phases reuse the same slot range, because they never overlap in time.
5. **Overrides.**
   - Per-edge views `slotA(e)`, `slotB(e)` (`int`, `−1` = none). Velocity: per manifold. Position:
     per unit.
   - Set only for hub-incident edges, to the assigned copy slot (the base slot for copy 0).
   - Every **state access** in the phase's sweep bodies uses
     `sA = (slotA.extent(0) && slotA(e) >= 0) ? slotA(e) : <today's index>`. Today's index is
     `realIdx(m.bodyA)` for velocity and `c.bodyA` for position.
   - Places that take the override:
     - `PGSManifoldSweep::solveOne` (`realA/realB` at `src/solver_velocity.hpp:1043-1044`);
     - the one-shot lambda (`:1606-1607`);
     - `PositionContactSweep::solveOne` (`idA/idB`);
     - the colouring vertex mapping in every fine colouring;
     - `debugColoringConflicts` (`src/sim.hpp:933`).
   - Keys, labels and side flags keep reading `m.bodyA` through `keyIdx` and `realIdx`, so the
     ledger, grounded levels and side flags stay per body.
6. **Seed** (one kernel per phase, once the solve views exist, §4.5). Each copy slot gets its base's
   state:
   - velocity: `velPred`, `angVelPred`, `quat`, `invInertia` (for `genInvMass`), `bodyOrphan`,
     `bodyOrphanVPeak`, `groundedLevel`;
   - position: `posPred`, `quatPred`, `quat`;
   - solve masses per §4.5.
   The seed value `σ` of every copy of `v` equals the base slot's value at that moment.

**Local fold** (one kernel, over hubs; captured inside the iteration lambdas):
- In every iteration of every loop of the phase, after the sweep and before any residual-based
  break, fold each hub:
  - `T = Σ_{its s copies} (x_q − σ)`;
  - set the base slot to `σ + T/s`;
  - re-seed the `s − 1` copy slots to the base's new value.
- The iteration lambdas are `emitVelIter`, `emitOsIter`, `emitMlIter` (between the fine sweep and
  the coarse cycle, with copies re-seeded again after the coarse cycle), `emitPosIter`, and the
  escalate and ordered loops.
- `σ` stays the **phase-start seed**, the owner seed or ghost baseline of §4.6, until the next rank
  sync. The local mean then keeps the sum of local increments, which is what the rank-level
  reconciliation needs (§4.6).
- At np 1 with no rank syncs, `σ` is the substep-start value and the local fold *is* the M-consensus
  with `k = s`.
- Orphan account (Poisson, velocity M phases): each active copy starts with `B/k` (§4.6). The local
  fold sums, sets every local copy to `(Σ local)/s`, and takes the max for `orphanVPeak`.
  **Superseded by §13.3:** the local fold divides by the local active count `a` (= `s`); shares and
  solve masses use the global `k`; the pack multiplies by `a`. The multilevel re-seed stays, with
  coarse vertex masses `a·m/k` (§13.2).

**Fast paths.**
- The single-rank fused device loops (`velLoopDone`, `osLoopDone`, `mlLoopDone`, `posLoopDone` in
  `src/solve_driver.hpp`) are **disabled while copies exist in that phase**. This is the same
  gating as today's `velLeftover == 0` / `posLeftover == 0`, and those conditions are replaced by
  "no copies in this phase".
- CUDA-graph capture of the iteration lambdas is unaffected: the fold kernel is part of the lambda.
- The per-colour fused sweep under MPI calls `solveOne`, so it inherits the overrides.

**The multilevel coarse colouring** (`src/solver_multilevel.hpp:320-335`) becomes
`while (c < kMlSlotSkip && (forbidden >> c) & 1) ++c; if (c == kMlSlotSkip) { mark pending-skip; }`.
The edge keeps slot 63 and is skipped at this level. That is the documented intent ("mask
saturation: leftovers keep slot 63"); only the forced colour 62 goes.

**Deleted as dead:**
- the velocity fallback (`velFallback`, `src/solve_driver.hpp:496-507`) and the position fallback
  (`posFallback`, `:917-927`), with their Jacobi applies;
- `leftover` stops meaning "Jacobi work" and means "invariant violated".
The `allMaxAny` vote stays and carries that flag.

### 4.5 Solve views and relaxation
- **Why separate views.** The Kokkos SoA keeps `P.invMass` as its own view, not in `pos.w`. It is
  also read by the bookkeeping kernels: the Poisson bank energy, the orphan mass weighting, the
  commit and the sleeping swap. Those must see true masses. The sweeps already take `invMass` and
  `invInertia` as parameters, so no sweep code changes for masses.
- **Construction.** Per phase, if any slot on this rank has `k_phase > 1` or any copies exist, build
  `invMassSolve` and `invInertiaSolve` (capacity-sized, grow-only members of `Particles`):
  - slot `q` of body `i` gets `k_i × P.invMass(base)` (and likewise the inertia);
  - otherwise pass `P.invMass` / `P.invInertia` themselves, which is byte-identical.
- **Which `k`.**
  - velocity phase, M (PGS, `g ≠ 0`): the global `k_vel` (§4.6);
  - velocity phase, X (one-shot, `g = 0`): `k_local = s` (hub copies only). Rank-split bodies keep
    their true mass;
  - position phase: the global `k_pos`.
- **Who uses them.** The sweeps, stabilization passes and multilevel (`effMass`, group masses) use
  the solve views. The warm start (before seeding), legacy friction, Poisson bookkeeping and commit
  use `P.invMass`.
- **Relaxation.** A per-slot byte `splitSlot(q) = (k of its body > 1)`, built with the solve views
  (empty when not needed). In the sweeps: if `splitSlot(sA) || splitSlot(sB)`, the step is scaled by
  `ω_phase` **before** the clamp:
  - PGS normal: `pNew = max(0, pOld + ω dp)`. The cone and Poisson release are never
    over-relaxed.
  - Position: `dLambda *= ω`, **and** `posLambdaContact` accumulates the relaxed value.
    **Superseded by §13.1:** the position phase is never relaxed; the branch is deleted.
- **Constants** (`src/solve_driver.hpp`, documented at definition):
  - `kSplitOmegaVelocity = 1.0f`;
  - `kSplitOmegaPosition = 1.5f` (**superseded by §13.1**: deleted, `ω_pos = 1`);
  - `kHubEdgeBudget = 32`.
- With `ω = 1` the velocity kernels need no relaxation code. Implement the hook anyway, because it
  is the pre-designed lever of R-F2.

### 4.6 Rank-level activity, `k`, masks and the reconciliation
**Activity.** After the colourings (and hub copies), each rank computes per slot:
- `a_vel(q)` = the number of local velocity copies of `q`'s body that have at least one owned
  active velocity edge. This is 0, 1, or `s` for a hub. It is accumulated at the body's canonical
  velocity slot (§6.2).
- `a_pos(q)` = the number of local position copies at raw slot `q` with at least one owned unit.
- `m_vel(q) = a_vel(q) > 0 ? (1ull << col(rank)) : 0`.

**Opening sync.** A new `ParticleHalo::openVelocityPhase(P, rotation)` replaces the PGS path's
post-warm-start `syncVelocities` (`src/solve_driver.hpp:430-431`). On the `g = 0` GS path it is a
new call at the same place (after `computeVn0` would be; before the first sweep).
- Under Jacobi (`velocityUseGS = false`, a global setting) it is not called.
- Collective rules as for every sync: skip only if `!exchanges()`.

Sequence:
0. **Owned rows first.** Before any exchange, set every owned row's
   `k_vel = max(1, a_vel(own))`, `mask = m_vel(own)` and `k_pos = max(1, a_pos(own))` from local
   activity. When `!exchanges()`, the method stops here, so np 1 closed and isolated ranks still get
   correct local `k` for hub copies. Local periodic images of `demStep` are handled by §WO-4 item 2,
   not here.
**Superseded by §13.3** in three respects:
- the payloads: no `velMask` before WO-6; `B` and the peak are forwarded; the `g = 0` path gets a
  counts-only exchange;
- step 5's balance source;
- the orphan baseline.

1. **Reverse**, payload
   `OpeningIncrement {float v[3], w[3], orphan, orphanPeak; uint64_t velMask; int velCount, posCount;}`.
   - `operator+` sums v, w, orphan and the counts, takes the max of `orphanPeak`, and ORs the mask.
   - The pack sends the warm-start increment against the ghost baselines, as today's
     `haloPackVelocityIncrement`, plus the slot's `a_vel`, `m_vel` and `a_pos`.
2. **Apply** on owned rows:
   - the raw increment, as today, because the warm start applies known impulses;
   - `k_vel = a_vel(own) + Σ velCount`, `mask = m_vel(own) | Σ velMask`,
     `k_pos = a_pos(own) + Σ posCount`, each floored at 1.
3. **Forward**, payload
   `OpeningState {float v[3], w[3]; uint64_t velMask; int kVel, kPos;}` (with rotation), or
   `{float v[3]; uint64_t velMask; int kVel, kPos;}` (without). Ghosts store `kVel`, `kPos` and
   `velMask` in per-slot views.
4. **Mark** the ghost baselines and the **owner seeds** (new members `ownerSeedVel_`,
   `ownerSeedAngVel_`, `ownerSeedOrphan_`, `ownerSeedPos_`, grow-only, `[no]`).
5. **Seed the orphan shares** (M velocity phase with Poisson only). Every *active* copy slot of a
   body with `k_vel > 1`, owned or ghost, sets `orphan := B/k_vel`, with `B` the forwarded balance.
   (**Superseded by §13.3**: `B` and the peak ride the opening forward and, under Poisson, every
   later velocity forward.)
   The baselines and seeds record `B/k` for ghosts and `B` for the owner seed.

**Rank-level reconciliation** (every `syncVelocities` / `syncPositions` after the opening). Local
hubs are folded first (§4.4).
- **Pack:** `inc_g = f_g × (x_g − baseline_g)`.
  - M phase: `f_g = a(g)`, the local copy count at that ghost slot (0 for an inactive slot). Its
    local copies are all equal after the fold.
  - X phase: `f_g = 1`.
  - Orphan under M: `inc = f × (orphan_g − baseline_g)`. The baseline is the share the slot was
    seeded with: `B/k` if active, `B` if not. (**Superseded by §13.3**: every slot of a `k > 1`
    body holds and records `B/k`; an inactive slot packs 0 through `f = 0`. Phase-end owner
    restore: §13.3.)
  - WO-5 adds a debug-build check that every slot with `a(g) = 0` has a zero increment. Nothing may
    write an inactive copy; a non-zero increment there is a bug and would be dropped silently.
- **Apply** on owned rows with `k > 1` in an M phase:
  `x_i := seed_i + (a(own) × (x_i − seed_i) + Σ inc) / k_i`, and for the orphan
  `B_new := max(0, B + a(own)(orphan_i − B/k) + Σ inc)`. `orphanPeak` takes the max.
- Owned rows with `k = 1`, and every row in an X phase or a raw sync (legacy friction, Jacobi): today's
  `x_i += Σ inc`, byte-identical.
- **Forward**, then mark baselines, owner seeds and (M) orphan shares again. Re-seed hub copies from
  their bases.

**Legacy-friction syncs.** They stay raw (`syncFrictionCounts`, then the friction
`syncVelocities`). They run after the loop, on reconciled state.

**X gate** (the `g = 0` one-shot only, and only on ranks that have X-split edges):
- a new kernel before each sweep computes `gate(e) ∈ {0, 1}` per owned manifold from `velMask`
  (per slot), `col(rank)`, `C` and `t` per §1.4;
- `t = P.solveEpoch + intervalIndex`, where `intervalIndex = it / syncEvery`;
- `P.solveEpoch` is a new `int64` member incremented at the top of every `demSolveContacts`, so it
  is identical on all ranks. Each substep starts the holder cycle one step later, so no rank is
  systematically first. Do not scale the epoch: a multiple of `C` or of `popcount` would cancel the
  rotation;
- the one-shot lambda takes `gate` (empty = fire all). It computes the approach and records
  `maxApproach` as today, then returns before writing if `!gate(e)`.

---

## 5. Visibility: band, drift slack, migration, periodic images

### 5.1 The per-substep vote (XPBD, `demStepMpi`)
At the top of `demStepMpi` (`src/step_solve_mpi.hpp`), **before** the growth update and predict,
compute over owned `i`:
- `d_i = |v_i| dt + |g + extForce_i · invMass_i| dt²`. This is the exact bound on
  `predictVelocityKokkos`'s centre displacement: `v' = v + a dt`, `x' = x + v' dt`.
- `E_i = min over periodic images of dist(x_i, block_rank) + d_i`.
- `D_i = |x_i − x_i^b| + d_i` when `verlet_skin > 0`, where `x^b` is the halo's build-time position
  `refPos_`; else 0.

**One** `MPI_Allreduce(MAX)` of `{R_max_owned, E_max, D_max, d_max}` replaces the pre-growth
`globalMaxRadius` Allreduce (`margin`), and folds in the halo's skin `MPI_LOR`
(`src/mpi_halo.hpp:617-621`: the rebuild predicate becomes `D_max ≥ skin`, computed here and passed
to `gather` as a new `bool displacementRebuild` argument; the internal `MPI_LOR` is deleted).

The other rebuild conditions are already the same on every rank and need no vote:
- `!haveTopo_` after a collective migration or rebalance;
- `band != lastBand_`, because the band is computed from reduced values;
- `skin ≤ 0`;
- an owned-count change, which only follows a collective migration or a user-level reset on all
  ranks.

The Hertz path calls `invalidateTopology()` before `gather` and passes `false`. Then:
- `margin = 0.1 R_max`.
- If `E_max ≥ S`: `halo.migrateToBlocks(P)`, which is collective, then force a topology rebuild.
- Growth update; the post-growth `R_max` Allreduce (unchanged).
- `band = max(rcut, reach + S + P)`, with `P = skin` if `skin > 0`, else `d_max` (the topology is
  rebuilt every gather at `skin = 0`, so a changing band costs nothing).
- predict, gather(band), … as today.

**Constants and rules:**
- `S = kDriftSlack × R_max`, `kDriftSlack = 0.25` (R-F1).
- `rcut` stays a lower bound (the user API is unchanged).
- The schedule depends only on reduced values, so it is identical on every rank (CLAUDE.md
  collective rule).

### 5.2 `migrateToBlocks`
- A new `ParticleHalo::migrateToBlocks(Particles&)`: `rebalance()` without the ORB re-init. It
  moves every owned particle to the rank whose **current** block contains its wrapped position,
  through the existing `ParticleMigrator::migrate` and `packState` / `unpackState` path, including
  the canonical `srcRank` order.
- Add `F3 extForce, extTorque` to `MigratePack` (`src/mpi_halo.hpp:113-132`). A coupled run sets
  forces before `step_mpi`, and a mid-call migration must carry them. The same applies, latently,
  to `rebalance_every > 0` today.
- Invalidate the topology (`haveTopo_ = false`), as after `rebalance`.
- `rebalance_every`, `rebalance()` and `migrate_to_weights` reset the drift too (they migrate). The
  vote simply rarely fires in coupled runs, which migrate every fluid step.

### 5.3 Periodic images: the core change (D4b; also fixes R8/P1)
- In core `ParticleHaloTopology::build` (`core/include/peclet/core/halo/particle_halo_topology.hpp:62-67`),
  add a parameter `bool allImages = false`.
- When true, the sender loop replaces `withinRcutOfBlock(pos[i], r, rcut, img)` with
  `imagesWithinRcutOfBlock(pos[i], r, rcut, /*allowIdentity=*/true, shifts)`
  (`particle_migrator.hpp:202-243`) and pushes **one send entry per shift**, so the same `i` may
  appear several times in `sendIdx[r]` with different shifts.
- Everything downstream already works per entry:
  - the shift exchange;
  - the per-ghost shift;
  - forward;
  - the reverse `atomic_add` through `sendIdx`, which sums both images' increments.
- Default false, so other consumers are byte-identical.
- Core tests (core's own suite):
  - an undecomposed periodic axis at np 2 delivers both images of an edge particle;
  - reverse sums both;
  - default-off byte-identical.
- Core is tagged (minor, 1.3.0) and published before dem uses it (R-U2).

**dem side** (WO-11):
- Call `halo_.build(pv, band, /*includePeriodicSelf=*/true, /*allImages=*/true)`.
- `copyRanks_` gains `copyImage_`, the integer image triple of the send entry (`int8_t[3]`).
- `ContactOwnership::partnerSees` checks for an entry of `o` with rank `src(g)` **and** image `−img(g)`,
  where `img(g)` is the ghost's integer image triple, computed from `shiftDev_` as
  `round(shift / L)` per periodic axis.

### 5.4 Ownership rule
Unchanged (c771e07 §2.1) apart from the image-aware `partnerSees`. With §5.1 in force, every
reportable pair is visible to both owners, so the "only one sees it" branch fires only at round-off
edges (gap ≈ margin, inactive). It is kept as defence in depth.

### 5.5 Hertz–Mindlin (confirmed unaffected in its conservation; the visibility fix applied)
- `demStepHertzMpi` evaluates every pair on **both** owners from identical forwarded state.
  Conservation needs both owners to hold every pair, i.e. symmetric visibility, which D4 breaks
  today in the same way (drift with `rebalance_every = 0`; single images).
- Apply the same fix:
  - `MpiForceHooks::gatherGhosts` runs at every pair-list rebuild. Before it, vote
    `E_max = max_i dist(x_i, block)` (committed positions; the list criterion covers motion until
    the next rebuild), folded into the rebuild's existing Allreduce. If `E_max ≥ S`, call
    `migrateToBlocks`.
  - `band_H = (2 + skinFrac) R_max + S`.
  - Same `allImages` build.
- Proof: at a rebuild, a listed pair has `|x_i − x_j| < (2 + skinFrac) R`, so
  `dist(x_j, block_r) < (2 + skinFrac) R + S = band_H`.

### 5.6 The Poisson ledger across ownership changes
See §6.3.

---

## 6. What happens to c771e07's machinery

### 6.1 Kept / changed / removed

| Item | Fate |
|---|---|
| Owner-exclusive contacts (`ContactOwnership`), owned-first contact and manifold lists, `kNonOwnedPairBit` | **Kept.** Units inherit the owned-first order. `partnerSees` becomes image-aware (§5.3). |
| Reverse primitive + ghost baselines + fused forwards (`VelocityState`, `PositionState`) | **Kept.** Payload grows only in the opening sync (§4.6). Apply gains the M-weighted branch for `k > 1` and owner seeds. |
| `publishPositions` (forward only) | **Kept.** Also marks owner position seeds and re-seeds position hub copies. |
| `syncFrictionCounts` | **Kept** (legacy friction). |
| `syncContactCounts` (WO-3b) | **Kept for mass-split Jacobi.** Its GS-fallback use is **removed** with the fallbacks. |
| `allMaxAny` | **Kept.** It now carries the colouring-invariant flag, no longer a fallback vote. |
| GS count-averaged fallbacks | **Removed** (dead; replaced by the invariant). |
| `ghost_band_*` unpinned, `ownership_*`, `momentum_*` gates | **Kept.** Thresholds tightened (§9). |
| The "redundant ghost-pair" `MigratePack` carry on both endpoints | **Kept.** Crediting is restricted (§6.3). |

### 6.2 The velocity-phase slot map under MPI (new)
**The map.** After `halo.gather` in `demStepMpi` (**not** inside `gather`, which the Hertz engine
shares), set `P.realIndices` for ghost slots so that each body has one velocity slot per rank:
- a self-image ghost (`gid` owned here) maps to the owned slot;
- otherwise, the lowest ghost slot with the same `gid` (from `ghostCanon_`, built at every topology
  rebuild on the host by sorting `(gid, slot)` over ghosts, then uploaded).

**Why it is required.** Without it, two images of one body on one rank are two concurrently
written copies. That breaks X and makes the np 1 periodic twin Jacobi-lagged.

**Consequences.**
- Velocity writes to any image land in the canonical slot, and non-canonical ghost slots report a
  zero increment.
- Positions stay per slot.
- `updateGroundedLevelsKokkos` and the orphan accounts become per body per rank, which is better.
- np 1 periodic `step_mpi` changes (named).

### 6.3 Warm-start ledger and Poisson bank under splitting
- **λ.**
  - `λ_c` is stored once, by the contact's owner. Under M, the λ a sweep accumulates is the **true**
    impulse: the copy's `Δṽ = kJ/m`, and consensus delivers `J/m`.
  - Warm start and commit are unchanged, and so is the next substep's true-mass warm start.
  - Hub overrides do not touch keys (`keyIdx(m.bodyA)`).
- **Orphan accounts** (Poisson, `g ≠ 0`, M). The per-copy share `B/k` (§4.6) makes concurrent draws
  from several copies bounded by the balance **exactly**. This fixes c771e07's R5 overdraw.
- **Double credit after migration** (review finding 3, `src/mpi_halo.hpp:826` carry,
  `src/solve_driver.hpp:403` scatter). `scatterOrphanBanksKokkos` gains an argument: under MPI,
  credit a dead entry only if its **lower-gid endpoint is owned by this rank**. Under §5.1's
  symmetric visibility that is exactly the pair's owner, so the entry is credited once. A stale
  copy on the other rank is never credited.
  - The single-rank path passes no restriction (byte-identical).
  - A dead pair whose entry lives only on the higher-gid owner (a round-off-edge ownership) is not
    credited. That is an under-credit, i.e. energy is not returned, which is conservative.

---

## 7. Cost model

**Reference.** The perf scenes, `test_momentum_mpi perf_{gas,pgs}`:
- N = 19683 periodic spheres, `dt = 2e-3`;
- 4 velocity and 8 position iterations, `sync_every = 1`, `forward_rotation = True`;
- quiet-host c771e07: 15.6 ms (np 4 × 4) and 8.1 ms (np 8 × 2);
- about 1.5k ghosts per rank at np 8.

### 7.1 Messages per substep (neighbourhood rounds; maximum, no early stop)
**Superseded by §13.4.** The `g = 0` opening's +2 rounds move to WO-5; the payloads are those of
§13.3.

| | c771e07 | after | Δ |
|---|---|---|---|
| gather | 3 | 3 | 0 |
| velocity opening (PGS: post-warm-start sync) | 2 (PGS) / 0 (gas) | 2 / 2 | 0 / **+2** |
| velocity iterations + final | 4 × 2 + 2 = 10 | 10 | 0 |
| legacy friction (gas) | 4 | 4 | 0 |
| position publish | 1 | 1 | 0 |
| position iterations + final | 8 × 2 + 2 = 18 | 18 | 0 |
| **total, pgs / gas** | 34 / 36 | 34 / 38 | 0 / +2 |
| Allreduce | 2 radius + stops | 1 fused (radius, drift, skin, d) + 1 radius + stops | 0 (the skin `LOR` folds in: −1 when skin > 0) |
| migration | rebalance only | vote-triggered: about 1 per 20 substeps in the perf scene (§7.4) | +1 migrate / 20 substeps |

**Bytes per ghost:**
- opening reverse: 32 → 48 B;
- opening forward: 24 → 40 B (rotation), 12 → 28 B (no rotation);
- every other exchange unchanged per ghost.
- The ghost count scales with the band: `2.1 → 2.1 + 0.25 + d` R_max, with `d ≈ 0.01 R` here, which
  is **+12 %** ghosts on every exchange, in the gather pack, in the broad phase's tree and in
  ghost-side narrow-phase work.

### 7.2 Kernels and launches
- **Per sweep:** unchanged, `numColors ≤ 64`.
  - Spheres: identical colour counts (12–20).
  - Ring beds, position phase: from 63 (capped, racy) to the pair degree + a few, **≈ 15–30
    launches**, each unit looping its points (5–60).
  - Ratio-6 bidisperse, position phase: ≈ 35–40 colours (hub virtual degree ≤ 32) against 63
    capped today, and ≈ 115 for an unbounded colouring.
  - The hub scene (242 manifolds), velocity phase: ≤ 64 against 63 capped.
- **Per iteration:** +1 fold kernel only in phases with hub copies. +1 gate kernel only at `g = 0`
  on ranks with X-split edges. The weighted apply replaces the existing apply kernel (no new
  launch).
- **Per substep:**
  - MPI: +2 activity kernels, and +1–2 solve-view kernels on ranks with `k > 1`;
  - the fused drift / skin reduction replaces two reductions;
  - hub setup (degree, sort, assign, seed) only after a failed colouring;
  - single-GPU without hubs: **nothing new**. The only added work is a cheap branch on the existing
    `leftover` readback.

### 7.3 Memory
Per rank, grow-only:
- owner seeds, 40 B × owned;
- `k` and mask views, 16 B × (owned + ghosts);
- solve views, 16 B × slots;
- unit CSR, 4 B × (nc + nu + 1);
- overrides, 8 B × edges (only with hubs);
- gate, 1 B × manifolds;
- copy slots, a full SoA row per copy, typically < 64 copies.

At N = 20k / np 8 this is < 1 MB per rank.

### 7.4 Migration frequency
- A migration fires when the first particle's predicted position passes `S` outside its block.
- For the perf scene, `v ~ N(0, 1)`, `dt = 2e-3`, the fastest near-face particle
  (`|v| ≈ 3`) covers `S = 0.125` (R = 0.5) in about 20 substeps.
- Cost of one host migrate ≈ 0.5–1 ms (fact to measure, R-F1), so ≤ 0.05 ms/substep amortized.

### 7.5 Expected ms/step against c771e07
**Superseded by §13.4:** +3–8 % in total, with the interface iteration risk R-F6.
- `perf_pgs` 8 × 2: +2–4 % (ghosts).
- `perf_gas` 8 × 2: +3–6 % (ghosts, +2 rounds ≈ 40–80 µs, gate ≈ 4 × 10 µs).
- np 4 × 4: about half of that.
- `forward_rotation=False`: the same rounds, +16 B per ghost on the opening forward, so within
  noise.
- CUDA single-GPU spheres: ≤ 1 %, i.e. noise.
- CUDA ring bed: expected **faster** than today (fewer position colours).
- Ratio-6: ≈ today.

The 10 % gate (§9 G12) holds with margin unless the migration cost (R-F1) or the struct-atomic cost
(R-F4) surprises.

---

## 8. Work orders

Each work order is one commit on `contacts`, in this order. Stage named paths only.

**Builds:**
- `build_ct` (host-openmp, Release, `PECLET_DEM_BUILD_TESTS=ON`, `PECLET_DEM_MPI=ON`,
  `-DMPIEXEC_PREFLAGS="--bind-to none"`);
- `build_ct_cuda` (nvidia-cuda, `nvcc` on PATH).

Test batteries run with `OMP_NUM_THREADS=2 OMP_PROC_BIND=false`.

**Stop rules, which apply to every work order:**
- If an acceptance number misses, stop and report the table.
- If a statistical-agreement test (`validate_exact`, `demstep_*`, `verify_distributed`, the
  rotating drum, coupling) fails, stop and report. Never loosen a tolerance.
- If a step appears to need anything this note leaves open (§11), stop.
- Never change a contact law or target.

### WO-0: Instrumentation and baselines (test-only; `git diff e90caff -- src` must stay empty)
Files: `tests/kokkos_mpi/test_momentum_mpi.cpp`, `tests/kokkos_mpi/test_ownership_mpi.cpp`,
`tests/kokkos_mpi/CMakeLists.txt`.
1. `test_momentum_mpi`:
   - Add the `KE` line (the review's CoM-frame KE per step, double, Allreduced) to every mode.
   - Add modes:
     - `tri` (the review scene from `docs/contact_evidence/review/review_test_patch_after.diff`:
       A1 (−0.42, 0.45), B (0.45, 0), A2 (−0.42, −0.45), `v_A = 1`, 3 steps, with an `--axis=` flag
       replacing the env var and no env reads);
     - `tri_pgs` (the same with g = (0, 0, −10), `--vel-iters=N`);
     - `cluster_e09` and `cluster_e10` (`cluster` with e 0.9 and 1.0);
     - `hub_pgs` (`hub` with g ≠ 0);
     - `cluster_poisson` and `cluster_{multilevel,escalate,ordered,onesided}` (`cluster_pgs` with
       those settings);
     - `ring_mini`: 27 hollow cylinders, D 1, H/D 1.5, wall 0.18, on a jittered 3 × 3 × 3 lattice
       at spacing 0.9 D, orientations from `mt19937(11)`, g = 0, friction 0.02, pos/vel iterations
       20/8, 10 steps.
   - All report-only (`kGate` stays false).
   - Add `--relabel=SEED` (permute gids before loading) to sample serial orderings.
2. Per-step colouring validity: after every step print `CONFLICTS vel=… pos=…` from
   `debugColoringConflicts`, max over steps and ranks.
3. `test_ownership_mpi`: a **dynamic oracle** mode `oracle_{closed,periodic,shear}`.
   - `step_mpi` for 50 steps with `rebalance_every = 0`: a sheared velocity field `v_x = 0.2 z`
     (in units of R per step) on a jittered lattice with ±10 % radii. Periodic uses strong jitter
     0.3.
   - After every step, gather to rank 0:
     - each rank's owned contacts of the last substep as `(min gid, max gid, image triple, dist)`,
       through a new C++-only `Simulation::debugCaptureContacts(bool)`, which copies them after the
       partition when on and never changes numerics;
     - the owned predicted positions and radii of that substep.
   - Rank 0 runs the serial narrow phase on `MPI_COMM_SELF` over the gathered predicted state (with
     periodic self-ghosts) and compares:
     - **required** = serial pairs with `dist < margin − 1e-4 R_max`;
     - report `missing` (required but owned by no rank), `dup` (owned by more than one) and `extra`
       (owned but not a serial pair with `dist < margin + 1e-4 R_max`).
4. Build `e90caff` + WO-0 as `build_base`. Store in `docs/contact_evidence/baseline/`:
   - np 1 OMP 1 `--dump` for `cluster`, `cluster_pgs`, `cluster_posonly`, `hertz`, `cluster_poisson`
     and the four stabilization modes (the byte-identity references);
   - the report lines of every new mode at np 1, 2, 4, 8 × OMP 1, 8;
   - `ring_mini` and `hub` on CUDA;
   - the oracle output at np 1, 2, 4, 8.

**Acceptance:**
- `src` is untouched and every new ctest passes.
- The baseline oracle shows `missing > 0` in `oracle_shear` at np ≥ 4 and in `oracle_periodic` at
  np 2 and 4 (it reproduces D4). Otherwise stop: the oracle is too weak.

### WO-1: Single application point (D2)
File: `src/solver_friction.hpp`.
- Every body–body use of `c.rA` / `c.rB` in the legacy pass switches to the midpoint arms. That is
  `accumulateNormalImpulseKokkos`'s pair branch (`:70-78`) and `solveContactFrictionKokkos`
  (`:165-217`).
- Add `KOKKOS_INLINE_FUNCTION void contactArmsMid(const ContactC&, F3& rA, F3& rB)`, identical to
  `transformContact`'s shift.
- Wall branches unchanged.

**Acceptance:**
- `friction_pair`: `|ΔL| ≤ 1e-6 × |dist n × J_t|` at every δ of FOLLOWUPS §2; set `kFollowupGate`
  for it.
- `cluster_friction` dLvel ≤ 1e-6 at np 1, 2, 4, 8.
- Every frictionless mode is byte-identical to baseline (np 1 OMP 1 dumps).

### WO-2: Mass-split Jacobi (D3)
Files: `src/solver_velocity.hpp` (`solveVelocityKokkos`, `applyVelocityDeltasAveragedKokkos`),
`src/solver_position.hpp` (`solvePositionKokkos`), `src/integration.hpp` (`applyUpdatesKokkos`),
`src/solve_driver.hpp` (the Jacobi branches `:518-526`, `:932-938`).
- Implement §3.1:
  - a count kernel first, then `hooks.syncContactCounts`;
  - the solve with `w̃ = n_a w_a + n_b w_b`;
  - true-mass accumulation;
  - apply factor 1.
- Rename nothing public.

**Acceptance:**
- `cluster_jacobi` dP ≤ 1e-6 and dXpos ≤ 1e-5 R at np 1, 2, 4, 8 (was 9e-3).
- `demstep_jacobi_*` pass unchanged.
- GS modes byte-identical to WO-1.

### WO-3: Position units
Files: `src/contact_preprocessing.hpp` (unit CSR outputs), `src/solver_position.hpp` (unit
colouring and sweep), `src/solve_driver.hpp`, `src/particles.hpp` (unit views,
grow-with-contacts: follow the "every maxContacts-sized view grown together" register rule).
- Implement §4.3.
- Move both colourings and their bucket builds to the top of `demSolveContacts` (§4.2 item 3).

**Acceptance:**
- Every sphere mode is byte-identical to WO-2 at np 1 OMP 1 (dumps).
- `ring_mini`: `CONFLICTS pos = 0` on CPU (OMP 8) and CUDA; dXpos ≤ 1e-5 R; the number of position
  colours is reported.
- `tests/python` (hollow cylinders, SDF particles, growth packing) pass. A failure is a stop.

### WO-4: Complete colouring, hub copies and local folds (np 1 hubs; images in `demStep`)
Files: `src/solver_velocity.hpp`, `src/solver_position.hpp`, `src/solver_multilevel.hpp`,
`src/solve_driver.hpp`, `src/particles.hpp`, `src/sim.hpp` (diagnostic), `src/step_solve.hpp`
(images).
1. Implement §4.2 items 1, 2 and 4, §4.4 and §4.5 (with `ω_vel = 1`, `ω_pos = 1.5`; `ω_pos` is
   **superseded by §13.1** and fixed in WO-4b), and the
   multilevel skip. Delete the fallbacks.
2. **Single-rank periodic images in the position phase** (`demStep`). Treat each image slot
   (`numReal ≤ q < numParticles`, `realIndices(q) = i`) as a local copy of `i`:
   - `a_pos` counts active slots;
   - seeds are `posPred(i) + shift_q`, where `shift_q = posPred(q) − posPred(i)` at generation
     (store it);
   - fold every position iteration with the §4.4 formula, using `k = a_pos(i)` and increments
     `x_q − seed_q`;
   - re-seed `posPred(q) = posPred(i) + shift_q`;
   - solve views with `k`.
   This is the same kernel as the hub fold, with shifts.
3. `debugColoringConflicts` uses the overrides and adds a multilevel check (per level and colour,
   group-disjointness).
4. Diagnostics: `Simulation::debugSplitStats()` returns
   `{velHubCopies, posHubCopies, lightHubs (hub mass < 10× mean partner mass), splitBodiesVel, splitBodiesPos, unfiredSplitContacts, driftMigrations}`.
   Bind it as `diagnostics.split_stats()` (tier 2, a dict).

**Acceptance:**
- `test_coloring_overflow`, star `D = 32..300`: 0 same-colour pairs; `leftover = 0` after copies
  (turn `kGate` on).
- `hub`, `hub_posonly` and `hub_pgs` at np 1, OMP 1 and 8 (3 runs each) and on CUDA (3 runs):
  - dP ≤ 1e-6;
  - dXpos ≤ 3e-5 R (the OMP 1 float floor is 1.1e-5);
  - `CONFLICTS = 0`.
- Every non-hub closed mode is byte-identical to WO-3.
- `cluster_periodic --solo` (`demStep`): CoM drift ≤ 1e-5 R (report the before value).

### WO-5: Rank-level M (projection phases)
**Superseded by §13.5.** WO-4b comes first.
Files: `src/mpi_halo.hpp`, `src/step_solve_mpi.hpp`, `src/solve_driver.hpp`.
- Implement §4.6 except the X gate:
  - `openVelocityPhase`;
  - the activity views;
  - `k` and mask views;
  - owner seeds;
  - the weighted apply;
  - orphan shares.
- Implement §6.2 (the slot map and `ghostCanon_`) and the MPI twin-dedup switch (§4.2 item 2).
- Add hooks `openVelocityPhase(P)` (Solo: no-op) and `rankK(…)`, which exposes the views to the
  solve-view builder.

**Acceptance:**
- G1 conservation for every M mode at np 1, 2, 4, 8 × OMP 1, 8: `cluster_pgs`, `cluster_poisson`,
  the stabilization modes, `cluster_posonly`, `hub_pgs`.
- G3 np 1 closed byte-identity.
- G7a: `tri_pgs` converges.
- G8 `ovl`.
- Report the `cluster_periodic` np 1 before and after.

### WO-6: Rank-level X (the `g = 0` one-shot)
**Superseded by §13.5.**
Files: `src/mpi_halo.hpp` (rank colouring from the decomposer's blocks, recomputed on
decomposition or band change; `C ≤ 64` else throw), `src/solver_velocity.hpp` (gate argument in the
one-shot lambda), `src/solve_driver.hpp` (gate kernel, `solveEpoch`, interval index),
`src/particles.hpp` (`solveEpoch`).

**Acceptance:**
- G2 energy.
- `tri` at np 2 (all three axes): KE equal to np 1 within 1e-6 relative.
- `cluster`, `cluster_e09`, `cluster_e10`, `hub`: KE non-increasing every step.
- G1 for `cluster`, `cluster_friction`, `cluster_sync3`, `cluster_norot`.
- Report `unfiredSplitContacts` per substep in `perf_gas`.

### WO-7: Drift vote, `migrateToBlocks` and the band (D4a), XPBD and Hertz
**Superseded by §13.5**, which keeps this text and adds to it.
Files: `src/step_solve_mpi.hpp`, `src/mpi_halo.hpp` (`migrateToBlocks`, `MigratePack` ext
force/torque, the gather rebuild predicate), `src/solve_driver_force.hpp` / `MpiForceHooks`,
`src/sim.hpp` (stepMpi loop unchanged except the counter), and `scatterOrphanBanksKokkos` +
`src/solve_driver.hpp:403` (§6.3 credit rule).

**Acceptance:**
- `missed_drift_pair` and `missed_drift_lattice`: 0 at every shift up to 6 R (turn `kMissedGate`
  on for them).
- `oracle_shear` and `oracle_closed`: `missing = dup = 0` at np 1, 2, 4, 8.
- `halo_schedule_*` pass (collective order).
- Hertz: a drift mode (`hertz` with shear advection, 200 steps): dP ≤ 1e-6.

### WO-8 (core repository, `suite/core` in its own worktree): `allImages`
Implement §5.3 in core with core tests. Tag and publish per `docs/RELEASE.md` (R-U2). dem work
continues meanwhile; only WO-9 waits.

### WO-9: dem uses all images (D4b)
Files: `src/mpi_halo.hpp` (the build flag, `copyImage_`, image-aware `ContactOwnership`),
`CMakeLists.txt` / `cmake` pin `PECLET_CORE_TAG` to the new core tag.

**Acceptance:**
- `missed_periodic` 0 at np 2, 4, 8 (`kMissedGate` fully on).
- `oracle_periodic` `missing = dup = 0`.
- `exactly_once_periodic` with the strong jitter 0.3 passes.

### WO-10: Gates on, mutations, docs
Files: tests (per-mode tolerance tables of §9), `tests/kokkos_mpi/CMakeLists.txt` (mutation
executables), `docs/mpi.md`, `docs/solver_details.md`, `CLAUDE.md`, the docstrings of
`enable_mpi_step` and `set_velocity_solver`.
- Mutations are compiled only into dedicated test executables with
  `-DPECLET_DEM_TEST_MUTANT=<n>`, guarded in `src` by `#if defined(PECLET_DEM_TEST_MUTANT)` (§9 G13).
  Production builds never define the macro.
- Hand the register text (§10) to the caller.

**Acceptance:**
- The full battery.
- Every mutation test detects its mutant.
- The grep of §10.3 is clean.

### WO-11: Evidence
Deliver `docs/contact_evidence/AFTER.md`: every gate table with the load, perf A/B, CUDA numbers,
the migration frequency, `split_stats`, and the before/after of every named np 1 change.

**Dependency graph.** WO-0 → {WO-1, WO-2} → WO-3 → WO-4 → WO-5 → WO-6 → WO-7 → WO-9 → WO-10 →
WO-11. WO-8 runs in parallel after WO-0. **Superseded by §13.5:** WO-4b goes between WO-4 and
WO-5, and the conditional WO-12 between WO-7 and WO-10.

---

## 9. Verification gates

"np set" = np 1, 2, 4, 8 × `OMP_NUM_THREADS` 1, 8, 3 repeats, max over repeats (`run_matrix.sh`).
dP is normalized by Σ m|v|. dX and dXpos are in units of R.

### G1: Conservation
In every mode, at the np set, plus CUDA single-GPU (`--solo` and `step_mpi` np 1).

| mode | dP | dX | dXpos | dLvel |
|---|---|---|---|---|
| `cluster`, `cluster_e09`, `cluster_e10`, `tri` | 1e-6 | 1e-5 | 1e-5 | 1e-6 |
| `cluster_friction`, `cluster_sync3`, `cluster_norot` (legacy friction) | 1e-6 | 1e-5 | 1e-5 | **1e-6** (was 1e-4) |
| `cluster_pgs` (+ cone), `cluster_poisson`, `cluster_{multilevel,escalate,ordered}` | 5e-6 (free-fall floor 7.9e-7) | 1e-5 | 1e-5 | 1e-6 |
| `cluster_onesided` | report only (a sink by design) | | | |
| `cluster_posonly` | 1e-12 | 1e-5 | 1e-5 | 1e-12 |
| `cluster_jacobi` | 1e-6 | 1e-5 | 1e-5 | 1e-6 |
| `hub`, `hub_pgs` | 1e-6 | 3e-5 | 3e-5 | 1e-6 |
| `hub_posonly` | 1e-12 | 3e-5 | 3e-5 | – |
| `ring_mini` | 1e-6 | 1e-5 | 1e-5 | 1e-5 (SDF arms) |
| `cluster_periodic` | 1e-6 | n/a | n/a | n/a |
| `hertz` | 1e-6 | 1e-5 | – | dL 1e-5 |

### G2: Energy
In force-free closed scenes with `e ≤ 1`, frictionless, at the np set and on CUDA. The scenes are
`cluster`, `cluster_e09`, `cluster_e10` and `hub`.
- `KE_s ≤ KE_{s−1} (1 + 1e-6)` at every step.
- `tri`, e = 0.5 and 0.8, np 2 on each axis: `|KE_np2 − KE_np1| ≤ 1e-6 KE_np1`. The review's
  0.291 / 0.555 must become 0.1379 / 0.2459.
- `tri_pgs`: the G7 criteria apply.

### G3: np 1 byte-identity
np 1, OMP 1, closed. The modes are `cluster`, `cluster_pgs`, `cluster_posonly`, `hertz`,
`cluster_poisson`, the stabilization modes and `cluster_e09`.
- `cmp` of dumps against the WO-0 baseline.
- `tests/kokkos` and `tests/arborx` pass.
- Changes reported and **expected**: `cluster_friction`, `cluster_jacobi`, `hub*`, `ring_mini`,
  `cluster_periodic` (np 1 `step_mpi` and `--solo`). Each is reported with its before/after
  numbers.

### G4: Run-to-run bitwise
np 4 and 8, OMP 1, 5 runs, `--dump`. Modes: `cluster`, `cluster_friction`, `cluster_pgs`, `tri`,
`hub`, `cluster_poisson`. All byte-identical.

### G5: Visibility oracle
`oracle_{closed,shear,periodic}` at np 1, 2, 4, 8, every step:
- `missing = 0`, `dup = 0`, `extra = 0`.
- Plus the missed-pair modes (FOLLOWUPS §4) at 0.
- Plus Hertz: every serial pair with `|x_i − x_j| < r_i + r_j` is in both owners' lists.

### G6: Race-freedom
- `CONFLICTS vel = pos = ml = 0` at every step for `hub`, `hub_posonly`, `hub_pgs`, `ring_mini`,
  hub scale 6 and 10, on host OMP 8 and CUDA.
- `test_coloring_overflow`: 0 same-colour pairs and `leftover = 0` for D ≤ 300.
- The D1 CUDA scene: dP back to ≤ 1e-6 (was 1e-2 to 4e-2).

### G7: Convergence to np 1
- (a) `tri_pgs` with `--vel-iters` 4, 8, 16, 32, 64:
  - `|KE_np2 − KE_np1|/KE_np1` is non-increasing in the iteration count;
  - ≤ 1e-3 at 16 and ≤ 1e-5 at 64 (model prediction: 3.6e-3 at 8, 0 at 16, Appendix A);
  - `KE_np2 ≤ KE_np1 (1 + 1e-6)` at every count.
- (b) The p–q–s chain (`ghost_band_margin`, 8 runs per cell, np 2 and 4 × OMP 1 and 8):
  `|comShift| ≤ 1e-5` and `posErr ≤ 1e-4`, 8/8.
- (c) Dense cluster, one substep from the same state (`cluster_pgs --steps=1`) with velocity
  iterations 8, 32, 128:
  - the RMS of per-body `|v_np − v_np1|/|v|` is non-increasing;
  - ≤ 1e-3 at 128, at np 2, 4, 8.
- (d) The `g = 0` one-shot:
  - `cluster --steps=1` at np 2, 4, 8: KE within the **ordering envelope** of np 1;
  - the envelope is the min and max over `--relabel` seeds 1..3 and 0 (identity), widened by
    ±0.1 %.
- (e) `ovl` (max over steps, np 2, 4, 8, `cluster`, `cluster_pgs`, `cluster_friction`): ≤ 1.5 × np 1
  and ≤ 2 × the c771e07 value. **Superseded by §13.6** (restated G7e; new G7f).

### G8: D4
`missed_drift_pair`, `missed_drift_lattice` (up to 6 R), `missed_periodic` (np 2, 4, 8): 0 missed.
`exactly_once_periodic` with jitter 0.3: pass.

### G9: Friction couple
`friction_pair` `|ΔL|/|prediction| ≤ 1e-6` at every δ and dt of FOLLOWUPS §2.

### G10: Jacobi
`cluster_jacobi` as in G1. `demstep_jacobi_*` pass.

### G11: Battery
- The full dem ctest (152 on `contacts` + new), `-DMPIEXEC_PREFLAGS="--bind-to none"`, OMP 2.
- The 12 `python_mpi_*` with core's Python build on `PYTHONPATH`, not skipped.
- coupling's MPI tests (Ergun, moving suspension): pass, with the deltas of their printed
  observables reported before and after.

### G12: Performance
**Host A/B.** `run_perf_ab.sh build_base build_after`, 5 interleaved repeats, load recorded:
- `perf_gas` and `perf_pgs` at 4 × 4 and 8 × 2: **median ratio ≤ 1.10 and min ratio ≤ 1.10**;
- `perf_*_norot` (a new `forward_rotation=False` variant): measured and reported against the same
  threshold.

**CUDA single-GPU, before/after ms/step:**
- `perf` spheres `--solo`: ratio ≤ 1.03;
- a ring bed at N = 80 (the RingBed config) and N = 500: reported (expected < 1);
- hub scale 6: reported.

**Also reported:** migrations per 100 substeps and the time of one migrate.

### G13: Mutation negative controls
§13.6 adds mutants 6 and 7.
Each must be **detected**. The test binary passes iff the gate fails.

| mutant `n` | what it changes | must trip |
|---|---|---|
| 1 | `ContactOwnership` returns false for every pair with a ghost endpoint (the review's mutant) | G5 `missing > 0` at np ≥ 2 (the momentum gates alone do not trip) |
| 2 | M apply uses weight 1 instead of `1/k` (raw sum with split masses) | G1 dP > 1e-4 in `cluster_pgs` np ≥ 2 and `hub_pgs` np 1 |
| 3 | colour search capped at 62 with the forced colour, no hub copies | G6 `CONFLICTS > 0` in `hub`; CUDA dP > 1e-4 |
| 4 | X gate disabled (every rank fires) | G2 `tri` np 2 KE ≠ np 1 (review value 0.555) |
| 5 | drift vote disabled | G5 / G8 `missed_drift_lattice > 0` at np 4 |

---

## 10. Register entries and documentation text

### 10.1 New register entries
These go in the umbrella `docs/decisions/dem.md`, indexed in `docs/DECISIONS.md`. The caller
commits them.

```
### A body updated in several places is solved through copies: mass splitting for projection-form updates, exclusive holding for the one-shot restitution sweep
- area: dem
- source: dem/docs/contact_solve_framework.md; evidence dem/docs/contact_evidence/{FOLLOWUPS,AFTER}.md
- decided: 2026-09-25
- status: settled
- quote: |
    Every contact acts on one copy of each body; each impulse lives once in its owner's lambda.
    Projection-form updates (PGS + cone + Poisson release, the overlap projection, every
    stabilization mode, multilevel, Jacobi) mass-split the copies (m/k, I/k, exact active count k)
    and reconcile by the mean -- conservative at every iterate, fixed point = the coupled solution.
    The g = 0 one-shot restitution sweep is event-form: rank-shared bodies are held by one rank
    per sync interval (rank colouring), so the distributed run is a legal serial Gauss-Seidel
    order and each update stays an exact dissipative binary collision.
- rejected: raw ghost-increment sums for the one-shot (tri e=0.8 KE 0.333 -> 0.555); mass
  splitting for the one-shot (compounding restitution: -14 % KE per substep at np 8, e = 0.9;
  never meets the stop); folding g = 0 into the current PGS form (creates energy at np 1: +6 % /
  +16 % at e = 0.9 / 1.0); exclusive holding for PGS (3-3.5x iterations at small blocks); k = ranks
  holding the body (3.4x iterations at np 8)
- why: conservation by construction; the one-shot's outcome depends on update order, so only an
  exact serial order preserves its energy behaviour; projection updates converge to the same
  fixed point for any positive copy mass
- supersedes: none (extends "Distributed XPBD contacts are owner-exclusive, with ghost->owner reverse accumulation")

### The Gauss-Seidel colourings are complete by construction: 64 colours, no forced colour, hub copies above 32 edges
- area: dem
- source: dem/docs/contact_solve_framework.md §4
- decided: 2026-09-25
- status: settled
- quote: |
    The greedy takes the lowest free colour of 64 and never forces one. When a colouring fails, every
    vertex above 32 edges is split into local mass-split copies (round-robin over its edges in
    canonical order) and the phase is recoloured -- guaranteed to succeed. Coarse multilevel edges
    without a free colour are skipped at that level. The count-averaged fallbacks are deleted.
- rejected: forcing colour 62 (a same-colour race: CUDA dP 1e-2); leaving edges to the per-body
  Jacobi fallback (non-conservative, D3); an unbounded palette (114-600 colours per sweep);
  time-sliced hubs (heavy hubs converge s times slower)
- why: race freedom from a counting lemma, cost confined to the hub
- supersedes: "Colouring stall-break needs a filtered count-averaged-Jacobi fallback for uncolourable manifolds"

### The overlap projection is coloured and swept per contact pair, all points of a pair sequentially in one work item
- area: dem
- decided: 2026-09-25
- status: settled
- quote: Ring beds carried 79-613 contact points per particle (per-point graph above the palette at
  every substep) but 5-20 pairs. Single-point pairs keep today's edge, key and arithmetic (bitwise).
- rejected: per-point colouring with hub copies (20+ copies per ring)
- why: exact GS inside a pair, colours per sweep from 63 (capped) to about 20

### Legacy friction applies +-J_t at one point, the contact midpoint
- area: dem
- decided: 2026-09-25
- status: settled
- quote: The two surface arms applied a couple dist*n x J_t (measured ratio 1.000; 62-79 % of friction
  contacts are speculative). Changes every g = 0 run with body-body friction at np 1.
- why: angular momentum conservation; the manifold path already used the midpoint

### The 'jacobi' velocity solver diagnostic is mass-split Jacobi
- area: dem
- decided: 2026-09-25
- status: settled
- quote: Each contact is solved against copies of mass m/count and the true-mass deltas are summed,
  replacing min(1, 2/count_i) and 1/count_i per body (dP 9e-3 at np 1).
- supersedes: "DEM velocity solve: over-relaxed min(1, 2/count) average, not raw Jacobi sum" (for the
  diagnostic path; the production path is Gauss-Seidel since 2026-07-10)

### Every pair within contact reach is visible to both owners: drift slack, vote-triggered migration, all periodic images
- area: dem
- decided: 2026-09-25
- status: settled
- quote: |
    Ghost band = reach + S + P (S = 0.25 R_max drift slack; P = Verlet skin, else the substep's
    predicted displacement). When any body's predicted position is more than S outside its owner's
    block (vote folded into the existing radius Allreduce) the step migrates to the current blocks.
    core's halo sends every periodic image within the band (allImages). Same for Hertz.
- rejected: zero slack (pairs lost at 1.857 R drift, np >= 4); per-step migration (a host round
  per substep); dem-side image completion
- why: the ownership rule presupposes that both owners see every pair

### Distributed dead-pair Poisson credit is given only by the pair's owner
- area: dem
- decided: 2026-09-25
- status: settled
- quote: scatterOrphanBanks credits a dead ledger entry only on the rank that owns its lower-gid
  endpoint; orphan balances are split B/k over mass-split copies.
- why: the migration carry put entries on both endpoints' new owners (double credit); concurrent
  draws overdrew
```

**Amend** the existing "Distributed XPBD contacts are owner-exclusive…" entry: add
`see also: "A body updated in several places is solved through copies"`.

**Finding for the register as an open issue, not a decision:** "dem's PGS restitution target of 0
for pre-separating contacts creates kinetic energy in dense random-velocity states: +6.4 % /
+16 % at e = 0.9 / 1.0 in one converged substep (framework note, Appendix A). The Moreau target
`−e·γ⁻` is dissipative. Unchanged pending a decision" (R-U1).

### 10.2 `CLAUDE.md` (dem)
**Settled decisions.**
- Replace the bullet "**The velocity solve uses the over-relaxed `min(1, 2/count)` average**…"
  with:
  > **A body touched from several places is solved through copies**: mass split (`m/k`, exact
  > active count, mean reconciliation) for PGS and the overlap projection; exclusive holding (one
  > rank per sync interval) for the `g = 0` one-shot restitution sweep. Never raw-sum ghost
  > increments in the one-shot, never mass-split the one-shot (it compounds restitution), never
  > force a colour.
- Add:
  > **The overlap projection sweeps whole contact pairs**, and **colourings never exceed 64
  > colours**: hubs get local copies.

**Replace** the paragraph "**Every contact is owned by exactly one rank…**" with the c771e07 text
plus:
> Since 2026-09-25 (`docs/contact_solve_framework.md`):
> - `openVelocityPhase` exchanges the active-copy counts `k` and holder masks with the warm-start
>   increments;
> - every later sync applies the mean for `k > 1` in projection phases;
> - the `g = 0` one-shot fires a rank-shared body's contacts only on its holder rank (a `gate` per
>   manifold);
> - the velocity phase maps every ghost image of a body to one slot (`realIndices`, set in
>   `demStepMpi`, not in `gather`).

**Replace** the ghost-band paragraph's first sentence with:
> **The distributed XPBD ghost band is `max(rcut, 2.1 R_max + 0.25 R_max + P)`** (`P` = Verlet
> skin, else the substep's predicted displacement), and the step migrates to the current blocks
> when any body is more than `0.25 R_max` outside its block. Both owners see every pair.

### 10.3 `docs/mpi.md`
Add a section "Bodies updated in several places" (the §0 principle in six sentences, the policy
table of §3, and the gates G1, G2, G5 and G6 as the validation statement). Replace the ghost-band
and periodicity bullets with §5's rules (band, vote, `migrateToBlocks`, all images).

Acceptance grep: `grep -n "min(1, 2/count)\|colour 62\|fallback" docs/*.md src/*.hpp` leaves no
statement describing the old behaviour as current.

---

## 11. Risks and open questions

Every item has a default, so work proceeds unattended.

| # | Item | Needs | Default |
|---|---|---|---|
| **R-U1** | dem's PGS restitution target (0 for pre-separating contacts) creates energy in dense random-velocity states even at np 1 (Appendix A: +6.4 % / +16 %). Changing it is a contact-law change, out of this package's scope. | **User preference** (physics model) | Record as an open issue (§10.1). Do not change here. Suggested follow-up: the Moreau target `−e·γ⁻` on every closed contact, gated `g ≠ 0` first. |
| **R-U2** | WO-8 needs a core minor release (tag + PyPI publish), which is outward-facing | **User go-ahead** under the standing "core first" directive | Implement and test in core and tag locally. The orchestrator confirms the publish step with the user per `docs/RELEASE.md`. dem WO-9 waits; every other WO proceeds. |
| **R-U3** | The np 1 changes beyond the brief's D1/D2/D3 list: periodic `demStep` and `step_mpi` position consensus (§WO-4 item 2) and the velocity slot map (§6.2) | **User preference** (the scope of "np 1 unchanged"; the c771e07 precedent: user accepted a periodic np 1 change for conservation) | Proceed (conservation is a directive). Report before/after in AFTER.md. |
| R-F1 | Migration cost and frequency at `S = 0.25 R_max`; the ghost overhead of the slack | **Fact** (G12) | `S = 0.25`. If migrations exceed 1 per 5 substeps in `perf_*`, or G12 fails on ghost volume, report. The lever is `S = 0.5` (fewer migrations, +24 % ghosts), a recorded decision. |
| R-F2 | Interface convergence of M with `ω_vel = 1` | **Fact** (G7c/e) | `ω_vel = 1`. If G7c or G7e fails, set `kSplitOmegaVelocity = 1.5` (pre-analysed: convergent for ω < 2, near-serial iterations; first-iterate KE overshoot like serial PGS), re-run G1/G7, record the decision. If it still fails, stop. **Superseded by §13.1 / §13.8:** this lever applies to the velocity phase only and is triggered by G7c only. The overlap projection is never relaxed; its lever is R-U4. |
| R-F3 | X starvation within a substep at small velocity caps: corner bodies (`popcount(mask)` up to 8) may not fire in a 4-iteration `g = 0` substep | **Fact** (`unfiredSplitContacts` in WO-6) | Accept; `solveEpoch` rotates the phase. If more than 1 % of split contacts are unfired per substep in `perf_gas`, report. A mitigation (X-split contacts fire first in the interval they hold) is not designed here. |
| R-F4 | CUDA and host cost of Kokkos's lock-based `atomic_add` on the 48 B `OpeningIncrement` (c771e07 R1, never timed) | **Fact** | Use it. If a reverse costs more than 5 % of a substep on CUDA, the fallback is c771e07's recorded lever (a core `reverse` returning the send-shaped buffer, dem scatters by component). That is another core release, so stop and report. |
| R-F5 | Ring-bed convergence and speed with pair units | **Fact** (G12) | Proceed. A slower ring bed than today's racy baseline is reported, not a stop. |
| R-P1 | Periodic boxes too small for the drift proof (block extent < 2·band on a decomposed axis) | Fact | `allImages` covers the multi-image need. The drift proof holds with images. No extra check. |
| R-P2 | Round-off-edge contacts (gap ≈ margin, or `dist ≈ 0` classified differently on two ranks) can make `k` off by one or drop a speculative pair | Fact (oracle tolerance `1e-4 R_max`) | Accepted. Conservation is unaffected (all copies use the forwarded `k`). |
| R-P3 | A light hub (> 32 velocity edges among comparable masses, e.g. a long fibre) in the `g = 0` one-shot is mass-split locally and over-damped | Fact (`lightHubs` counter) | Accept. It is geometrically impossible for spheres. Report non-zero counts. Time-slicing for light hubs is the designed-away alternative, not built. |
| R-P4 | Legacy friction on speculative contacts (`dist > 0`), a modelling question raised by D2 | User preference | Unchanged. |
| R-P5 | c771e07's quiet-host perf re-measure is still owed | Fact | G12 is measured on a quiet host if one is available; otherwise interleaved with the load recorded (as in c771e07). |

---

## Appendix A. The numerical experiments behind the decisions

**The models.** Throwaway Python models (not committed) of the velocity kernels, for frictionless
spheres, equal density, unit-norm manifolds:
- the event-form one-shot exactly as `solveVelocityColoredGSKokkos` (`(1 + e)·approach`, applied
  while approaching, colour order);
- PGS exactly as `PGSManifoldSweep`'s normal update (`target = −e·max(vn0, 0)`, accumulated, clamp
  ≥ 0);
- the overlap projection as `PositionContactSweep` (translation only).
Contacts are owned by the lower-gid body's rank. There is one sync (and one consensus) per
iteration.

**Validation of the model.** It reproduces the review's tri numbers to 4 digits:
- np 1: 0.1379 and 0.2459;
- c771e07 np 2: 0.2907 and 0.5546;
- PGS np 1: 0.2234.

### A.1 The tri scene (A1 (−0.42, 0.45), B (0.45, 0), A2 (−0.42, −0.45), R = 0.5, v_A = 1; KE in the CoM frame, initial 0.3333)

| e | serial one-shot | c771e07 raw np 2 | M one-shot np 2 | X np 2 | PGS np 1 (converged) | M-PGS np 2, iter 1 / 2 / 4 / 8 / 16 | M-PGS ω 1.5, iter 1 / 8 | serial PGS iter 1 / 2 / 4 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.1379 | 0.2907 | 0.0370 | = serial | 0.1042 | 0.0370 / 0.0789 / 0.1006 / 0.1039 / 0.1042 | 0.1818 / 0.1042 | 0.1709 / 0.1082 / 0.1043 |
| 0.8 | 0.2459 | 0.5546 | 0.0715 | = serial | 0.2234 | 0.0715 / 0.1717 / 0.2161 / 0.2226 / 0.2234 | 0.3624 / 0.2233 | 0.3418 / 0.2308 / 0.2234 |
| 1.0 | 0.3333 | 0.7809 | 0.1118 | = serial | 0.3333 | 0.1118 / 0.2602 / 0.3231 / 0.3323 / 0.3333 | – | – |

- **M one-shot:** A1 is split because the A1–A2 overlap contact lives on the other rank. With A1
  unsplit the values are 0.0527 / 0.1176 / 0.1833.
- **X np 2:** equals serial by construction. All six serial orders of the three contacts give the
  same KE (0.137904 / 0.245928), so the G2 tri criterion is order-proof.

### A.2 Dense cluster
N = 343, a jittered 7³ lattice, radii ±10 %, 536 overlapping contacts, mean degree 3.1, Gaussian
velocities. One substep, KE₀ = 527.2. np 2 = x-slab (35 split bodies, 10 %); np 8 = octants (87,
25 %).

| e | serial one-shot (6 orderings) | M one-shot np 2 / 8 | raw np 2 / 8 | X np 2 / 8 | dem-PGS np 1 | Moreau-PGS np 1 |
|---|---|---|---|---|---|---|
| 0.0 | 341.36 | – | – | – | 341.27 | 341.27 |
| 0.5 | 368.6–369.2 | 363.1 / 354.8 | 368.4 / 367.8 | 368.76 / 368.71 | 405.9 | 387.7 |
| 0.9 | 475.9–476.6 | 449.0 / 409.2 | 473.5 / 469.4 | 476.04 / 475.9 | **560.9** | 491.9 |
| 1.0 | 527.2 | 489.2 / 434.0 | 523.5 / 516.9 | 527.18 / 527.18 | **614.1** | 527.2 |

One-shot iterations to "no approach": serial 5; X np 2 → 8; X np 8 → 13–17; M → never (cap 200).

### A.3 PGS iterations to a relative complementarity residual of 1e-4
e = 0, cold start, one sync per iteration.

| N (per-rank at np 8) | serial | np 2: X / raw / M ω1 / M ω1.5 | np 8: X / raw / M ω1 / M ω1.5 | np 8 M with k = ranks holding |
|---|---|---|---|---|
| 343 (43) | 25 | 39 / 30 / 45 / 31 | 74 / 30 / 54 / 33 | 181 |
| 1728 (216) | 49 | 59 / 49 / 61 / 49 | 171 / 56 / 93 / 57 | – |
| 5832 (729) | 93 | 94 / 94 / 94 / 94 | 118 / 98 / 108 / 97 | – |

ω sweep at N = 343 (np 2 / np 8): 1.3 → 35 / 38; 1.5 → 31 / 33; 1.7 → 28 / 30; 1.9 → 61 / 61.

### A.4 Heavy hub, overlap projection
One hub (R = 6) with 195 overlapping leaves (R = 0.5), leaves not touching each other. Iterations to
max overlap < 1e-4 R:

| | iterations | colours |
|---|---|---|
| serial (unbounded) | 3 | 195 |
| time-slice s = 2 | 5 | 98 |
| time-slice s = 4 | 9 | 49 |
| mass-split copies s = 2 | 3 | 98 |
| mass-split copies s = 4 | 3 | 49 |
| mass-split copies s = 4, ω = 1.5 | 2 (over-separation, not convergence: superseded by §13.1) | 49 |

## 12. Session decisions during implementation (2026-09-25, binding on the work orders)

- **S1: the WO-1 gate is restated at the float floor.** The relative 1e-6 is unreachable in
  single precision: after the fix |dL| is 7e-9 to 3e-8, which is 7.7e-6 to 1.3e-3 of
  |dist n × J_t|. That is also true of the existing midpoint path, the PGS cone. The new gates:
  - `friction_pair`: couple ratio |dL| / |dist n × J_t| ≤ 1e-2. Before the fix it was 1.000.
  - `cluster_friction`: dLvel ≤ 1e-7 at np 1 to 8. Measured ≤ 6.4e-9.
- **S2: `test_cooling_slope_vs_enskog` loses its Jacobi comparison.** The assertion
  `r_gs >= 0.9 * r_j` took the old per-body count-averaged Jacobi as its reference. That Jacobi is
  gone: under D3 it became mass-split Jacobi, which is conservative and at a fixed iteration count
  more dissipative (2.14 × Enskog, against GS at 1.81 × Enskog). The test keeps its physics check,
  the Enskog band 0.5 < r_gs < 2.5, and prints r_j for information only. A mass-split Jacobi that
  over-dissipates at finite iterations is expected (§2, one-shot argument), and the path is legacy
  and not a default.
- **S3: the "dead fallback" premise was false.** Each colouring stops after
  `maxRounds = numBodies + 2` rounds. On `ring_mini` (27 rings, a multigraph of contact points)
  that left 24052 contacts uncoloured, so the count-averaged position fallback ran every substep at
  np 1 (dXpos 3.5e-2 R). The design already removes the cause: WO-3 sweeps position per contact
  pair, and WO-4 gives complete colouring with hub copies. The decisions:
  - Keep the round cap.
  - Once WO-4 lands, an edge the colouring leaves uncoloured violates the colouring invariant. It
    throws, with a clear message naming the body and its degree. It is never fallback work, and
    the count-averaged fallbacks are deleted as the note says.
  - WO-3 must show, on `ring_mini` and on the ring and hub survey scenes, that the cap is no longer
    exhausted. If any scene still exhausts it, STOP before WO-4.
- **S4: the WO-2 per-body counts are confirmed as implemented.** Velocity counts skip periodic
  twins. Position counts every contact in [0, nc), with no dist filter. Both match the contact sets
  each kernel actually applies. Re-verify this after WO-3 changes the position sweep to per-pair.
- **S5: the WO-0 `debugCaptureContacts` hook in `src` (+83 lines) is accepted.** It was proved
  inert: 127/127 comparisons identical.
- **S6: wall units get a wall id (option c).** This settles the WO-3 stop of 2026-09-25. §4.3
  defines a wall unit as "one body and one wall", but `pairKey` (`contact_preprocessing.hpp:396`)
  gives every boundary contact of a body the same key, and `ContactC::bodyB` is -1 for every wall.
  The decisions:
  - **Encoding.** A wall contact carries `bodyB = -1 - wallIndex`. `wallIndex` is unique across the
    analytic planes and the SDF walls of the scene, and stable across steps. Every test for "is a
    wall" becomes `bodyB < 0`: audit every `== -1` / `!= -1` / `>= 0` use. `ContactC` does not
    change size. If an existing encoding already uses negative `bodyB` values other than -1, STOP.
  - **Keys.** Units are keyed by (body, wallIndex). The manifold-reduction key and its segments
    stay EXACTLY as today (`pairKey` unchanged), so that np 1 spheres and analytic walls remain
    byte-identical (§0). Unit construction uses its own key, or a secondary sort key within a
    segment.
  - **Acceptance, in addition to WO-3's own:** byte-identity at np 1 for the sphere scenes with
    2–3 simultaneous wall contacts, i.e. a sphere in a box corner and the drum. Add such a dump if
    WO-0 has none.
  - Rejected: (a) units = the sort's segments, which changes np 1 spheres at box edges and corners;
    (b) one unit per wall contact, which pushes ring beds in containers onto hub copies.
- **S7: the colouring round cap bounds the colouring graph, not the bodies.** `maxRounds = numBodies + 2`
  (`solver_position.hpp:204, 358`, and `numReal + 2` in `solver_velocity.hpp:348, 501`) is not a
  termination bound: the vertices being coloured are units or contacts, not bodies. On `ring_mini`
  at OMP 8 it ran out in 4 of 6 runs. Change every colouring to
  `maxRounds = numVerticesColoured + 2`. The bound holds because in each round the globally
  highest-key uncoloured vertex is the maximum at both its bodies and is always coloured, provided
  a colour is free, which WO-4 guarantees. That argument needs **unique keys**:
  - Verify that `colorKey` cannot tie between two vertices that share a body. Otherwise two winners
    would both write the non-atomic `bodyMask |=`, which is a race.
  - If ties are possible, break them with the vertex index inside the key, as long as that leaves
    single-thread np 1 byte-identical. If it cannot, STOP.
  - Rounds stay O(log n) in practice. The stall break stays; after WO-4 an exhausted cap or a stall
    throws (S3).
- **S8: `test_hollow_cylinder_overlap` asserts detection, not residual overlap.** WO-3 sweeps a
  pair's 30 contact points in order, which removes the overlap within the substep, so the
  post-step assertion `max_overlap > 0` now reads 0.0. The bodies still separate (dx 0.54 after 99
  steps). Replace that assertion:
  - assert that step 0's narrow phase detected at least one contact between the two cylinders,
    using an existing diagnostic contact count;
  - if no such diagnostic exists, assert that the initial configuration's overlap is > 0, measured
    before the step with the same overlap query;
  - keep the separation assertion.
- **S9: a known temporary regression between WO-3 and WO-5.** Once WO-3 lands, `ring_mini` at np 4
  and 8 diverges (np 8: dXpos 0.14, max overlap 3.6e6). The per-point Jacobi fallback no longer
  damps the raw cross-rank sum. WO-5, the rank-level mass split, is the designed fix. `contacts`
  must NOT be pushed between WO-3 and WO-5, and WO-5's acceptance must include `ring_mini` at
  np 4 and 8.

## 13. Amendment after WO-4 (2026-09-25, binding; brief `docs/contact_evidence/ARCHITECT_BRIEF_2.md`)

WO-0 to WO-4 are on `contacts` (ca32026). Implementation falsified one premise (A: over-relaxing
the overlap projection), exposed one conservation hole (B: the multilevel coarse cycle at a folded
hub), and found WO-5 under-specified in four places (C). This section resolves them, restates the
interface convergence and the cost model (D), and replaces the texts of WO-5, WO-6 and WO-7. Where
it conflicts with §0–§12 it wins; the superseded places are marked in place. Evidence:
`docs/contact_evidence/IMPL_A.md` (WO-4, Stop A, finding B, open points 1–4) and the model of §13.4.

### 13.0 Decisions at a glance

| # | Decision | Lands in |
|---|---|---|
| A | The overlap projection is never over-relaxed: `ω_pos = 1` on every slot. `kSplitOmegaPosition` and the position relaxation branch are deleted. The PGS normal keeps its hook (`ω_vel = 1`; the 1.5 lever stays legitimate there because that multiplier is accumulated). | WO-4b |
| B | A multilevel coarse vertex carries the mass of the rank's folded copy set, `μ(q) = a(q)·m/k(q)`: the true mass at np 1. "Fold before the coarse cycle, re-seed after" is kept. | WO-4b (np 1), WO-5 (MPI) |
| C1 | The holder mask exists only for X. WO-5 carries no mask; WO-6 adds it with the rank colouring. No reordering. | WO-5, WO-6 |
| C2 | Under Poisson, every velocity-phase forward carries the owner's balance `B` and peak; every copy re-seeds the share `B/k`. No new message. | WO-5 |
| C3 | IMPL_A's phase-end formula is verified. The local fold divides by the local active count `a`; solve masses and orphan shares use the global `k`; the pack multiplies by `a`. | WO-5 |
| C4 | `a` comes from a per-substep activity pass (MPI only). `k` comes from one opening exchange per substep: on the PGS path it rides the existing post-warm-start sync; on the `g = 0` path it is a new counts-only exchange (2 rounds), the one WO-6 needed anyway. | WO-5 |
| D | Interface convergence and cost restated for `ω_pos = 1`; G7e restated; G7f, `hub_static` and `hub_ml` added. | §13.4, §13.6 |
| new | Accumulated (retractable) position PSOR is sound, has a unique fixed point, needs about 3× fewer iterations at np 1, and makes `ω_pos = 1.5` legitimate. It also changes every np 1 run: **user decision R-U4**. WO-12 is specified but conditional. | – |

### 13.1 A: the overlap projection is not over-relaxed

**What was wrong.** §1.3 P5 proves convergence as block coordinate descent on the dual and allows
any relaxation ω ∈ (0, 2). That proof needs an **accumulated multiplier with a retractable
projection**, `λ ← max(0, λ + ω r/w̃)`, so that an overshoot is taken back by a later negative
increment. The PGS normal has that form. The overlap projection does not:
- `PositionContactSweep::solveContact` (`src/solver_position.hpp`) applies `Δλ = −ω C/w̃` only while
  `C < 0`, and never applies a negative increment.
- `posLambdaContact` only records the increments for the friction bound; the update never reads it.

For that form the applicable theory is relaxed POCS (Agmon–Motzkin–Schoenberg, Gubin–Polyak–Raik).
For ω ∈ (0, 2) it converges to *some* feasible point, and for ω > 1 that point lies strictly
inside the feasible set. **The overshoot is permanent.**

**The quantity that decides it.** Take one update of contact `c` and hold every other update
fixed. After reconciliation, its own constraint moves by `ω_eff·(−C_c)`, where

    ω_eff = ω · w_c / w̃_c,   w_c = Σ_ends J M⁻¹ Jᵀ with true masses,   w̃_c = the same with k × the inverse masses.

- **ω = 1.** Since `k ≥ 1`, `ω_eff = w_c/w̃_c ∈ (0, 1]`. A contact's own update moves it toward
  its boundary, never past it. Gaps come only from other contacts' pushes (the coupling), exactly
  as in serial POCS.
  - demStep's periodic twins (`i–j′` and `j–i′`, `k = 2` at every end) are two copies of one
    constraint. Each contributes 1/2, together exactly 1.
- **ω = 1.5.** `ω_eff > 1` whenever `w̃_c < 1.5 w_c`:
  - **The twins:** 2 × 0.75 = 1.5, so an isolated wrap pair ends 1.5 δ apart (Stop A: 0.150
    against 0.100).
  - **A light unsplit body against a heavy split one** (`w̃ ≈ w`): every leaf of a hub is pushed
    1.5 δ and left 0.5 δ clear. The "s = 4, ω = 1.5: 2 iterations" row of A.4 was this
    over-separation, not faster convergence.
  - **At rank faces:** any mix of the two.
- **Physical harm.** Position corrections do not feed velocities in dem (`finalCommitKokkos`
  commits `posPred`), so the over-separation is a pure displacement.
  - Under gravity it lifts grains, which is potential energy from nothing.
  - It makes np 1 and np N disagree (`validate_periodic`).

**Decision.**
- `ω_pos = 1` on every slot.
  - Delete `kSplitOmegaPosition` (`src/solve_copies.hpp`) and the `ov.relax(...)` / `ov.omega`
    branch in `PositionContactSweep::solveContact`.
  - The position phase's `SlotOverride` carries only its slot overrides.
- The velocity hook stays: `kSplitOmegaVelocity = 1`.
  - R-F2's lever (1.5 on split PGS-normal edges) remains legitimate, because that multiplier is
    accumulated and clamped.
  - The cone and the Poisson release are never relaxed.
- **P5 restated for the overlap projection.** The M iteration is cyclic projection in the split
  metric `M̃`, onto two kinds of set:
  - the half-spaces `{C_c ≥ 0}`: each update is an exact `M̃`-projection at ω = 1. Copies on
    different ranks are disjoint variables, so concurrent updates are sequential ones;
  - the consensus subspaces: the local fold and the rank reconciliation are exact
    `M̃`-projections, because equal copy masses give the arithmetic mean.

  Cyclic projection onto finitely many half-spaces and subspaces with a non-empty intersection
  converges to a point of the intersection. It is Fejér-monotone with respect to every feasible
  point. Its fixed-point set is the feasible set, and which point it reaches depends on the order,
  as for serial POCS. §1.3 P5 as written remains valid for the accumulated forms: the PGS family,
  and position PSOR if R-U4 adopts it.

**Rejected.**

| Alternative | Why rejected |
|---|---|
| ω = 1.5 only at hubs | The hub is exactly the light-against-heavy case, the worst place for it. |
| Keep 1.5 and change the three tests | The in-box pair's answer (exact contact) is not a tolerance question. |
| Retractable PSOR on split contacts only, ω = 1.5 there, POCS elsewhere (np 1 stays bitwise) | Converges in the model with the ω = 1.5 counts, but it has three flaws. It mixes a projection-finding and a feasibility-finding method, with no convergence proof for the mix. Its fixed point is mixed: split contacts are complementary, the rest only feasible. It needs a second stop metric, because retractions are invisible to `maxOverlap`: the wrap pair would stop at iteration 2 with its gap open. |
| Copy masses weighted by the edge count at each copy (`θ_q ∝ d_q`, `Σθ = 1`; conservative for any θ) | 1.2–1.5× more iterations at np 2 and no better at np 8 (§13.4). This answers "should `k` be exact per unit or contact": `k` is already exact per body (active copies), and a finer weighting does not help. |
| Accumulated PSOR on every contact, ω = 1.5 | The principled form, and faster (§13.4), but it changes every np 1 run. R-U4. |

**Cost.**
- At rank faces the overlap projection needs 1.0–2.4× the serial iterations at np 2 and 1.6–3.2×
  at np 8 (43 to 729 bodies per rank; §13.4). The withdrawn estimate of 1.04–1.2× was bought with
  over-separation.
- At hubs, ω = 1 costs nothing measurable: A.4's split hubs converge in 3 iterations, as serial
  does.

### 13.2 B: the coarse vertex mass

**The hole.** WO-4 passes the solve view (`k × invMass`) to `buildContactHierarchyKokkos` and
`multilevelCoarseCycleKokkos`. Take a hub with `s` local copies at np 1 (`k = s`). It is folded
before the coarse cycle, so its base stands for all `s` copies.
- The base enters its group with mass `m/s`, and a coarse impulse moves the group by `dV`.
- Prolongation adds `dV` to the base; the re-seed copies it to the other `s − 1` copies; the next
  fold (÷ `s` over `s` copies, each `+dV`) keeps `dV`.
- The body's momentum therefore changes by `m dV`, against the `(m/s) dV` the coarse problem
  accounted for. The error is `(1 − 1/s) m dV` per cycle.

**The rule.** After the fold that precedes the coarse cycle, the `a(q)` active local copies at
vertex `q` are equal and move together until the next fold. They are **one** coarse vertex, with
the mass they carry together:

    μ(q) = a(q) · m_q / k(q),     invMassCoarse(q) = invMass(q) · ( float(k(q)) / float(max(1, a(q))) )

- Compute the ratio first. It is then exactly 1.0f when `k = a`, and `invMassCoarse` equals
  `invMass` bit for bit.
- `invMassCoarse` replaces `invMassVel` in the hierarchy build (group masses) and in the coarse
  cycle (restriction weights and group inverse masses), on both the Kokkos path and the fused CUDA
  path.
- The sequence is unchanged: fine sweep → fold → coarse cycle → re-seed.

What the rule gives in each case:
- **np 1:** `a = k = s` at a hub base and 1 elsewhere, so `μ = m`. The coarse cycle uses
  `P.invMass`.
- **np ≥ 2, a rank-split vertex without hub copies:** `a = 1`, so `μ = m/k`. §3's statement holds
  for exactly this case.
- **`a(q) = 0`** (a non-canonical ghost image, §6.2): such a vertex never joins a group with a
  coarse edge. An eligible manifold is coloured, so the copies at its ends are active. `max(1, a)`
  only keeps a singleton's restriction finite. A debug build asserts that no multi-member group
  holds a vertex with `a = 0`.

**Proof (linear momentum).**
1. Vertex `(i, r)` has `a` local copies, each carrying `m_i/k_i` in the split system. Its
   split-system mass is therefore `μ`.
2. The coarse cycle restricts with weights `μ`, applies equal and opposite impulses between groups
   (walls are external), and prolongs `dV_g` to every member vertex.
3. The re-seed writes `dV_g` into all `a` local copies. The split-system momentum change of vertex
   `(i, r)` is then `a (m_i/k_i) dV_g = μ dV_g`, and summed over `g` it is `M_g dV_g`: exactly the
   coarse impulses on `g`.
4. The fold and the rank reconciliation map split-system momentum to true momentum exactly
   (§1.3 P1: `m_i Δv_i = Σ_q (m_i/k_i) Δṽ_q`).
5. Hence `Σ_i m_i Δv_i` equals the coarse wall impulses. ∎

The shared accumulator `lambdaAcc` receives the coarse impulse, which is the true impulse because
the vertex masses are the split-system masses. The ledger therefore stays the true λ (§6.3).

**Rejected: fold after the coarse cycle instead of re-seeding** (IMPL_A's first option).
- It is also conservative: vertex mass `m/k`, prolongation to the base only, then the fold dilutes
  by `a`.
- But the coarse problem then sees a fully local heavy hub at `1/s` of its mass. The coarse
  transport exists to use a supported column's true inertia, so this is wrong by a factor of `s`
  at np 1.

**Gate scene `hub_ml`** (new mode in `tests/kokkos_mpi/test_momentum_mpi.cpp`):
- **The hub** is `makeHub(10)`'s hub, placed as the **last** body (highest gid). At np ≥ 2 every
  leaf–hub contact is then owned by the leaf's owner, and the hub is mass-split across ranks. Its
  mass is set explicitly to `scale³` × a leaf's (equal density).
- **The leaves** are only those with `dir.z < 0`, about 150, which gives `s = 5` velocity copies
  at np 1. They carry `makeHub`'s common drift and a radial approach speed
  `0.3 (1 + 0.1 N(0, 1))`.
  - This speed lies between `vRestS = 2 g dt = 0.2` and `qsThr = 8 g dt = 0.8` at `g = 10`,
    `dt = 1e-2`.
  - The leaves are therefore quasi-static and eligible, yet the residual still triggers the pass.
- **Settings:** `g = (0, 0, −10)`, frictionless, stabilization `multilevel`, velocity iterations
  1, default position iterations, 10 steps.
- **Positive controls**, asserted by the mode (it fails otherwise, because it would not test what
  it claims), each in at least one substep:
  - velocity hub copies > 0 (np 1);
  - the multilevel pass built ≥ 1 level;
  - `mlHubAggregated ≥ 1`, a new `split_stats` counter (max over the step's substeps): the hub's
    base at np 1, or the hub's slot at np ≥ 2, sits in a level-1 group with ≥ 2 members.
- **Discrimination:** on ca32026 plus the new test, np 1 must give `dP > 1e-4`. Otherwise apply
  R-F8.

### 13.3 C: rank-level M, complete

**Per-slot quantities.** Phase `p` is vel or pos. Both quantities are recomputed every substep, and
nothing crosses substeps.
- `a_p(q)`: the number of local copy slots of vertex `q` that are an endpoint, through the
  overrides, of an owned edge with colour ≥ 0 in phase `p`'s final colouring.
  - A non-hub vertex has `a ∈ {0, 1}`.
  - A hub base has `a = s`, because every copy carries at least ⌊d/s⌋ ≥ 16 edges. This is WO-4's
    `groupK`.
  - Velocity vertices are canonical slots (§6.2), so a non-canonical ghost image has `a = 0`.
    Position vertices are raw slots, so every image slot is its own copy.
- `k_p(q) = max(1, a_p(owner row) + Σ a_p over every ghost slot of the body on every rank)`. It is
  identical on every slot of the body (it is forwarded) and on its hub copies.

| | solve multiplier κ (`invMassSolve = κ·invMass`) | local-fold divisor | pack factor `f` | owner apply |
|---|---|---|---|---|
| M phases (PGS family, stabilization sweeps, overlap projection) | `k` | `a` | `a` | `k ≤ 1`: `x += Σ inc` (c771e07, bitwise); `k > 1`: `x = seed + (a_own (x − seed) + Σ inc)/k` |
| X phase (`g = 0` one-shot, WO-6) | `a` (local hub copies only) | `a` | 1 | `x += Σ inc` |
| raw syncs (legacy friction, Jacobi) | 1 | – | 1 | `x += Σ inc` |
| multilevel coarse vertex | `invMassCoarse = invMass·k/a` (§13.2) | – | – | – |

The local fold divides by `a`, not by `k`:
- It is the exact `M̃`-projection onto "the `a` local copies are equal". The remote copies'
  increments are unknown until the sync, and treating them as zero would not be a projection.
- The pack `a·(x − baseline)` then hands the rank's summed increment `T` to the owner.
- This is WO-4's fold unchanged (`groupK = a`). Under MPI only the solve multiplier and the orphan
  share change: both use the global `k`.

**Orphan accounts** (Poisson, M velocity phase). This verifies IMPL_A open point 3.
- **Shares.** Every copy slot of a body with `k_vel > 1` holds `orphan = B/k`; with `k = 1` it
  holds `B`. `B` is the owner's balance, forwarded. The baseline and the seed record the share,
  and the owner keeps `B` in `ownerSeedOrphan_`.
- **Local fold.** The mean of the accounts over the active local copies (÷ `a`), which preserves
  the local total. WO-4 already does this with `groupK`.
- **Pack.** `inc = a(g)·(orphan_g − share_g)`.
- **Apply.** `B_new = max(0, B + a_own (orphan_own − B/k) + Σ inc)`; the peak takes the max.
- **Forward.** `B_new` and the peak. Every copy re-seeds its share `B_new/k` and its baseline.
- **Phase end.** Owner row `:= B_seed + a_own (orphan_own − B_seed/k)` (IMPL_A's formula).
  - Every distributed M velocity phase ends with a sync, after which the formula returns `B_seed`
    exactly.
  - At np 1 there is no sync and `a = k = s`, so it returns `s × share`, which is
    `unfoldOrphanKokkos`'s sum. One formula serves both.
  - `unfoldOrphanKokkos` is therefore no longer called before a rank sync.
  - Ghost accounts are dead after the phase: the next `gather` forwards `B` again.
- **No overdraw, by construction.** The accessible total is `Σ_copies share = k·B/k = B`. This
  closes c771e07's R5.
  - A debug build asserts that the pre-clamp `B_new ≥ −1e-6·max(B, 1e-30)`.
  - `split_stats.orphanClamps` counts clamp hits and must stay 0.

**The substep under MPI, in order** (PGS path):
1. `gather`. `beginSolve` marks the velocity baselines, orphan included (unchanged).
2. Both colourings at the top of `demSolveContacts`, with hub copies on failure (WO-3/WO-4). Then
   the **activity pass**: `a_vel` and `a_pos` per slot (MPI only).
3. Orphan decay and scatter (unchanged; they write owned and ghost slots).
4. `computeVn0`, and the warm start on true masses (unchanged).
5. **Opening** (`openVelocityPhase`, replacing the post-warm-start `syncVelocities`):
   1. Reverse `OpeningIncrement`: the warm-start `v`, `w` and the orphan increments, raw (they are
      known impulses), plus `aVel` and `aPos`.
   2. Owner apply: the raw add, then `k_vel = max(1, a_vel(own) + Σ aVel)`, and `k_pos` likewise.
   3. Forward `OpeningState`: `v`, `w`, `B`, the peak, `kVel`, `kPos`. Ghosts store them.
   4. Mark the ghost baselines and the owner seeds (`v`, `w`, `B`), and set the orphan shares.
   5. Build the velocity solve views with `k_vel`, and `invMassCoarse`.
   6. Seed the local hub copies: state, shares `B/k`, and `σ` = the base's value.
6. **Velocity loop.** Each iteration: sweep, then the local fold (÷ `a`). At every sync point:
   1. pack `a·(x − baseline)` for `v`, `w` and the orphan;
   2. weighted apply;
   3. forward `VelocityState`, plus `B` and the peak iff Poisson;
   4. re-mark the baselines, owner seeds and shares;
   5. re-mark the local groups' `σ`, and re-seed their copies from the bases.

   The final `syncVel` follows as before.
7. Stabilization passes (same sync rule); the multilevel pass takes `invMassCoarse`.
8. Phase end: restore the owner's orphan balance (Poisson).
9. Integrate and predict, then `publishPositions` (forward only).
   - Mark the position baselines and the owner position seeds.
   - Seed the position hub copies.
   - Build the position solve views with `k_pos`.
10. **Position loop.** Each iteration: sweep (ω = 1), then the local fold (÷ `a_pos`). At every
    sync: `syncPositions` with the weighted apply, then re-mark. The final sync follows as before.

**The `g = 0` path** has no warm start and no Poisson (the orphan machinery is PGS-only).
- Its opening is a **new counts-only exchange** at the same place, before the first one-shot
  sweep: reverse `aPos`, forward `kPos`.
- In WO-5 its velocity phase stays as WO-4 left it: c771e07 raw across ranks, with local hubs at
  κ = `a`. WO-6 turns it into X and adds the mask to this exchange.
- Under Jacobi (`velocityUseGS = false`) no opening runs (unchanged).

**np 1 closed, and any rank with `!exchanges()`,** stop after step 0 of §4.6
(`k = max(1, a(own))`). That reproduces WO-4's `groupK`, so they are byte-identical.

**Payloads.** All are POD and sent as `MPI_BYTE`. `operator+` sums floats and ints, takes the max
of the peak, and ORs the mask.

| exchange | payload | B per ghost: c771e07 → WO-5 (→ WO-6) |
|---|---|---|
| PGS opening, reverse | `OpeningIncrement {float v[3], w[3], orphan, orphanPeak; int aVel, aPos;}` | 32 → 40 |
| PGS opening, forward | `OpeningState {float v[3], w[3], B, peak; int kVel, kPos;}`; without rotation, drop `w` | 24 → 40 (no rotation: 12 → 28) |
| `g = 0` opening, reverse | `{int aPos;}`; WO-6: `{uint64 velMask; int aPos; int pad;}` | 0 → 4 (→ 16) |
| `g = 0` opening, forward | `{int kPos;}`; WO-6: `{uint64 velMask; int kPos; int pad;}` | 0 → 4 (→ 16) |
| later velocity forwards | `VelocityState`, plus `{float B, peak}` **iff Poisson** | 24 → 24 (32 with Poisson) |
| every reverse after the opening; position exchanges | unchanged | – |

**Messages.**
- The PGS path gets no new round.
- The `g = 0` path gets +2 rounds for the counts exchange. WO-6 needs this exchange anyway, and
  §7.1 already charged it there; it now appears at WO-5.
- The orphan balance rides existing forwards.

**Activity and `k`: computation and cost** (C4).
- **Activity pass** (per phase):
  - a memset of a per-slot byte `hit` over `[0, numParticles + nCopies)`;
  - one kernel over the owned edges with colour ≥ 0 of the final colouring. It resolves both ends
    through the overrides and stores `hit(slot) = 1` (a same-value store; use
    `Kokkos::atomic_store`).
  - Then `a(q) = hit(q)` for a vertex outside any group, and `groupK` for a hub group. `groupK` is
    computed from `groupActive`, which this pass also fills.
- `k` needs no kernel of its own: the owner computes it in the opening apply, and the ghosts
  receive it in the forward.
- **Cost under MPI.** Two launches and O(edges + slots) work per phase per substep. The solve-view
  build (one launch per phase) already exists; `invMassCoarse` is one launch, and only when the
  multilevel pass runs.
- **Single rank.** The pass runs only where WO-4 already builds groups, so it is unchanged and
  costs nothing.
- **Memory.** `aVel`, `aPos`, `kVel`, `kPos` (`int32` per slot, grow-only: 16 B per slot), plus
  `hit` (1 B) and `invMassCoarse` (4 B).

**C1.** `m_vel` and `col(rank)` exist only for X. WO-5 computes `a_vel` on both paths (the mask
will need it) but carries no mask. WO-6 adds the rank colouring and the mask field.

### 13.4 D: interface convergence and cost, restated

**Model.** A throwaway model, not committed, with Appendix A's conventions:
- **Scene:** `PositionContactSweep`'s translation-only projection on spheres, jittered cubic
  lattices, radii ±10 %.
  - "Loose": spacing 1.04 D, jitter ±0.10 D.
  - "Compressed": spacing 1.00 D, jitter ±0.05 D. This is a jammed crystal, the worst case.
  - Initial max overlap 0.16–0.24 R; two seeds per size.
- **Colouring:** greedy per rank, in a random key order.
- **Ownership:** each contact belongs to the lower-gid body's rank, with random gids.
- **Ranks:** np 2 = x-slab, np 8 = octants.
- **Iteration:** one consensus (sync) per iteration.
- **Stop:** max overlap < 1e-4 R, measured on the reconciled state.

| scene | N (per rank at np 8) | POCS ω 1: np 1 / np 2 / np 8 | np 8 ÷ np 1 | POCS ω 1.5, np 8 (unsound) | PSOR ω 1.5: np 1 / np 8 |
|---|---|---|---|---|---|
| loose | 343 (43) | 24–27 / 39–43 / 52–53 | 1.9–2.2 | – | 13 / 27–31 |
| loose | 1728 (216) | 40–62 / 67–71 / 71–124 | 1.8–2.0 | – | 14–18 / 42–69 |
| loose | 5832 (729) | 50–63 / 63–71 / 80–128 | 1.6–2.0 | – | 14–18 / 40–70 |
| compressed | 343 (43) | 40–43 / 73–103 / 118–127 | 2.7–3.2 | 73–79 | 14 / 69–80 |
| compressed | 1728 (216) | 118–125 / 189–197 / 291–301 | 2.4–2.5 | 151–168 | 38–42 / 152–164 |
| compressed | 5832 (729) | 248–249 / 335–>400 / >400 | > 1.6 | 276–279 | 88 / 244–267 |

- **np 2 ÷ np 1** (POCS, ω 1): loose 1.0–1.8, compressed 1.35–2.4.
- **Correction norm `Σ m|Δx|²` against the QP optimum** (the least-displacement feasible point):
  POCS 1.002–1.013 at every np and ω; PSOR 1.000.
- **Fixed budget of 8 iterations** (the perf scenes' position budget), loose scene, residual as a
  fraction of the initial max overlap: np 1 0.02–0.12; np 8 0.11–0.23 (1.5–9× np 1); ω 1.5
  (unsound) 0.06–0.13.
- **Edge-count-weighted copy masses**, compressed: np 2 111–131 against 73–103 uniform; np 8
  137–161 against 118–144.
- **Velocity phase** (A.3, unchanged, `ω_vel = 1`): np 8 needs 54 / 93 / 108 iterations against
  serial 25 / 49 / 93, at 43 / 216 / 729 bodies per rank.

Readings:
1. **Rank faces at ω = 1.** A face costs the position phase about what it costs the PGS velocity
   phase: about 2× at np 8, up to 3× in a jammed crystal. At a small fixed budget this shows as
   larger residual overlaps at interfaces. G7e's 1.5 × np 1 bound was derived from the withdrawn
   estimate and is restated (§13.6).
2. **PSOR.** Accumulated PSOR at ω 1.5 needs 2.8–3.5× fewer iterations than today's POCS at np 1,
   and at np 8 about today's np 1 count. It also lands on the least-displacement point at every np.
   This is the evidence behind R-U4.

**Messages per substep** (maximum, no early stop; perf scenes: 4 velocity and 8 position
iterations, `sync_every = 1`):

| | c771e07 | WO-5 | WO-6 | WO-7 |
|---|---|---|---|---|
| gather | 3 | 3 | 3 | 3 |
| opening, PGS / gas | 2 / 0 | 2 / 2 | 2 / 2 | 2 / 2 |
| velocity iterations + final | 10 | 10 | 10 | 10 |
| legacy friction (gas) | 4 | 4 | 4 | 4 |
| position publish, iterations + final | 19 | 19 | 19 | 19 |
| **total, pgs / gas** | 34 / 36 | 34 / 38 | 34 / 38 | 34 / 38 |
| Allreduce | §7.1 | same | same | fused vote (§5.1) |

**Expected ms/step against c771e07** (np 8 × 2):
- **`perf_pgs` at WO-5:** +1–3 %. This covers about 6 activity and solve-view launches, 8–16 B
  more per ghost on the opening, and the weighted apply, which replaces the apply kernel.
- **`perf_gas` at WO-5:** +2–5 %, the above plus the 2 rounds (≈ 40–80 µs).
- **Both:** if the early stop fires later at interfaces under ω = 1, up to +3 position iterations
  (≈ 60 µs each) until the budget of 8 caps it (R-F6).
- **WO-7:** adds the ghosts of the drift slack, +2–3 %.
- **Total:** +3–8 %, inside G12's 10 % with less margin than §7.5 claimed.
- **CUDA single-GPU spheres:** unchanged, since none of this runs without hubs.
- **CUDA hub and ring runs:** ω 1 against 1.5 is within noise (A.4: 3 against 2 iterations, and
  the 2 was over-separation).

### 13.5 Work orders (replacing §8 WO-5, WO-6, WO-7; adding WO-4b and the conditional WO-12)

**Dependency.** WO-4 → **WO-4b** → WO-5 → WO-6 → WO-7 → WO-9 → WO-10 → WO-11.
- WO-8 is unchanged.
- WO-12 runs only if R-U4 = yes, after WO-7 and before WO-10.
- S9 stands: no push before WO-5 lands.

#### WO-4b: fix A and B on the committed code (np 1)
A new commit on top of ca32026; do not amend it.

Files:
- `src/solve_copies.hpp`: delete `kSplitOmegaPosition`, and correct the comment's convergence
  claim to §13.1.
- `src/solver_position.hpp`: remove the relaxation branch of `solveContact`.
- `src/solve_driver.hpp`:
  - The position `SlotOverride` loses its omega.
  - Both multilevel calls (hierarchy build and coarse cycle) take `P.invMass`. On a single rank
    `a = k`, so `invMassCoarse ≡ invMass`. This is also right for WO-4's interim np ≥ 2 local hubs.
- `src/particles.hpp`, `src/sim.hpp`, bindings: `split_stats` gains `mlHubAggregated`,
  `velItersUsed` and `posItersUsed` (last substep; tier 2).
- `tests/kokkos_mpi/test_momentum_mpi.cpp`:
  - the modes `hub_static` and `hub_ml` (§13.2, §13.6);
  - a `--pos-iters=N` flag;
  - an `ITERS vel= pos=` line;
  - the leaf-gap metric.
- `tests/kokkos_mpi/CMakeLists.txt`: ctests for `hub_static` and `hub_ml`, gated at np 1 and
  report-only at np 2 and 4 until WO-5.

Acceptance:
1. **The three Stop-A tests pass:** `test_wrap_pair_matches_in_box_pair`, and
   `python_mpi_validate_periodic_np2` / `_np4` (0.800 / 0.800, straddlers ≤ 1e-5).
2. **`hub_static` at np 1** (`step_mpi` and `--solo`; OMP 1, OMP 8 × 3; CUDA × 3):
   - max leaf gap ≤ 0.05 δ;
   - residual overlap ≤ 1e-4 R;
   - dXpos ≤ 3e-5 R;
   - velocities unchanged (absolute `|P| = 0`).

   Discrimination, run once: the same mode on ca32026 must give a max gap ≥ 0.3 δ (record it).
   Otherwise stop: the mode is too weak.
3. **`hub_ml` at np 1** (`step_mpi` and `--solo`; OMP 1, OMP 8 × 3; CUDA × 3): dP ≤ 5e-6, and
   every positive control holds. Discrimination, run once: ca32026 plus the test gives
   dP > 1e-4 (R-F8 otherwise).
4. **WO-4's gates re-run** for `hub`, `hub_posonly`, `hub_pgs`: dP ≤ 1e-6, dXpos ≤ 3e-5 R,
   CONFLICTS 0, on host and CUDA.
5. **`cluster_periodic --solo`:** CoM drift ≤ 1e-5 R.
6. **Byte-identical to ca32026** at np 1, OMP 1: every non-hub closed dump (plus np 4 `cluster`
   and `cluster_pgs`), the 5 S6 wall scenes, and `ring_mini`. Named changes: `hub*`,
   `cluster_periodic --solo`, and multilevel runs with velocity hub copies.
7. **Battery:** all pass (195 plus the new ctests).

#### WO-5: rank-level M (projection phases)
Files: `src/mpi_halo.hpp`, `src/step_solve_mpi.hpp`, `src/solve_driver.hpp`,
`src/solve_copies.hpp`, `src/particles.hpp`, `src/sim.hpp` (`orphanClamps`).
1. **Implement §13.3:**
   - the activity pass;
   - `openVelocityPhase` on both paths, with §13.3's payloads;
   - the `k` views and the owner seeds;
   - the weighted apply for `k > 1`, with the raw form kept for `k ≤ 1`;
   - the `a`-weighted pack;
   - the orphan shares, forwards and phase-end restore;
   - `invMassCoarse`;
   - solve views with a per-slot `k` (hub copies take their base's `k`).
2. **Implement §6.2:** `ghostCanon_` and the velocity slot map, set in `demStepMpi` and not in
   `gather`. Add the MPI twin-dedup switch (§4.2 item 2).
3. **Hooks:** `openVelocityPhase(P)` (on Solo it does step 0 only) and `rankK(…)`, which exposes
   `kVel` / `kPos` / `aVel` / `aPos` to the solve-view builder.
4. **Debug-build checks:**
   - every slot with `a = 0` packs a zero increment;
   - no multi-member coarse group holds a vertex with `a = 0`;
   - the orphan clamp never binds.

Acceptance. The np set is np 1, 2, 4, 8 × OMP 1, 8 × 3; CUDA is np 1 `step_mpi` and `--solo` × 3.
1. **Conservation matrix (G1)** on the np set plus CUDA, with the rows of §9 and §13.6:
   `cluster_pgs`, `cluster_poisson`, `cluster_{multilevel,escalate,ordered}`, `cluster_posonly`,
   `hub_pgs`, `hub_posonly`, `hub_static`, `hub_ml`, `ring_mini`.
2. **`ring_mini` at np 2, 4, 8** (closes S9):
   - dP ≤ 1e-6;
   - dXpos ≤ 1e-5 R;
   - CONFLICTS 0;
   - ovl (max over steps) ≤ max(10 × ovl_np1, 1e-3 R), and ≤ 0.1 R at every step.

   Report `ovl_npN / ovl_np1`.
3. **Hub modes at np 2, 4, 8:**
   - `hub_static`: max leaf gap ≤ 0.05 δ and residual ≤ 1e-4 R;
   - `hub_ml`: the positive controls (`mlHubAggregated ≥ 1`) and dP ≤ 5e-6.
4. **G3:** every np 1 closed dump byte-identical to WO-4b (`step_mpi` and `--solo`). Named change,
   reported before and after: `cluster_periodic` at np 1 under `step_mpi` (§6.2 and the
   self-image position consensus).
5. **G4:** at np 4 and 8, OMP 1, 5 runs byte-identical for `cluster_pgs`, `cluster_poisson`,
   `cluster_posonly`, `hub_pgs`, `hub_ml`, `ring_mini`.
6. **G7a** (`tri_pgs`), **G7c**, **G7e** (restated) and **G7f** (new): §13.6.
7. **Orphan accounts:** `orphanClamps = 0` in `cluster_poisson` at every np.
8. **Periodic regressions:** the Stop-A tests and `validate_periodic` at np 2 and 4 still pass
   under rank-level M.
9. **Performance:** `perf_pgs` and `perf_gas` at 8 × 2 and 4 × 4, 5 interleaved repeats against
   `build_base`, with the load recorded. Report the medians. Stop if a median ratio exceeds 1.10,
   because later work orders only add cost.

#### WO-6: rank-level X (the `g = 0` one-shot)
Files: as §8 WO-6.
1. **Rank colouring** `col(r)` per §1.4, recomputed on a decomposition or band change. `C ≤ 64`,
   else throw.
2. **Mask.** `m_vel(q) = a_vel(q) > 0 ? 1ull << col(rank) : 0` at canonical velocity slots.
   - The `g = 0` opening becomes `{uint64 velMask; int aPos; int pad;}` in both directions: the
     reverse ORs the masks and sums the counts; the forward carries the owner's OR and `kPos`.
   - Ghosts store `velMask` per slot.
   - No new round.
3. **Gate.** The gate kernel, `solveEpoch` and the interval index per §4.6 "X gate". The one-shot
   lambda takes `gate`.
4. **Local hub copies under X** keep κ = `a`, fold ÷ `a` and `f = 1` (§13.3 table). That is WO-4's
   behaviour, now stated in code comments; the numerics are unchanged.

Acceptance:
1. §8 WO-6's list.
2. **Inertness:** at np 1, 2, 4, 8, OMP 1, every `g ≠ 0` mode and every position-only mode
   (`cluster_posonly`, `hub_posonly`, `hub_static`) is byte-identical to WO-5. Only the `g = 0`
   velocity phase changes.
3. **G4** for `cluster`, `tri` and `hub` at np 4 and 8.
4. **Report:** the bytes per ghost on the `g = 0` opening (16 B each way), and
   `unfiredSplitContacts` per substep in `perf_gas`.

#### WO-7: drift vote, `migrateToBlocks` and the band (D4a), XPBD and Hertz
Files and items as §8 WO-7, with two additions:
- A vote-triggered migration or a band change alters the ghost sets, and with them `a` and `k`.
  Both are recomputed every substep, and nothing of §13.3 crosses substeps. `MigratePack`
  therefore carries no new state beyond `extForce` / `extTorque` (§5.2) and the orphan balance it
  already carries.
- The migration runs at the top of `demStepMpi`, before `gather`. It can never fall between an
  opening and its phase.

Acceptance:
1. §8 WO-7's list.
2. The WO-5 conservation matrix and G4, re-run once on the final band. The band change alters the
   ghosts and therefore `k`: a changed number is expected, a failed gate is not.
3. The G5 oracle on its three scenes.
4. **Report:** migrations per 100 substeps and the time of one migrate (R-F1).

#### WO-12 (conditional on R-U4 = yes): accumulated position PSOR
Files: `src/solver_position.hpp`, `src/solve_driver.hpp` (stop metric), tests.
- **Update.** Per contact of a unit, in order:
  - `Λ = posLambdaContact(c)` (already zeroed per substep);
  - `Λ′ = max(0, Λ − ω_pos C / w̃)`, and `d = Λ′ − Λ`;
  - apply `d`, which may be negative, then store `Λ′`. A plain read-modify-write is safe: the
    unit's work item owns the contact.
  - `ω_pos = kPositionOmega = 1.5` on every contact, np 1 included.
- **Stop.** The loop stops on `r_pos = max |d|·w̃` at `posTol`. This is the position change,
  retractions included, Allreduce-MAXed as today. `max_overlap` keeps its meaning: the largest
  violation seen.
- **Ledger.** `posLambdaContact` becomes the net position impulse. The friction bound reads it
  unchanged.
- **Named change.** Every run with coupled contacts at np 1. Re-baseline the G3 references once,
  with before/after statistics.
- **Acceptance:**
  1. `cluster_posonly --steps=1` with a large cap converges at every np.
  2. np agreement: `‖x_npN − x_np1‖∞ ≤ 1e-4 R`. This is a new, strong G7 gate, possible because the
     fixed point is unique.
  3. KKT residual ≤ 1e-4 R.
  4. `tests/python` (packing, drum, statics) passes within its bands.
  5. G1 for every mode.
  6. Iteration counts at ω 1.3 / 1.5 / 1.7 reported, and the constant chosen from them.
  7. Performance not worse than WO-7.

### 13.6 Gates, revised

**G1 additions.**

| mode | dP | dX | dXpos | dLvel |
|---|---|---|---|---|
| `hub_ml` | 5e-6 | 3e-5 | 3e-5 | report (the coarse cycle is translation-only by design) |
| `hub_static` | velocities unchanged (absolute \|P\| = 0; Σ m\|v\| = 0, so no normalized dP) | – | 3e-5 | – |

**`hub_static`** (new mode).
- **Scene:** `makeHub(10)`'s geometry, with the hub **last** (highest gid) and its mass set to
  `scale³` leaf masses. Leaf scale 1. Shell radius `r_h + R − 0.04 R`, so every leaf overlaps the
  hub by δ = 0.04 R.
- **Settings:** zero velocities, `g = 0`, velocity iterations 0, position iterations 64, 1 step.
- **Gate:** after the step, the max leaf–hub gap ≤ 0.05 δ and the max residual overlap
  ≤ 1e-4 R.
- **Why the bound discriminates:** coupling gaps are ≈ `k·m_leaf/m_hub·δ` ≤ 0.02 δ for `k ≤ 16`,
  while ω 1.5 gives ≈ 0.5 δ.

**G7e, restated.** ovl (max over steps) at np 2, 4, 8 for `cluster`, `cluster_pgs` and
`cluster_friction`.
- **Hard** (a divergence and bug guard): ≤ max(10 × np 1, 1e-3 R), and ≤ 0.1 R.
- **Report** the ratio. A ratio above 1.5 is evidence for R-U4, not a stop.

**G7f, new: the interface iteration ratio.**
- **Runs:** `cluster_posonly --steps=1 --pos-iters=400` and `cluster_pgs --steps=1
  --vel-iters=400`, at np 2, 4, 8 against np 1, OMP 1.
- **Metric:** the iterations the loop ran (`ITERS`).
- **Hard:** np 2 ≤ 3.5×; np 4 and 8 ≤ 5×. The model gives ≤ 2.4 and ≤ 3.2. A wrong `k` ("ranks
  holding") gives about 7.
- **Report** anything above 2.5×.

**G12.** Unchanged (≤ 1.10); §13.4 restates the expectation.

**G13 additions** (compiled per WO-10):

| mutant `n` | what it changes | must trip |
|---|---|---|
| 6 | the multilevel takes `invMassSolve` (WO-4's behaviour) | `hub_ml` np 1: dP > 1e-4 |
| 7 | `ω_pos = 1.5` on split slots (WO-4's behaviour) | `hub_static` np 1 and np 2: max gap > 0.3 δ; `test_wrap_pair_matches_in_box_pair` fails |

### 13.7 Register entries (the caller commits them) and superseded places

```
### The overlap projection is never over-relaxed: omega_pos = 1
- area: dem
- source: dem/docs/contact_solve_framework.md §13.1; evidence dem/docs/contact_evidence/IMPL_A.md (WO-4 Stop A)
- decided: 2026-09-25
- status: settled
- quote: |
    The overlap projection applies -C/w only while C < 0 and never retracts (non-accumulated
    POCS). Over-relaxing it leaves a permanent gap (omega w/w~ - 1)|C|: omega_pos = 1.5 on
    mass-split slots put an isolated periodic wrap pair 1.5 x its overlap apart and every leaf of a
    split hub 0.5 x its overlap clear. Over-relaxation is legitimate only on an accumulated
    multiplier with a retractable clamp (the PGS normal).
- rejected: omega_pos = 1.5 on split slots (3 periodic tests failed; the faster hub convergence of
  A.4 was over-separation); PSOR on split contacts only (no convergence proof for the mix, a second
  stop metric); edge-count-weighted copy masses (1.2-1.5x more iterations at np 2)
- why: a projection that cannot retract must not overshoot

### A multilevel coarse vertex carries the mass of its folded copies, a*m/k
- area: dem
- source: dem/docs/contact_solve_framework.md §13.2
- decided: 2026-09-25
- status: settled
- quote: After the fold, the a active local copies at a vertex move together; the coarse cycle
    treats them as one vertex of mass a*m/k (the true mass at np 1). Solve-view masses (m/k) at a
    folded hub gained (1 - 1/s) m dV per coarse cycle.
- rejected: solve-view masses (non-conservative at a folded hub); fold after the coarse cycle
  (conservative, but the coarse problem sees a local hub at 1/s of its mass)
```

Open issue for the register (R-U4): "the overlap projection is non-accumulated POCS. Accumulated
PSOR would have a unique least-displacement fixed point and legitimate over-relaxation (model:
2.8–3.5× fewer iterations at np 1). Unchanged pending the user's decision."

**Superseded in place** (each marked "superseded by §13.x"):
- §1.3 P5 (the scope of ω ∈ (0, 2));
- §3, the table rows "Multilevel" and "Overlap projection";
- §4.4, the local fold's orphan text and the multilevel note;
- §4.5, the position relaxation and `kSplitOmegaPosition`;
- §4.6, the opening payloads, step 5, and the reconciliation's orphan baseline;
- §7.1 and §7.5;
- §8 WO-5, WO-6, WO-7 and the dependency graph;
- §9 G7e and G13;
- §11 R-F2 (the position part);
- Appendix A.4, the ω 1.5 row.

### 13.8 Risks and open questions (additions to §11)

| # | Item | Needs | Default |
|---|---|---|---|
| **R-U4** | Adopt accumulated position PSOR (WO-12). Gains: a unique fixed point (the least-displacement correction, np-independent); `ω_pos = 1.5` becomes legitimate; the model needs 2.8–3.5× fewer position iterations at np 1, and at np 8 about today's np 1 count (§13.4). Cost: it changes every np 1 run with coupled contacts, and the position stop metric. | **User preference** (the scope of the np 1 change; solver semantics) | Not in this package: WO-4b → WO-7 proceed with `ω_pos = 1`. **The architect recommends yes** as the next package. WO-12 is specified so it can run unattended once approved. |
| R-F6 | Interface convergence of the position phase at ω = 1 in real scenes. The model gives 1.6–3.2× the iterations at np 8, and 1.5–9× np 1's residual overlap at a fixed budget. | Fact (G7e report, G7f, G12) | Proceed and report. If G7e's ratio exceeds 1.5 or the WO-5 perf median exceeds 1.07, put the numbers to the user together with R-U4. |
| R-F7 | `ring_mini` at np 8 has 3–4 rings per rank, so nearly every ring is split several ways. At ω = 1 it converges, but slowly. | Fact | The WO-5 gate is a divergence guard (≤ 10 × np 1, ≤ 0.1 R); report the ratio. |
| R-F8 | `hub_ml` may not reproduce B on ca32026 (dP ≤ 1e-4) if the fine sweep leaves the coarse cycle too little to do. | Fact | Retry in this order: approach speed 0.5 (still below `qsThr`); hub density 1/8 (mass `scale³/8`). If neither reproduces B, land the fix on the proof, keep `hub_ml` as a conservation gate (its positive controls still assert aggregation), drop mutant 6 from G13, and record why. |
| R-F9 | Cost of the per-substep activity pass and of the larger opening payload on CUDA under MPI. | Fact | Expected ≤ 1 % of a substep; report in G12. |
| R-P6 | A coloured edge that never writes (e.g. `num_points = 0`) still counts its copies as active: `k` one larger, slightly more under-relaxation, conservation intact. | Fact | Accept (the same class as R-P2). |
- **S10: the `hub_ml` scene is the dense shell** (settles WO-4b Stop 1). As §13 specifies it, the
  leaves never touch each other. The contact graph is then a star, the matching merges only the hub
  and one leaf, and the 90 % stall rule rejects the level, so no level is ever built. Use the
  denser shell in which the leaves touch: `ns = 3.0 (rs/R)^2`, N = 181. It builds a level and
  aggregates the hub. Measured: dP 1.5e-2 on ca32026 and 1.7e-7 to 2.1e-7 with fix B, at OMP 1 and
  8, for both step_mpi and `--solo`. The positive controls (levels > 0, `mlHubAggregated` > 0, and
  dP > 1e-4 on ca32026) apply to this scene. The observed s = 3 velocity copies is accepted.
- **S11: the `hub_static` leaf-gap bound is 0.15 δ** (settles WO-4b Stop 2). The ≤ 0.02 δ estimate
  missed one effect: each copy takes about 32 sequential pushes of about 0.01 δ before the fold.
  Their random-walk residual is about 0.06 δ, and a projection that never takes a push back cannot
  remove it. Measured: 0.034 δ at OMP 1, up to 0.061 δ at OMP 8, and up to 0.053 δ on CUDA. The
  bound of 0.15 δ still separates cleanly from the ω = 1.5 defect (0.584 δ), by 3.9×. This residual
  is the same non-retractable-projection limitation that R-U4 (WO-12) would remove.
- **S12: the iteration counters must work in the fused device loop** (G7f needs them). The fused
  loop writes its iteration count to a device scalar. Read it back to the host only when
  diagnostics are enabled, which the test harness does. With diagnostics off there must be no
  extra fence or copy.
