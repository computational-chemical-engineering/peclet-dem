# Momentum-conserving cross-rank contacts in the distributed XPBD step

Design note, 2026-09-25. Branch `momentum`, base dem `c79dea5` (= `e4e17f0` plus the report-only
diagnostic). Brief: `docs/momentum_evidence/ARCHITECT_BRIEF.md`. Evidence: `docs/momentum_evidence/BEFORE.md`.
The note stands on its own; the implementer does not need the brief.

**Decision in one line.** Each contact is **solved by exactly one rank**: the only owner that sees
it or, when both owners see it, the owner of its lower-gid body. Every rank then sweeps only the
contacts it owns, with the same Gauss–Seidel kernels as before. At every existing ghost-sync point,
each ghost's accumulated change is **reverse-accumulated** onto its owner through core's existing
`ParticleHalo<3>::reverse`, and the forward refresh follows. That makes each impulse (velocity
phase) and each correction (position phase) exactly equal and opposite, applied once. The fix needs
no new synchronisation points and no kernel changes to the sweeps. After the fused forwards
(WO-5), the default `forward_rotation=True` configuration needs about the same number of message
rounds as today.

---

## 0. Scope and invariants

**In scope.**
- `demStepMpi` (`src/step_solve_mpi.hpp`) and the parts of `demSolveContacts` (`src/solve_driver.hpp`)
  that its hooks reach.
- The dem particle halo (`src/mpi_halo.hpp`).
- The contact→manifold reduction (`src/contact_preprocessing.hpp`).
- The distributed tests, the docs, and the register text.

**Out of scope.**
- Bitwise reproducibility across threads or on the GPU.
- gamma calibration, flow, and the collocated variable-density issue.
- The single-GPU solver (`demStep`).
- The Hertz–Mindlin engine, which already conserves (§1.3).
- The serial model's own non-conservations: the legacy-friction lever arm (§1.4) and the
  count-averaged Jacobi fallbacks (§4.2). Both are recorded as separate issues.
- Pairs that no rank sees (§9, R4).

**Invariants the design must keep.**

| # | Invariant | How it is kept |
|---|---|---|
| I1 | Single-rank `demStep` bitwise unchanged, all configurations | It never calls the partition or the reverse. The new hook members are no-ops in `SoloSolveHooks`. The new reduction argument defaults to "all owned". `decodeKey`'s mask is the identity on keys below 2^63. |
| I2 | `demStepMpi` at np = 1 on a **closed** domain bitwise unchanged (reference: `c79dea5` + the WO-0 dump option) | With no ghosts, every contact is owned, so the partition is the identity. `exchanges()` is false, so every reverse/forward returns before any work. The visible count equals the owned count. |
| I3 | Linear momentum and CoM conserved to round-off at every np, thread count and `sync_every`. The velocity phase conserves angular momentum whenever the serial model does. | §2.4 |
| I4 | Run-to-run bitwise at `OMP_NUM_THREADS=1`, np 4/8 | Every new kernel is element-wise or a `parallel_scan`. Core's reverse scatter runs in `sendIdx` order, which comes from `std::map` over destination ranks and is deterministic. No arrival-order dependence. |
| I5 | The collective schedule is identical on all ranks (deadlock fixes stay) | Reverse calls happen only inside the existing sync hooks, plus one call at a globally uniform point (legacy friction). The only skip is the existing `exchanges()` predicate. |
| I6 | Ledgers stay gid-keyed; `MigratePack` layout unchanged | §4.1, §4.8 |
| I7 | Same `demSolveContacts` for single-GPU and MPI; the solver stack is unchanged | Only the contact/manifold **lists** and the **hooks** change. |
| I8 | Kokkos device code, `.hpp`/`.cpp`, names say what a thing is | New kernels are namespace-scope free functions, following the existing `halo*` kernel pattern (nvcc forbids `KOKKOS_LAMBDA` in member functions). |

**One consequence to state up front.** `demStepMpi` at np = 1 on a **periodic** domain changes. It
had the same defect through its local periodic self-ghosts, so it is part of the fix (§2.6, Q1).

---

## 1. Root cause, in physical terms

### 1.1 Redundant two-owner solves
Take a contact between body A (owned by rank r) and body B (owned by rank s).
- Today both ranks see the contact. Each solves it by Gauss–Seidel from its **own** current state:
  - r uses fresh v_A and a ghost v_B, stale since the last sync;
  - s uses fresh v_B and a stale ghost v_A.
- r keeps its impulse +J_r on A. s keeps −J_s on B. Each ghost-side half is discarded at the next
  refresh.
- In GS, J_r ≠ J_s whenever either body has any other contact swept earlier in the interval, so
  every such contact injects momentum J_r − J_s.
- The same happens in the position projection with the corrections Δλ_r ≠ Δλ_s, which moves the
  CoM.

This double solve is not confined to cross-rank pairs. A local periodic self-ghost produces two
**twin** manifolds on one rank, (A, B′) and (B, A′), and both are swept. Under MPI `realIndices` is
self-mapped, so the kernels' `realA > realB` twin dedup never fires. np = 1 on a periodic domain
therefore double-solves every wrap pair. So does any np with an undecomposed periodic axis.

### 1.2 One-sided solves of drifted particles (a second, unreported mechanism)
- `step_mpi` migrates ownership only when rebalancing (`rebalance_every`, default 0 = never).
- The halo is **block**-based: a rank receives the particles within `band` of its **block**, not
  within `band` of its particles.
- The band is 2.1 R_max, exactly the maximum contact reach, so there is **zero slack**. Once a
  particle has drifted out of its owner's block, some of its pairs are seen by only one owner:
  - the other owner no longer receives the partner as a ghost;
  - today that seeing owner solves the pair and keeps only its own body's half;
  - the drifted body gets nothing, so the contact is one-sided.
- The BEFORE cluster drifts about 1.5 R in 50 steps, so this mechanism contributes to the measured
  defect.
- A rule that assigns every pair to the lower-gid owner alone would **drop** these pairs entirely.
  That is conservative but lets the bodies interpenetrate. The ownership rule below is built to
  handle this case.

### 1.3 Why Hertz–Mindlin already conserves (confirmed, left alone)
`demStepHertzMpi` is explicit:
- Each step forwards the full state (x, v, ω, q) owner→ghost before any force evaluation.
- Both owners evaluate the pair force from bit-identical inputs. The only difference is argument
  order, since slot order differs, so F_BA = −F_AB to round-off.
- The two gid-keyed Mindlin ξ copies are updated by the same deterministic function of the same
  inputs, so they stay equal.
- There is no sweep order, and the skin/rebuild decisions are Allreduced, so both ranks hold the
  same pair list.

Measured: dP about 2e-8 at every np and thread count, identical to np = 1. The redundant evaluation
is correct there **because the evaluation is a pure function of a state that both ranks hold
identically**. That is exactly the property the GS sweeps lack.

### 1.4 The friction serial floor (dLvel = 1.9e-5 at np = 1, `cluster_friction`) is real
`solveContactFrictionKokkos` applies:
- +J_t at A's surface point p_A = x_A + r_A;
- −J_t at B's surface point p_B = x_B + r_B.

For two spheres with overlap δ, p_A − p_B = δ n. The pass therefore changes the total angular
momentum by ΔL = (p_A − p_B) × J_t = δ n × J_t ≠ 0, because J_t ⊥ n.

This is a couple of first order in the overlap: a discretisation error of the legacy pass, not
round-off. The PGS friction cone measures 1.9e-8 and is effectively conserving.

The fix changes np = 1 results, so it is **out of scope**. The fix is to use one common application
point p = ½(p_A + p_B) for both arms, i.e. r_A′ = p − x_A and r_B′ = p − x_B, which gives ΔL = 0.
Record it as a separate dem issue (§8.4). The gate for `cluster_friction` is set against this floor
(§7).

---

## 2. The scheme

### 2.1 Ownership rule (decides which rank solves a contact)
Notation for one rank:
- slots [0, numReal) are owned and [numReal, numReal + numGhost) are ghosts;
- `gid(slot)` is the global id;
- for a ghost slot g, `src(g)` is the rank it was received from (this rank for a periodic
  self-ghost);
- for an owned slot o, `copies(o)` is the set of ranks o is ghosted to (cross-rank only), i.e. o's
  entries in the halo send lists.

```
owns(a, b):                              // contact between body slots a, b (b < 0: wall/plane)
  if b < 0:                return a < numReal            // wall contact: its body's owner
  ao = a < numReal; bo = b < numReal
  if ao and bo:            return true                   // both owned here
  if !ao and !bo:          return false                  // cannot occur (queries come from owned); defensive
  (o, g) = ao ? (a, b) : (b, a)                          // o owned, g ghost
  if gid(o) == gid(g):     return false                  // a body against its own periodic image
  s = src(g)
  if s != thisRank and s not in copies(o):
                           return true                   // the partner's owner cannot see this pair
  return gid(o) < gid(g)                                 // both owners see it (or a self-image twin)
```

**Why this is exactly-once** (proof, modulo precondition P1 below). Take a cross-rank physical
contact κ = {A owned by r, B owned by s}.
- r's narrow phase reports κ ⟺ B is a ghost on r ⟺ B ∈ send_s[r]. Likewise s reports it ⟺
  A ∈ send_r[s]. Queries are issued from owned bodies; reach ≤ band.
- On r, `s ∈ copies(A)` is exactly A ∈ send_r[s], i.e. "s reports κ". On s, `r ∈ copies(B)` is
  exactly "r reports κ". Each rank therefore knows whether the other rank reports the pair.
- Both report it: both evaluate `partnerSees = true` and apply the same gid tie-break, so exactly
  one owns it.
- One reports it: that rank finds `partnerSees = false` and owns it. The other rank never evaluates
  it.
- Neither reports it: not solved (pre-existing; §9 R4).
- Self-image twins (A, B′) and (B, A′) on one rank:
  - both twins are always present: positions on an undecomposed periodic axis are wrapped at
    commit, and if |A − B′| < reach then A lies within reach of the face, so A′ exists within the
    band;
  - the tie-break keeps the twin whose **real** endpoint has the lower gid;
  - at np = 1, gid = real index, so this is the same twin single-rank `demStep` keeps
    (`realA < realB`).

The gid tie-break is deterministic, rank-independent and stable across substeps. Ownership of a
pair changes only when visibility changes, and that happens only for a particle that has left its
owner's block (§1.2).

The one boundary case is a pair at gap ≈ margin, where the two ranks' shifted coordinates round
differently. It may be dropped, but such a contact has `dist > 0`, so it is **inactive**: zero
manifold points (`transformContact`) and C ≥ 0 in the projection. It has no effect.

**Precondition P1 (pre-existing, not added here).** On every decomposed periodic axis, each block
extent is ≥ 2·band. Then a particle has at most one image within the band of any other rank's block
(core's `withinRcutOfBlock` keeps a single image). A smaller box can already miss contacts today.
Recorded in §9 R8.

### 2.2 Owned and visible lists (filter at the source; no kernel changes)
The narrow phase keeps reporting every contact with at least one owned body (the *visible* set).
Right after it:
1. **Stable partition of contacts.** `partitionContactsKokkos(P.contacts, nc, ownership)` puts owned
   contacts first, [0, ncOwned), then non-owned contacts, [ncOwned, nc). Each part keeps its
   relative order. It is not a sort; see §6 WO-2 for the exact algorithm.
2. **Manifold reduction with owned manifolds first.** `reduceContactsToManifoldsKokkos(...,
   numOwnedContacts = ncOwned, &nmOwned)` ORs bit 63 (`kNonOwnedPairBit`) into the local sort key
   of every contact at index ≥ ncOwned. After the existing `sort_by_key`, the owned manifolds are
   [0, nmOwned) in their old relative order and the non-owned ones are [nmOwned, nmVisible).
   - `decodeKey` masks bit 63. Slots are < 2^31, so the mask is the identity for every
     single-rank key.
   - `contactSlot` of an owned contact points into [0, nmOwned).
3. `demSolveContacts` receives `nc = ncOwned` and `nm = nmOwned`. Every sweep, warm start, clamp,
   commit, colouring and friction pass therefore runs on owned items only. The visible count
   `nmVisible` is used **only** by the three label computations that must see the whole
   neighbourhood and write no body state:
   - `gatherWarmLambdaKokkos`, so that a stale ledger copy of a pair owned elsewhere is "matched"
     and never orphaned (§4.6);
   - `updateGroundedLevelsKokkos`;
   - `computeHeightLevelsKokkos`.

   `P.contactCount` and `P.manifoldCount` keep reporting the visible counts, so the getters are
   unchanged.

### 2.3 Reconciliation: baseline differencing and reverse accumulation
**Ghost slots evolve in place, as today.** The owned kernels write the ghost half of each impulse
into the ghost slot (self-mapped `realIndices`), so the local Gauss–Seidel sees it. What changes is
the sync: the ghost's accumulated change is no longer discarded but delivered to its owner.

For each reconciled field F, the halo keeps a **baseline** per ghost: F_ghost at the last moment the
ghost was set or reconciled. At a sync:

```
increment(g) = F(ghostSlot g) − baseline(g)          // exactly what this rank's owned contacts put there
owner row   += Σ_{all ghost copies g of it, all ranks, incl. self-images} increment(g)
                                                      // core ParticleHalo<3>::reverse, atomic scatter
forward owner → ghosts                               // existing refresh (for the forwarded fields)
baseline(g) = F(ghostSlot g)                         // after the forward (or current value if not forwarded)
```

Reconciled fields and payloads:

| phase | fields | reverse payload (one exchange) | forward |
|---|---|---|---|
| velocity | `velPred` (3), `angVelPred` (3), `bodyOrphan` (1), `bodyOrphanVPeak` (1) | `VelocityIncrement {float v[3], w[3], orphan, orphanPeak}`: 32 B per ghost | `velPred`, plus `angVelPred` if `forwardRotation` (fused into one message by WO-5) |
| position | `posPred` (3) | `PositionIncrement {float x[3]}`: 12 B per ghost | `posPred` with the periodic shift, plus `quatPred` if `forwardRotation` (fused by WO-5) |
| legacy friction count | `planeFriction(:,1)` (1) | `float`: 4 B | `planeFriction(:,1)` |

`operator+` of `VelocityIncrement` **sums** v, w and orphan and takes the **max** of orphanPeak
(an event peak speed). Core's reverse accumulates with `Kokkos::atomic_add`, which applies
`operator+`, so the operator is defined that way and documented at the type. For the peak the pack
sends the ghost value only if it rose above its baseline, else 0, so a decayed owner peak is never
undone. The owner applies:
- v and w by adding the increment;
- `bodyOrphan(i) = max(0, bodyOrphan(i) + orphan)`;
- `bodyOrphanVPeak(i) = max(bodyOrphanVPeak(i), orphanPeak)`.

Without Poisson restitution the orphan fields are exactly 0, so this is a no-op.

The quaternion needs no reverse: the position projection is translation-only
(`PositionContactSweep`: "rotation discarded"), and the velocity phase does not write `quatPred`.

**Reconciliation points.** They are the existing sync calls; none is added in any loop.

| where in `demSolveContacts` | hook today | hook after |
|---|---|---|
| top of the function, before any body write | none | `beginSolve` = mark velocity baselines (no MPI) |
| after `warmStartApplyKokkos` (PGS) | `syncVelocities` | `syncVelocities` = reverse, then forward |
| main, stabilization, escalate and ordered loops: `syncPoint(it)` and the final call of each phase | `syncVelocities` | same (now reverse, then forward) |
| legacy friction, after `countFrictionContactsKokkos` | none | `syncFrictionCounts` = reverse, then forward of the count column |
| legacy friction, after `applyVelocityDeltasKokkos` | `syncVelocities` | same |
| after `applyVelocityAndPredictPositionKokkos` | `syncPositions` | **`publishPositions`**: forward and mark baselines, **no reverse**. The ghost's change there is integration, not an interaction. |
| position loop `syncPoint(it)` and final | `syncPositions` | `syncPositions` = reverse, then forward |

### 2.4 Proof sketch: equal and opposite, applied once
- **(a) Partition of the constraint set.** By §2.1 the owned sets C_ρ of all ranks partition the
  set of active contacts seen by at least one rank. That set is the serial contact set up to
  pre-existing invisibility (R4).
- **(b) Local conservation.** On rank ρ, between two reconciliations, every write to a body slot is
  one of:
  - an impulse pair +J/m_a at slot a and −J/m_b at slot b (same J, angular parts r_a × J and
    −r_b × J through the slot's inverse inertia);
  - a wall impulse on one slot (external);
  - a one-sided stabilization impulse (a sink by design, identical to serial);
  - a count-averaged Jacobi fallback (non-conservative in serial too, §4.2).

  Ghost slots carry the owner's real mass, inertia and quaternion (gathered), so for the first two
  kinds Σ_{slots of ρ} m Δv = Σ_{walls in C_ρ} J.
- **(c) Exact delivery.** The reconciliation adds to owned row i the sum of the increments of all
  ghost copies of i (core `reverse`: cross-rank by MPI, self-ghosts by the device self-scatter).
  It then resets the baselines, so no increment is delivered twice or lost.
- **(d) Linear momentum and CoM.** Summing over ranks after a reconciliation:
  - ΔP_global = Σ_ρ Σ_{owned} m Δv + Σ_ρ Σ_{ghost} m Δv = Σ_walls J, which is 0 in a closed,
    force-free box.
  - The position phase is identical with Δx and n·dλ: per contact m_a Δx_a + m_b Δx_b =
    n dλ − n dλ = 0, so the CoM is conserved.
  - Δx is shift-invariant (a difference of two values in one image frame), so periodic ghosts need
    no shift in the reverse.
- **(e) Angular momentum, velocity phase.** Positions are fixed during the phase. A ghost copy g of
  body i sits at x_i + σ_g (σ_g = its periodic shift). Per contact the local change is
  (p_a − p_b) × J, i.e. the serial model's value, plus −Σ_g σ_g × m Δv_g after delivery.
  - Closed domain (σ = 0): ΔL_global equals exactly the serial model's ΔL.
  - That is zero for central normal impulses and for the PGS cone (measured), and δ n × J_t for the
    legacy friction (§1.4).
  - Periodic domain: L is not a conserved quantity, in serial either.
- **(f) Round-off.**
  - Forming an increment as a difference adds one rounding of order ε|F| per ghost per sync. That is
    the same order as the rounding an owned row incurs when the same impulses are added to it, so
    the np ≥ 2 floor is expected at np = 1's order: dP ~1e-8, dX ~1e-6 R.
  - An explicit per-kernel accumulator would avoid that rounding at the cost of touching about ten
    kernels, including the fused CUDA paths (rejected, §3).
- **(g) Any `sync_every = M`.** (b) and (c) do not depend on how many sweeps lie between two
  reconciliations, so conservation is exact for every M. M only trades lag for rounds.

### 2.5 Convergence expectation
- **Interior contacts** (both bodies owned by the sweeping rank) keep the serial GS order
  restricted to the rank: unchanged.
- **Interface contacts** are GS on the owning side with the far body up to one sync interval old,
  i.e. Jacobi-coupled across ranks. The far body also receives the impulse one interval late.
  - This is the same lag class as today: today each side uses a stale partner.
  - The difference is that today each body's own update was GS-consistent from its owner's view.
    That consistency is exactly what made the two halves unequal.
- **Linear chain across a face** (the p–q–s case):
  - serial GS contracts the overlap excess by about ¼ per sweep;
  - the cross-rank coupling contracts it by about ½ per iteration at `sync_every=1`. Worked out: an
    excess of 0.08 after the first iteration falls below 1e-4 R in ≈ 11 iterations. The test runs
    20 position iterations.
- **Overshoot.** A body touched in one interval by interface contacts owned by k different ranks
  gets a superposition of k independently computed impulses. For equal-mass spheres the pairwise
  Jacobi coupling ratio is ≤ ½, so the coupling is diagonally dominant for k ≤ 3, which covers every
  face and edge. Corner bodies with k up to 8 are rare and the PGS clamp (accumulated λ ≥ 0)
  absorbs over-push in the next iteration.
- **Prediction:** unchanged convergence in the interior, and at most +1–3 iterations where the
  interface dominates the residual.
- **Premise check.** This prediction is a premise of the design and gate G8 checks it; a failure
  goes back to the architect.

### 2.6 What changes numerically
- **Single-rank `demStep`:** nothing (I1).
- **`step_mpi` np = 1, closed:** nothing (I2).
- **`step_mpi` np = 1, periodic:** changes whenever a wrap pair exists. Each wrap pair is now solved
  once (by the twin that single-rank also keeps) instead of twice, and its warm start is loaded
  once. `cluster_periodic` gates it (§7).
- **`step_mpi` np ≥ 2:** every run with a cross-rank or self-image contact changes.
  - Linear momentum and CoM are now conserved to round-off.
  - Runs are still bitwise run-to-run at 1 thread, still round-off-nondeterministic across threads
    and on the GPU, and still not bitwise equal to np = 1.
  - Agreement with single-rank remains statistical for trajectories; the numbers are re-measured in
    WO-7.
- **`step_hertz_mpi`:** nothing.

---

## 3. Rejected alternatives

| Candidate | Why rejected |
|---|---|
| **(b) Globally consistent colouring of interface contacts**, so that both sides sweep them in the same colour from the same refreshed state | Equal J on both sides needs both ranks to hold identical states of both bodies at the moment the colour is swept. That means an owner→ghost exchange before every colour that touches an interface body: a sync per colour, about 20–60 per iteration, against today's one per iteration. It also yields equality only to round-off: manifold orientation differs by slot order, the in-manifold contact order is thread order, and the periodic shift rounds differently. So the two warm-start copies drift and can cross a clamp (λ ≥ 0, cone) differently. That is conservation by coincidence of floating point, not by construction. |
| **(c) Interface contacts solved Jacobi-style, symmetric by construction** | To be symmetric, an interface impulse must be computed from the post-sync snapshot and nothing later, so interface contacts get one Jacobi update per interval. They also need a relaxation factor known identically on both sides (a count forwarded per body), kernels that canonicalise orientation by gid, and two ledger copies kept equal. Equality is again only to round-off. Interface restitution becomes count-averaged Jacobi physics, which the project rejected for the whole solver ("correct multi-contact dissipation with no count-averaging"), so rank faces would carry a visible artefact. Its one saving, the reverse round, is recovered anyway by fusing the forwards (§5). |
| **Lower-gid owner alone, ignoring visibility** | Drops every pair that only the non-owning rank sees. That is common, because the band equals the maximum reach exactly (§1.2), and dropped pairs interpenetrate. |
| **Lower-rank owner** | Cannot choose between self-image twins, since both are on one rank. Ownership would also move with every rebalance. |
| **Predictor–corrector overlap** (keep today's redundant sweep as a prediction, then replace the non-owner's prediction by the owner's impulse at the sync) | Conservative and more GS-like at the interface, but every sweep kernel (PGS, coloured GS, Jacobi, warm start, friction, multilevel, fused CUDA) would have to book non-owned impulses separately, and two ledger copies would need reconciling per manifold. The complexity is not justified by an unmeasured convergence gain. |
| **Explicit per-kernel reverse accumulators** instead of baseline differencing | About 10 kernels touched, including the fused CUDA sweep and the multilevel coarse cycle. It removes one rounding per ghost per sync, and that rounding is already at np = 1's floor. |
| **Per-step migration**, so that ownership is always by position | It would restore "both owners see every pair" and fix R4 too, but it changes `step_mpi`'s ownership model: `rebalance_every=0` fixes the partition by design and the ledger migration costs host work every step. It is a separate decision (§9 R4). |
| Sorting contacts by gid pair; whole-solver count-averaged Jacobi; accepting statistical agreement | Already rejected by the user or the register. |

**This reverses a documented historical choice.** `docs/mpi.md:145` says step_mpi follows "the
EXACT variant ... rather than the reverse-reduction schemes B/C". The register has no entry
rejecting reverse reduction. The EXACT variant is exact only for Jacobi-type deltas; under the
modern GS stack it is the redundant solve that breaks conservation. The conservation decision
(register, 2026-09-25) supersedes it, and §8.3 records the new decision.

---

## 4. Interaction with the rest of the stack

**4.1 Warm start, persistent keys, accumulated-impulse clamps and the friction cone.**
- Only the owner sweeps a pair, so only the owner accumulates its λ, λ_T and posImpulse and commits
  them (`commitPairKeysLambdaKokkos` over [0, nmOwned)). The clamps λ ≥ 0 and |λ_T| ≤ μ·λ act on
  that single copy.
- A ledger entry of a pair this rank no longer owns is still gathered (`gatherWarmLambdaKokkos`
  over nmVisible, harmless per-manifold writes above nmOwned), is never applied, and is dropped at
  the next commit.
- If ownership moves (a drifted particle, §1.2), the new owner warm-starts that pair cold once.
  Accepted (R3).
- `markPersistentManifoldsKokkos`, `computeVn0Kokkos`, `warmStartApplyKokkos` and
  `commitPosImpulseKokkos` all run on the owned range.

**4.2 Count-averaging factors (per body, and they differ between ranks).**
- **Legacy friction.** `inv_n = 1/max(n_A, n_B, 1)` is **per contact and symmetric**, so it is
  conservative. What was wrong at np ≥ 2 was the counts themselves: a ghost's count covered only
  this rank's contacts. `syncFrictionCounts` (reverse-sum, then forward of `planeFriction(:,1)`,
  after `countFrictionContactsKokkos` over owned contacts) makes every count the serial count,
  because each owned contact is counted once at each endpoint. The friction impulse therefore
  equals serial's except for the lag.
- **The colour-saturation Jacobi fallbacks** (`applyVelocityDeltasAveragedKokkos` with
  `min(1, 2/count)` and `applyUpdatesKokkos` with `1/count`) use a **per-body** factor. They are
  non-conservative in serial as well whenever count_A ≠ count_B. After the fix the ghost's averaged
  increment is delivered exactly as applied, so np ≥ 2 inherits np = 1's property, no worse. They
  trigger only above colour-mask degree 62, which is impossible for the gate scenes.
- Recorded as a separate issue (§8.4). The `'jacobi'` diagnostic solver is the same case.

**4.3 Statics and stabilization.**
- The one-sided pass remains a momentum sink by design. The manifold's owner computes its side
  flags and applies the one-sided impulse once, and a ghost-side half is delivered like any other.
- Grounded and height levels are labels, not interactions. They are still propagated over **all
  visible** manifolds, as today, so a pile straddling a face keeps its support paths.
- Multilevel (mode 2) builds its hierarchy over owned manifolds (it sweeps coarse impulses).
  Aggregates may contain ghosts, whose velocity changes are delivered. The coarse cycle is locally
  momentum-conserving, so it is globally conserving by §2.4.

**4.4 Adaptive stops.** Unchanged. Residuals come from owned contacts and manifolds only and are
Allreduce-MAXed. The break test still precedes the sync call. The final sync of each phase delivers
the last sweep's increments.

**4.5 Sync schedule and deadlock safety.** No new sync point in any loop.
- Each sync becomes two rounds: reverse (tag 7604, core's default), then forward (tag 7603).
- Each rank finishes its `MPI_Waitall` for the reverse and applies the received increments before
  it packs its forward. So a forward always carries the owner's post-reverse value.
- The one-sided halo is handled: the reverse is the transpose of the forward graph, so a rank that
  only sends in the forward only receives in the reverse. Both calls are made by all ranks.
- The only skip is `if (!exchanges()) return;`. It is safe because such a rank has no neighbour in
  either direction.
- **Forbidden:** skipping a reverse because "this rank wrote no ghost". That is rank-local and
  deadlocks.
- `syncFrictionCounts` is called at a globally uniform point: `legacyFriction` depends only on
  global material parameters and gravity.

**4.6 Poisson (event-level) restitution.**
- The per-pair bank lives in the owner's ledger.
- Orphan credits to a ghost endpoint (`scatterOrphanBanksKokkos`) and draws from a ghost's account
  inside the sweeps are delivered by the velocity reverse.
- This replaces today's "the owner's redundant ledger copy applies the same credit
  authoritatively", which no longer exists.
- Dead-pair detection matches against visible manifolds (§2.2), so a stale copy of a live pair
  owned elsewhere is not orphaned.
- Known limitation (R5): two ranks can draw from one body's account within one interval, each
  bounded by its own view of the balance. The owner clamps at 0.

**4.7 Colouring and GS race-freedom.**
- Colouring runs over owned manifolds and contacts. Ghost slots stay graph vertices, so two owned
  contacts that touch the same ghost slot never share a colour. The in-place RMW stays race-free.
- Distinct periodic images of one body are distinct vertices. Within an interval they act
  Jacobi-coupled, as today under MPI, and their increments sum at the owner.

**4.8 Migration ledger (`MigratePack`, `packState` / `unpackState`).**
- **No code change.** Entries still ride on both locally owned endpoints and are deduplicated by
  key.
- After a move, the pair's new owner (by §2.1 on the new topology) uses its copy. A non-owner's
  copy is gathered but never applied or committed (§4.1), so it vanishes after one substep.
- Update the `MigratePack` / `packState` comments: the "redundant ghost-pair pattern" rationale is
  gone.

**4.9 GPU, atomics and reproducibility.**
- Sweep kernels are untouched.
- New kernels: pack and apply are element-wise per ghost or per owned row. The partition is one
  `parallel_scan` plus one scatter. The baseline marks are element-wise.
- Core's reverse scatters with `Kokkos::atomic_add` on the payload type. For 32 B and 12 B structs
  this is Kokkos's generic lock-based atomic, which needs `operator+`. That is correct on every
  backend; the risk is performance only (R1).
- At 1 host thread every loop runs in index order (I4). GPU and multi-thread atomics reorder sums
  at round-off, which is accepted.
- CUDA graphs and the device-side fused loops are already off under MPI. The fused colour sweep
  ("auto" = on under MPI) only sees the owned list.

---

## 5. Communication cost

**Rounds per substep, `sync_every = 1`.** One round = one neighbourhood exchange (pack, host
staging, `MPI_Isend`/`Irecv`/`Waitall`, unpack).
- S_v = velocity syncs executed: 1 after the warm start in PGS, plus the executed velocity
  iterations, plus the phase-final sync, plus the stabilization syncs if the pass triggers, plus 1
  for legacy friction.
- S_p = position syncs executed: the executed position iterations plus the final one.

| | today, spheres `forward_rotation=False` | today, default `forward_rotation=True` | after WO-3 | after WO-5 (fused forwards) |
|---|---|---|---|---|
| gather | 3 | 3 | 3 | 3 |
| velocity sync | 1 | 2 | 2 / 3 | 2 / 2 |
| position publish | 1 | 2 | 1 / 2 | 1 / 1 |
| position sync | 1 | 2 | 2 / 3 | 2 / 2 |
| legacy-friction counts | 0 | 0 | 2 | 2 |
| Allreduce | unchanged | unchanged | unchanged | unchanged |

Pairs of numbers read (spheres / rotation).

**Volume per ghost per sync:**
- velocity: 32 B reverse, plus 12 B (spheres) or 24 B (rotation) forward;
- position: 12 B reverse, plus 12 B or 28 B forward.

With about 1–2k ghosts per rank at N ≈ 20k and np 8, this is ≤ 90 kB per sync. Latency dominates.
The partition pass costs O(nc) memory traffic, about 3 × nc × 92 B: under 2 MB per substep, about
0.1 ms on two host cores.

**A-priori overhead** for the perf scenes (`run_perf.sh`: forward rotation on, 4 velocity and
8 position iterations):
- after WO-5 the solve needs the same number of rounds as today: one fewer at the position
  publish (position and quaternion fused), two more for the legacy-friction counts in `perf_gas`;
- the added cost is the reverse payload and host staging, plus the partition pass (about 0.1 ms);
- with 15–40 µs per round on this host (host-staged, OpenMP fork/join per kernel) this is:
  - **np 8 × 2: +1–4 %** (+0.1–0.3 ms on 8.1 ms);
  - **np 4 × 4: +1–2 %**.
- Without WO-5 it would be +17 rounds per step, about +3–9 % at np 8. That is why WO-5 is in the
  main line.
- Spheres with `forward_rotation=False` pay about 2× the solve's rounds: an a-priori +4–9 % at np 8.

**If gate G9 fails**, apply these levers in order. Each is a separate decision, not part of this
package.
1. A **fused bidirectional exchange** per sync:
   - each message to a neighbour carries both the reverse increments and the forward values for
     that neighbour;
   - the ghost is then set to the owner's value plus this rank's own not-yet-included increment;
   - this halves the rounds while staying exactly conservative, but ghosts miss third-party
     increments of the last interval;
   - it needs a new core primitive, hence a core release.
2. A **two-ended narrow-phase append** (owned from the front, non-owned from the back) instead of
   the partition pass. It changes the narrow-phase kernels.

---

## 6. Work orders

Each order is a separate commit on `momentum`, in this order. Stage named paths only. Build
`build_mom` as in `BEFORE.md` (host-openmp, Release, tests + MPI, `-DMPIEXEC_PREFLAGS="--bind-to none"`).
Run test batteries with `OMP_NUM_THREADS=2 OMP_PROC_BIND=false`.

**Do not touch:**
- `solver_velocity.hpp`, `solver_position.hpp`, `solver_friction.hpp`, `solver_multilevel.hpp`,
  `solver_fused*.hpp`;
- `solve_driver_force.hpp`;
- `step_solve.hpp`;
- `packState`'s logic;
- anything in `../core`.

If a step below seems to need any of these, **stop and report**.

### WO-0: Baseline instrumentation (test-only; commit before any `src/` change)
Files: `tests/kokkos_mpi/test_momentum_mpi.cpp`, `tests/kokkos_mpi/CMakeLists.txt`.
1. Add an optional second argument `--dump=<path>`. After the last step, gather to rank 0 every
   owned particle's `gid` (int32), `pos` (3), `vel` (3), `angVel` (3) and `quat` (4) as raw
   float32, sorted by gid, and write them as one binary file. Hertz uses its committed state.
2. Add `ovl=` to the MOMENTUM line. It is the maximum over steps of the Allreduce-MAX of
   `max_overlap` (the position loop's last residual; consistent before and after, even though it
   under-reports).
3. Add three modes, all report-only (`kGate` stays false), registered for np 1, 2, 4, 8:
   - `cluster_sync3`: `cluster_friction` with `enableMpiStep(0.0, 3, true)`;
   - `cluster_norot`: `cluster_friction` with `enableMpiStep(0.0, 1, false)`;
   - `cluster_periodic`: `cluster` in a fully periodic box [−16, 16]³ with the cluster centred on
     the box corner (−16, −16, −16), so wrap pairs exist on every axis (self-ghost twins at np = 1,
     cross-rank wraps at np ≥ 2). Only dP is meaningful there: print dX, dXpos and dLvel as `n/a`.
4. Build `c79dea5` + WO-0 as `build_base` and store in `docs/momentum_evidence/baseline/`:
   - np-1 dumps, `OMP_NUM_THREADS=1`, for `cluster`, `cluster_friction`, `cluster_pgs`,
     `cluster_posonly`, `hertz`;
   - `run_matrix.sh` output for the three new modes;
   - `ovl` for all modes at np 1, 2, 4, 8, 1 thread.

Acceptance: `git diff c79dea5 --stat -- src` is empty. The battery count grows by 12 and every test
passes (report-only).

### WO-1: Halo reconciliation primitives (inert)
File: `src/mpi_halo.hpp`.

1. At namespace scope, next to `MpiGatherPack`, define:
   - `struct VelocityIncrement { float v[3]; float w[3]; float orphan; float orphanPeak; };` with a
     free `KOKKOS_INLINE_FUNCTION VelocityIncrement operator+(const VelocityIncrement&, const
     VelocityIncrement&)`. It sums v, w and orphan and takes the max of orphanPeak. Put a comment at
     the type: core's `reverse` accumulates through `Kokkos::atomic_add`, which applies this
     operator.
   - `struct PositionIncrement { float x[3]; };` with an `operator+` that sums.
2. Add free kernels at namespace scope. In each, g ∈ [0, ng) is the **topology** ghost index and
   s = no + slot(g):
   - `haloMarkVelocityBaseline(velPred, angVelPred, orphan, orphanPeak, baseV, baseW, baseO, basePk, slot, no, ng)`:
     base*(g) = field(s).
   - `haloPackVelocityIncrement(..., inc, ...)`: inc(g).v = velPred(s) − baseV(g) and
     inc(g).w = angVelPred(s) − baseW(g), component-wise;
     inc(g).orphan = orphan(s) − baseO(g);
     inc(g).orphanPeak = orphanPeak(s) > basePk(g) ? orphanPeak(s) : 0.
   - `haloApplyVelocityIncrement(velPred, angVelPred, orphan, orphanPeak, ownedInc, no)`: for
     i ∈ [0, no), add v and w, orphan(i) = fmax(0, orphan(i) + inc.orphan),
     orphanPeak(i) = fmax(orphanPeak(i), inc.orphanPeak).
   - `haloMarkPositionBaseline`, `haloPackPositionIncrement` and `haloApplyPositionIncrement`: the
     same three for `posPred` alone.
   - `haloPackGhostColumn(pf, ghostVal, slot, no, ng)`: ghostVal(g) = pf(s, 1).
   - `haloAddOwnedColumn(pf, ownedVal, no)`: pf(i, 1) += ownedVal(i).
   - `haloPackOwnedColumn(pf, ownedVal, no)` and `haloUnpackGhostColumn(pf, ghostVal, slot, no, ng)`:
     the forward counterparts.
3. `ParticleHalo` members, allocated grow-only in `allocBuffers(no, ng)`:
   - baselines, each [ng] in topology ghost order: `baseVel_`, `baseAngVel_`, `basePos_` (`V3`),
     `baseOrphan_`, `baseOrphanPeak_` (`Vf`);
   - `ghostVelInc_` [ng] and `ownedVelInc_` [no] (`peclet::core::View<VelocityIncrement>`);
   - `ghostPosInc_` [ng] and `ownedPosInc_` [no] (`peclet::core::View<PositionIncrement>`);
   - `ghostVal_` [ng] and `ownedVal_` [no] (`peclet::core::View<float>`);
   - `ghostSource_` (`Vi` [ng], indexed by the **canonical** offset s − no);
   - `copyOffsets_` (`Vi` [no+1]) and `copyRanks_` (`Vi` [nSend]).
4. `buildOwnershipMaps(const FlatTopo& t, int no, int ng)`, called in `gather()` right after
   `uploadGhostSlots` in the rebuild branch:
   - `ghostSource_`: for each receive block k and each j < `t.recvCounts[k]`, set
     `ghostSource_(hSlot(t.recvOffsets[k] + j)) = t.recvRanks[k]`, where hSlot is the host copy of
     the canonical map `uploadGhostSlots` builds. Have it return its host vector rather than
     recomputing the map. Set the self tail [numReceived, ng) to `rank_`.
   - `copyOffsets_` / `copyRanks_`: the CSR over owned rows from `t.sendRanks`, `t.sendOffsets` and
     `t.sendIdx`. Within a row the ranks are ascending (`sendRanks` is ascending).
   - Build on the host and upload with `deep_copy`.
5. Public methods:
   - `markVelocityBaseline(Particles&)`.
   - `syncVelocities(Particles& P, bool rotation)`: `if (!exchanges()) return;` → pack →
     `deep_copy` `ownedVelInc_[0,no)` to zero → `dev_.reverse(ghostVelInc_, ownedVelInc_)` → apply
     → `forward(P.velPred)`, then `forward(P.angVelPred)` if rotation → `markVelocityBaseline(P)`.
   - `publishPositions(Particles& P, bool rotation)`: `if (!exchanges()) return;` →
     `forwardPositions(P.posPred)`, then `forward4(P.quatPred)` if rotation → mark the position
     baseline.
   - `syncPositions(Particles& P, bool rotation)`: `if (!exchanges()) return;` → pack → zero →
     `dev_.reverse(ghostPosInc_, ownedPosInc_)` → apply → `publishPositions(P, rotation)`.
   - `syncFrictionCounts(Particles& P)`: `if (!exchanges()) return;` → `haloPackGhostColumn` →
     zero `ownedVal_` → `dev_.reverse(ghostVal_, ownedVal_)` → `haloAddOwnedColumn` →
     `haloPackOwnedColumn` → `dev_.forward(ownedVal_, ghostVal_)` → `haloUnpackGhostColumn`.
   - `ContactOwnership contactOwnership(const Particles& P) const` (struct in WO-2).
6. New test `tests/kokkos_mpi/test_ownership_mpi.cpp`, following `test_migrate_mpi.cpp`'s direct
   `Particles` + `ParticleHalo` construction. Modes `reverse_closed` and `reverse_periodic`
   (periodic on all axes, so self-ghosts appear at np = 1). The test:
   - gathers ghosts over a jittered lattice;
   - marks the baselines, then adds to every ghost slot's `velPred` the integer vector
     (gid+1, 2(gid+1), 3(gid+1));
   - calls `syncVelocities(P, true)`;
   - checks that each owned row gained exactly c(gid)·(gid+1, 2(gid+1), 3(gid+1)), where c is the
     number of ghost copies of that gid over all ranks (Allreduce of a per-gid count). Integers
     make the sum order-independent;
   - checks that every ghost now equals its owner (exact for velocities; exact after subtracting
     the shift for positions via `syncPositions`);
   - calls sync a second time with no writes and checks that the owned rows are unchanged
     bit for bit.

   Register it for np 1, 2, 4, 8.

Acceptance:
- compiles on host-openmp and on nvidia-cuda (compile only if no GPU);
- `ownership_reverse_*` passes;
- the rest of the battery is unchanged, since nothing calls the new code yet.

### WO-2: Contact ownership and the owned-first lists (inert)
Files: `src/contact_preprocessing.hpp`, `src/mpi_halo.hpp`, the test from WO-1.

1. In `contact_preprocessing.hpp`:
   - Add `inline constexpr std::uint64_t kNonOwnedPairBit = 1ull << 63;`.
   - In `decodeKey`, change `bodyA = static_cast<int>(key >> 32)` to
     `bodyA = static_cast<int>((key >> 32) & 0x7FFFFFFFu)`.
2. In `contact_preprocessing.hpp`, add
   `template <class Owns> inline int partitionContactsKokkos(Kokkos::View<ContactC*, CpMem> contacts, int n, const Owns& owns)`:
   1. `parallel_scan` over i with `f = owns(contacts(i)) ? 1 : 0`, storing `before(i)` (exclusive).
      The return value is ncOwned.
   2. Scatter into a per-call scratch `View<ContactC*>` (`view_alloc(space, "peclet::dem::cp::part",
      WithoutInitializing)`, the same pattern as the reduction's scratch):
      dst(i) = f ? before(i) : ncOwned + (i − before(i)).
   3. `deep_copy` scratch → `contacts[0, n)`.
   4. Return ncOwned.

   If `n == 0`, return 0.
3. Extend `reduceContactsToManifoldsKokkos` with `int numOwnedContacts = -1, int* numOwnedManifolds = nullptr`:
   - step 1 becomes `keys(i) = pairKey(contacts(i)) | ((numOwnedContacts >= 0 && i >= numOwnedContacts) ? kNonOwnedPairBit : 0)`;
   - after the segment scan, if `numOwnedManifolds`, `parallel_reduce` the count of leaders
     (p == 0 or k(p) ≠ k(p−1)) with (k(p) & kNonOwnedPairBit) == 0 into it;
   - the default arguments reproduce today's behaviour exactly.
4. In `mpi_halo.hpp`, add `struct ContactOwnership { int rank; int numReal; Kokkos::View<const int*, CpMem> gid, ghostSource, copyOffsets, copyRanks; KOKKOS_INLINE_FUNCTION bool operator()(const ContactC& c) const; }`,
   implementing §2.1's `owns(c.bodyA, c.bodyB)` exactly. `copies(o)` is a linear scan of
   `copyRanks[copyOffsets(o), copyOffsets(o+1))`.
5. In `test_ownership_mpi`, add modes `exactly_once_closed`, `exactly_once_periodic` and
   `exactly_once_drift`, registered for np 1, 2, 4, 8:
   - build the global set, distribute it by the ORB, gather, run `findCollisionsGrow` +
     `narrowPhaseGrow` + `partitionContactsKokkos`;
   - collect every rank's owned **active** (`dist <= 0`) pair keys `pairKeyFromGids(gid_a, gid_b)`
     (walls: `(gid, 0xFFFFFFFF)`) and gather them to rank 0;
   - run the same narrow phase on the global set on `MPI_COMM_SELF`;
   - require **no key owned twice** and **owned union == serial active set**.
   - In `drift`, after distribution (no migration), displace every particle that lies within 1 R of
     an **interior** block face (a face shared with another rank's block) by 1.0 R across that
     face into the neighbour's block, **excluding** any particle with another displaced particle
     within reach (greedy in gid order). np = 1 has no interior face, so there it equals `closed`. Every active pair then has at least one endpoint
     inside its owner's block, so it is seen (§1.2). This mode fails with the lower-gid-only rule;
     it is the regression test for §2.1.

Acceptance:
- `tests/kokkos` (contact preprocessing vs serial) passes unchanged;
- `ownership_exactly_once_*` passes at np 1, 2, 4, 8;
- the battery is otherwise unchanged.

### WO-3: Wire the scheme (the conservation commit)
Files: `src/step_solve_mpi.hpp`, `src/solve_driver.hpp`.

1. `MpiSolveHooks` fields, in this order: `halo`, `syncEvery`, `forwardRotation`,
   `numManifoldsVisible`. Methods:

   | method | behaviour |
   |---|---|
   | `allMax` | unchanged |
   | `syncPoint` | unchanged |
   | `int visibleManifolds(int) const` | return `numManifoldsVisible` |
   | `beginSolve(P)` | `halo.markVelocityBaseline(P)` |
   | `syncVelocities(P)` | `halo.syncVelocities(P, forwardRotation)` |
   | `publishPositions(P)` | `halo.publishPositions(P, forwardRotation)` |
   | `syncPositions(P)` | `halo.syncPositions(P, forwardRotation)` |
   | `syncFrictionCounts(P)` | `halo.syncFrictionCounts(P)` |

   Rewrite the struct comment: single owner per contact, reverse-then-forward at every sync, no
   redundant solve.
2. `SoloSolveHooks`: add `int visibleManifolds(int nm) const { return nm; }` and no-op
   `beginSolve`, `publishPositions` and `syncFrictionCounts`.
3. `demStepMpi`, after `narrowPhaseGrow`:
   ```
   const int ncOwned = partitionContactsKokkos(P.contacts, nc, halo.contactOwnership(P));
   int nmOwned = 0;
   reduceContactsToManifoldsKokkos(P.contacts, nc, P.manifolds, P.manifoldCount, P.contactSlot, ncOwned, &nmOwned);
   const int nmVisible = readInt(P.manifoldCount);
   demSolveContacts(P, ncOwned, nmOwned, P.numParticles, P.gid,
                    MpiSolveHooks{halo, syncEvery < 1 ? 1 : syncEvery, forwardRotation, nmVisible});
   ```
   Rewrite the `demStepMpi` comment ("identical physics ... not bit-exact") to §2.6's wording.
4. `demSolveContacts`, exactly these edits and no others:
   1. After `CpExec space;`: `hooks.beginSolve(P); const int nmVisible = hooks.visibleManifolds(nm);`.
   2. `gatherWarmLambdaKokkos(P.manifolds, nm, ...)` → `nmVisible`.
   3. `updateGroundedLevelsKokkos(P.manifolds, nm, ...)` → `nmVisible`.
   4. `computeHeightLevelsKokkos(P.manifolds, nm, ...)` → `nmVisible`. `buildLevelColorBucketsKokkos`
      keeps `nm`.
   5. After `countFrictionContactsKokkos(...)`: `hooks.syncFrictionCounts(P);`.
   6. After `applyVelocityAndPredictPositionKokkos(...)`, replace `hooks.syncPositions(P)` with
      `hooks.publishPositions(P)`.
   7. Rewrite the file comment (lines 10–17): single owner, reverse accumulate, visible vs owned
      counts.

   Every other `nm` and `nc` stays: those are now the owned counts under MPI.
5. Update the comments in `mpi_halo.hpp`: the file header ("ghost deltas are discarded ...") and
   the `MigratePack` / `packState` notes (§4.8). No logic change.

Acceptance, all measured, numbers in the commit message:
- gates G1–G5 and G8 (§7) hold with `kGate` still false, read from the report lines;
- G3 np-1 closed dumps are byte-identical to the baseline;
- the full battery passes.

If G8 fails, **stop**: a design premise failed, so report to the architect with the `ovl` table.

### WO-4: Turn the gates on
Files: `tests/kokkos_mpi/test_momentum_mpi.cpp`, `tests/kokkos_mpi/CMakeLists.txt`.
1. Set `kGate = true` and replace the three scalar tolerances with a per-mode table (§7 G1). Gate
   dLvel for XPBD modes and dL for `hertz`.
2. Delete `set_tests_properties(ghost_band_${mode}_np${np} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1)`
   and its comment block. Replace the comment with one line: the chain is conservative, so the flake
   is gone.

Acceptance: G1, G2 and G7 pass.

### WO-5: Fused forwards (performance; bitwise-inert)
File: `src/mpi_halo.hpp`.
1. When `rotation` is true, `syncVelocities` forwards velPred and angVelPred as **one**
   `dev_.forward` of `struct VelocityState { float v[3]; float w[3]; }`, with pack and unpack
   kernels. This replaces the two forwards.
2. When `rotation` is true, `publishPositions` forwards `struct PositionState { float x[3]; float q[4]; }`,
   applying the periodic shift to x on unpack, exactly as `haloUnpackF3` does. This replaces
   `forwardPositions` + `forward4`.
3. The gather, Hertz and `forward`/`forwardPositions`/`forward4` stay as they are.

Acceptance:
- at `OMP_NUM_THREADS=1`, np 2, 4, 8, the dumps of all gated modes are byte-identical to WO-4;
- G9 is measured and reported.

### WO-6: Docs, comments and CLAUDE.md
Files: `docs/mpi.md`, `docs/solver_details.md` §10, `CLAUDE.md`, and the docstrings of
`src/sim.hpp:811` and `src/dem_bindings.cpp` (`enable_mpi_step`: "sync_every: owner/ghost
reconciliation interval; conservation is exact at any value, larger = more lag, fewer messages").
Use the text in §8. Hand the register text in §8.3 to the caller, who commits it in the umbrella.

Acceptance: `grep -n "EXACT\|redundant\|discarded" docs/mpi.md docs/solver_details.md src/*.hpp`
leaves no statement that describes the old scheme as current.

### WO-7: Evidence
Deliver `docs/momentum_evidence/AFTER.md`, with the same tables as `BEFORE.md`, containing:
- the matrix and chain runs;
- perf (G9) with the load;
- the battery and python_mpi counts;
- the coupling MPI deltas (G10);
- `ovl` before and after;
- a re-measured "statistical agreement with single-rank" line for `docs/mpi.md`, same protocol as
  2026-09-08: N = 200, 15 steps, mean, 95 % quantile, max.

---

## 7. Verification gates

| Gate | What | Configuration | Threshold |
|---|---|---|---|
| **G1 conservation** | `test_momentum_mpi`, `kGate=true`; the maximum over 3 repeats (`run_matrix.sh`) | np ∈ {1, 2, 4, 8} × `OMP_NUM_THREADS` ∈ {1, 8} | See the table below |
| **G2 chain** | `run_chain.sh` (margin mode), 8 runs per cell | np 2 and 4 × OMP 1 and 8 | \|comShift(dist)\| ≤ 1e-5 in **8/8** and posErr ≤ 1e-4 in **8/8**. `ghost_band_*` ctests pass without the pin. |
| **G3 np-1 bitwise** | `--dump` byte comparison with `docs/momentum_evidence/baseline/` | np 1, OMP 1; `cluster`, `cluster_friction`, `cluster_pgs`, `cluster_posonly`, `hertz` | `cmp` identical. `tests/kokkos`, `tests/arborx` and `python_tests` pass. `cluster_periodic` np 1 is **expected to change**: report dP before and after. |
| **G4 run-to-run** | 5 runs, `--dump` | np 4 and 8, OMP 1; `cluster`, `cluster_friction`, `cluster_pgs` | All 5 byte-identical |
| **G5 exactly-once** | `ownership_exactly_once_{closed,periodic,drift}` | np 1, 2, 4, 8 | 0 duplicate owned keys; owned union = serial active set |
| **G6 reverse primitive** | `ownership_reverse_{closed,periodic}` | np 1, 2, 4, 8 | Exact integer sums; ghost = owner after sync; a second sync changes nothing, bit for bit |
| **G7 battery** | the full ctest (98 + 12 + 20 new), `--bind-to none`, OMP 2; the 12 `python_mpi_*` with core's Python build on PYTHONPATH | as registered | All pass. A statistical-agreement test (`validate_exact`, `demstep_*`, `verify_distributed`, rotating drum) failing is a **stop and report**, never a tolerance change. |
| **G8 convergence premise** | `ovl` | np 2, 4, 8, OMP 1; `cluster`, `cluster_friction`, `cluster_pgs` | after ≤ 2 × baseline, per cell |
| **G9 performance** | `run_perf.sh`, 5 runs; report medians and the load | `perf_gas` and `perf_pgs` at np 4 × 4 and np 8 × 2 | The min of 5 must be ≤ 1.10 × the baseline min: 15.2 → 16.7, 7.9 → 8.7, 15.5 → 17.1, 8.2 → 9.0 ms. Expected +1–5 %. |
| **G10 coupling** | coupling MPI tests (Ergun, moving suspension) | as registered | Pass; report deltas of their printed observables before and after |
| **G11 CUDA** | build + `ownership_reverse_*` | nvidia-cuda prefix | Compiles; the test passes if a GPU is present, otherwise say "compile-only" |

G1 thresholds per mode (max over runs):

| mode | dP | dX (R) | dXpos (R) | dLvel | basis |
|---|---|---|---|---|---|
| `cluster` | 1e-6 | 1e-5 | 1e-5 | 1e-6 | np 1: 2.8e-9 / 3e-7 / 3e-9; broken ≥ 3.3e-3 |
| `cluster_friction`, `cluster_sync3`, `cluster_norot` | 1e-6 | 1e-5 | 1e-5 | **1e-4** | the serial friction floor is 1.9e-5 (§1.4); broken ≥ 1.1e-3 |
| `cluster_pgs` | **5e-6** | 1e-5 | 1e-5 | 1e-6 | the free-fall float accumulation floor is 7.9e-7 at np 1 |
| `cluster_posonly` | 1e-12 | 1e-5 | 1e-5 | 1e-12 | the increments are exact zeros |
| `cluster_periodic` | 1e-6 | n/a | n/a | n/a | dP only |
| `hertz` | 1e-6 | 1e-5 | n/a | dL 1e-5 | regression guard; today 2.8e-8 / 8.8e-7 / 4.4e-7 |

Expected after the fix: dP ~1e-8 and dX ~1e-6 R, at least 300× below the broken values.

---

## 8. Doc and register text

### 8.1 `docs/mpi.md`
Replace the section "The EXACT scheme + the `sync_every` (M) knob" (lines 68–72) with:

> ### Contact ownership, reconciliation and the `sync_every` (M) knob
> Every contact is solved by **exactly one** rank: the only owner that sees it, or, when both
> owners see it, the owner of its lower-gid body (for a periodic self-image twin: the twin whose
> real body has the lower gid). Each rank sweeps only the contacts it owns, writing the partner's
> half of every impulse into the partner's ghost slot. At every sync, each ghost's accumulated
> change is **added onto its owner** (core `ParticleHalo::reverse`) and the owners then republish
> their state (forward). Every impulse and every overlap correction is therefore applied once,
> equal and opposite: linear momentum and the centre of mass are conserved to round-off at any np,
> thread count and `sync_every`, and the velocity phase conserves angular momentum wherever the
> serial model does. `sync_every=M` sets how many sweeps pass between reconciliations: conservation
> does not depend on it; larger M means more lag at rank faces and fewer messages.
> Interior contacts keep the serial Gauss–Seidel order; a contact across a rank face sees its far
> body as of the last reconciliation, so trajectories agree with single-rank statistically, not
> bit-for-bit (numbers under *What is validated*).

In the paragraph at line 145, replace "The shipped `step_mpi` follows the **EXACT** variant (...)
rather than the reverse-reduction schemes B/C." with:

> Since 2026-09 the shipped `step_mpi` uses reverse reduction: owner-exclusive contacts plus a
> ghost→owner accumulation at every sync. The earlier "EXACT" variant solved each cross-rank
> contact on both owners, which is exact only for Jacobi-type deltas and broke momentum
> conservation under the Gauss–Seidel stack.

In "What is validated", replace the np = 2/4 sentence with the WO-7 numbers, and add:

> `momentum_*` gates conservation (dP ≤ 1e-6, CoM ≤ 1e-5 R, velocity-phase dL ≤ 1e-6 frictionless,
> ≤ 1e-4 with the legacy friction pass) at np 1/2/4/8, and `ownership_*` gates that every contact
> is owned exactly once.

### 8.2 `CLAUDE.md` (dem)
Two edits:
- In "Reproducibility trap", replace the sentence from "**Threads are a second, separate
  source:**" to the end of the paragraph with:

  > **Threads are a second, separate source:** the narrow phase appends contacts in thread order,
  > so runs are bitwise reproducible only at `OMP_NUM_THREADS=1` (or Serial). Conservation does
  > not depend on it (see below).

- Add a paragraph after it:

  > **Every contact is owned by exactly one rank, and ghost increments are reverse-accumulated**
  > (2026-09-25, `docs/mpi_momentum_conservation.md`). `ContactOwnership` (`mpi_halo.hpp`):
  > the only owner that sees the pair, else the lower-gid body's owner. `demStepMpi` partitions
  > the contacts owned-first and reduces manifolds owned-first (key bit 63), and
  > `demSolveContacts` sweeps only `[0, ncOwned)` / `[0, nmOwned)`. Only the grounded/height
  > levels and the warm-ledger match read the visible range. Every `syncVelocities` /
  > `syncPositions` is **reverse, then forward**. The position phase opens with
  > `publishPositions` (forward only: the integration of a ghost is not an interaction). NEVER
  > re-introduce a redundant two-owner solve, never skip a reverse on rank-local grounds (it is
  > collective), and any new kernel that writes body state during the solve must write ghost
  > slots only as the partner half of an impulse pair: the reverse delivers exactly what is
  > there. `ghost_band_*` no longer pins one thread.

In "Settled decisions", add the bullet:

> **Distributed contacts are owner-exclusive with reverse accumulation**, not solved redundantly
> on both owners (the redundant solve was momentum-non-conserving under Gauss–Seidel).

### 8.3 Register entry (umbrella `docs/decisions/dem.md`, index in `docs/DECISIONS.md`)
```
### Distributed XPBD contacts are owner-exclusive, with ghost→owner reverse accumulation at every sync
- area: dem
- source: dem/docs/mpi_momentum_conservation.md; evidence dem/docs/momentum_evidence/{BEFORE,AFTER}.md
- decided: 2026-09-25
- status: settled
- quote: |
    Each contact is solved by exactly one rank -- the only owner that sees it, else the owner of
    its lower-gid body -- and every ghost's accumulated velocity/position change is added onto its
    owner (core ParticleHalo::reverse) before each forward refresh. Linear momentum and CoM are
    conserved to round-off at any np, thread count and sync_every.
- rejected: redundant two-owner solves ("EXACT" variant); globally consistent colouring of
  interface contacts (a sync per colour); symmetric Jacobi for interface contacts (count-averaged
  physics at rank faces, equality only to round-off); lower-gid ownership alone (drops pairs only
  one owner sees -- zero slack, band = max reach); per-kernel reverse accumulators (baseline
  differencing is exact to the same round-off with no kernel change)
- why: conservation by construction, not by coincidence of floating point; no new sync point;
  the sweep kernels stay the single-GPU ones
- supersedes: docs/mpi.md "EXACT variant rather than the reverse-reduction schemes B/C"
```

### 8.4 Separate issues to record (not in this package)
1. **Legacy friction lever-arm couple.** `solveContactFrictionKokkos` applies ±J_t at two different
   surface points, which gives ΔL = δ n × J_t per contact (serial dLvel 1.9e-5). Fix: one common
   application point. It changes np = 1.
2. **The count-averaged Jacobi fallbacks** (velocity `min(1, 2/count)`, position `1/count`, and the
   `'jacobi'` diagnostic) use per-body factors, so they are momentum-non-conserving at every np
   when triggered.
3. **Pairs that no rank sees**: both endpoints drifted out of their owners' blocks with
   `rebalance_every=0` (R4).

---

## 9. Risks and open questions

Each item has a default, so the work proceeds unattended.

| # | Item | Needs | Default |
|---|---|---|---|
| **Q1** | `step_mpi` at np = 1 on a **periodic** domain changes (§2.6). It double-solved wrap twins. The brief's "np 1 bitwise unchanged" is kept for closed domains and for single-rank `demStep`. | **User preference** (scope of "np 1 unchanged") | Proceed; conservation requires it. Report the before/after dP of `cluster_periodic` at np 1. |
| Q2 | The performance gate threshold of +10 % (min of 5) | User preference | 10 % |
| R1 | Kokkos' generic `atomic_add` on the 32 B / 12 B payloads (lock-based) must compile on CUDA and cost ≤ 5 % of a reverse | **Fact** (WO-1 build + a `ownership_reverse` timing on CUDA) | Use it. If it does not compile or is slow, the fallback is a core addition `reverse(ghost, recvBuf)` that returns the `sendIdx`-shaped buffer, with dem scattering components using float atomics. That needs a core minor release, so stop and report. |
| R2 | The convergence premise (§2.5) | **Fact** (G8) | No interface relaxation. If G8 fails, stop and consult the architect with the `ovl` table. A symmetric relaxation of cross-rank manifolds is the pre-identified candidate; it is deliberately not designed here. |
| R3 | An ownership flip (a drifted particle's visibility changes) cold-starts that pair's warm start once. Under Poisson it can also drop that pair's bank once. | Fact (rare; only for particles outside their owner's block) | Accept |
| R4 | Pairs whose two endpoints have both drifted out of their owners' blocks can be seen by nobody (pre-existing; unsolved before as well) | User preference (`rebalance_every=0` fixes the partition by design) | Out of scope; record (§8.4). Recommended follow-up: per-step migration of particles that left their block. |
| R5 | Poisson orphan accounts can be overdrawn by concurrent draws from two ranks within one interval (bounded by the balance; clamped at 0) | Fact | Accept; document in `mpi.md` |
| R6 | Legacy friction lever-arm couple (§1.4) | User preference (it changes np = 1) | Separate issue |
| R7 | The count-averaged Jacobi fallbacks are non-conservative at any np (§4.2) | User preference | Separate issue |
| R8 | P1: block extent ≥ 2 · band on decomposed periodic axes (one image per particle and rank) | Fact (pre-existing assumption) | No check added here; record |
| R9 | G9 fails | Fact | Levers in order (§5): a fused bidirectional exchange (core primitive), then a two-ended narrow-phase append. Each is a separate decision. |
| R10 | A statistical-agreement test regresses at np ≥ 2 | Fact | Stop and report; never loosen a tolerance in this package |

**R10 addendum (WO-3b, 2026-09-25).** It happened: `demstep_jacobi_closed` np 2/4 regressed to
posErr 0.157 (tol 1e-2), because the count-averaged solves divide by **per-body** counts (serial
Jacobi: `min(1, 2/count_i)` for velocity, `1/count_i` for position, each body by its own count)
and after WO-3 a rank counts only the contacts it owns. Fix: `syncContactCounts` (the
`syncFrictionCounts` pattern: ghost partial counts reverse-summed to the owner, totals forwarded)
between each Jacobi kernel and its apply, so every copy divides by the serial count. It is
unconditional under the Jacobi A/B (a global setting); for the GS loops' rank-local
colour-saturation fallback the activation is voted in the existing stop Allreduce (`allMaxAny`,
two floats in the same message), so no message is added to the GS production path, which stays
byte-identical. Result: posErr 5.9e-6 / 7.6e-6 at np 2 / 4, tolerance untouched; the new
`momentum_cluster_jacobi` gate holds np 2..8 at the np 1 drift level (per-body averaging is not
conservative in serial, R7). Found on the way: the saturation fallback is unreachable (the
colourings assign colour 62 to every contact beyond the 62nd at a body instead of leaving it
uncoloured). Evidence: `momentum_evidence/AFTER.md`, WO-3b section.
