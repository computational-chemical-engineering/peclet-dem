# Architect brief: one principled, conservative contact-solve framework for dem (serial, threaded, GPU, MPI)

## 1. The question
Design ONE principled method for how dem's contact solve updates bodies that receive several contact
updates at the same time. Such bodies arise in three places:
- across ranks;
- inside one colour, at a hub body with more contacts than colours;
- on the Jacobi paths.

The method must:
- conserve linear and angular momentum by construction;
- never create kinetic energy for restitution e ≤ 1;
- converge to the coupled (serial) solution;
- be race-free on threads and the GPU;
- be fast.

The same package must also:
- fix the friction application point;
- guarantee that every pair within contact reach is solved by exactly one rank;
- specify tests strong enough to catch dropped contacts and energy creation.

Deliver a design note with an implementation plan in work orders, a performance model, and gates.

## 2. Why it needs the architect, and what the user said
USER, 2026-09-25, after the first fix and its review: "It feels that we are trying to patch this
issue while it needs a rigorous design. Think of a principled method to handle this issue. Then
make an implementation plan. Take performance also into account. Test it well etc."

The first fix is dem c771e07 (design `docs/mpi_momentum_conservation.md`). It made each contact
owner-exclusive, with ghost→owner reverse accumulation. It conserves momentum, but:
- the review found that it creates energy at rank faces;
- four further defects were measured, three of them reachable from production defaults.

This package replaces patching with a framework. It is release-blocking. Earlier user decisions still
hold:
- "That momentum is not conserved is not acceptable."
- Bitwise reproducibility across threads or the GPU is NOT required. Do not sort contacts for it.

## 3. The evidence (read these; they are curated)
All paths are relative to the worktree `suite/dem-contacts` (branch `contacts`, based on main
c771e07).
- `docs/contact_evidence/FOLLOWUPS.md` (620 lines): the four defects, each with mechanism,
  file:line excerpts, repro command, numbers, whether production reaches it, and what a fix must
  decide. The report-only tests are in commit 3e72ad5.
- `docs/contact_evidence/review/review_momentum.md`: the independent review of c771e07, with its
  raw evidence files.
- `docs/mpi_momentum_conservation.md` and `docs/momentum_evidence/{BEFORE,AFTER}.md`: the landed
  scheme and its numbers.
- `docs/momentum_evidence/recon_digest.md`: a code digest of the solve driver, kernels, colouring and
  halo. It is from before c771e07, so check the details against the source.

Summary of what they establish:

**R1. Energy creation at rank faces (review, CONFIRMED).**
- The g = 0 restitution sweep (`solve_driver.hpp:484`, kernel `solver_velocity.hpp:1652-1694`) is a
  one-shot incremental impulse from the current relative velocity. It never retracts.
- Under owner-exclusive solving, two ranks push the same face body from its stale state, and the
  pushes add. That is the raw Jacobi sum dem's register rejected.
- 3-body scene across the face, KE relative to the centre of mass (0.333 at start):

  | e | np 1 | np 2 before c771e07 | np 2 after |
  |---|---|---|---|
  | 0.5 | 0.138 | 0.189 | 0.291 |
  | 0.8 | 0.246 | 0.340 | 0.555 (energy created) |

- The warm-started PGS path is fine.

**R2. No gate detects a dropped contact (review, CONFIRMED by mutation).**
- A mutant that solves no contact with a ghost endpoint passes every momentum gate.
- The ovl metric covers owned contacts only (`solve_driver.hpp:883-887, 919`), so it rewarded
  dropping.

**D1. Colour overflow (a production race).**
- Five colourings, including multilevel at `solver_multilevel.hpp:328`, cap at 63 colours. Past
  that, colour 62 is reused, `leftover` is always 0, and the fallback is dead code.
- Same-colour pairs share a body, so there are concurrent read-modify-writes.
- On CUDA, a big-grain scene gives dP 1–4e-2 against 9.5e-8 below the cap.
- Production reaches it through the position colouring:
  - RingBed-style ring beds have 79–613 contact points per particle, with 1240–6439 same-colour
    pairs per step;
  - sphere beds at size ratio 6 reach degree 114.

**D2. Friction couple.**
- The legacy friction pass applies ±J_t at two surface points separated by dist. The measured
  angular-momentum error equals dist·n × J_t, ratio 1.000.
- 62–79 % of friction-active contacts are speculative, i.e. a gap rather than an overlap.
- The PGS cone is clean.
- It is reached by any g = 0 run with friction, including RingBed packing.

**D3. Per-body Jacobi factors.**
- `min(1, 2/count_i)` and `1/count_i`: np 1 dP is 9e-3.
- 4 call sites: two need `set_velocity_solver('jacobi')`, and two are the dead fallbacks from D1.
- The friction count-averaging uses one factor per pair and conserves.

**D4. Pairs no rank sees.**
- (a) Drift with `rebalance_every=0` (the step_mpi default):
  - a pair is lost once both bodies are more than √(band² − d²/4) ≈ 1.857 R past their blocks;
  - it needs np ≥ 4;
  - a shifted grid loses 10–21 pairs at 4 R.
- (b) Periodic: core's halo sends ONE image per (particle, destination rank). An undecomposed
  periodic axis therefore loses edge-wrap pairs whose needed image was inside the band: np 2 loses
  2, np 4 loses 3.

Performance before c771e07, quiet host: N = 19683 periodic, 15.6 ms/step at np 4 × 4 threads and
8.1 ms/step at np 8 × 2. c771e07 measured within ±5 % at load 55–60. `forward_rotation=False` and
the CUDA struct atomics have never been timed.

## 4. The session's proposed principle
This is a hypothesis. Test it and reject it if it is wrong. Derive from first principles, not by
porting literature (standing user feedback).

**Invariant.** Each contact c has ONE accumulated impulse λ_c, owned by one rank. Every body state
always equals its substep-start value plus M⁻¹ Σ_c J_cᵀ λ_c, over the single global λ vector.
Any iteration that preserves this conserves linear momentum, and angular momentum when every J_c
acts at one point, whether or not it has converged.

**Mechanism: mass splitting with consensus.**
- A body updated simultaneously in k places is split into k copies. Each copy carries m/k and I/k.
- The k places are ranks, same-colour hub slots, or Jacobi contacts.
- Each contact acts, equal and opposite, on one copy of each body, and the local solve (GS/PGS)
  sees the copy's reduced mass.
- At each sync, copies are averaged with mass weights. That averaging exactly reproduces
  v_sync + Σ_c J_c/m. The following then hold:
  - momentum is conserved;
  - averaging is KE-non-increasing (Jensen), so with e ≤ 1 per local solve, no energy can be
    created — which rules out R1;
  - at a fixed point the copies agree, so the solution solves the unsplit problem.
- Interior bodies have k = 1, so np 1 on a closed domain with hub degree ≤ the colour cap stays
  bitwise.
- **Over-splitting is safe.** An idle copy contributes Δv = 0 to the average. So k may be any upper
  bound on the number of active copies, for example the number of ranks holding the body. That is
  known to the owner at ghost-gather time with no extra message, at the price of
  under-relaxation.
- dem stores inverse mass in the `(N,4)` w column, so pre-scaling w and the inverse inertia by k
  for the substep may need no kernel change. Verify this.

**One unified treatment of**:
- (i) rank faces: copies on different ranks, reconciled by the existing reverse+forward, now
  weighted;
- (ii) D1 hubs: a body with degree > the colour cap is split into local copies, i.e. extra virtual
  vertices in the colouring graph, so the colour count stays bounded and every colour is race-free;
- (iii) D3 Jacobi paths: they become mass-split Jacobi, which is conservative serially too.

**The other two invariants:**
- **Single application point.** Every contact impulse acts at one point x_c, so r_A = x_c − x_A and
  r_B = x_c − x_B. That fixes D2. It changes np 1 on the legacy friction path; say whether that is
  acceptable in the note. My view: yes, it is a physics bug.
- **Visibility.** Every pair within contact reach is visible to at least one rank and solved by
  exactly one. That means:
  - band = reach + skin;
  - an ownership refresh (migration) triggered by a global vote when any body has moved more than
    skin/2 since its last migration (dem already has a global Verlet-skin vote);
  - periodic ghosts identified by (gid, image shift), with every image within the band sent,
    which fixes D4b in core's halo or in dem's use of it.

## 5. What the note must decide (genuinely open)
1. **Accept, amend or reject** the mass-splitting framework. If rejected, say what replaces it, and
   show it meets every property in §1.
2. **The choice of k.** Either the exact active-copy count (one extra reverse+forward per substep)
   or an upper bound known for free (ranks holding the body). Weigh convergence against messages.
   Also say how k is defined for hubs.
3. **Hub splitting in detail.** How copies are allocated (virtual body slots? the SoA layout?), how
   contacts are assigned to copies (balanced, deterministic given the manifold order), and when
   copies reconcile: after each colour sweep, each iteration, or each sync. This must hold for all
   five colourings, including multilevel. Would a bounded-colour greedy with splitting be cheaper on
   the GPU than an unbounded colouring? Quantify the launches per sweep.
4. **What happens to the one-shot g = 0 restitution sweep.** Is it kept as a mass-split one-shot, or
   folded into the accumulated-impulse (PGS) form? Check what the register says before changing the
   restitution model (`docs/decisions/dem.md`: colored GS restitution, the Poisson bank,
   restitution threshold).
5. **What is kept from c771e07.** Owner-exclusive contacts, owned-first lists, the reverse primitive,
   baselines, WO-3b count sync, allMaxAny. Remove whatever mass splitting makes redundant; say
   which.
6. **The warm-start ledger and the Poisson bank under splitting.** The review's plausible double
   credit of the Poisson bank after migration (`mpi_halo.hpp:826`, `solve_driver.hpp:403`) is part
   of this.
7. **Migration trigger and skin value.** The default, and its interaction with
   `rebalance_every`, `sync_every` and the existing skin vote. Core versus dem ownership of the
   multi-image fix. A change to core means core is tagged and published before dem; say if it is
   needed.
8. **Hertz–Mindlin engine.** It already conserves by construction. Confirm it is unaffected by D4
   (visibility), and apply the visibility fix to it too.

## 6. Constraints and invariants (settled; do not relitigate)
- **Every method on-device and MPI-distributable.** No host serial production path. Kokkos
  CUDA/HIP/OpenMP; sources `.hpp`/`.cpp`, never `.cu`. Identifiers never name where code runs.
- **Keep the modern stack** (`demSolveContacts` shared with single-GPU): coloured GS restitution,
  warm-started PGS with gid-keyed persistent contacts, statics/stabilization (one-sided, multilevel,
  escalate, ordered), friction cone, and adaptive stops Allreduce-MAXed. Do not replace the whole
  solver with Jacobi.
- **Local particle order is canonical** (ascending source rank). np 4/8 must stay run-to-run bitwise
  at 1 thread.
- **Deadlock-safe collectives**: the one-sided halo, the global skin vote, the empty-rank vote, and
  `enable_mpi_step(sync_every=M)` semantics. All ranks take every collective in the same order.
- **The `MigratePack` ledger stays gid-keyed.**
- **np 1 closed bitwise unchanged** wherever no body exceeds the colour cap and the legacy friction
  pass is not active. Name every np 1 change explicitly: the D1 hubs, D2 friction, D3 Jacobi.
- **Never change numerics silently.** Every intended change is listed with its measured magnitude.
- **USER DIRECTIVE: match or exceed SOTA massively-parallel performance.** No per-contact syncs, and
  no per-colour messages.

## 7. Performance requirements
Give an a-priori cost model:
- messages and bytes per substep, against c771e07;
- extra kernels and launches per sweep, especially hub copies and reconciliation on the GPU;
- extra memory for the copies;
- the migration frequency the skin implies.

Gates:
- ms/step at N ≈ 20000 periodic, np 4 × 4 and np 8 × 2, interleaved A/B against c771e07 with
  `docs/momentum_evidence/run_perf_ab.sh`, load recorded;
- one CUDA single-GPU ms/step, spheres and a ring bed, before/after;
- `forward_rotation=False` measured.

Target: no regression beyond noise for spheres. For ring beds, state the expected cost of hub
splitting against today's racy baseline.

## 8. Verification the note must specify
- **Conservation:** dP, CoM and dLvel to round-off, at np 1/2/4/8 × OMP 1/8, and a CUDA single-GPU
  run, in every solver mode:
  - gas / e;
  - PGS + cone;
  - friction legacy and cone;
  - Jacobi;
  - multilevel / escalate / ordered stabilization;
  - Poisson;
  - Hertz.
- **Energy:** KE non-increasing for e ≤ 1 in a force-free closed scene, every step. Also the
  review's 3-body face scene: np ≥ 2 KE within a stated tolerance of np 1 once converged, and never
  above np 1 by more than round-off.
- **Visibility oracle** (catches R2): gather the committed positions and enumerate all pairs within
  reach with an O(N²) oracle (or a cell list). Each such pair must have been solved exactly once
  globally (gather the solved gid pairs). Also compare the overlap of the committed state with
  np 1. The oracle must fail the review's mutant: include that mutant as a negative control.
- **Race-freedom:** a colouring validity check (no two same-colour contacts share a body copy) in
  every colouring, on the hub scenes and a ring bed, on CPU and CUDA. Also the D1 CUDA scene: dP
  back to round-off.
- **Convergence:** np ≥ 2 converges to np 1 as the iteration count grows (the fixed point is the
  coupled solution), shown on the 3-body chain, the p–q–s chain and a dense cluster.
- **D4:** the drift and periodic modes lose 0 pairs, including at a 4 R shift and on an
  undecomposed periodic axis.
- **Existing gates:**
  - the full dem battery (152 ctests on branch `contacts`), built with
    `-DMPIEXEC_PREFLAGS="--bind-to none"` (with a space);
  - the 12 `python_mpi` tests with core's Python build on PYTHONPATH, not skipped;
  - coupling's MPI tests (Ergun, moving suspension);
  - np 4/8 run-to-run bitwise at 1 thread.
- **Mutation tests**, as negative controls for the new gates:
  - drop ghost contacts;
  - remove the averaging weights;
  - reintroduce the colour cap.

## 9. Deliverable
A design note at `docs/contact_solve_framework.md` in the worktree, committed on branch `contacts`
(stage only that path). The commit message ends with `Co-Authored-By: Claude Opus 5.5
<noreply@anthropic.com>`. Do not push. Sections:
1. The invariant and a derivation of the framework, with proofs or proof sketches: conservation
   (linear, angular), KE monotonicity, fixed point = coupled solution, race-freedom.
2. The rejected alternatives, and why.
3. How each existing phase maps onto it: restitution, PGS, cone friction, legacy friction,
   position, the stabilization modes, Poisson, Jacobi.
4. Hub splitting and colouring.
5. Visibility: band, skin, migration trigger, periodic images.
6. What happens to c771e07's machinery.
7. The cost model.
8. Work orders, in dependency order, each with its acceptance check. Each is implementable by an
   Opus implementer who must not decide anything open; say where the implementer must stop.
9. Gates with thresholds.
10. Register entries (new ones, and those superseded), plus the text for `docs/mpi.md` and
    `CLAUDE.md`.
11. Risks.

Return a summary of 20 lines or fewer: the commit, the framework as decided, the expected cost, and
any question that genuinely needs the user.

## 10. Out of scope
- Bitwise thread/GPU reproducibility.
- gamma calibration.
- flow.
- The collocated variable-density issue.
- New contact laws.
