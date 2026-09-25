# Architect brief 2: amend the contact-solve framework where implementation falsified its premises

## 1. The question
The design note `docs/contact_solve_framework.md` (dem 0e5a9eb, with session decisions §12 S1–S9)
has been implemented through WO-4. Implementation has now falsified one premise and exposed one
conservation hole, and WO-5 is under-specified in four places. Write an amendment, a new §13 in the
same note. It must make WO-5, WO-6 and WO-7 implementable without further decisions, and it must
fix the two defects in a principled way. Say where the fix belongs: in WO-4's committed code, or in
a new work order.

## 2. Why the architect
The user asked for a rigorous, principled design, not patches (2026-09-25): "It feels that we are
trying to patch this issue while it needs a rigorous design ... Take performance also into account.
Test it well." These issues are in the method, not the code.

## 3. State
Worktree: `suite/dem-contacts`, branch `contacts`. Not pushed, and must not be pushed before WO-5
lands (S9).

Commits on the branch:

| work order | commit |
|---|---|
| WO-0 | 3e9a870 |
| WO-1 (midpoint friction) | 07114fb |
| WO-2 (mass-split Jacobi) | 5c91df9 |
| WO-3 (units, S6 wall ids, S7 round cap) | 785d984 |
| WO-4 (complete colouring, hub copies, local folds) | ca32026 |

All implementation numbers are in `docs/contact_evidence/IMPL_A.md`, with raw data in
`docs/contact_evidence/impl_a/`.

WO-4 results:
- **Hubs conserve:** `hub*` dP ≤ 6.1e-7 on host and CUDA, against a 1e-2 baseline. Colouring has 0
  conflicts and 0 leftovers.
- **Non-hub runs are unchanged:** byte-identical.
- **Periodic self-images conserve:** `cluster_periodic --solo` CoM drift 1.8e-6 R, down from 2.3e-3 R.
- **Battery:** 192/195. The 3 failures are item A below.

## 4. What is broken, and what is open

**A. Premise falsified: over-relaxing the position projection (`kSplitOmegaPosition = 1.5`).**
- With ω_pos = 1.5 on demStep's split periodic image copies, an isolated wrap pair is pushed apart
  by 1.5 × its overlap. The gap is never pulled back.
- Failures it causes:
  - `test_wrap_pair_matches_in_box_pair`: 0.150 vs 0.100.
  - `validate_periodic` np 2: serial 0.900 vs distributed 0.800. Rank 0 asserts and rank 1 hangs.
  - `validate_periodic` np 4: 0.846 vs 0.800, and the straddler check fails.
- With ω = 1.0 all three pass (tried, not committed).
- WO-5 would give every rank-split pair the same over-separation.
- The session's reading:
  - Over-relaxing an incremental, non-accumulated projection of a unilateral constraint overshoots
    into a gap that the constraint (λ ≥ 0) cannot retract.
  - Over-relaxation is legitimate only on an accumulated multiplier with a projection that can
    retract (projected SOR on the accumulated λ, clamped at 0), or not at all.
- The note's interface convergence estimate relied on ω = 1.5: 1.04–1.2× serial iterations, against
  1.16–1.9× with ω = 1.
- **Decide:** ω = 1 everywhere, or an accumulated-λ PSOR form for the position projection (and its
  cost), or another principled form. Give the convergence cost at interfaces and hubs under the
  choice, and whether the split k should be exact per unit/contact to reduce under-relaxation.

**B. Conservation hole: the multilevel stabilizer at a hub (implemented as the note specified).**
- "Re-seed after the coarse cycle" (§4.4), combined with solve-view group masses (§4.5), does not
  conserve momentum at an aggregated hub: the hub keeps m·dV where it should keep (m/s)·dV.
- No gate scene aggregates a hub, so there are no numbers yet.
- **Decide** the correct formulation, prove it conserves, and specify a gate scene that aggregates a
  hub.

**C. WO-5 is under-specified** (the implementer's list):
1. The velocity mask needs `col(rank)`, which is WO-6's rank colouring. Either reorder the work
   orders or restate the dependency.
2. The forward payloads carry no orphan balance, yet §4.6 step 5 seeds ghost shares from "the
   forwarded balance". Specify the payload and the message count.
3. Under rank-level mass splitting, local hub copies must take the base's B/k share, and the
   phase-end owner balance must be restored. The implementer's formula is in IMPL_A.md; verify or
   replace it.
4. The per-slot degree and the per-slot global k are needed every substep under MPI. Specify how
   they are computed and communicated, fused with existing messages where possible, and their cost.

**D. Re-derive the interface convergence and the cost model under the answer to A**, and restate the
WO-5 acceptance: `ring_mini` at np 4/8 (currently divergent, S9), the conservation matrix, the
interface-iteration ratio, np 4/8 bitwise, and the performance gate.

## 5. Settled (do not relitigate)
- Mass splitting with consensus for the projection-form phases; exclusive holding (X) for the g = 0
  one-shot.
- Complete colouring with hub copies, and S1–S9.
- Round-off nondeterminism across threads and the GPU is accepted; no contact sorting.
- np 1 closed runs are byte-identical except where the note lists a change.
- Every method is on-device and MPI-distributable; collectives are deadlock-safe; ledgers are
  gid-keyed.
- Performance: match or exceed the state of the art. No per-contact or per-colour messages.

## 6. Deliverable
Append §13 "Amendment after WO-4" to `docs/contact_solve_framework.md`. It must contain:
- the resolution of A–D, with proofs or proof sketches;
- revised WO-5, WO-6 and WO-7 texts, plus any new work order, e.g. a WO-4b fix for A/B on the
  committed code;
- the revised gates;
- anything superseded in §2–§9, marked in place with "superseded by §13".

You may run throwaway experiments in the scratchpad
`/tmp/claude-1003/-home-frankp-Codes-suite/53a85a4c-bcc1-4b4e-8987-b25c5c7a74d8/scratchpad`.

Commit only the note on `contacts`. The message ends with `Co-Authored-By: Claude Opus 5.5
<noreply@anthropic.com>`. Do not push.

Return a summary of 15 lines or fewer, including any question that genuinely needs the user.

## 7. Out of scope
- WO-8/9 (periodic images, done in core).
- The serial PGS energy creation from separating contacts (recorded as an open issue).
- gamma calibration.
- flow.
