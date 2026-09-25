# Review: owner-exclusive contacts + reverse accumulation (dem e4e17f0..c771e07)

Reviewed against `docs/mpi_momentum_conservation.md` (9fb8972). All line numbers are at c771e07.

## Verdict

**The conservation machinery is correct. Its verification cannot detect the two failures that
conservation hides.**

- **Correct:** linear momentum and the centre of mass are conserved by construction, and I found no
  path that double-counts or loses an increment. I re-ran it in configurations the brief did not
  cover (below), and it conserves in all of them.
- **Two findings to act on:**
  1. **Energy at rank faces, g = 0 path (confirmed).** The one-shot coloured-GS restitution path
     (g = 0) now sums impulses from different ranks' contacts on the same body with no way to take
     an over-push back. A three-body scene creates kinetic energy at np = 2.
  2. **Dropped contacts are invisible to the gates (confirmed).** A mutant that drops every contact
     with a ghost endpoint passes every momentum gate, and it lowers `ovl` 17-fold, so G8 cannot
     catch dropped pairs.

Evidence files (in this scratchpad): `tri.txt`, `tri_pgs.txt`, `ke_cluster.txt`,
`ke_cluster_e09.txt`, `mutant.txt`, `stab.txt`, `poisson.txt`. The test patches that produced them
are `review_test_patch_{after,before}.diff`: a KE print plus env-driven `tri` / `REVIEW_*` hooks, in
scratch worktrees that have since been removed.

All runs: host-openmp, Release, `--bind-to none`, OMP_NUM_THREADS=1 unless stated, load ≈ 57.

---

## Findings, most severe first

### 1. The g = 0 one-shot restitution path is a raw Jacobi sum at rank faces: under-dissipation, and energy creation for e ≳ 0.6

**Severity: should fix. Needs a decision; I would fix it. Status: CONFIRMED.**

**Where**
- `src/solve_driver.hpp:484`: `solveVelocityColoredGSKokkos`, the non-PGS path taken whenever g = 0.
- `src/solver_velocity.hpp:1652-1694`: the kernel applies the full `(1+e)` impulse whenever the pair
  is approaching. It keeps no accumulator, so a later iteration can never take back an over-push.

**Mechanism**
- Under owner-exclusive solving, a body B whose contacts are owned by two ranks is pushed by both
  within one sync interval, each push computed from B's stale state.
- This is common, not a corner case. Any face body with one local contact and one cross-face contact
  whose partner has the lower gid has k = 2.
- Design §2.5 ("same lag class as today"; diagonal dominance for k ≤ 3) argues only convergence. It
  relies on the PGS clamp to absorb over-push, and the one-shot path has no such clamp.
- The old scheme gave each body a GS-consistent update from its owner's view. It was not
  conserving, but it did not superpose.

**Reproduced: the `tri` scene.**
- Setup:
  - equal spheres, R = 0.5;
  - A1 and A2 at ±0.45 transverse, moving at 1 toward B, which sits across the x = 0 rank face;
  - A2B is owned by rank 0 and A1B by rank 1;
  - measured quantity is the kinetic energy in the CoM frame; the initial value is 0.333.
- Results:

| e | np 1 | np 2 before | np 2 after |
|---|---|---|---|
| 0.5 | 0.138 | 0.189 (dP 0.14) | **0.291** (dP 0) |
| 0.8 | 0.246 | 0.340 (dP 0.20) | **0.555** (dP 0), i.e. **above the initial 0.333** |

- The same scene on the PGS path (g = −10) is unaffected: 0.2229 at np 2 against 0.2234 at np 1
  (e = 0.8). The accumulated-λ pull-back works.

**In the gate scene (`cluster`, g = 0)**
- The error is sign-indefinite. Squeezed bodies are over-damped; same-side impacts are
  under-damped.
- KE relative to np 1:

| e | step | np 8 before | np 8 after |
|---|---|---|---|
| 0.5 | 2 | −0.2 % | −1.4 % |
| 0.9 | 45 | −3.2 % | −4.4 % |

- Energy is gated nowhere.

**Why it matters**
- dem's register rejected the raw Jacobi sum on exactly these grounds ("diverges hard for e ≳ 0.5").
- g = 0 distributed runs, i.e. HCS and growth packing, now carry that artefact at every face.

**Minimal correction:** go back to the architect.
- R2's pre-identified candidate: relax the cross-rank manifolds.
- Alternative: an accumulated, clampable impulse for the one-shot sweep under MPI.

**Verification to add:** the `tri` scene as a gated KE regression, np 2 against np 1 within a few
percent, at e = 0.5 and e = 0.8.

### 2. The gates cannot see a dropped contact, and G8/`ovl` is inverted by one

**Severity: should fix (verification). Status: CONFIRMED by mutation.**

**Where**
- `src/solve_driver.hpp:883-887` and `:919`: `maxOverlap` is written only by sweeps over
  `[0, ncOwned)`. A pair that no rank owns never enters it.
- `tests/kokkos_mpi/test_momentum_mpi.cpp:454`: `ovl` reads that value.

**Mutant**
- `ContactOwnership` returns false for every pair with a ghost endpoint, so no cross-rank or wrap
  contact is solved at all.
- `momentum_cluster`, `_pgs` and `_friction` at np 2 and np 8 all print `OK`:
  - dP 2e-9 to 8e-7;
  - dXpos ≤ 3e-7;
  - `ovl` at np 8 **drops** to 1.7e-3 (`cluster`) and 7.7e-4 (`cluster_pgs`), against the correct
    2.87e-2 / 2.47e-2;
  - ghosts rise 881 → 1154, because bodies interpenetrate across the faces.
- G8 ("after ≤ 2 × before") therefore rewards dropping.

**Consequences**
- The AFTER.md ratio of 0.61–1.00 partly reflects the metric narrowing:
  - before the change, `ovl` took the max over both ranks' views of every cross pair;
  - now it takes one view, and only of owned pairs.
- The only guard against dropping is the static `ownership_exactly_once_*` snapshot. Nothing guards
  the dynamic 50-step run, where drift and image effects occur.

**Correction:** measure the committed state independently in `test_momentum_mpi`.
- Gather positions to rank 0 each step.
- Compute an O(N²) max overlap (N = 925, so this is cheap) and/or the active-pair count.
- Gate both against np 1's envelope.
- Alternatively, compare Σ-over-ranks owned active keys with a serial narrow phase on the gathered
  state.

### 3. Poisson bank double-credit after a migration

**Severity: note. Status: PLAUSIBLE, by construction.**

**Where**
- `src/mpi_halo.hpp:826`: MigratePack carries each ledger entry on both locally owned endpoints.
- `src/solve_driver.hpp:403`: `scatterOrphanBanksKokkos`.

**Scenario**
1. A and B are owned by one rank and migrate to different ranks. Both ranks now hold the entry.
2. The pair separates in the first substep after the move.
3. Both ranks find the entry dead and credit both endpoints.

- Credits to ghost endpoints are now delivered (before, they were overwritten), so the orphan
  account receives 2× the bank.
- This affects the energy budget only, not momentum.

**Correction:** carry the entry only with the endpoint whose new owner would own the pair.
Alternatively, have the scatter credit only pairs this rank would own under `ContactOwnership`.

### 4. `partnerSees` does not check which periodic image the partner's owner holds; the margin case is not quite "no effect"

**Severity: note. Status: PLAUSIBLE, analytic.**

**Where:** `src/mpi_halo.hpp:155-165`.

**Image-blindness**
- `s ∈ copies(o)` is true if o is sent to s under *any* image.
- Where one rank holds the partner at the contact's image and the other rank holds a different one
  (P1 violated, or corner wraps), a pair that used to be solved one-sidedly is now dropped.
- This interacts with the known corner-wrap follow-up. When that follow-up is fixed, carry the shift
  in `copyRanks_`.

**Margin case (design §2.1)**
- The design says a dropped pair with gap ≈ margin "has no effect". That holds only at the
  narrow-phase configuration.
- The same pair can overlap at posPred within the substep and then escape that substep's
  projection.
- This happens only when rounding puts the gap exactly at the margin, so it is rare.

### 5. Performance: two items unmeasured

**Severity: note.**

- **Spheres with `forward_rotation=False`** now pay two rounds per sync instead of one; the design
  predicts +4–9 % at np 8. The perf A/B used rotation on, where the round count is unchanged. At
  load 55–60 its ±5 % noise could not resolve an effect of that size anyway.
- **CUDA cost of the struct lock atomics (R1)** was never timed.
- Both need a quiet or GPU measurement before the next scaling campaign.

### 6. The voted colour-saturation fallback is correct but relies on an unenforced invariant

**Severity: note.**

**Where:** `src/solve_driver.hpp:496-507` and `:917-927`.

**The invariant**
- A rank with no leftover still runs `syncContactCounts` and the apply on `deltaVel` / `deltaPos` /
  `constraintCounts` it never filled this iteration.
- This is harmless only because every apply zeroes what it consumed.
- Deadlock-safe: yes.

**Correction:** the velocity-side Jacobi pass does not write `maxApproach`, so it can run after the
vote on every rank. Also add a test when the colour-62 fix makes this path reachable.

---

## What I checked and found sound

- **Baseline differencing (your Q1).**
  - Every body write between `beginSolve` and the last `syncPositions` lands in a phase whose next
    sync delivers it:
    - warm start;
    - orphan scatter, where orphan decay is owned-only;
    - sweeps;
    - stabilization, all three modes;
    - multilevel coarse cycle;
    - legacy friction;
    - Jacobi applies.
  - Baselines are re-marked after each forward, so an in-loop sync followed by the final sync
    delivers exactly zero.
  - Break-before-sync is covered by the unconditional phase-final sync.
  - `publishPositions` is correctly forward-only.
  - M > 1 adds nothing to the logic.
  - `exchanges()` counts self-ghosts, so the np 1 periodic reverse runs.
- **Ownership.**
  - The exactly-once proof holds under P1, given core's single nearest image per destination
    (`particle_migrator.hpp` `withinRcutOfBlock`).
  - Self-image twins are handled correctly at np 1 and np 2.
  - Flips cold-start a pair's warm start, as accepted in R3.
- **Collective schedule.**
  - `allMaxAny` sits in the same place, with the same count, on every rank.
  - The count and friction syncs run only under global predicates.
  - The reverse is the transpose of the forward, so the one-sided halo is safe.
  - Empty and zero-contact ranks still take every Allreduce.
- **Device.** Core's reverse scatter uses `Kokkos::atomic_add` on the struct, which is a lock-based
  desul atomic and correct on every backend. `orphanPeak` combines by max as documented.
- **Extra runs (np 1–8), all at the free-fall dP floor of 7.8–8.0e-7:**
  - `cluster_pgs` with the multilevel, escalate and ordered stabilization modes;
  - `cluster_pgs` with Poisson restitution, OMP 1 and 8.

  Neither is in the gate battery; both are cheap to add.

**Not examined:**
- CUDA;
- python_mpi;
- coupling;
- the chain test;
- docs;
- the onesided stabilization mode;
- a dynamic run with `rebalance_every` > 0 or Verlet skin > 0 under the new scheme.
