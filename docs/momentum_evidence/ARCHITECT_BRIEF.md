# Architect brief: momentum-conserving cross-rank contacts in dem's distributed XPBD step

## 1. The question
Design how each cross-rank contact in dem's distributed contact solve gets ONE impulse (velocity solve)
and ONE correction (position projection), applied equal and opposite to both bodies, so that total
linear momentum and the centre of mass are conserved to round-off at any np and thread count. The
velocity phase must also conserve angular momentum. The design must keep Gauss–Seidel convergence and
must not add a sync per contact. Deliver a design note with the scheme, the rejected alternatives,
work orders and gates.

## 2. Why it needs the architect
This changes the parallel structure of the solver: what state each rank sweeps from, what is
communicated, and when. It is release-blocking (USER, 2026-09-25): "That momentum is not conserved
is not acceptable. This should be solved." It changes numerics at np ≥ 2. It interacts with the
warm-started PGS ledger, the colouring, the adaptive stops, statics/stabilization, and the collective
sync schedule that took several deadlock fixes to get right.

## 3. Current state
Paths are relative to the worktree `suite/dem-momentum` (branch `momentum`, dem e4e17f0 + c79dea5).

- **Code digest with excerpts:** `docs/momentum_evidence/recon_digest.md` (≈600 lines, file:line
  anchors). Read it before the source; open the source only where the digest is insufficient.
- **Measured evidence:** `docs/momentum_evidence/BEFORE.md`, and the new report-only test
  `tests/kokkos_mpi/test_momentum_mpi.cpp`.

Structure in brief:
- `demStepMpi` (`src/step_solve_mpi.hpp:93`) runs: predict owned → gather ghosts → narrow phase over
  owned + ghost → `demSolveContacts(P, …, MpiSolveHooks)` (`src/solve_driver.hpp:239`) → commit.
- `demSolveContacts` is the same modern sequence as the single-GPU step:
  - colouring;
  - warm-started PGS with gid-keyed persistent contacts;
  - colored GS restitution;
  - velocity sweeps with adaptive stop;
  - statics/stabilization (one-sided/multilevel);
  - friction cone;
  - position projection.
- `MpiSolveHooks` has:
  - `syncVelocities` / `syncPositions`: forward owner→ghost refresh every `syncEvery` iterations
    and after every phase;
  - `allMax`: Allreduce-MAX of the adaptive-stop residuals.
- Colouring is rank-local over owned + ghost manifolds. Ghost bodies participate.
- Kernels scatter `+J/m_A`, `−J/m_B` (linear + angular) atomically by local index. They write to
  ghost slots too; ghost deltas are discarded and only owned rows are committed.
- Pair identity: `pairKeyFromGids(min, max)`. The persistent/warm-start and Hertz–Mindlin histories
  are gid-pair keyed, so each owner of a cross-rank pair holds its **own copy** of that pair's ledger
  entry.
- There is **no reverse (ghost→owner accumulate) halo operation** in dem today, only forward. Check
  core's `ParticleMigrator` / `gatherGhosts` (`../core/include/peclet/core/halo/`) for anything
  reusable.
- The ghost band is `max(rcut, 2.1 R_max_global)`, with a global narrow-phase margin, so both owners
  of a cross-rank pair see it as a contact.

## 4. The defect, measured (c79dea5, `BEFORE.md`)
A cross-rank contact is solved redundantly by both owners, each from its own state after its own
earlier sweeps. The two impulses are not equal and opposite.

Test scene: 925 polydisperse spheres centred on the corner all rank blocks share; closed and
force-free; 50 steps. The table gives the median of 3 runs.
- dP = |P(t) − P(0)| / Σ m|v0|
- dX = centre-of-mass drift in radii
- dLvel = angular-momentum drift over the velocity phase

| mode | np 1 | np 2/4/8, 1 thread (8 threads: same order) |
|---|---|---|
| cluster (plain XPBD) | dP 2.8e-9, dX 3e-7, dLvel 3e-9 | dP 3.9e-3 / 6.2e-3 / 3.3e-3; dX 8e-3 to 1.4e-2 |
| cluster_friction | dP 3.7e-9, dX 1e-7, dLvel 1.9e-5 (serial floor) | dP ~4e-3; dX ~1e-2; dLvel 1e-3 to 3e-3 |
| cluster_pgs (production PGS + cone, gravity) | dP 7.9e-7 (gravity accumulation) | dP 7e-3 to 1.3e-2; dX 1.3e-2 to 2.9e-2 |
| cluster_posonly (velocity solve off) | dX 5e-7 | dX 3e-3 to 1e-2 |
| hertz (force engine) | dP 2.8e-8 | **identical to np 1: already conserves** |

Findings:
- **Not thread order.** The defect is present at 1 thread, where runs repeat to every digit. Threads
  only add run-to-run variation.
- **Both phases are at fault.** The velocity solve breaks P; the position projection alone breaks
  the centre of mass.
- **The p–q–s chain** (`test_ghost_band_mpi` mode `margin`, pin removed by override): it shifted
  rigidly by +2.133e-2 in 4 of 8 runs at np 2 and 5 of 8 at np 4 with 8 threads, and in 0 of 8 at
  1 thread. Serial moves ≤ 1.4e-6.
- **Angular momentum in plain XPBD is not conserved even at np 1** (dL 3e-4 to 3e-3), because the
  position projection moves bodies without changing their velocities. The gate therefore applies to
  the velocity phase, dLvel.
- **The friction pass has a serial dLvel floor of 1.9e-5.** The agent says this is because each
  body's impulse acts at its own surface point. Judge whether that floor is a real non-conservation
  in the serial model, i.e. a lever-arm asymmetry. If it is, say whether it is in scope or should be
  a separate recorded issue.

Performance before the fix: N = 19683, periodic, 16 cores per rank pinned with `taskset`, median of 5
runs, load 19–30. Script: `docs/momentum_evidence/run_perf.sh`.

| | np 4 × 4 threads | np 8 × 2 threads |
|---|---|---|
| gas | 15.6 ms/step | 8.1 ms/step |
| pgs | 15.9 ms/step | 9.3 ms/step |

## 5. Constraints and invariants
- **np 1 must stay bitwise unchanged.** np ≥ 2 may change.
- **Conservation to round-off.** Target relative ≲ 1e-6 in float for P and the centre of mass;
  dLvel likewise, except where the serial model itself has a floor.
- **Bitwise reproducibility across threads or the GPU is NOT required** (USER). **Do not sort
  contacts** to get it.
- **Keep run-to-run bitwise reproducibility at 1 thread** (canonical local order, ascending source
  rank, dem f7b7b22).
- **Every method must be on-device and MPI-distributable.** No host serial production path.
- **Keep the modern solver stack.** The distributed step must drive the SAME `demSolveContacts` as
  single-GPU: colored GS restitution, warm-started PGS with gid-keyed persistent contacts,
  statics/stabilization, friction cone, and Allreduce-MAXed adaptive stops. Do not revert the whole
  solver to count-averaged Jacobi. A conservative Jacobi-type treatment of **cross-rank contacts
  only** is legitimate.
- **Deadlock fixes stay**:
  - one-sided halo;
  - global skin vote;
  - ghost band max(rcut, 2.1 R_max global);
  - empty-rank vote;
  - `enable_mpi_step(sync_every=M)` semantics.

  Collective calls must be taken by all ranks together. A rank-local break deadlocks them.
- **Keep the ledgers gid-keyed.** The `MigratePack` rebalance/migration ledger stays gid-keyed, and
  so do the persistent contacts.
- **Kokkos device code.** CUDA/HIP/OpenMP. Sources are `.cpp`/`.hpp`, never `.cu`.
- **Naming.** Identifiers name what a thing is, never where it runs.

## 6. Already decided (not open)
- Conservation is required (register: "The distributed contact solve must conserve linear and
  angular momentum across rank boundaries").
- Accepting redundant two-owner solves is rejected.
- Round-off nondeterminism is accepted.
- Hertz–Mindlin already conserves. Measured: identical to np 1 at every np and thread count. Confirm
  why in one paragraph (both owners evaluate the same pair from the same synced state), then leave it
  alone.

## 7. Genuinely open (decide these)
1. **The scheme.** The handoff lists these candidates, which are not prescribed:
   - (a) **Single owner per cross-rank pair.** For example the lower-gid owner computes J and sends
     −J to the partner's owner, accumulated like a force. This needs a reverse halo accumulate.
   - (b) **Globally consistent colouring of interface contacts**, so both sides sweep them in the
     same colour from the same refreshed state.
   - (c) **Interface contacts solved Jacobi-style**, symmetric by construction, with interior
     contacts kept GS.

   Any of these must be made compatible with the rest of the stack:
   - the **warm start**: each owner holds its own copy of a cross-rank pair's accumulated lambda, so
     the two copies must stay identical or have one owner;
   - the **accumulated-impulse clamps** (cone, λ ≥ 0);
   - **count-averaging factors**, which are per-body and differ between ranks;
   - **statics/stabilization**, which has one-sided passes by design;
   - the **position projection**.
2. **Angular momentum:** friction impulses applied at an offset.
3. **Cost per substep** (messages, syncs) against today's: syncs every `syncEvery` iterations plus
   one per phase, forward only.
4. **The gate definitions:**
   - which quantity is gated in which mode;
   - the free-fall mode threshold (np 1 is already 7.9e-7 from gravity accumulation);
   - the friction serial floor.

## 8. Already tried / rejected
- Pinning the ctests to one thread (c64e117) hides the symptom. Remove the pin once conservation
  holds.
- Sorting contacts by gid pair every substep: rejected by the user.
- Count-averaged Jacobi as the whole distributed solver: superseded by the shared modern stack;
  rejected.
- Accepting "statistical" agreement (`docs/mpi.md:69-72, 89-96`): that covers trajectories only, not
  conservation. The doc must be fixed.

## 9. Verification (the note must turn these into concrete gates)
- **`test_momentum_mpi` as a gate** at np 1/2/4/8 × OMP 1/8 threads. It covers:
  - dP and dX to round-off;
  - dLvel for the frictionless and friction modes;
  - the p–q–s chain with the pin removed, stable at 8 threads (count over 8 runs);
  - a random dense cluster straddling rank corners, which is what `cluster` already is.
- **The full dem battery.** It is now 98 ctests with the new test; `align_np8` runs last. Build with
  `-DMPIEXEC_PREFLAGS="--bind-to none"` (with a space).
- **The 12 `python_mpi_*` tests**, with core's Python build on PYTHONPATH.
- **np 4/8 bitwise run-to-run at 1 thread.**
- **np 1 bitwise unchanged.** Name the reference. State explicitly what np 2 does.
- **Coupling MPI tests** (Ergun, moving suspension): report the deltas.
- **ms/step** at N ≈ 20000 periodic, np 4 and 8, with `run_perf.sh`, state the load. State the
  expected overhead a priori.

## 10. Deliverable
A design note at `docs/mpi_momentum_conservation.md` in the worktree. Commit it on branch `momentum`,
staging only that path. The commit message ends with `Co-Authored-By: Claude Opus 5.5
<noreply@anthropic.com>`. Sections:
1. Root cause, in physical terms.
2. The chosen scheme, with a proof sketch of equal-and-opposite (linear + angular, velocity +
   position) and of the convergence expectation.
3. The rejected candidates, and why.
4. Interaction with the warm start, clamps, stabilization, adaptive stops, sync schedule, migration
   ledger, and GPU/atomics.
5. Communication cost.
6. Work orders, numbered and implementable by an Opus implementer who must not decide anything open.
   Include the exact files and functions to touch.
7. Gates with thresholds.
8. Doc/register text to change: `docs/mpi.md`, the `dem/CLAUDE.md` trap text.
9. Open risks.

Return a ≤ 20-line summary: the commit hash, the scheme, the expected cost, and any question that
genuinely needs the user.

## 11. Out of scope
- Bitwise thread/GPU reproducibility.
- gamma calibration.
- flow.
- The collocated variable-density issue.
- Redesigning the single-GPU solver.
- Changing the Hertz engine, unless the proof fails.
