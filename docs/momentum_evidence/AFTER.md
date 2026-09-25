# dem distributed contact solve: momentum conservation, AFTER the fix

Measured 2026-09-25 on branch `momentum` (dem worktree `suite/dem-momentum`), the tree of
`docs/mpi_momentum_conservation.md` WO-0..WO-6 (final code = WO-6; WO-5 and WO-6 are
byte-identical to WO-3/WO-4 at OMP_NUM_THREADS=1). "Before" is `BEFORE.md` plus, for the three
modes added in WO-0, `baseline/matrix_new.txt` (a build of WO-0, `src/` = `c79dea5` = `e4e17f0`).

Host: the same Threadripper PRO 5965WX, shared. **The load was 95–120 (1-min average) during
every run below** (BEFORE: 3–43), from other sessions; see each section.

## Summary

- **Linear momentum and the centre of mass are conserved to round-off at every np, thread count
  and `sync_every`.** dP ≤ 9.5e-9 (≤ 8.0e-7 under free fall, the np-1 float floor; Hertz
  unchanged at ≤ 3.0e-8), dX ≤ 1.0e-6 R,
  dXpos ≤ 1.0e-6 R, over np 1/2/4/8 × OMP 1/8 × 3 repeats, all 8 modes. Before: dP 3.3e-3..1.3e-2,
  dX 7.5e-3..2.9e-2 R at np ≥ 2. `momentum_*` is now a gate (G1): no threshold exceeded.
- **Velocity-phase angular momentum** is at the serial model's level: dLvel ≤ 5e-9 frictionless,
  ≤ 2.5e-8 PGS cone; with the legacy friction pass 1.6e-5..4.0e-5 against its serial floor of
  1.9e-5 (§1.4 of the note; gate 1e-4).
- **Periodic wrap pairs**: `cluster_periodic` np 1 dP 4.1e-3 → 2.7e-9 (the self-image twins are
  solved once), np 2..8 3.9e-2..5.2e-2 → ≤ 4.2e-9.
- **The p–q–s chain**: |comShift| = 5.6e-7 and posErr = 1.9e-6 in 8/8 runs at np 2 and 4, OMP 1
  and 8 (before: a rigid 2.133e-2 shift in 4/8 and 5/8 runs at 8 threads). `ghost_band_*` run
  unpinned.
- **Bitwise**: np 1 closed byte-identical to the baseline (5 modes); np 4/8 5/5 runs
  byte-identical at OMP 1; every inert work order (WO-1, WO-2, WO-5, the WO-6 reformat)
  byte-identical to its predecessor.
- **Statistical agreement with single-rank improved**: np 2 mean 3.9e-3 / q95 3.2e-2 / max 6.6e-2
  (was 5.3e-3 / 3.6e-2 / 0.10), np 4 4.0e-3 / 3.2e-2 / 7.2e-2 (was 5.0e-3 / 3.5e-2 / 0.11).
- **Convergence premise (G8)**: ovl after/before = 0.61..1.00 in every cell — the interface
  coupling did not slow the position loop; it ends with less overlap than the redundant solve.
- **Battery: 134/134 after WO-3b** (was 128/130: `demstep_jacobi_closed_np2` and `_np4` failed
  with posErr 0.157, tol 1e-2; fixed by global Jacobi counts, below; +4 `momentum_cluster_jacobi`).
  All 12 `python_mpi_*` pass (not skipped).
- **Coupling (Ergun, moving suspension, np 1/2/4)**: every printed observable identical before
  and after.

## WO-3b: global per-body counts for the count-averaged Jacobi solves (was open, G7 / R10)

**The failure.** The count-averaged solves (`velocityUseGS = false`: `solveVelocityKokkos` →
`applyVelocityDeltasAveragedKokkos`, `solvePositionKokkos` → `applyUpdatesKokkos`, plus the
colour-saturation fallbacks of the GS loops) divide each body's summed correction by its per-body
`constraintCounts`. Serial semantics: the factor is **per body**, not per contact (velocity
`min(1, 2/count_i)`, position `1/count_i`; body A and B of one contact are scaled by their own
counts). Since WO-3 each rank counts only the contacts it owns, so a count missed the pairs the
partner rank solves:

```
[jacobi_closed  ] np=2 particles=64 ghosts(total)=64 posErr=1.566e-01 (tol 1e-02)
```

**The fix** (`ParticleHalo::syncContactCounts`, the `syncFrictionCounts` pattern on the int
column): between each Jacobi kernel and its apply, the ghosts' partial counts are reverse-summed
onto their owners and the totals forwarded back, so the owner and every ghost copy divide by the
serial count; the ghost's scaled increment is then delivered by the existing reverse. The Jacobi
A/B is a global setting, so that sync is unconditional there (one extra count exchange per Jacobi
iteration, velocity and position; this path is diagnostic). The GS loops' saturation fallback is
decided per rank (`velLeftover`, `posLeftover` come from the rank-local colouring), so its
activation is voted inside the loop's existing stop Allreduce (`allMaxAny`: two floats, MAX, the
same one message); only if some rank has leftovers do all ranks sync counts and apply. No message
was added to the GS production path.

| gate | result |
|---|---|
| `demstep_jacobi_closed` posErr, np 1 / 2 / 4 (tol 1e-3 / 1e-2 / 1e-2) | 0 / 5.9e-6 / 7.6e-6 (was 0 / 0.157 / 0.157) |
| `demstep_jacobi_periodic` posErr, np 2 / 4 (tol 0.30) | 4.8e-7 / 0.106 (was 2.3e-2 / 0.107) |
| `cluster_jacobi` (new ctest mode), np 1 | dP 9.0e-3, dX 2.1e-2 R, dXpos 8.2e-4 R, dLvel 3.2e-3: serial per-body averaging is not conservative (§4.2, R7) |
| `cluster_jacobi` np 2/4/8 × OMP 1/8 × 3 (`after/matrix_jacobi.txt`) | max dP 9.1e-3, dX 2.2e-2, dXpos 8.3e-4, dLvel 3.2e-3 = the np 1 level (before WO-3b: dXpos 2.4e-2..3.1e-2, dP 1.1e-2..1.3e-2) |
| `cluster_jacobi` gate (tolOf) | dP 1.5e-2, dX 3e-2, dXpos 2e-3, dLvel 5e-3 |
| np 1 closed vs the WO-0 dumps (5 modes, OMP 1) | byte-identical; `cluster_jacobi` np 1 byte-identical to its parent |
| GS path np 2/4/8, 6 modes, vs the parent commit (OMP 1) | 18/18 byte-identical (the vote does not change a bit) |
| run-to-run np 4/8, OMP 1, 3 runs: cluster, cluster_friction, cluster_pgs, cluster_jacobi | 1 distinct hash in all 8 cells |
| GS modes matrix, np 1..8 × OMP 1/8 (`after/matrix_gs_wo3b.txt`) | all within G1 (dP ≤ 9.5e-9, 8.0e-7 free fall; dXpos ≤ 1.1e-6 R) |
| battery (`after/battery_wo3b.txt`), OMP 2, -j1 | 134/134, 12 `python_mpi_*` ran |

`jacobi_periodic` np 4 (0.106) is unchanged by WO-3b and within its divergence guard; the scene
(L = 11, rcut 3) violates P1 on its decomposed axes (block 5.5 < 2 · 3, R8).

**Finding, pre-existing, not changed here:** the saturation fallback is unreachable. Both
colourings pick `col` with `while (col < 62 && forbidden bit)`, so a body whose 62 low colours are
taken gives colour 62 to every further contact instead of leaving it uncoloured (-1); the
arbitration colours at least one contact per round, so the stall break never fires and
`leftover` is always 0. A body of degree > 63 therefore has several same-colour contacts: an
in-place race on multi-thread and GPU backends (reading of the code; not reproduced). A trial
scene, the `cluster` grains with a radius-4 grain at the centre in place of those within 3.9, gave
leftover 0 on every rank at np 1..8, so the fallback branch with the vote taken is untested; it
was not kept as a test. The WO-3b vote is in place for when the colouring leaves leftovers.

## Conservation matrix (G1)

Command: `REPS=3 MODES="cluster cluster_friction cluster_pgs cluster_posonly hertz cluster_sync3
cluster_norot cluster_periodic" docs/momentum_evidence/run_matrix.sh build_mom`, raw lines in
`after/matrix.txt`. `run_matrix.sh` now exports `OMP_WAIT_POLICY=passive`: at load ~110 an
np 8 × 8-thread run with spinning OpenMP waits exceeded the 600 s timeout (26 s passive). Timing
only; numerics unchanged. Load 97–105 throughout (`after/load.txt`).

Cells show the median [min–max] of 3 runs; "(=)" means all 3 agreed to every printed digit.

#### `cluster`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.8e-09 (=) | 3.2e-07 (=) | 3.3e-07 (=) | 8.2e-04 (=) | 8.2e-04 (=) | 2.9e-09 (=) |
| 1 | 8 | 3.9e-09 [1.9e-09–4.6e-09] | 3.6e-07 [2.2e-07–4.6e-07] | 3.6e-07 [2.1e-07–4.5e-07] | 8.1e-04 [7.9e-04–8.2e-04] | 8.1e-04 [7.9e-04–8.2e-04] | 2.1e-09 [1.4e-09–2.4e-09] |
| 2 | 1 | 1.8e-09 (=) | 1.4e-07 (=) | 1.4e-07 (=) | 8.1e-04 (=) | 8.1e-04 (=) | 4.2e-09 (=) |
| 2 | 8 | 2.5e-09 [1.8e-09–3.2e-09] | 2.1e-07 [1.8e-07–2.5e-07] | 2.0e-07 [1.8e-07–2.5e-07] | 8.2e-04 [8.2e-04–8.3e-04] | 8.2e-04 [8.2e-04–8.3e-04] | 3.2e-09 [3.2e-09–3.3e-09] |
| 4 | 1 | 2.0e-09 (=) | 1.6e-07 (=) | 1.6e-07 (=) | 8.1e-04 (=) | 8.1e-04 (=) | 2.1e-09 (=) |
| 4 | 8 | 2.6e-09 [2.4e-09–3.6e-09] | 2.6e-07 [1.2e-07–2.8e-07] | 2.6e-07 [1.2e-07–2.8e-07] | 8.0e-04 [8.0e-04–8.1e-04] | 8.0e-04 [8.0e-04–8.1e-04] | 2.3e-09 [2.1e-09–2.6e-09] |
| 8 | 1 | 2.7e-09 (=) | 2.4e-07 (=) | 2.4e-07 (=) | 7.9e-04 (=) | 7.9e-04 (=) | 2.4e-09 (=) |
| 8 | 8 | 2.9e-09 [2.7e-09–3.7e-09] | 3.6e-07 [2.0e-07–3.9e-07] | 3.6e-07 [2.0e-07–4.0e-07] | 7.7e-04 [7.6e-04–7.8e-04] | 7.7e-04 [7.6e-04–7.8e-04] | 2.6e-09 [2.3e-09–2.9e-09] |

#### `cluster_friction`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.7e-09 (=) | 1.3e-07 (=) | 1.3e-07 (=) | 3.2e-04 (=) | 3.2e-04 (=) | 1.9e-05 (=) |
| 1 | 8 | 3.2e-09 [2.0e-09–3.9e-09] | 1.5e-07 [1.4e-07–2.1e-07] | 1.5e-07 [1.4e-07–2.1e-07] | 3.1e-04 [3.1e-04–3.2e-04] | 3.1e-04 [3.1e-04–3.2e-04] | 2.0e-05 [1.6e-05–2.2e-05] |
| 2 | 1 | 9.5e-09 (=) | 1.6e-07 (=) | 1.6e-07 (=) | 3.3e-04 (=) | 3.3e-04 (=) | 2.1e-05 (=) |
| 2 | 8 | 6.4e-09 [4.3e-09–7.9e-09] | 1.6e-07 [1.1e-07–2.5e-07] | 1.6e-07 [1.1e-07–2.6e-07] | 3.2e-04 [3.2e-04–3.3e-04] | 3.2e-04 [3.2e-04–3.3e-04] | 2.1e-05 [1.9e-05–2.4e-05] |
| 4 | 1 | 3.0e-09 (=) | 2.5e-07 (=) | 2.5e-07 (=) | 3.2e-04 (=) | 3.2e-04 (=) | 2.5e-05 (=) |
| 4 | 8 | 3.7e-09 [2.4e-09–4.5e-09] | 1.6e-07 [1.5e-07–2.7e-07] | 1.6e-07 [1.5e-07–2.7e-07] | 3.2e-04 [3.2e-04–3.2e-04] | 3.2e-04 [3.2e-04–3.2e-04] | 2.2e-05 [1.8e-05–2.4e-05] |
| 8 | 1 | 3.1e-09 (=) | 1.4e-07 (=) | 1.4e-07 (=) | 3.0e-04 (=) | 3.0e-04 (=) | 3.5e-05 (=) |
| 8 | 8 | 2.9e-09 [2.1e-09–3.2e-09] | 2.0e-07 [1.7e-07–2.1e-07] | 2.0e-07 [1.7e-07–2.1e-07] | 3.0e-04 [3.0e-04–3.1e-04] | 3.0e-04 [3.0e-04–3.1e-04] | 3.5e-05 [3.3e-05–3.7e-05] |

#### `cluster_pgs`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 7.9e-07 (=) | 4.2e-07 (=) | 1.4e-07 (=) | 3.8e-04 (=) | 3.8e-04 (=) | 1.9e-08 (=) |
| 1 | 8 | 7.9e-07 [7.9e-07–7.9e-07] | 4.3e-07 [3.5e-07–5.3e-07] | 2.5e-07 [2.1e-07–2.9e-07] | 3.8e-04 [3.7e-04–3.8e-04] | 3.8e-04 [3.7e-04–3.8e-04] | 1.8e-08 [1.8e-08–2.5e-08] |
| 2 | 1 | 8.0e-07 (=) | 3.7e-07 (=) | 2.6e-07 (=) | 3.8e-04 (=) | 3.8e-04 (=) | 1.7e-08 (=) |
| 2 | 8 | 8.0e-07 [7.9e-07–8.0e-07] | 4.6e-07 [4.5e-07–6.0e-07] | 2.3e-07 [1.9e-07–2.4e-07] | 3.8e-04 [3.8e-04–3.8e-04] | 3.8e-04 [3.8e-04–3.8e-04] | 1.6e-08 [1.5e-08–1.7e-08] |
| 4 | 1 | 7.9e-07 (=) | 4.9e-07 (=) | 1.0e-07 (=) | 3.7e-04 (=) | 3.7e-04 (=) | 1.6e-08 (=) |
| 4 | 8 | 7.9e-07 [7.9e-07–7.9e-07] | 6.0e-07 [5.3e-07–6.3e-07] | 2.6e-07 [1.4e-07–4.4e-07] | 3.7e-04 [3.7e-04–3.8e-04] | 3.7e-04 [3.7e-04–3.8e-04] | 1.7e-08 [1.5e-08–1.8e-08] |
| 8 | 1 | 7.9e-07 (=) | 4.8e-07 (=) | 4.4e-07 (=) | 3.8e-04 (=) | 3.8e-04 (=) | 1.8e-08 (=) |
| 8 | 8 | 7.8e-07 [7.8e-07–7.9e-07] | 5.3e-07 [4.4e-07–6.2e-07] | 2.8e-07 [1.2e-07–3.5e-07] | 3.8e-04 [3.8e-04–3.9e-04] | 3.8e-04 [3.8e-04–3.9e-04] | 1.7e-08 [1.5e-08–1.8e-08] |

#### `cluster_posonly`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.0e+00 (=) | 4.9e-07 (=) | 4.9e-07 (=) | 2.9e-03 (=) | 2.9e-03 (=) | 0.0e+00 (=) |
| 1 | 8 | 0.0e+00 (=) | 4.9e-07 [4.1e-07–8.5e-07] | 4.9e-07 [4.1e-07–8.5e-07] | 2.9e-03 [2.9e-03–2.9e-03] | 2.9e-03 [2.9e-03–2.9e-03] | 0.0e+00 (=) |
| 2 | 1 | 0.0e+00 (=) | 7.0e-07 (=) | 7.0e-07 (=) | 2.9e-03 (=) | 2.9e-03 (=) | 0.0e+00 (=) |
| 2 | 8 | 0.0e+00 (=) | 8.1e-07 [5.1e-07–8.8e-07] | 8.1e-07 [5.1e-07–8.8e-07] | 2.9e-03 [2.9e-03–3.0e-03] | 2.9e-03 [2.9e-03–3.0e-03] | 0.0e+00 (=) |
| 4 | 1 | 0.0e+00 (=) | 4.6e-07 (=) | 4.6e-07 (=) | 2.9e-03 (=) | 2.9e-03 (=) | 0.0e+00 (=) |
| 4 | 8 | 0.0e+00 (=) | 3.9e-07 [2.6e-07–9.1e-07] | 3.9e-07 [2.6e-07–9.1e-07] | 2.9e-03 [2.9e-03–2.9e-03] | 2.9e-03 [2.9e-03–2.9e-03] | 0.0e+00 (=) |
| 8 | 1 | 0.0e+00 (=) | 1.0e-06 (=) | 1.0e-06 (=) | 2.9e-03 (=) | 2.9e-03 (=) | 0.0e+00 (=) |
| 8 | 8 | 0.0e+00 (=) | 4.4e-07 [3.1e-07–6.9e-07] | 4.4e-07 [3.1e-07–6.9e-07] | 2.9e-03 [2.9e-03–2.9e-03] | 2.9e-03 [2.9e-03–2.9e-03] | 0.0e+00 (=) |

#### `hertz`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.8e-08 (=) | 8.8e-07 (=) | n/a | 4.4e-07 (=) | 4.4e-07 (=) | n/a |
| 1 | 8 | 2.2e-08 [2.1e-08–2.9e-08] | 8.7e-07 [8.7e-07–8.8e-07] | n/a | 4.3e-07 [4.3e-07–4.4e-07] | 4.3e-07 [4.3e-07–4.4e-07] | n/a |
| 2 | 1 | 1.5e-08 (=) | 8.7e-07 (=) | n/a | 4.4e-07 (=) | 4.4e-07 (=) | n/a |
| 2 | 8 | 2.5e-08 [2.1e-08–2.8e-08] | 8.8e-07 [8.8e-07–8.8e-07] | n/a | 4.4e-07 [4.3e-07–4.4e-07] | 4.4e-07 [4.3e-07–4.4e-07] | n/a |
| 4 | 1 | 2.4e-08 (=) | 8.7e-07 (=) | n/a | 4.3e-07 (=) | 4.3e-07 (=) | n/a |
| 4 | 8 | 2.2e-08 [1.5e-08–2.6e-08] | 8.7e-07 [8.7e-07–8.8e-07] | n/a | 4.4e-07 [4.3e-07–4.4e-07] | 4.4e-07 [4.3e-07–4.4e-07] | n/a |
| 8 | 1 | 2.5e-08 (=) | 8.7e-07 (=) | n/a | 4.3e-07 (=) | 4.3e-07 (=) | n/a |
| 8 | 8 | 2.3e-08 [2.1e-08–3.0e-08] | 8.7e-07 [8.7e-07–8.8e-07] | n/a | 4.3e-07 [4.3e-07–4.4e-07] | 4.4e-07 [4.3e-07–4.4e-07] | n/a |

#### `cluster_sync3`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.7e-09 (=) | 1.3e-07 (=) | 1.3e-07 (=) | 3.2e-04 (=) | 3.2e-04 (=) | 1.9e-05 (=) |
| 1 | 8 | 3.3e-09 [3.3e-09–4.6e-09] | 1.8e-07 [1.6e-07–2.5e-07] | 1.8e-07 [1.6e-07–2.5e-07] | 3.1e-04 [3.1e-04–3.1e-04] | 3.1e-04 [3.1e-04–3.1e-04] | 1.7e-05 [1.5e-05–1.8e-05] |
| 2 | 1 | 5.8e-09 (=) | 9.8e-08 (=) | 9.9e-08 (=) | 3.3e-04 (=) | 3.3e-04 (=) | 2.3e-05 (=) |
| 2 | 8 | 5.6e-09 [4.1e-09–6.4e-09] | 2.9e-07 [1.7e-07–3.8e-07] | 2.9e-07 [1.7e-07–3.9e-07] | 3.1e-04 [3.1e-04–3.2e-04] | 3.1e-04 [3.1e-04–3.2e-04] | 2.4e-05 [2.2e-05–2.4e-05] |
| 4 | 1 | 5.6e-09 (=) | 2.4e-07 (=) | 2.3e-07 (=) | 3.1e-04 (=) | 3.1e-04 (=) | 2.7e-05 (=) |
| 4 | 8 | 3.9e-09 [3.1e-09–5.5e-09] | 1.3e-07 [9.0e-08–2.7e-07] | 1.3e-07 [9.0e-08–2.7e-07] | 3.1e-04 [3.1e-04–3.2e-04] | 3.1e-04 [3.1e-04–3.2e-04] | 2.8e-05 [2.6e-05–2.8e-05] |
| 8 | 1 | 5.8e-09 (=) | 2.8e-07 (=) | 2.8e-07 (=) | 3.0e-04 (=) | 3.0e-04 (=) | 2.9e-05 (=) |
| 8 | 8 | 3.9e-09 [3.3e-09–5.9e-09] | 2.0e-07 [1.2e-07–2.4e-07] | 2.0e-07 [1.3e-07–2.4e-07] | 2.9e-04 [2.9e-04–3.0e-04] | 2.9e-04 [2.9e-04–3.0e-04] | 2.8e-05 [2.8e-05–2.9e-05] |

#### `cluster_norot`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.7e-09 (=) | 1.3e-07 (=) | 1.3e-07 (=) | 3.2e-04 (=) | 3.2e-04 (=) | 1.9e-05 (=) |
| 1 | 8 | 2.6e-09 [2.5e-09–4.2e-09] | 2.1e-07 [1.4e-07–3.3e-07] | 2.1e-07 [1.4e-07–3.3e-07] | 3.1e-04 [3.1e-04–3.2e-04] | 3.1e-04 [3.1e-04–3.2e-04] | 1.8e-05 [1.7e-05–2.4e-05] |
| 2 | 1 | 6.6e-09 (=) | 4.6e-07 (=) | 4.5e-07 (=) | 3.3e-04 (=) | 3.3e-04 (=) | 1.8e-05 (=) |
| 2 | 8 | 6.8e-09 [5.3e-09–7.3e-09] | 1.5e-07 [1.0e-07–2.1e-07] | 1.5e-07 [1.0e-07–2.0e-07] | 3.2e-04 [3.2e-04–3.2e-04] | 3.2e-04 [3.2e-04–3.2e-04] | 2.1e-05 [1.8e-05–2.4e-05] |
| 4 | 1 | 5.3e-09 (=) | 3.2e-07 (=) | 3.1e-07 (=) | 3.2e-04 (=) | 3.2e-04 (=) | 2.4e-05 (=) |
| 4 | 8 | 3.8e-09 [3.8e-09–4.2e-09] | 1.7e-07 [1.5e-07–2.6e-07] | 1.8e-07 [1.5e-07–2.5e-07] | 3.2e-04 [3.2e-04–3.2e-04] | 3.2e-04 [3.2e-04–3.2e-04] | 2.3e-05 [2.2e-05–2.5e-05] |
| 8 | 1 | 3.4e-09 (=) | 4.0e-07 (=) | 4.0e-07 (=) | 3.0e-04 (=) | 3.0e-04 (=) | 3.3e-05 (=) |
| 8 | 8 | 5.0e-09 [4.9e-09–6.2e-09] | 1.5e-07 [1.4e-07–2.0e-07] | 1.4e-07 [1.3e-07–2.1e-07] | 3.0e-04 [2.9e-04–3.0e-04] | 3.0e-04 [2.9e-04–3.0e-04] | 3.6e-05 [3.3e-05–4.0e-05] |

#### `cluster_periodic`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.7e-09 (=) | n/a | n/a | 2.9e-02 (=) | 2.9e-02 (=) | n/a |
| 1 | 8 | 3.7e-09 [2.9e-09–4.2e-09] | n/a | n/a | 2.9e-02 [2.8e-02–3.0e-02] | 2.9e-02 [2.8e-02–2.9e-02] | n/a |
| 2 | 1 | 3.8e-09 (=) | n/a | n/a | 2.6e-02 (=) | 2.6e-02 (=) | n/a |
| 2 | 8 | 3.8e-09 [3.7e-09–3.8e-09] | n/a | n/a | 2.6e-02 [2.5e-02–2.7e-02] | 2.6e-02 [2.6e-02–2.6e-02] | n/a |
| 4 | 1 | 3.3e-09 (=) | n/a | n/a | 2.4e-02 (=) | 2.7e-02 (=) | n/a |
| 4 | 8 | 2.9e-09 [1.6e-09–3.0e-09] | n/a | n/a | 2.3e-02 [2.2e-02–2.3e-02] | 2.7e-02 [2.7e-02–2.7e-02] | n/a |
| 8 | 1 | 2.6e-09 (=) | n/a | n/a | 2.7e-02 (=) | 2.7e-02 (=) | n/a |
| 8 | 8 | 2.0e-09 [2.0e-09–2.6e-09] | n/a | n/a | 2.6e-02 [2.5e-02–2.6e-02] | 2.6e-02 [2.6e-02–2.7e-02] | n/a |


### Before, for the three modes added in WO-0

`baseline/matrix_new.txt` (build of WO-0, spinning waits: two np 8 × 8 repeats of
`cluster_periodic` hit the 600 s timeout at load ~115).

#### `cluster_sync3`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.7e-09 (=) | 1.3e-07 (=) | 1.3e-07 (=) | 3.2e-04 (=) | 3.2e-04 (=) | 1.9e-05 (=) |
| 1 | 8 | 4.8e-09 [3.6e-09–6.0e-09] | 2.3e-07 [1.4e-07–4.5e-07] | 2.3e-07 [1.4e-07–4.5e-07] | 3.2e-04 [3.1e-04–3.2e-04] | 3.2e-04 [3.1e-04–3.2e-04] | 1.8e-05 [1.6e-05–2.0e-05] |
| 2 | 1 | 3.2e-03 (=) | 8.8e-03 (=) | 2.5e-03 (=) | 3.4e-03 (=) | 3.4e-03 (=) | 3.6e-03 (=) |
| 2 | 8 | 2.3e-03 [2.3e-03–3.0e-03] | 6.0e-03 [5.9e-03–8.1e-03] | 1.6e-03 [1.3e-03–1.7e-03] | 2.9e-03 [2.8e-03–3.5e-03] | 2.9e-03 [2.8e-03–3.5e-03] | 3.1e-03 [3.0e-03–3.7e-03] |
| 4 | 1 | 7.8e-03 (=) | 2.1e-02 (=) | 3.1e-03 (=) | 4.5e-03 (=) | 5.1e-03 (=) | 5.2e-03 (=) |
| 4 | 8 | 7.5e-03 [6.2e-03–7.7e-03] | 1.9e-02 [1.5e-02–2.0e-02] | 2.4e-03 [1.9e-03–2.7e-03] | 4.2e-03 [3.9e-03–4.8e-03] | 4.5e-03 [4.4e-03–5.2e-03] | 4.6e-03 [4.5e-03–5.3e-03] |
| 8 | 1 | 1.1e-02 (=) | 2.6e-02 (=) | 3.2e-03 (=) | 3.5e-03 (=) | 3.6e-03 (=) | 3.3e-03 (=) |
| 8 | 8 | 8.8e-03 [8.8e-03–8.8e-03] | 2.2e-02 [2.1e-02–2.2e-02] | 2.9e-03 [2.9e-03–2.9e-03] | 3.8e-03 [3.6e-03–4.3e-03] | 4.1e-03 [3.6e-03–4.4e-03] | 4.1e-03 [3.5e-03–4.3e-03] |

#### `cluster_norot`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.7e-09 (=) | 1.3e-07 (=) | 1.3e-07 (=) | 3.2e-04 (=) | 3.2e-04 (=) | 1.9e-05 (=) |
| 1 | 8 | 4.1e-09 [3.0e-09–7.9e-09] | 2.0e-07 [9.8e-08–2.2e-07] | 2.0e-07 [9.1e-08–2.2e-07] | 3.1e-04 [3.1e-04–3.1e-04] | 3.1e-04 [3.1e-04–3.1e-04] | 1.6e-05 [1.5e-05–2.0e-05] |
| 2 | 1 | 4.1e-03 (=) | 9.8e-03 (=) | 1.2e-03 (=) | 2.1e-03 (=) | 2.0e-03 (=) | 2.3e-03 (=) |
| 2 | 8 | 4.6e-03 [3.5e-03–5.2e-03] | 9.1e-03 [8.7e-03–1.2e-02] | 8.8e-04 [3.2e-04–1.9e-03] | 1.6e-03 [9.9e-04–2.6e-03] | 1.5e-03 [8.8e-04–2.4e-03] | 1.7e-03 [1.1e-03–2.6e-03] |
| 4 | 1 | 6.0e-03 (=) | 1.7e-02 (=) | 4.4e-03 (=) | 1.6e-03 (=) | 1.5e-03 (=) | 1.7e-03 (=) |
| 4 | 8 | 4.3e-03 [3.9e-03–5.8e-03] | 1.1e-02 [7.7e-03–1.4e-02] | 1.3e-03 [1.1e-03–2.6e-03] | 1.6e-03 [1.5e-03–2.1e-03] | 1.6e-03 [1.4e-03–2.0e-03] | 1.8e-03 [1.5e-03–2.2e-03] |
| 8 | 1 | 4.1e-03 (=) | 9.1e-03 (=) | 2.3e-03 (=) | 2.9e-03 (=) | 2.9e-03 (=) | 2.7e-03 (=) |
| 8 | 8 | 3.0e-03 [2.7e-03–3.3e-03] | 7.2e-03 [6.6e-03–1.0e-02] | 4.1e-03 [2.1e-03–4.2e-03] | 2.5e-03 [2.4e-03–3.0e-03] | 2.7e-03 [2.5e-03–3.1e-03] | 2.5e-03 [2.3e-03–3.1e-03] |

#### `cluster_periodic`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.1e-03 (=) | n/a | n/a | 2.9e-02 (=) | 2.8e-02 (=) | n/a |
| 1 | 8 | 4.9e-03 [4.1e-03–6.7e-03] | n/a | n/a | 3.2e-02 [2.9e-02–3.3e-02] | 3.1e-02 [2.7e-02–3.1e-02] | n/a |
| 2 | 1 | 3.9e-02 (=) | n/a | n/a | 5.5e-02 (=) | 5.2e-02 (=) | n/a |
| 2 | 8 | 4.6e-02 [1.7e-02–4.9e-02] | n/a | n/a | 4.7e-02 [4.6e-02–4.7e-02] | 4.7e-02 [4.6e-02–4.7e-02] | n/a |
| 4 | 1 | 4.1e-02 (=) | n/a | n/a | 3.9e-02 (=) | 4.0e-02 (=) | n/a |
| 4 | 8 | 4.2e-02 [3.7e-02–5.2e-02] | n/a | n/a | 5.5e-02 [4.6e-02–5.9e-02] | 5.5e-02 [4.6e-02–6.0e-02] | n/a |
| 8 | 1 | 4.6e-02 (=) | n/a | n/a | 7.8e-02 (=) | 7.9e-02 (=) | n/a |
| 8 | 8 | 5.2e-02 (=) | n/a | n/a | 6.2e-02 (=) | 6.5e-02 (=) | n/a | (2 timed out)


## The p–q–s chain (G2)

`docs/momentum_evidence/run_chain.sh build_mom` (default OpenMP wait policy, as BEFORE), 8 runs
per cell, raw lines in `after/chain.txt`. Load 99.

| np | threads | runs passing (\|comShift\| ≤ 1e-5 and posErr ≤ 1e-4) | max \|comShift\| (dist) | max posErr | serial reference CoM |
|---|---|---|---|---|---|
| 2 | 8 | **8 / 8** | 5.6e-7 | 1.9e-6 | −1.4e-6 … −1.2e-6 |
| 2 | 1 | 8 / 8 | 5.6e-7 | 1.9e-6 | −1.2e-6 |
| 4 | 8 | **8 / 8** | 5.6e-7 | 1.9e-6 | −1.2e-6 … −0.7e-6 |
| 4 | 1 | 8 / 8 | 5.6e-7 | 1.9e-6 | −1.2e-6 |

The `ghost_band_*` ctests pass with the one-thread pin removed (OMP 2 in the battery).

## Bitwise gates (G3, G4, inertness)

`after/bitwise.txt`; dumps from `test_momentum_mpi --dump` at OMP_NUM_THREADS=1.

- **G3** np 1 closed, final build vs `baseline/np1_*_base.bin`: `cluster`, `cluster_friction`,
  `cluster_pgs`, `cluster_posonly`, `hertz` byte-identical. `cluster_periodic` np 1 changes, as
  the note states (dP 4.103e-3 → 2.706e-9). np 2 (and every np ≥ 2) changes wherever a
  cross-rank or self-image contact exists; single-rank `demStep` is untouched (`tests/kokkos`,
  `tests/arborx`, `python_tests` pass).
- **G4** np 4 and np 8, `cluster` / `cluster_friction` / `cluster_pgs`: one distinct hash over
  5 runs in all 6 cells.
- **Inert orders**: WO-1 and WO-2 vs the WO-0 build, all 8 modes × np 1/2/4/8: 32/32 identical
  each; WO-5 (fused forwards) vs WO-4: 32/32 identical; the WO-6 clang-format rebuild vs WO-4 at
  np 1 and 4: identical.

## Convergence premise (G8): `ovl`

`baseline/ovl.txt` vs `after/ovl.txt`, one run per cell at OMP 1 (max over steps of the global
position-loop residual). Gate: after ≤ 2 × before.

| mode | np 2 | np 4 | np 8 |
|---|---|---|---|
| cluster | 3.26e-2 → 2.79e-2 | 3.48e-2 → 2.79e-2 | 3.59e-2 → 2.87e-2 |
| cluster_friction | 3.08e-2 → 2.42e-2 | 3.21e-2 → 2.49e-2 | 3.04e-2 → 2.47e-2 |
| cluster_pgs | 3.44e-2 → 2.46e-2 | 4.08e-2 → 2.49e-2 | 3.65e-2 → 2.49e-2 |
| cluster_posonly | 3.29e-2 → 2.82e-2 | 3.51e-2 → 2.80e-2 | 3.64e-2 → 2.89e-2 |
| cluster_sync3 | 3.83e-2 → 3.56e-2 | 4.02e-2 → 3.56e-2 | 3.97e-2 → 3.60e-2 |
| cluster_periodic | 1.67e-1 → 2.95e-2 | 1.32e-1 → 2.80e-2 | 1.90e-1 → 2.81e-2 |

np 1 is unchanged in every closed mode (the same bits).

## Ownership and the reverse primitive (G5, G6, G11)

`after/ownership.txt`. Every contact is owned exactly once in the closed, periodic and drift
layouts at np 1/2/4/8 (0 duplicates, owned union = serial active set); `drift` has 2/7/8
cross-rank pairs that only one rank sees and fails, as designed, with the lower-gid-only rule
(2/4/5 pairs missing). The reverse primitive sums exact integers, leaves ghost = owner and is
idempotent. **G11**: the same 20 cells pass on nvidia-cuda (RTX 5080; `build_cuda_mom`), where
the struct `Kokkos::atomic_add` compiles and runs; `cluster` / `cluster_pgs` at np 1/2/4 on the
GPU give dP 2.2e-9..3.9e-9 / 7.6e-7..7.7e-7. R1's "≤ 5 % of a reverse" cost was not timed.

Finding (pre-existing, outside the ownership rule): with a strongly jittered lattice in a fully
periodic box, np 2 and 4 lose 2–3 corner-wrap pairs that **no** rank reports. A pair that wraps
across a decomposed and an undecomposed periodic axis at once needs a second image of the partner
on one destination rank, and core's halo keeps one image per (particle, rank). The committed
`exactly_once_periodic` uses the weak jitter. It belongs with R4/R8 of the note.

## Battery (G7)

`OMP_NUM_THREADS=2 OMP_PROC_BIND=false PYTHONPATH=core/build_rel_py ctest --test-dir build_mom -j1`
(130 = 98 + 12 WO-0 modes + 8 `ownership_reverse_*` + 12 `ownership_exactly_once_*`; `align_np8`
last): **128 passed, 2 failed** (`demstep_jacobi_closed_np2`, `_np4`, above). All 12
`python_mpi_*` ran and passed; `python_tests`, `tests/kokkos`, `tests/arborx` pass.
`after/battery.txt`.

## Statistical agreement with single-rank (`test_validate_exact`, N = 200, 15 steps, OMP 1)

`after/statistical_agreement.txt`; before = a build of `e4e17f0`'s `src/`, same session.

| np | before: mean / q95 / max | after: mean / q95 / max |
|---|---|---|
| 1 | 2.2e-6 / 2.4e-5 / 8.5e-5 | 2.2e-6 / 2.4e-5 / 8.5e-5 (same bits) |
| 2 | 5.3e-3 / 3.6e-2 / 1.03e-1 | 3.9e-3 / 3.2e-2 / 6.6e-2 |
| 4 | 5.0e-3 / 3.5e-2 / 1.09e-1 | 4.0e-3 / 3.2e-2 / 7.2e-2 |

## Coupling (G10)

`coupling/tests/test_mpi_moving_suspension.py` and `test_mpi_fixed_bed_ergun.py` at np 1/2/4,
copied to a scratch directory (the moving test writes its np-1 reference next to itself), with
coupling built from `coupling` `8191a1e` and flow built from `flow` `f4b105e` (`-DPECLET_FLOW_MPI=ON`,
host-openmp), both in scratch trees. Before = dem `e4e17f0` `src/`; after = this branch.
`after/coupling_{before,after}.txt`. **Every printed observable is identical** (the moving
suspension's mean_vx to 9 digits: np 1 −2.27080469e-01, np 2 −2.27080478e-01, np 4
−2.27080473e-01; Ergun U = 1.5042e-3 / 1.4982e-1 / 6.4216 at every np); all PASS. Neither test has
body–body contacts (the suspension is dilute, the bed fixed), so this checks the plumbing only.

## Performance (G9)

**Not measurable at BEFORE's conditions in this session; the gate is open.** The 1-min load stayed
at 85–120 for the whole afternoon (BEFORE: 19–30). With the default (spinning) OpenMP waits, one
`perf_gas` 4 × 4 run of the BEFORE build took 8444 ms/step (15.6 ms in BEFORE.md), so the
protocol as written cannot run here. `docs/momentum_evidence/run_perf_ab.sh` interleaves the two
builds (each repeat runs before, then after, back to back; otherwise the `run_perf.sh` protocol:
N = 19683 periodic, np 4 × 4 and np 8 × 2 on cores 8–23, taskset-pinned, 10 warm-up + 50 timed
steps). It was run with `OMP_WAIT_POLICY=passive` at load 85–93. Before = `build_base` (WO-0,
`src/` = `c79dea5`); after = the final build. Raw: `after/perf_ab_passive.txt`.

| mode | np × thr | before runs (ms/step) | after runs (ms/step) | min before → after | ratio (min) | median ratio |
|---|---|---|---|---|---|---|
| perf_gas | 4 × 4 | 68.7, 74.2, 69.0, 66.3, 73.1 | 71.6, 78.3, 71.5, 87.3, 89.0 | 66.3 → 71.5 | 1.079 | 1.135 |
| perf_gas | 8 × 2 | 87.2, 85.4, 97.4, 95.6, 108.9 | 112.2, 111.8, 111.2, 112.4, 98.2 | 85.4 → 98.2 | **1.150** | 1.169 |
| perf_pgs | 4 × 4 | 100.8, 89.9, 97.4, 82.4, 102.5 | 86.6, 98.8, 83.3, 93.0, 88.5 | 82.4 → 83.3 | 1.011 | 0.909 |
| perf_pgs | 8 × 2 | 98.1, 89.5, 106.6, 101.6, 108.5 | 102.2, 122.9, 105.9, 109.1, 98.5 | 89.5 → 98.5 | 1.100 | 1.043 |

These absolute times are 5–12 × BEFORE's, and the spread inside one cell (e.g. 71–89 ms) is larger
than the effect being measured. Taken at face value, `perf_gas` at np 8 × 2 misses the 1.10 gate
(1.150). That configuration has the most syncs per rank, and `perf_gas` adds the two
legacy-friction count rounds. Each reconciliation adds four small element-wise kernels (pack,
zero, apply, mark); on an overloaded host with passive waits every OpenMP fork/join is expensive.
Neither this measurement nor the a-priori +1–4 % of §5 settles the gate. **Re-run it on a quiet
host** (expected output: 40 `PERF ... rep=R build={before,after}` lines, then `EXIT perf`), and
compare the min of 5 per cell against 1.10 × before:

```bash
cd suite/dem-momentum
mkdir -p /tmp/perf_after/tests/kokkos_mpi && cp build_mom/tests/kokkos_mpi/test_momentum_mpi /tmp/perf_after/tests/kokkos_mpi/
docs/momentum_evidence/run_perf_ab.sh build_base /tmp/perf_after > docs/momentum_evidence/after/perf_ab.txt
```

(`build_base/tests/kokkos_mpi/test_momentum_mpi` is the frozen WO-0 binary in the worktree. If
the after build misses the gate on a quiet host, §5's levers apply in order: a fused
bidirectional exchange, which needs a core primitive, then a two-ended narrow-phase append. Each
is a separate decision.)
