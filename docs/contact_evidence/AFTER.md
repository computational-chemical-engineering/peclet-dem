# Contact-solve framework: evidence (WO-11, 2026-09-26)

- **Design:** `docs/contact_solve_framework.md`, i.e. §1–§11, the §12 session decisions S1–S24,
  and the §13 amendment.
- **Per-work-order records:** `IMPL_A.md`.
- **Commits** (dem, all on main):

  | work order | commit | work order | commit |
  |---|---|---|---|
  | WO-0 | 3e9a870 | WO-5b | 4665e0f |
  | WO-1 | 07114fb | WO-6 | 4441c2e |
  | WO-2 | 5c91df9 | WO-7 | aea487a, 3b11393 |
  | WO-3 | 785d984 | WO-9 | 4035ac2 |
  | WO-4 | ca32026 | WO-10 | 274f4bb, b4ed7ae, 1cbf3c0 |
  | WO-4b | 8b4a5f3 | performance | ffa41b4 |
  | WO-5 | 0ea32ba | | |

- **Core** (main, unreleased 1.3.0): efa9b0d (allImages), 9625788 (sendShift), 3e51d39 (the exact
  per-axis image prefilter).

## 1. Conservation (G1). Host, OMP 1 and 8, np 1/2/4/8; and CUDA np 1/2

| scene | before (dem e4e17f0) | after |
|---|---|---|
| frictionless cluster, dP / CoM | 3–6e-3 / ~1e-2 R | ≤ 5e-9 / ≤ 4e-7 R |
| friction cluster, dP / dLvel | 4–5e-3 / 1e-3 | ≤ 8e-9 / ≤ 7e-9 |
| free fall, production PGS + cone | 0.7–1.3e-2 | ≤ 8e-7 (the float floor of gravity) |
| hub bodies (> 63 contacts), CUDA | 1–4e-2 | ≤ 6e-7 |
| rings, dLvel | 5e-4 (np 1 too) | 2–4e-8 |
| Jacobi path, np 1 | 9e-3 | ≤ 1e-8 |
| periodic box | 4e-3 (np 1 step_mpi) – 5e-2 | ≤ 4e-9 |
| sheared, bodies migrating (XPBD / Hertz) | — / 9e-4 at the first migration | ≤ 1.6e-8 / ≤ 4e-8 |

The ctests `momentum_*` gate all of these, with thresholds 10–20× above the measured maxima.

## 2. Energy (G2)
The review's 3-body face scene, KE after the collision:
- e = 0.5: np 1/2/4/8 all give 0.1379044264;
- e = 0.8: all give 0.2459279908.

Before, c771e07 gave 0.2907 and 0.555, i.e. energy created. Axes 0, 1 and 2 all match. KE is
non-increasing in `cluster`, `cluster_e09` and `hub`; `cluster_e10`, the elastic case, is flat to
2e-8. The `tri` ctest gates the np 1 value to 1e-6.

## 3. Convergence (G7)
With every adaptive stop off:
- np 2/4/8 converge to np 1 at the float floor: 1.4e-7 for N ≥ 32 on the 3-body scene;
- in the dense cluster the gap is ≤ 1.2e-4 at 256 iterations.

Interface iteration ratio: 1.3–2.4× the np 1 iterations.

## 4. Visibility (G5, G6)

| probe | before | after |
|---|---|---|
| `oracle_closed`, np 1–8 | 0 missing | missing = dup = 0 |
| `oracle_shear` (bodies leave an unwalled domain) | 592–737 missing (WO-0) | 0 |
| `oracle_periodic` | 12–30 missing | 0 |
| drift probes to 6 R | up to 54 missed | 0 |
| colouring conflicts on hubs / ring beds | 1240–6439 same-colour pairs per step | 0 |

## 5. Reproducibility
- np 4/8 run-to-run are bitwise identical at 1 thread in every scene tested, drift migrations
  included.
- np 1 closed is byte-identical except for the named changes: rings (the S15 friction inertia),
  the Jacobi path, hubs, periodic `step_mpi`, and legacy friction (the midpoint).

## 6. Negative controls (G13)
Seven mutants, each compiled only under `PECLET_DEM_TEST_MUTANT`. All 8 WILL_FAIL ctests detect
their mutant.

## 7. Cost
Pinned, interleaved A/B, `run_perf_ab.sh`, against c771e07; host load 3–10; ms/step, min of 5.

| scene | before | after | ratio |
|---|---|---|---|
| N = 19683, gas, np 4×4 | 14.95 | 17.43 | 1.17 |
| N = 19683, gas, np 8×2 | 7.85 | 9.10 | 1.16 |
| N = 19683, PGS, np 4×4 | 14.83 | 17.91 | 1.21 |
| N = 19683, PGS, np 8×2 | 7.72 | 9.52 | 1.23 |
| N = 157464, gas, np 4×4 | 100.2 | 116.4 | 1.16 |
| N = 157464, gas, np 8×2 | 59.7 | 68.4 | 1.15 |
| N = 157464, PGS, np 4×4 | 102.7 | 119.3 | 1.16 |
| N = 157464, PGS, np 8×2 | 58.1 | 68.7 | 1.18 |

**The design's 10 % gate is NOT met; the cost is +15–23 %.** Two items were found and fixed:
- core's image enumeration walked all 27 shifts for every particle × rank × rebuild. That was
  +19 % at np 8; core 3e51d39 fixes it exactly;
- a separate X-gate pass per iteration; it is now inline, bitwise identical.

**Where the rest goes** (Kokkos Tools regions, np 4×2, N = 157464, cores 0–7; the two runs used
the same profiler):
- **All of the extra time is inside `demSolveContacts`:** 65.5 → 86.7 ms/step. Broad phase,
  narrow phase and gather are unchanged.
- **About half is kernels (+10 ms):**
  - the per-pair position sweep, +6.1: two indirections per contact even when every unit is a
    single contact, as for spheres;
  - velocity PGS, +1.4: 30 % more iterations, because the stop now waits for the copies to agree
    (S14);
  - the rank-level M halo kernels, +2.2;
  - the drift-vote reduction, +1.0;
  - colouring, +1.5.
- **The other half is MPI and host time:** the M openings and syncs, and the consensus residual.

Candidate optimizations, recorded as a follow-up:
1. A singleton fast path for position units: when every unit is one contact, sweep contacts
   directly. It must keep colouring and sweep indices consistent, i.e. stay bitwise identical.
2. Fuse the drift vote into the existing radius reduction.
3. Fuse the M pack, apply and seed kernels.
4. Measure the consensus-stop iteration cost against a tolerance scaled to the interface.

Also, host-side topology work is O(N_local × P) per rebuild (pre-existing). It should use the
block-neighbour list before large rank counts.

## 8. Open, outside this package
- The multilevel coarse cycle's angular momentum (S17). The mode is opt-in; it needs rigid-body
  aggregates.
- R-U4 / WO-12: an accumulated, retractable position projection. It is a user decision.
- Serial PGS gains energy on contacts that were separating (+6 % at e = 0.9), a contact-law issue.
- The position phase's effective mass carries rotational terms although the rotation is never
  applied (`computeW`).
- Core 1.3.0 publish; then move dem's `PECLET_CORE_TAG`.

## 9. WO-12: the accumulated, retractable overlap projection (USER decision 2026-09-26)

The change: each contact's net position push Λ ≥ 0 is projected, Λ' = max(0, Λ − ω C / w̃), and the
change applied, so an overshoot is retracted. ω_pos = 1.5 everywhere, np 1 included. The stop is
max |d| w̃ < 1e-4 R. This is a named numerics change for every run with coupled contacts.

**Uniqueness.** The overlap-only substep, converged with 2000 iterations and the stops off, gives
positions whose max difference from np 1 is:

| | np 2 | np 4 | np 8 |
|---|---|---|---|
| after | 1.5e-5 R | 2.0e-5 R | 4.7e-5 R |
| before (non-accumulated POCS) | 1.0e-2 R | 9.5e-3 R | 8.6e-3 R |

The `position_agreement_np{2,4,8}` ctests gate this at 1e-4 R.

**Iterations to converge** at np 1, stops on (`after11/wo12_omega_scan.txt`):

| scene | old | ω 1.0 | 1.3 | **1.5** | 1.7 |
|---|---|---|---|---|---|
| dense cluster, overlap only | 97 | 97 | 56 | **34** | 30 |
| cluster, PGS with gravity | 105 | 106 | 60 | **35** | 29 |
| hub | 8 | 10 | 9 | **13** | 25 |

That is 2.9× fewer on clusters. The hub pays a few iterations, and ω = 1.7 is worse on hubs, so
the design's 1.5 stays. `ring_mini` does not converge under either form (overlap ~0.2–0.25 R);
that is the pre-existing `computeW` / translation-only issue, open.

**Cost:** within ±1 % of the pre-WO-12 build in the benchmark (clean cores 0–15;
`after11/wo12_perf.txt`), where the iteration cap of 8 binds. The gain shows only where the
adaptive stop ends the loop.

**Gates:** battery 264/264, the 3 new agreement gates included; mutant 7 is redefined as "no
retraction" and detected. CUDA: the subset passes 59/59; the device loop converges in 33
iterations (host 34).

## 10. Performance package (2026-09-26 evening): the framework's cost, closed

**Result.** Interleaved three-way A/B against c771e07 and main 0520b21 (`after12/perf_ab3.txt`;
cores 0–15, min of 3, host load 5–15, ms/step):

| scene | c771e07 | main 0520b21 | this package | new / c771e07 |
|---|---|---|---|---|
| N = 157464, gas, np 4×2 | 137.84 | 150.55 | 141.08 | 1.023 |
| N = 157464, gas, np 8×2 | 64.38 | 69.93 | 65.38 | 1.016 |
| N = 157464, PGS, np 4×2 | 136.46 | 151.25 | 139.46 | 1.022 |
| N = 157464, PGS, np 8×2 | 62.74 | 70.37 | 64.71 | 1.031 |
| N = 19683, gas, np 4×2 | 16.69 | 18.63 | 17.50 | 1.048 |
| N = 19683, gas, np 8×2 | 10.38 | 11.44 | 10.74 | 1.034 |
| N = 19683, PGS, np 4×2 | 17.16 | 19.45 | 17.68 | 1.030 |
| N = 19683, PGS, np 8×2 | 10.44 | 12.26 | 11.06 | 1.060 |

Every change is bit-identical: byte-equal state dumps against main in 29 scenes at np 1/2/4/8 (1
thread), friction-pair output equal, battery 264/264 host, CUDA build + ctests.

**Where the §7 cost really was.** The kernels were not the problem: summed kernel time at N = 157464
np 4×2 was 131.3 (c771e07) against 139.5 ms/step (main), and MPI time (a PMPI interposer,
`after12/pmpi.cpp`) +1.6 ms/step. The rest was host work invisible to a kernel timer; an
exclusive-host-time Kokkos Tools timer (`after12/ktimer.cpp`: region time minus the kernels and
child regions inside it) found it:

1. **The position-unit build sorted every contact key on the host** (Kokkos sort_by_key is a
   serial std::sort on the host backends) only to discover that every unit is one contact:
   4.9 ms/step. The manifold count already proves it (the manifold key is never finer than the unit
   key on the owned range), so `buildPositionUnitsKokkos(..., numManifolds)` writes the identity
   CSR directly when nm == nc.
2. **`mapVelocitySlots` sorted every owned and ghost gid on the host** each rebuild: 1.8 ms/step.
   The topology already names the owned row of every self ghost (selfIdx), and a cross-rank ghost's
   gid is never owned here, so only the cross-rank ghosts are sorted (checked equal to the old map
   at every gather in 14 scenes × np 2/4/8 before the check was removed).
3. **The owner-side halo kernels ran over all owned rows at every sync**: seeds, the M applies,
   the zero fill and the forward packs (~3.7 ms/step at 157k). They now run over the interface rows
   (every row a forward reads or a reverse writes) plus the k > 1 rows (built once per opening).
4. **Launch count at small N** (490 against 406 launches/step at 20k np 8; ~9 µs each on 2 OpenMP
   threads): each M reconciliation is now three dem kernels (MPackZero, MApply with the payload,
   seeds and owner share, MUnpack with the consensus, ghost share and baseline) around core's
   reverse and forward, instead of eight.

**What remains** (+2–6 %): the S14 consensus stop's extra velocity iterations (+27 % sweeps, each
with a sync; principled, registered), MPI waits in the extra syncs, and the drift vote (1 ms/step
at 157k np 4). The halo topology is rebuilt every step at the default `verlet_skin = 0`
(pre-existing, O(N_local) host work, ~2.5 ms/step at 157k np 4): `verlet_skin > 0` is the lever
there, and a device-side topology build the long-term one.

**Also found:** a fresh build tree took ParaView's MPICH `mpiexec` from PATH, so every np ≥ 2
ctest ran singletons; the mutation controls caught it (mutants 1, 2, 4, 5 "passed"). dem now pins
the launcher beside mpicxx (`cmake/PecletDemPinMpiexec.cmake`, core's rule). And CI was red since
f44fba7: clang-format violations (fixed tree-wide, whitespace only) and core 1.2.0's rename of
`peclet.core.mpi` to `peclet.halo` (the CI job and the Python MPI drivers now use `peclet.halo`).
