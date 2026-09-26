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
