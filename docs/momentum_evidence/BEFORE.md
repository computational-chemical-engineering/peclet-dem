# dem distributed contact solve: momentum conservation, BEFORE the fix

Measured 2026-09-25 on the tree **dem `e4e17f0` + the diagnostic commit that carries this file**
(`tests/kokkos_mpi/test_momentum_mpi.cpp`, a CoM print in `test_ghost_band_mpi.cpp`; `src/` untouched).
The task is `suite/docs/HANDOFF_DEM_MPI_MOMENTUM.md`. Nothing here is fixed; this is the baseline.

Host: AMD Threadripper PRO 5965WX (24 cores / 48 threads, one NUMA node), shared with other sessions.
Build (host OpenMP backend, Release):

```bash
cd suite/dem-momentum && source ../.venv/bin/activate
cmake -S . -B build_mom -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp" -DCMAKE_BUILD_TYPE=Release \
  -DPECLET_DEM_BUILD_TESTS=ON -DPECLET_DEM_MPI=ON -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun \
  -DMPIEXEC_PREFLAGS="--bind-to none"
cmake --build build_mom -j16
```

## Summary

- **np 1 conserves** to float round-off: linear momentum 3e-9 (7.9e-7 under free fall, from float
  accumulation of g dt into |v| ~ 5), CoM 1e-7 to 5e-7 R, velocity-phase angular momentum 3e-9
  (frictionless). The diagnostic is therefore sound.
- **np 2, 4 and 8 do not**, at 1 thread as well as at 8 threads. Linear momentum: 3e-3 to 1.3e-2
  relative in 50 steps. CoM: 1e-2 to 3e-2 R. Angular momentum: 1e-3 to 6e-3. The 1-thread runs agree to every
  printed digit over 3 repeats, so the defect is deterministic. Threads only change which wrong
  answer comes out.
- **Both phases are broken.** The velocity phase alone (dP) is 3e-3 to 1.3e-2. The position
  projection alone moves the CoM by 3e-3 to 1e-2 R (dXpos in `cluster_posonly`, where the velocity
  solve is off) against 5e-7 R at np 1. With the velocity solve on, the projection's share (dXpos)
  is 5e-4 to 5e-3 R.
- **The Hertz–Mindlin engine is already pairwise-symmetric.** dP is about 2e-8, dX 8.8e-7 R and dL
  4.4e-7 at every np and thread count, the same as np 1.
- **The p–q–s chain without the pin:** a rigid CoM shift of **2.133e-2** in 4 of 8 runs at np 2 and
  5 of 8 at np 4 with 8 threads; 0 of 8 at 1 thread. The serial reference moves by at most 1.4e-6.
- **XPBD does not conserve the total angular momentum even on one rank** (dL at np 1: 3e-4 to 3e-3).
  The position projection moves x without touching v, so sum m dx × v ≠ 0. The g = 0 friction pass
  (the legacy Jacobi `solveContactFriction`) also leaves 1.9e-5 in the velocity phase at np 1,
  because it applies each body's impulse at that body's own surface point. The PGS friction cone
  gives 1.9e-8. So an angular-momentum gate "to round-off" can only be stated for the velocity phase
  (dLvel), and in the g = 0 friction mode only against the 2e-5 serial floor. See Questions.

## What the diagnostic measures

`test_momentum_mpi <mode>` runs a closed, force-free, **polydisperse** cluster (scale 1 ± 0.1,
m ∝ s³, I = 2/5 m (s r)²). It has **N = 925** spheres (base r = 0.5) on a jittered cubic lattice with
spacing = base diameter, so the larger grains start overlapping. They fill a ball of radius 6
centred at (0.3, −0.2, 0.1) in the box [−16, 16]³. The ORB puts that centre on the common corner of
all blocks at np 2, 4 and 8. There are no walls within reach and no gravity, except in `cluster_pgs`.
Initial velocities are Gaussian (σ = 1.5), plus a drift (0.7, −0.4, 0.3), plus an inward radial term
of magnitude 1·r/6. e = 0.5, 20 position and 8 velocity iterations, dt = 1e-2, 50 steps. Hertz uses
E = 1e5, ν = 0.25, dt = 1e-4 and 40 × 25 substeps. After every step the global sums are
Allreduce'd in double over the owned bodies.

| column | definition | phase |
|---|---|---|
| dP | max_t \|P(t) − P(0) − M g t\| / Σ m\|v(0)\| | velocity (the position phase never writes v) |
| dX | max_t \|X(t) − X(0) − V(0) t − g t²/2\| / r | both |
| dXpos | max_t \|Σ_steps [X(n+1) − X(n) − dt (P(n)+P(n+1))/2M]\| / r | position projection |
| dL | max_t \|L(t) − L(0)\| / Lscale, about the origin (about the CoM under gravity) | both |
| dLcm | the same about the moving CoM | both |
| dLvel | max_t \|Σ_steps Σ_i [m (x_pred − X_pred) × (Δv − g dt) + I Δω]\| / Lscale | velocity |

Here Lscale = Σ (m\|x − X\|\|v − V\| + I\|ω\|) at t = 0. The step integrates
x(n+1) = x(n) + dt (v(n) + v(n+1))/2 and then projects, so dXpos isolates the projection exactly.
dLvel uses the predicted positions x + (v + g dt) dt, which are where the velocity phase evaluates its
lever arms.

Modes:
- `cluster`: g = 0, frictionless. This is the one-shot coloured-GS restitution followed by the
  coloured-GS overlap projection.
- `cluster_friction`: mu = 0.4 and random spins (σ = 3), g = 0. This adds the legacy friction pass.
- `cluster_pgs`: free fall under g = (0, 0, −10), mu = 0.4, spins, `set_stabilization('off')`. This
  is the production warm-started PGS with the friction cone. The one-sided stabilization is left off
  because it is a momentum sink by design.
- `cluster_posonly`: the velocity solve is off (the dem default of 0 velocity iterations), so this
  is the position projection alone.
- `hertz`: `step_hertz_mpi`, mu = 0.4, spins.

The test is REPORT-ONLY. `kGate = false`, and the gate thresholds `kTolP = 1e-6`, `kTolX = 1e-5` and
`kTolL = 1e-5` (on dLvel) sit at the top of the file. The 20 new ctests are
`momentum_<mode>_np{1,2,4,8}`; the battery now has 98 tests.

## Conservation matrix

Command: `docs/momentum_evidence/run_matrix.sh build_mom`, i.e. for each mode, np ∈ {1, 2, 4, 8},
OMP_NUM_THREADS ∈ {1, 8} and 3 repeats:
`OMP_NUM_THREADS=$thr OMP_PROC_BIND=false mpirun --bind-to none -np $np build_mom/tests/kokkos_mpi/test_momentum_mpi $mode`.
Raw lines are in `matrix.txt`. Load average (1/5/15 min) was 2.7/21.9/24.8 at the start and
32.8/43.0/37.2 at the end.

Cells show the median [min–max] of 3 runs; "(=)" means all 3 agreed to every printed digit (4 significant).

#### `cluster`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.8e-09 (=) | 3.2e-07 (=) | 3.3e-07 (=) | 8.2e-04 (=) | 8.2e-04 (=) | 2.9e-09 (=) |
| 1 | 8 | 2.6e-09 [2.2e-09–2.9e-09] | 2.4e-07 [1.9e-07–2.7e-07] | 2.4e-07 [1.8e-07–2.8e-07] | 8.0e-04 [7.9e-04–8.1e-04] | 8.0e-04 [7.9e-04–8.1e-04] | 3.2e-09 [1.8e-09–3.7e-09] |
| 2 | 1 | 3.9e-03 (=) | 8.3e-03 (=) | 6.9e-04 (=) | 2.3e-03 (=) | 2.1e-03 (=) | 1.4e-03 (=) |
| 2 | 8 | 3.6e-03 [3.3e-03–4.3e-03] | 8.3e-03 [5.0e-03–1.1e-02] | 1.2e-03 [1.2e-03–1.3e-03] | 2.4e-03 [2.4e-03–3.0e-03] | 2.3e-03 [2.2e-03–2.9e-03] | 1.6e-03 [1.4e-03–2.1e-03] |
| 4 | 1 | 6.2e-03 (=) | 1.4e-02 (=) | 1.9e-03 (=) | 2.3e-03 (=) | 2.2e-03 (=) | 2.3e-03 (=) |
| 4 | 8 | 4.2e-03 [3.2e-03–5.3e-03] | 7.8e-03 [7.4e-03–1.2e-02] | 1.7e-03 [5.4e-04–2.0e-03] | 2.4e-03 [2.2e-03–3.0e-03] | 2.3e-03 [2.1e-03–3.1e-03] | 2.4e-03 [2.4e-03–2.8e-03] |
| 8 | 1 | 3.3e-03 (=) | 7.5e-03 (=) | 3.3e-03 (=) | 5.7e-03 (=) | 5.6e-03 (=) | 5.5e-03 (=) |
| 8 | 8 | 3.6e-03 [3.0e-03–3.6e-03] | 8.7e-03 [7.5e-03–1.0e-02] | 2.4e-03 [2.1e-03–2.4e-03] | 5.1e-03 [3.9e-03–5.3e-03] | 5.2e-03 [3.8e-03–5.3e-03] | 4.6e-03 [3.5e-03–5.3e-03] |

#### `cluster_friction`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.7e-09 (=) | 1.3e-07 (=) | 1.3e-07 (=) | 3.2e-04 (=) | 3.2e-04 (=) | 1.9e-05 (=) |
| 1 | 8 | 4.0e-09 [3.9e-09–7.1e-09] | 1.5e-07 [1.4e-07–3.3e-07] | 1.5e-07 [1.3e-07–3.3e-07] | 3.1e-04 [3.1e-04–3.3e-04] | 3.1e-04 [3.1e-04–3.3e-04] | 2.1e-05 [1.4e-05–2.2e-05] |
| 2 | 1 | 3.7e-03 (=) | 9.7e-03 (=) | 1.3e-03 (=) | 1.1e-03 (=) | 9.9e-04 (=) | 1.2e-03 (=) |
| 2 | 8 | 3.6e-03 [3.1e-03–4.2e-03] | 7.7e-03 [7.3e-03–1.0e-02] | 6.1e-04 [5.5e-04–1.1e-03] | 1.4e-03 [8.0e-04–1.7e-03] | 1.3e-03 [7.1e-04–1.6e-03] | 1.6e-03 [9.8e-04–1.8e-03] |
| 4 | 1 | 5.0e-03 (=) | 1.4e-02 (=) | 4.5e-03 (=) | 1.1e-03 (=) | 1.1e-03 (=) | 1.3e-03 (=) |
| 4 | 8 | 5.4e-03 [4.4e-03–7.5e-03] | 1.6e-02 [1.1e-02–1.8e-02] | 9.5e-04 [8.0e-04–3.7e-03] | 1.7e-03 [1.6e-03–2.3e-03] | 1.6e-03 [1.6e-03–2.0e-03] | 1.8e-03 [1.7e-03–2.0e-03] |
| 8 | 1 | 3.8e-03 (=) | 1.0e-02 (=) | 2.3e-03 (=) | 3.2e-03 (=) | 3.3e-03 (=) | 3.1e-03 (=) |
| 8 | 8 | 3.6e-03 [3.5e-03–3.6e-03] | 9.5e-03 [5.8e-03–1.0e-02] | 3.6e-03 [3.1e-03–3.8e-03] | 2.5e-03 [2.4e-03–3.1e-03] | 2.4e-03 [2.3e-03–3.0e-03] | 2.1e-03 [2.1e-03–2.9e-03] |

#### `cluster_pgs`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 7.9e-07 (=) | 4.2e-07 (=) | 1.4e-07 (=) | 3.8e-04 (=) | 3.8e-04 (=) | 1.9e-08 (=) |
| 1 | 8 | 7.9e-07 [7.8e-07–7.9e-07] | 4.9e-07 [4.2e-07–5.3e-07] | 1.4e-07 [1.3e-07–2.2e-07] | 3.8e-04 [3.8e-04–3.8e-04] | 3.8e-04 [3.8e-04–3.8e-04] | 1.7e-08 [1.6e-08–1.9e-08] |
| 2 | 1 | 6.8e-03 (=) | 1.7e-02 (=) | 1.3e-03 (=) | 3.7e-03 (=) | 3.7e-03 (=) | 4.0e-03 (=) |
| 2 | 8 | 7.4e-03 [6.2e-03–8.2e-03] | 1.7e-02 [1.7e-02–2.2e-02] | 1.4e-03 [5.7e-04–3.0e-03] | 2.8e-03 [2.7e-03–4.0e-03] | 2.8e-03 [2.7e-03–4.0e-03] | 3.2e-03 [3.0e-03–4.3e-03] |
| 4 | 1 | 1.1e-02 (=) | 2.9e-02 (=) | 4.8e-03 (=) | 1.6e-03 (=) | 1.6e-03 (=) | 1.9e-03 (=) |
| 4 | 8 | 9.5e-03 [9.1e-03–9.9e-03] | 2.5e-02 [2.2e-02–2.5e-02] | 2.0e-03 [1.3e-03–2.5e-03] | 2.6e-03 [2.2e-03–2.8e-03] | 2.6e-03 [2.2e-03–2.8e-03] | 2.9e-03 [2.5e-03–2.9e-03] |
| 8 | 1 | 1.3e-02 (=) | 1.3e-02 (=) | 2.1e-03 (=) | 4.2e-03 (=) | 4.2e-03 (=) | 3.8e-03 (=) |
| 8 | 8 | 1.2e-02 [1.0e-02–1.3e-02] | 1.5e-02 [1.3e-02–2.0e-02] | 2.9e-03 [1.8e-03–3.6e-03] | 5.0e-03 [4.3e-03–5.3e-03] | 5.0e-03 [4.3e-03–5.3e-03] | 4.6e-03 [3.8e-03–5.2e-03] |

#### `cluster_posonly`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.0e+00 (=) | 4.9e-07 (=) | 4.9e-07 (=) | 2.9e-03 (=) | 2.9e-03 (=) | 0.0e+00 (=) |
| 1 | 8 | 0.0e+00 (=) | 5.3e-07 [4.5e-07–7.8e-07] | 5.3e-07 [4.5e-07–7.8e-07] | 2.9e-03 [2.9e-03–2.9e-03] | 2.9e-03 [2.9e-03–2.9e-03] | 0.0e+00 (=) |
| 2 | 1 | 0.0e+00 (=) | 4.8e-03 (=) | 4.8e-03 (=) | 2.8e-03 (=) | 2.8e-03 (=) | 0.0e+00 (=) |
| 2 | 8 | 0.0e+00 (=) | 3.1e-03 [2.9e-03–3.6e-03] | 3.1e-03 [2.9e-03–3.6e-03] | 2.7e-03 [2.7e-03–2.8e-03] | 2.8e-03 [2.7e-03–2.8e-03] | 0.0e+00 (=) |
| 4 | 1 | 0.0e+00 (=) | 7.0e-03 (=) | 7.0e-03 (=) | 2.8e-03 (=) | 2.7e-03 (=) | 0.0e+00 (=) |
| 4 | 8 | 0.0e+00 (=) | 5.4e-03 [4.6e-03–6.0e-03] | 5.4e-03 [4.6e-03–6.0e-03] | 2.9e-03 [2.9e-03–2.9e-03] | 2.8e-03 [2.8e-03–2.9e-03] | 0.0e+00 (=) |
| 8 | 1 | 0.0e+00 (=) | 7.2e-03 (=) | 7.2e-03 (=) | 2.4e-03 (=) | 2.2e-03 (=) | 0.0e+00 (=) |
| 8 | 8 | 0.0e+00 (=) | 7.5e-03 [6.0e-03–1.0e-02] | 7.5e-03 [6.0e-03–1.0e-02] | 2.7e-03 [2.7e-03–3.0e-03] | 2.7e-03 [2.7e-03–2.9e-03] | 0.0e+00 (=) |

#### `hertz`

| np | thr | dP | dX | dXpos | dL | dLcm | dLvel |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.8e-08 (=) | 8.8e-07 (=) | n/a | 4.4e-07 (=) | 4.4e-07 (=) | n/a |
| 1 | 8 | 2.6e-08 [2.1e-08–2.7e-08] | 8.8e-07 [8.7e-07–8.8e-07] | n/a | 4.3e-07 [4.3e-07–4.4e-07] | 4.4e-07 [4.3e-07–4.4e-07] | n/a |
| 2 | 1 | 1.5e-08 (=) | 8.7e-07 (=) | n/a | 4.4e-07 (=) | 4.4e-07 (=) | n/a |
| 2 | 8 | 2.4e-08 [2.2e-08–2.5e-08] | 8.7e-07 [8.7e-07–8.8e-07] | n/a | 4.4e-07 [4.4e-07–4.4e-07] | 4.4e-07 [4.4e-07–4.4e-07] | n/a |
| 4 | 1 | 2.4e-08 (=) | 8.7e-07 (=) | n/a | 4.3e-07 (=) | 4.3e-07 (=) | n/a |
| 4 | 8 | 2.2e-08 [1.9e-08–2.3e-08] | 8.8e-07 [8.7e-07–8.8e-07] | n/a | 4.3e-07 [4.3e-07–4.4e-07] | 4.4e-07 [4.4e-07–4.4e-07] | n/a |
| 8 | 1 | 2.5e-08 (=) | 8.7e-07 (=) | n/a | 4.3e-07 (=) | 4.3e-07 (=) | n/a |
| 8 | 8 | 2.0e-08 [1.8e-08–2.3e-08] | 8.8e-07 [8.8e-07–8.8e-07] | n/a | 4.4e-07 [4.3e-07–4.4e-07] | 4.4e-07 [4.4e-07–4.4e-07] | n/a |

## The p–q–s chain (`ghost_band_*` mode `margin`) without the one-thread pin

The pin in `tests/kokkos_mpi/CMakeLists.txt` is still in place. The runs override OMP_NUM_THREADS on
the command line: `docs/momentum_evidence/run_chain.sh build_mom`, 8 runs per cell. The instrument is
a new `comShift` field on the `[margin]` line, the x-CoM displacement over the 4 steps for the
distributed run and for the serial reference. Every body has unit mass and starts at rest, so a
conserving solve gives 0. Raw lines are in `chain.txt`; the load average was 36.3 at the start and
28.2 at the end.

| np | threads | runs with the rigid shift | shift (dist CoM) | serial reference CoM |
|---|---|---|---|---|
| 2 | 8 | **4 / 8** | +2.133e-2 (all 4) | −0.7e-6 … −1.4e-6 |
| 2 | 1 | 0 / 8 | −1.19e-6 | −1.19e-6 |
| 4 | 8 | **5 / 8** | +2.133e-2 (all 5) | −1.0e-6 … −1.4e-6 |
| 4 | 1 | 0 / 8 | −1.19e-6 | −1.19e-6 |

In every flake, posErr equals comShift (2.133e-2): the whole chain translates rigidly. The flake
rate is higher than in the c64e117 investigation (2 of 8), which ran on a less loaded host; the rate
depends on thread timing. A clean 1-thread run is not evidence of conservation. The cluster matrix
above is non-conserving at 1 thread too.

## "Before" performance: distributed XPBD step, N = 19683 (27³), fully periodic

Command: `docs/momentum_evidence/run_perf.sh build_mom`, test modes `perf_gas` and `perf_pgs`.
- Scene: a jittered lattice at spacing 1.02 d (φ ≈ 0.49), Gaussian velocities σ = 1, e = 0.5,
  mu = 0.4, 8 position and 4 velocity iterations, dt = 2e-3, ORB grid 32³.
- `perf_gas` runs g = 0 (the one-shot GS and legacy friction path). `perf_pgs` runs g = −10 (the
  warm-started PGS path, with the default one-sided stabilization).
- Timing: 10 warm-up steps are excluded, 50 steps are timed, and `MPI_Barrier` brackets the timed
  window.
- Placement: 16 physical cores 8–23 in both configurations, np 4 × 4 threads and np 8 × 2 threads.
  Each rank is `taskset`-pinned to its own contiguous core block by `pin_rank.sh`, keyed on
  `OMPI_COMM_WORLD_RANK`, with `OMP_PROC_BIND=false`.
- 5 runs per cell. The 1-minute load average was 19.3–30.5 (my 16 threads included). The raw
  per-run load lines are in `perf.txt`.

| mode | np × thr | runs (ms/step) | **median** | min |
|---|---|---|---|---|
| perf_gas | 4 × 4 | 48.0, 15.2, 27.2, 15.6, 15.6 | **15.6** | 15.2 |
| perf_gas | 8 × 2 | 7.9, 16.4, 8.1, 9.3, 8.1 | **8.1** | 7.9 |
| perf_pgs | 4 × 4 | 15.5, 119.8, 26.2, 15.9, 15.6 | **15.9** | 15.5 |
| perf_pgs | 8 × 2 | 98.8, 8.2, 8.3, 10.2, 9.3 | **9.3** | 8.2 |

The host is shared, and the outliers (27–120 ms) are contention from other sessions. Compare the
"after" numbers to these medians and minima under the same script.

## Questions this baseline raises (for the design)

1. **Angular-momentum gate.** The XPBD step does not conserve the total L on one rank (dL = 3e-4 to
   3e-3 at np 1), because the position projection moves x without touching v. The handoff's
   "angular momentum with friction to round-off" can only be gated on the velocity phase (dLvel).
   Even there, the g = 0 legacy friction pass has a 1.9e-5 serial floor, from each body's lever arm
   pointing to its own surface point. The PGS cone is at 1.9e-8. The proposal is to gate dLvel ≲ 1e-6
   on `cluster` and `cluster_pgs`, and to gate `cluster_friction` against its np 1 value instead of
   round-off.
2. **Free-fall dP floor.** `cluster_pgs` at np 1 gives dP = 7.9e-7, which is float accumulation of
   g dt into v. kTolP = 1e-6 is therefore marginal for that mode. Either normalise by Σ m|v(t)| or
   gate it at about 5e-6.
