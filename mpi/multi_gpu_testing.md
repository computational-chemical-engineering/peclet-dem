# dem — multi-GPU testing & profiling guide

The MPI-aware XPBD step (`Simulation::step_mpi`, see [README.md](README.md)) is **correctness-complete
and validated** (`tests/kokkos_mpi`, np=1,2,4, closed + periodic) but at-scale multi-GPU tuning is the
remaining roadmap work. On a single GPU, `np>1` ranks **share one device** (the GPU contexts serialise
and the wall-clock is contention-bound). This document is the plan for running and profiling on **real
multi-GPU hardware** (one rank per GPU, single- or multi-node), where the design is meant to scale. It
captures the prerequisites, launch recipes, what to measure, the profiling toolchain, and the
optimisation backlog ranked by expected payoff.

The engine is header-only **Kokkos + ArborX** (CUDA retired); the backend (CUDA/HIP/OpenMP) is chosen
by the `extern/install/<backend>` prefix the build is pointed at, not hard-coded. The profiling tools
below assume the **CUDA backend** (`nvidia-cuda` prefix); for HIP use the ROCm equivalents
(`rocprof`/`omnitrace`).

Read alongside: `../../core/docs/cuda-aware-mpi.md` (the CUDA-aware-MPI diagnosis &
sysadmin ask), the "MPI / sdflow" section of `../../flow/CLAUDE.md` (the Eulerian precedent), and
`../../docs/ROADMAP.md` Phase 4 / Phase 7.

---

## 1. Prerequisites

### 1.1 One rank per GPU — device binding (REQUIRED)
The `dem` module calls `Kokkos::initialize()` (no args) at import, so each rank takes whatever device
Kokkos selects by default — **device 0** unless the environment restricts it. There is **no**
`set_cuda_device` helper in the Kokkos module (that was a CUDA-era API). Bind each MPI rank to its own
GPU by restricting visibility to the **node-local rank** *before* `import dem`:

```python
import os
from mpi4py import MPI

world = MPI.COMM_WORLD
local = world.Split_type(MPI.COMM_TYPE_SHARED)          # ranks sharing a node
os.environ["CUDA_VISIBLE_DEVICES"] = str(local.rank)    # one visible GPU -> Kokkos uses it as device 0
import dem                                               # <-- import AFTER setting visibility
```

Equivalent launcher binding: `mpirun --map-by ppr:1:gpu` (sets `CUDA_VISIBLE_DEVICES` per rank), or a
wrapper script. Verify the mapping at startup — a silent fallback to a shared device 0 looks like a
correctness pass but with zero speedup. (For HIP, use `ROCR_VISIBLE_DEVICES`.)

### 1.2 Launcher & build
Use the system MPI, not ParaView's bundled one (it launches OpenMPI binaries as singletons):
`-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun`. Build the `dem` module with `-DDEM_MPI=ON` against the
bootstrapped backend prefix:
```bash
cd dem && source .venv/bin/activate
export PATH=/usr/local/cuda-13.2/bin:$PATH              # nvcc on PATH for the CUDA backend
cmake -S . -B build -DDEM_MPI=ON -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build -j$(nproc)
```
`mpi4py` must live in the same Python (`.venv`) as the `dem` module.

### 1.3 CUDA-aware MPI (unlocks the device-resident halo)
The stock `/usr/bin` OpenMPI is built without CUDA (device-pointer MPI segfaults). A user-space
CUDA-aware stack (OpenMPI + UCX, both `--with-cuda`) is required for device→device transfers; see
[`../../core/docs/cuda-aware-mpi.md`](../../core/docs/cuda-aware-mpi.md) for the
build/runtime recipe. core's `GridHalo` device-pointer branch is runtime-gated on
`TPX_CUDA_AWARE_MPI` (not `MPIX_Query_cuda_support()`, which mis-reports here). Bringing the `dem`
particle halo's gather/scatter onto the device-pointer path (§5.1) is the remaining piece.

```bash
source ~/opt/cudampi-env.sh                              # PATH/LD_LIBRARY_PATH/OPAL_PREFIX + OMPI_MCA_pml=ucx
TPX_CUDA_AWARE_MPI=1 mpirun -x TPX_CUDA_AWARE_MPI -np N ... # device-pointer path
```

---

## 2. Launch recipes

The mpi4py drivers in this directory each construct a `dem.Simulation` per rank and drive `step_mpi`:

```bash
cd dem && source .venv/bin/activate
PYP=$PWD/build                                           # the -DDEM_MPI=ON module build dir

# Correctness across GPUs (per-particle vs a serial reference; spheres settling on a floor):
PYTHONPATH=$PYP mpirun -np 2 --map-by ppr:1:gpu python3 mpi/validate_exact.py
PYTHONPATH=$PYP mpirun -np 4 --map-by ppr:1:gpu python3 mpi/validate_exact.py

# Cross-rank physics (restitution across a split, settled packing fraction/overlap):
PYTHONPATH=$PYP mpirun -np 2 --map-by ppr:1:gpu python3 mpi/verify_distributed.py

# Steady-state throughput (env knobs PI/VI/M/R):
PYTHONPATH=$PYP mpirun -np 4 --map-by ppr:1:gpu python3 mpi/bench_step.py
M=4 PYTHONPATH=$PYP mpirun -np 4 --map-by ppr:1:gpu python3 mpi/bench_step.py
```

`validate_exact.py`/`bench_step.py` env: `M`=`sync_every` (1=EXACT), `R`=`forward_rotation` (0 for
spheres), `PI`/`VI`=position/velocity iterations. The Kokkos `tests/kokkos_mpi` ctests are the
primary correctness gate; the Python drivers are for at-scale throughput/observable checks.

---

## 3. What to measure

| Experiment | Setup | Metric | Expectation |
|---|---|---|---|
| **Strong scaling** | fixed global N (e.g. 1e6), np = 1,2,4,8 | ms/step, speedup, parallel efficiency | near-linear until comm/halo dominates |
| **Weak scaling** | fixed N **per rank** (e.g. 2e5), grow np | ms/step ~ flat | flat = comm not growing with np |
| **Comm fraction** | per-step time split | gather + per-iter forward vs solve | shrinks with N/rank; sets the device-pack payoff |
| **Ghost fraction** | `num_ghost()` / owned | surface/volume ratio | drops as N/rank grows; drives redundant compute |
| **M-knob sweep** | M=1,2,4,8 | ms/step **and** mean‖dist−serial‖ | trade boundary error for fewer exchanges |
| **Load balance** | per-rank owned count + ms/step | max/mean | weighted-ORB split quality; imbalance caps speedup. Try `rebalance_every=N` |

Record alongside each run: np, N, N/rank, ghost fraction, `gsize` (ORB cell grid), `rcut`,
`sync_every`, `forward_rotation`, `rebalance_every`, GPU model, MPI build (CUDA-aware?), interconnect
(NVLink/PCIe/IB).

A clean scaling story needs the **comm fraction**: instrument `step_mpi` (`Kokkos::Profiling` regions
or `MPI_Wtime`) around (a) gather, (b) the per-iteration forwards, (c) the solver kernels, and
reduce-max across ranks. That single breakdown tells you whether to spend effort on the device pack
(§5.1), load balance, or larger blocks.

---

## 4. Profiling toolchain (CUDA backend)

- **Nsight Systems** (timeline; comm vs compute, H2D/D2H stalls, context serialisation):
  ```bash
  PYTHONPATH=$PYP mpirun -np 2 --map-by ppr:1:gpu \
    nsys profile -t cuda,mpi,nvtx -o nsys_rank_%q{OMPI_COMM_WORLD_RANK} \
    python3 mpi/bench_step.py
  ```
  Look for: the synchronous host-staged copies per step (each a GPU bubble), MPI wait time, and whether
  ranks rendezvous in lock-step (over-synchronisation).
- **Nsight Compute** (`ncu`) — only once comm is hidden; the XPBD kernels (`dem::solve_position`,
  narrowphase, the ArborX BVH build/query) are the compute targets. Profile a single rank.
- **Kokkos profiling** — `KOKKOS_TOOLS_LIBS=<kp_*.so>` (kernel-timer / space-time-stack) attributes
  time to named kernels (`dem::solve_velocity`, `dem::bp::*`, …) without a vendor tool; the kernels are
  already labelled. NVTX ranges still show on the Nsys timeline.
- **MPI**: `OMPI_MCA_pml_base_verbose`, or mpiP/Score-P for message-size/latency histograms. The
  forwards are many small messages — latency-bound — so message **count** matters more than bytes.
- **Quick host-side breakdown** without external tools: accumulate `std::chrono`/`MPI_Wtime` around the
  gather / per-iter forward / solve phases in `demStepMpi` behind an env flag and print rank-0 +
  reduce-max each N steps.

---

## 5. Optimisation backlog (ranked)

### 5.1 Device-resident halo pack
Keep the owner→ghost forward fully on-device: a gather kernel packs the owned records into a contiguous
device buffer, MPI transfers it **device→device** (CUDA-aware), and received ghosts land directly in the
contiguous ghost slab `[num_real, num_real+num_ghost)` (positions get a per-ghost shift kernel). This is
the Kokkos counterpart of core's device `GridHalo` path; it is **groundwork for real
multi-GPU/multi-node** — on a single shared GPU the per-transfer overhead beats the tiny host bounce,
but across NVLink / GPUDirect RDMA it eliminates the D2H + network + H2D round trip. Build against the
CUDA-aware MPI (§1.3) and gate on `TPX_CUDA_AWARE_MPI`.

### 5.2 Overlap comm with compute
Post the ghost forward asynchronously and compute the **interior** (owned particles with no ghost
neighbour) while the halo is in flight, then apply the boundary. Needs an interior/boundary partition of
the owned set (particles within `rcut` of the block edge). core's `GridHalo` already has the
overlap-capable `start()/wait()` split to mirror.

### 5.3 Avoid per-step rebuilds
`step_mpi` rebuilds the halo each substep. When the neighbour list is stable (small motion per step),
skip the `ParticleHalo::build()` and reuse the correspondence for several substeps (rebuild on a
skin-distance trigger). Saves the build round and the topology recompute.

### 5.4 Load balance
For evolving packings, sweep `enable_mpi_step(rcut, rebalance_every=N)` (weighted-ORB SoA ownership
migration via `Sim.rebalance()`): too-frequent rebalancing pays migration cost, too-rare lets imbalance
cap speedup. Track per-rank owned-count max/mean alongside ms/step.

### 5.5 Bigger blocks / fewer ranks per GPU
Ghost (surface) cost scales as $N^{2/3}$; larger blocks amortise comm. One fat rank per GPU beats many
thin ones. Tune `gsize` so the ORB blocks are compact (low surface/volume).

---

## 6. Correctness caveats on multi-GPU

- **Periodicity needs ≥2 ranks per *decomposed* periodic axis.** A rank never ghosts to itself for the
  cross-rank wrap; undecomposed periodic axes use the local periodic self-ghosts instead.
  `validate_exact.py` is deliberately non-periodic (floor + gravity) so ghosts arise only at the
  inter-rank split. For periodic validation, decompose the periodic axis across ≥2 ranks and compare
  aggregate observables (`verify_distributed.py`).
- **Bit-exactness is not expected** at np>1 even when physics matches: Jacobi accumulates contact deltas
  with atomic adds whose order differs between the serial (all-N) and distributed (per-block) contact
  lists. Compare to ~1e-3 per-particle and to tight tolerance on aggregate observables. np=1 (0 ghosts)
  is bit-exact to the single-rank step.
- **`sync_every>1` is an approximation** (boundary error grows with M); only `M=1` is EXACT.
- Re-run the suite (`tests/kokkos_mpi` np=1,2,4 + the Python drivers) after any change to `demStepMpi`,
  the gather, or the core `ParticleHalo`.
