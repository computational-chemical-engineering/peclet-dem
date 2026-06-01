# packing-gpu — multi-GPU testing & profiling guide

The MPI-aware XPBD step (`Simulation::step_mpi`, see [README.md](README.md)) is **correctness-complete
and validated on one GPU** but has only ever run with `np>1` ranks **sharing a single RTX 5080**, where
the CUDA contexts serialise and the wall-clock is contention-bound. This document is the plan for
running and profiling it on **real multi-GPU hardware** (one rank per GPU, single- or multi-node),
where the design is meant to scale. It captures the prerequisites, launch recipes, what to measure,
the profiling toolchain, and the optimisation backlog ranked by expected payoff.

Read alongside: `../../transport-core/docs/cuda-aware-mpi.md` (the CUDA-aware-MPI diagnosis &
sysadmin ask), `../../cfd-gpu/doc/mpi_parallelization_status.md` (the Eulerian precedent), and
`../../docs/ROADMAP.md` Phase 4.

---

## 1. Prerequisites

### 1.1 One rank per GPU — device binding (REQUIRED)
Nothing in demgpu called `cudaSetDevice` historically, so every rank defaulted to device 0. Map each
MPI rank to its own GPU using the **node-local rank**, before constructing any `Simulation`:

```python
from mpi4py import MPI
import demgpu

world = MPI.COMM_WORLD
local = world.Split_type(MPI.COMM_TYPE_SHARED)          # ranks sharing a node
ndev = demgpu.Simulation.cuda_device_count()
demgpu.Simulation.set_cuda_device(local.rank % ndev)    # <-- bind BEFORE initialize()
```

`set_cuda_device` / `cuda_device_count` are `@staticmethod`s on `Simulation`. Equivalent CLI binding:
`mpirun --map-by ppr:1:gpu` or setting `CUDA_VISIBLE_DEVICES` per rank via a wrapper script. Verify
the mapping at startup (each rank prints its bus id) — a silent fallback to device 0 looks like a
correctness pass but with zero speedup.

### 1.2 Launcher
Use the system MPI, not ParaView's bundled one (it launches OpenMPI binaries as singletons):
`-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun`. Build demgpu with `-DDEMGPU_ENABLE_MPI=ON` (the default) and
the sm-matching arch (`CMAKE_CUDA_ARCHITECTURES=native`/`120` on Blackwell). mpi4py must live in the
same Python as the `demgpu`/`tpx_mpi` modules.

### 1.3 CUDA-aware MPI (optional, unlocks the biggest optimisation)
The current exchange is **host-staged** (device→host→MPI→host→device) because the box's stock OpenMPI
is built without CUDA support (device-pointer MPI segfaults — see the transport-core doc). With a
CUDA-aware MPI (OpenMPI+UCX built `--with-cuda`, or a vendor MPI) the device-resident-pack path
(§5.1) can hand device pointers straight to MPI and keep the field on the GPU. Confirm with
`ompi_info --parsable | grep mpi_built_with_cuda_support:value`.

---

## 2. Launch recipes

```bash
cd packing-gpu
export PATH=/usr/local/cuda-13.2/bin:$PATH
PYP=build_sm120:../transport-core/python/build

# Correctness across GPUs (per-particle vs a serial reference; spheres settling on a floor):
PYTHONPATH=$PYP mpirun -np 2 --map-by ppr:1:gpu python3 mpi/validate_exact.py
PYTHONPATH=$PYP mpirun -np 4 --map-by ppr:1:gpu python3 mpi/validate_exact.py

# Cross-rank physics (restitution across a split, settled packing fraction/overlap):
PYTHONPATH=$PYP mpirun -np 2 --map-by ppr:1:gpu python3 mpi/verify_distributed.py

# Steady-state throughput (rebuild-free; env knobs PI/VI/M/R):
PYTHONPATH=$PYP mpirun -np 4 --map-by ppr:1:gpu python3 mpi/bench_step.py
M=4 R=0 PYTHONPATH=$PYP mpirun -np 4 --map-by ppr:1:gpu python3 mpi/bench_step.py
```

`validate_exact.py` env: `M`=`sync_every` (1=EXACT), `R`=`forward_rotation` (0 for spheres).
`bench_step.py` env: `PI`/`VI` iterations, `M`, `R`.

---

## 3. What to measure

| Experiment | Setup | Metric | Expectation |
|---|---|---|---|
| **Strong scaling** | fixed global N (e.g. 1e6), np = 1,2,4,8 | ms/step, speedup, parallel efficiency | near-linear until comm/halo dominates |
| **Weak scaling** | fixed N **per rank** (e.g. 2e5), grow np | ms/step ~ flat | flat = comm not growing with np |
| **Comm fraction** | per-step time split | gather + per-iter forward vs solve | shrinks with N/rank; sets the device-pack payoff |
| **Ghost fraction** | `num_particles(True)-num_particles(False)` / owned | surface/volume ratio | drops as N/rank grows; drives redundant compute |
| **M-knob sweep** | M=1,2,4,8 | ms/step **and** mean‖dist−serial‖ | trade boundary error for fewer exchanges |
| **Load balance** | per-rank owned count + ms/step | max/mean | ORB split quality; imbalance caps speedup |

Record alongside each run: np, N, N/rank, ghost fraction, `gsize` (ORB cell grid), `rcut`,
`sync_every`, `forward_rotation`, GPU model, MPI build (CUDA-aware?), interconnect (NVLink/PCIe/IB).

A clean scaling story needs the **comm fraction**: instrument `step_mpi` with `cudaEvent`/`MPI_Wtime`
around (a) gather, (b) the per-iteration forwards, (c) the solver kernels, and reduce-max across ranks.
That single breakdown tells you whether to spend effort on the device pack (§5.1), load balance, or
larger blocks.

---

## 4. Profiling toolchain

- **Nsight Systems** (timeline; comm vs compute, H2D/D2H stalls, context serialisation):
  ```bash
  PYTHONPATH=$PYP mpirun -np 2 --map-by ppr:1:gpu \
    nsys profile -t cuda,mpi,nvtx -o nsys_rank_%q{OMPI_COMM_WORLD_RANK} \
    python3 mpi/bench_step.py
  ```
  Look for: the ~18 synchronous `cudaMemcpy`/step in the host-staged path (each a GPU bubble), MPI
  wait time, and whether ranks rendezvous in lock-step (over-synchronisation).
- **Nsight Compute** (`ncu`) — only once comm is hidden; the XPBD kernels (`solve_position_jacobi`,
  narrowphase, BVH) are the compute targets. Profile a single rank.
- **NVTX ranges** — wrap gather / forward / solve so they're named on the Nsys timeline (add
  `nvtxRangePush/Pop` in `step_mpi`, guarded by a build flag).
- **MPI**: `OMPI_MCA_pml_base_verbose`, or mpiP/Score-P for message-size/latency histograms. The
  forwards are many small messages — latency-bound — so message **count** matters more than bytes.
- **Quick host-side breakdown** without external tools: accumulate `std::chrono` around the three
  phases in `step_mpi` behind an env flag and print rank-0 + reduce-max each N steps.

---

## 5. Optimisation backlog (ranked)

### 5.1 Device-resident pack (biggest lever)
Today each gather/forward does **separate synchronous `cudaMemcpy`s per field** (~18/step), and each
serialises the GPU. Replace with:
1. a **gather kernel** that packs the `sendIdx_` owned records into one contiguous **device** buffer,
2. **one** D2H of that compact buffer (or, with CUDA-aware MPI, hand the device pointer to MPI),
3. MPI exchange,
4. one H2D into the contiguous ghost slots (`[num_real, num_real+num_ghost)` — already contiguous, no
   scatter needed for the gather; per-iteration forwards write straight into the ghost slab).

This cuts ~18 `cudaMemcpy`/step to ~2 and removes the per-field GPU bubbles. With CUDA-aware MPI it
becomes a true on-device halo (cf. `tpx::halo::DeviceGridExchange` on the Eulerian side). Expose the
`sendIdx_` flattened list from `tpx::halo::ParticleHalo` to drive the gather kernel.

### 5.2 Overlap comm with compute
Post the ghost forward asynchronously and compute the **interior** (owned particles with no ghost
neighbour) while the halo is in flight, then apply the boundary. Needs an interior/boundary partition
of the owned set (particles within `rcut` of the block edge). `GridHalo` already has the
overlap-capable `start()/wait()` split to mirror.

### 5.3 Avoid per-step rebuilds
The drivers currently build a fresh `Simulation` each step (re-alloc + BVH buffers). Add a
fixed-capacity sim with a settable **active count** and in-place owned/ghost refresh so migration just
updates counts. Removes per-step allocation and `mpi_init` re-setup.

### 5.4 Persistent halo across substeps
When the neighbour list is stable (small motion per step), skip `ParticleHalo::build()` and reuse the
correspondence for several substeps (rebuild on a skin-distance trigger). Saves the build NBX round
and the topology recompute.

### 5.5 Bigger blocks / fewer ranks per GPU
Ghost (surface) cost scales as N^(2/3); larger blocks amortise comm. One fat rank per GPU beats many
thin ones. Tune `gsize` so ORB blocks are compact (low surface/volume).

---

## 6. Correctness caveats on multi-GPU

- **Periodicity needs ≥2 ranks per periodic axis.** A rank never ghosts to itself, so a single rank on
  a periodic axis loses the wrap. `validate_exact.py` is deliberately non-periodic (floor + gravity)
  so ghosts arise only at the inter-rank split. For periodic validation, decompose the periodic
  axis across ≥2 ranks and compare aggregate observables (§ `verify_distributed.py`).
- **Bit-exactness is not expected** even when physics matches: Jacobi accumulates contact deltas with
  atomic adds whose order differs between the serial (all-N) and distributed (per-block) contact
  lists. Compare to ~1e-3 per-particle and to tight tolerance on aggregate observables.
- **`sync_every>1` is an approximation** (boundary error grows with M); only `M=1` is EXACT.
- Re-run the full suite (`validate_exact.py` np=1/2/4 + `verify_distributed.py`) after any change to
  `step_mpi`, the gather, or `ParticleHalo`.
```
