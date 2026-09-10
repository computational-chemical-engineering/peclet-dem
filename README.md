# peclet-dem (`peclet.dem`)

[![PyPI version](https://img.shields.io/pypi/v/peclet-dem.svg)](https://pypi.org/project/peclet-dem/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/peclet-dem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/computational-chemical-engineering/peclet-dem/blob/main/LICENSE)
[![CI](https://github.com/computational-chemical-engineering/peclet-dem/actions/workflows/ci.yml/badge.svg)](https://github.com/computational-chemical-engineering/peclet-dem/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21132441.svg)](https://doi.org/10.5281/zenodo.21132441)

Performance-portable Discrete Element Method (DEM) particle simulation: an XPBD solver with SDF-based point-shell collision detection. Built on **Kokkos + ArborX**, so the same source runs on **CUDA, HIP (AMD/LUMI), and OpenMP** backends (selected at build time by the install prefix). Optional MPI for domain partitioning, with **nanobind** Python bindings (zero-copy, via scikit-build-core) for scripting and visualization.

> The CUDA implementation was retired (2026-06): the Kokkos `peclet.dem` module was validated against it before the CUDA sources were removed. Restore point: git tag `pre-cuda-retirement`.

## Features

- **Hybrid XPBD Solver**: Two-pass velocity/position solver for stable high-density packing.
- **SDF Collision**: Point-shell collision detection using Signed Distance Fields (supports analytic shapes like hollow cylinders).
- **Periodicity**: Full periodic boundary conditions (Ghost Particles).
- **Python Bindings**: Control simulation logic, data initialization, and export entirely from Python.
- **MPI Support**: Optional Multi-GPU/Node support via domain decomposition.

## Folder Structure

```text
├── CMakeLists.txt              # Build configuration (find_package Kokkos + ArborX; version from pyproject.toml)
├── pyproject.toml              # scikit-build-core packaging (peclet-dem); THE version source
├── packaging/                  # peclet/dem/__init__.py, particle_builder.py, scene_particle.py, CUDA-wheel pyproject
├── src                         # Kokkos sources (header-only, namespace peclet::dem)
│   ├── dem_bindings.cpp          # nanobind module entry point (the `peclet.dem` module)
│   ├── sim.hpp                   # Simulation facade: the host-facing setters/getters, walls, steppers, MPI driver state
│   ├── shape_registry.hpp        # ShapeRegistry (Simulation's base): shapes, shells, inertias + the device upload
│   ├── step_solve.hpp            # Single-rank step drivers: demStep, computeOverlapsKokkos, contact-buffer sizing
│   ├── step_solve_mpi.hpp        # Distributed step drivers + MPI hook policies, gated PECLET_DEM_MPI
│   ├── dem_portable.hpp          # POD types + math + analytic SDFs shared by every kernel
│   ├── particles.hpp             # Particle SoA container (the Kokkos Views)
│   ├── shapes_portable.hpp       # Surface-shell point generators for the analytic shapes
│   ├── broadphase_arborx.hpp     # ArborX BVH broad-phase
│   ├── narrowphase.hpp           # Narrow-phase point-shell-vs-SDF collision + boundary planes
│   ├── contact_preprocessing.hpp # Contact -> manifold reduction
│   ├── solve_driver.hpp          # The shared XPBD contact-solve driver (velocity + position, coloured GS)
│   ├── solve_driver_force.hpp    # The force-based (explicit Hertz-Mindlin) step driver
│   ├── solver_velocity.hpp       # Manifold velocity solve (restitution impulse)
│   ├── solver_position.hpp       # XPBD position solve (overlap removal)
│   ├── solver_friction.hpp       # Coulomb friction cluster
│   ├── solver_fused.hpp          # Fused colour sweeps (one persistent kernel per sweep)
│   ├── solver_multilevel.hpp     # Multilevel (GraphMG-style) contact stabilization
│   ├── solver_hertz.hpp          # Soft-sphere Hertz-Mindlin force model
│   ├── sleeping.hpp              # Island sleeping / freezing for the statics path
│   ├── integration.hpp           # Time integration & prediction
│   ├── periodicity.hpp           # Periodic ghost generation
│   ├── output_sdf.hpp            # Packed-bed SDF grid reconstruction (get_sdf_grid)
│   ├── io.hpp                    # LAMMPS-dump + SDF-VTI export
│   └── mpi_halo.hpp              # Distributed particle halo (core), gated PECLET_DEM_MPI
├── tests                       # ctest suites, built by -DPECLET_DEM_BUILD_TESTS=ON: kokkos/ (kernels),
│                               #   arborx/ (broad-phase + pipeline), kokkos_mpi/ (distributed step, np=1,2,4),
│                               #   python/ (pytest) + python/mpi/ (mpirun-launched pytest files)
├── examples                    # demos: packing / collision / stacking / precession / thermostat, pack.py,
│                               #   shape + packing generators, the distributed driver + microbenchmark
├── docs                        # Reference docs: solver_details.md, mpi.md, multi_gpu_testing.md,
│                               #   visualization.md, Doxyfile; archive/ = dated campaign records
└── notebooks                   # packing_analysis.ipynb
```

## Prerequisites

- **Linux**
- **CMake** >= 3.24
- **Kokkos 5.x + ArborX** (C++20) — provisioned by `../tools/bootstrap_deps.sh` into
  `../extern/install/<backend>` (`nvidia-cuda` / `host-openmp` / `lumi-hip`). A **hard build dependency**.
- **nanobind** + **scikit-build-core** (found via the active Python interpreter; see `pyproject.toml`)
- a backend compiler: **nvcc** (CUDA) on `PATH`, **hipcc** (ROCm), or just a host C++ compiler (OpenMP)
- **Python** >= 3.10
- **MPI** (optional, `-DPECLET_DEM_MPI=ON`) — OpenMPI or MPICH

## Build Instructions

```bash
source ../.venv/bin/activate                      # the ONE suite venv (nanobind, numpy, ...)
export PATH=/usr/local/cuda-13.2/bin:$PATH        # if building the CUDA backend

# Canonical: build + install the module via scikit-build-core
CMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda" pip install .

# Or a dev cmake build (nanobind is found via the active interpreter, no cmakedir needed):
cmake -B build -S . -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build -j$(nproc)
```
*Swap the prefix to `../extern/install/host-openmp` for the OpenMP backend. `-DPECLET_DEM_MPI=ON` links MPI
and exposes the distributed step (`init_mpi` / `enable_mpi_step` / `step_mpi`), including dynamic load
balancing — `enable_mpi_step(..., rebalance_every=N)` or an explicit `rebalance()` re-decomposes by
particle count (weighted ORB) and migrates ownership so each rank keeps a near-equal share.*

The compiled `peclet.dem` extension is placed in `build/peclet/dem/`; run scripts with `build/` on `PYTHONPATH` (`import peclet.dem`).

## Quick start

```python
import numpy as np
from peclet import dem

sim = dem.Simulation(capacity=20000)             # owned + periodic ghost slots
sim.initialize_shape('sphere', 0.5)              # one shape; add_shape(...) appends more
sim.set_global_scale(0.01)                       # grain size -- set it BEFORE set_domain (both reset the skin)
sim.set_domain(extent=(0.2, 0.2, 0.4), origin=(0, 0, 0), periodic=(True, True, False))
sim.add_plane(point=(0, 0, 0), normal=(0, 0, 1)) # there is no implicit floor

xyz = np.random.rand(5000, 3) * [0.2, 0.2, 0.4]
sim.set_positions(xyz.astype(np.float32))        # (N, 3), or (N, 4) whose 4th column is the INVERSE mass
sim.set_gravity((0, 0, -9.81))                   # the default is ZERO
sim.set_material_params(restitution_normal=0.3, friction=0.4)   # friction defaults to ZERO
sim.set_solver_iterations(pos=10, vel=8)         # vel defaults to 0 = no restitution

sim.set_dt(1e-4)                                 # every stepper RAISES before this
sim.step(500)                                    # advance 500 substeps
sim.relax(50)                                    # dynamics-free overlap removal (no dt needed)

pos = sim.get_positions()                        # (N, 3) float32
print(sim.num_particles, sim.num_contacts, sim.compute_overlaps())
```

Counts and stored scalars are **properties** (`num_particles`, `num_contacts`, `max_overlap`, `dt`,
`gravity`, `stabilization`, ...); computed scalars are methods (`compute_overlaps()`); array copies
are `get_*`. Developer instruments, ablations and GPU execution policies live on `sim.diagnostics`.
See `CLAUDE.md` for the call-order requirements and `docs/solver_details.md` for what the step does.

## Running Simulations

`examples/` holds the Python demos: `verify_*.py` (sphere / hollow-cylinder packing, collisions,
stacking, precession, thermostat), the `pack.py` / `pack_meter.py` packing protocol + meter,
`generate_particles.py` (Ovito shape mesh) and the packing generators, plus the distributed
`driver_distributed.py` skeleton and `bench_step.py` (`mpirun -np N python examples/bench_step.py`,
needs a built `peclet.core.mpi` on `PYTHONPATH`). All run from the build tree:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/build        # import peclet.dem from the dev build
python examples/verify_packing_spheres.py
```

## Tests

Configure with `-DPECLET_DEM_BUILD_TESTS=ON` (add `-DPECLET_DEM_MPI=ON` for the distributed suites)
and run `ctest`: the kernel unit tests (`tests/kokkos`), the ArborX broad-phase + pipeline tests
(`tests/arborx`), the distributed ctests (`tests/kokkos_mpi`, np=1,2,4) and the Python suite
(`tests/python`, pytest: Hertz + non-spherical Hertz, cone friction, pair materials, coloured
Gauss-Seidel, statics battery, restitution, SDF particles, rotating drum, ...; `tests/python/mpi` is
launched through `mpirun` on top of `peclet.core.mpi`). See `CLAUDE.md` for the exact recipe; CI
(`.github/workflows/ci.yml`) runs all of it on the host OpenMP backend.

## Output & Visualization

The simulation supports two primary output formats:

### 1. LAMMPS + STL (Ovito)

For particle visualization (especially non-spherical shapes), we use the LAMMPS dump format combined with an STL mesh.

1.  **Generate Output**: The simulation writes `dump.custom.*` files.
2.  **Generate Shape**: Run `python examples/generate_particles.py` to create `particle_shape.stl`.
3.  **Visualize**:
    - Open **Ovito**.
    - Load the `dump.custom.*` sequence.
    - Add a **Particle Types** modifier.
    - Set the shape visualization to **Mesh/User-defined** and load `particle_shape.stl`.
    - Ovito will automatically scale the mesh by the particle radius.

*See `docs/visualization.md` for a detailed guide.*

### 2. VTI (ParaView)

For visualizing fields (like the Signed Distance Field or occupancy grids), the simulation exports VTI files (`.vti`).

1.  **Generate Output**: `sim.export_sdf("bed.vti", resolution=(128, 128, 128))` (the resolution is an `(rx, ry, rz)` triple; `sim.get_sdf_grid((rx, ry, rz))` returns the same field as an array indexed `[x, y, z]`).
2.  **Visualize**:
    - Open **ParaView**.
    - Load the `.vti` file.
    - Use "Volume" representation or "Slice" filter to inspect the field.

## Status

The single-GPU engine is complete and validated: it reaches stable high-density (random close)
packing, and energy is conserved to ~0.3% (see `docs/archive/packing_investigation.md`). Friction is
stabilized for spheres; body-body tangential friction is a known follow-up (currently weaker than
ideal). Active work is at-scale multi-GPU/MPI tuning.

> [!NOTE]
> The distributed (MPI) step is validated against the single-rank result (`tests/kokkos_mpi`,
> np=1,2,4 on OpenMP + CUDA) and supports dynamic load rebalancing; remaining MPI work is at-scale
> multi-GPU tuning.
