# peclet-dem (`peclet.dem`)

[![PyPI version](https://img.shields.io/pypi/v/peclet-dem.svg)](https://pypi.org/project/peclet-dem/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/peclet-dem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
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
├── CMakeLists.txt              # Build configuration (find_package Kokkos + ArborX)
├── src                         # Kokkos sources (header-only, namespace peclet::dem)
│   ├── dem_bindings.cpp        # nanobind module entry point (the `peclet.dem` module)
│   ├── sim.hpp                 # Simulation facade + the demStep XPBD substep
│   ├── integration.hpp         # Time integration & prediction
│   ├── broadphase_arborx.hpp   # ArborX BVH broad-phase
│   ├── narrowphase.hpp         # Narrow-phase point-shell-vs-SDF collision
│   ├── solver_velocity.hpp     # Velocity solver kernels
│   ├── solver_position.hpp     # Position solver kernels (XPBD overlap removal)
│   ├── solver_friction.hpp     # Coulomb friction cluster
│   ├── output_sdf.hpp          # SDF/VTI grid generation (Eikonal)
│   ├── shapes_portable.hpp     # Analytic shapes (sphere / hollow cylinder / box)
│   ├── io.hpp                  # LAMMPS-dump + SDF-VTI export
│   └── mpi_halo.hpp            # Distributed particle halo (core), gated PECLET_DEM_MPI
├── tests                       # C++ unit tests: kokkos/ (kernels), arborx/, kokkos_mpi/
├── docs                        # Documentation
└── *.py                        # Python verification/example scripts (verify_*.py)
```

## Prerequisites

- **Linux**
- **CMake** >= 3.24
- **Kokkos 5.x + ArborX** (C++20) — provisioned by `../tools/bootstrap_deps.sh` into
  `../extern/install/<backend>` (`nvidia-cuda` / `host-openmp` / `lumi-hip`). A **hard build dependency**.
- **nanobind** + **scikit-build-core** (found via the active Python interpreter; see `pyproject.toml`)
- a backend compiler: **nvcc** (CUDA) on `PATH`, **hipcc** (ROCm), or just a host C++ compiler (OpenMP)
- **Python** >= 3.10
- **MPI** (optional, `-DDEM_MPI=ON`) — OpenMPI or MPICH

## Build Instructions

```bash
python -m venv .venv && source .venv/bin/activate && pip install nanobind numpy
export PATH=/usr/local/cuda-13.2/bin:$PATH        # if building the CUDA backend

# Canonical: build + install the module via scikit-build-core
CMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda" pip install .

# Or a dev cmake build (nanobind is found via the active interpreter, no cmakedir needed):
cmake -B build -S . -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build -j$(nproc)
```
*Swap the prefix to `../extern/install/host-openmp` for the OpenMP backend. `-DDEM_MPI=ON` links MPI
and exposes the distributed step (`init_mpi` / `enable_mpi_step` / `step_mpi`), including dynamic load
balancing — `enable_mpi_step(..., rebalance_every=N)` or an explicit `rebalance()` re-decomposes by
particle count (weighted ORB) and migrates ownership so each rank keeps a near-equal share.*

The compiled `peclet.dem` extension is placed in `build/peclet/dem/`; run scripts with `build/` on `PYTHONPATH` (`import peclet.dem`).

## Running Simulations

Example scripts are provided in the root directory:

```bash
# Add build artifact to python path if needed (or symlink it)
export PYTHONPATH=$PYTHONPATH:$(pwd)/build

# Run a verification script
python verify_packing_hollow_cylinders.py
```

## Output & Visualization

The simulation supports two primary output formats:

### 1. LAMMPS + STL (Ovito)

For particle visualization (especially non-spherical shapes), we use the LAMMPS dump format combined with an STL mesh.

1.  **Generate Output**: The simulation writes `dump.custom.*` files.
2.  **Generate Shape**: Run `python generate_particles.py` to create `particle_shape.stl`.
3.  **Visualize**:
    - Open **Ovito**.
    - Load the `dump.custom.*` sequence.
    - Add a **Particle Types** modifier.
    - Set the shape visualization to **Mesh/User-defined** and load `particle_shape.stl`.
    - Ovito will automatically scale the mesh by the particle radius.

*See `docs/visualization.md` for a detailed guide.*

### 2. VTI (ParaView)

For visualizing fields (like the Signed Distance Field or occupancy grids), the simulation exports VTI files (`.vti`).

1.  **Generate Output**: Use `Simulation.export_sdf("filename.vti", resolution=...)`.
2.  **Visualize**:
    - Open **ParaView**.
    - Load the `.vti` file.
    - Use "Volume" representation or "Slice" filter to inspect the field.

## Status

The single-GPU engine is complete and validated: it reaches stable high-density (random close)
packing, and energy is conserved to ~0.3% (see `docs/packing_investigation.md`). Friction is
stabilized for spheres; body-body tangential friction is a known follow-up (currently weaker than
ideal). Active work is at-scale multi-GPU/MPI tuning.

> [!NOTE]
> The distributed (MPI) step is validated against the single-rank result (`tests/kokkos_mpi`,
> np=1,2,4 on OpenMP + CUDA) and supports dynamic load rebalancing; remaining MPI work is at-scale
> multi-GPU tuning.
