# dem-gpu

GPU-accelerated Discrete Element Method (DEM) particle simulation prototype. combines a custom CUDA XPBD solver with SDF-based point-shell collision detection. Supports optional MPI for domain partitioning and features full Python bindings via pybind11 for easy scripting and visualization.

## Features

- **Hybrid XPBD Solver**: Two-pass velocity/position solver for stable high-density packing.
- **SDF Collision**: Point-shell collision detection using Signed Distance Fields (supports analytic shapes like hollow cylinders).
- **Periodicity**: Full periodic boundary conditions (Ghost Particles).
- **Python Bindings**: Control simulation logic, data initialization, and export entirely from Python.
- **MPI Support**: Optional Multi-GPU/Node support via domain decomposition.

## Folder Structure

```text
├── CMakeLists.txt              # Build configuration
├── src
│   ├── main_binding.cpp        # Pybind11 module entry point
│   ├── simulation.cpp          # Core Simulation class implementation
│   ├── cuda                    # CUDA Kernels
│   │   ├── integration.cu      # Time integration & prediction
│   │   ├── broadphase.cu       # BVH construction & traversal (cuBQL)
│   │   ├── narrowphase.cu      # Narrowphase collision detection
│   │   ├── solver_velocity.cu  # Velocity solver kernels
│   │   ├── solver_position.cu  # Position solver kernels
│   │   └── output_sdf.cu       # SDF/VTI grid generation
│   ├── shapes                  # Shape utilities
│   │   ├── ShapeManager.cpp    # GPU shape data management
│   │   └── point_sampler.cpp   # Point generation for shells
│   └── io
│       └── Exporter.cpp        # File export (LAMMPS, etc.)
├── python                      # Python utilities
├── tests                       # C++ Unit Tests
├── docs                        # Documentation
│   └── visualization.md        # Detailed visualization guide
└── *.py                        # Python verification/example scripts
```

## Prerequisites

- **Linux** (Recommended) or Windows
- **CMake** >= 3.24
- **CUDA Toolkit** (11.0+) & Compatible Driver
- **Python** >= 3.10
- **Pybind11** (Usually installed via pip or system package)
- **MPI** (Optional) - OpenMPI or MPICH

## Build Instructions

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install pybind11 numpy
```

### 2. Configure & Build
```bash
cmake -B build -S . -DDEMGPU_ENABLE_MPI=OFF
cmake --build build -j$(nproc)
```
*Note: Set `-DDEMGPU_ENABLE_MPI=ON` to enable MPI support.*

The compiled shared library (`demgpu.cpython-....so`) will be placed in the `build/` directory (or root depending on configuration). Ensure this file is accessible or simpler, run scripts from the root with the build directory in `PYTHONPATH`.

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

Prototype stage. Active development on high-density packing stability and friction handling.

> [!NOTE]
> MPI support is currently scaffolding and not fully implemented.
