# dem-gpu

GPU-accelerated Discrete Element / Particle simulation prototype combining CUDA kernels (XPBD solver, collision via SDF point-shell), optional MPI for domain partitioning, and Python bindings via pybind11 for scripting + visualization.

## Folder Structure

```text
├── CMakeLists.txt          # Build configuration (CUDA + MPI + pybind11)
├── src
│   ├── cuda
│   │   ├── bvh_adapter.cu   # Placeholder BVH wrapper (future cuBQL integration)
│   │   ├── xpbd_solver.cu   # XPBD constraint solver kernels
│   │   ├── collision.cu     # SDF point-shell collision kernels
│   │   └── memory_utils.cuh # CUDA RAII helpers / SoA utilities
│   ├── mpi
│   │   ├── domain.cpp       # Morton/Z-curve partitioning
│   │   └── communicator.cpp # Halo exchange helper wrappers
│   ├── shapes
│   │   ├── sdf_generator.cpp# CPU SDF grid generation
│   │   └── point_sampler.cpp# CPU shell point sampling
│   └── engine_binding.cpp   # pybind11 engine exposure
├── python
│   ├── simulation_script.py # Simple example entry point
│   └── visualizer.py        # Placeholder visualization utilities
```

## Prerequisites

- CMake >= 3.24
- A CUDA-capable GPU + NVIDIA drivers
- CUDA Toolkit (matching driver) for nvcc
- Python >= 3.10 plus `pybind11` development package (CMake config)
- MPI (MS-MPI on Windows or OpenMPI/MPICH on Linux) if `DEMGPU_ENABLE_MPI=ON`

On Windows you may install MS-MPI: [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) and ensure `MPI_HOME` is set or CMake can locate it.

## Configure & Build

```pwsh
git clone <repo-url> dem-gpu
cd dem-gpu
python -m venv .venv
Activate: .\.venv\Scripts\Activate.ps1
pip install pybind11

cmake -B build -S . -DPython_EXECUTABLE=$((Get-Command python).Source) -DDEMGPU_ENABLE_MPI=ON -DDEMGPU_CUDA_ARCH=native
cmake --build build --config Release
```

After build, the Python extension `demgpu` will reside in `build/` (or a config-specific subfolder). You can run the example:

```pwsh
python .\python\simulation_script.py
```

Disable MPI if not available:

```pwsh
cmake -B build -S . -DDEMGPU_ENABLE_MPI=OFF
cmake --build build --config Release
```

## Next Steps

- Replace placeholder BVH in `bvh_adapter.cu` with cuBQL integration.
- Flesh out XPBD solver with real constraints and time integration.
- Implement real SDF loading/generation (multi-shape compositions).
- Add asynchronous MPI halo exchanges overlapped with CUDA streams.
- Visualization: integrate matplotlib/pyvista for point clouds & grids.

---

## License

Add chosen license details (e.g., Apache-2.0, MIT) here.

## Contributing

PRs welcome. Please open issues for discussion before large changes.

## Status

Early prototype / scaffolding. APIs and kernels will change.
