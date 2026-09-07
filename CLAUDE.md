# CLAUDE.md — dem (`peclet.dem`)

Kokkos + ArborX XPBD Discrete Element Method, header-only C++20 under `src/` behind one nanobind
class (`Simulation`, `src/dem_bindings.cpp`). Drive it from Python; there is no C++ main. Read the
umbrella `../CLAUDE.md` first (one venv, prefixes, `docs/NAMING.md`); `README.md` here has the folder map.

## Build

```bash
source ../.venv/bin/activate                      # the ONE suite venv (nanobind is found through it)
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp"   # or nvidia-cuda (nvcc on PATH)
cmake --build build -j8                           # -> build/peclet/dem/_dem.*.so ; PYTHONPATH=build; import peclet.dem
```
`-DPECLET_DEM_MPI=ON` links MPI and exposes `init_mpi` / `enable_mpi_step` / `step_mpi` / `step_hertz_mpi`
/ `rebalance` (see `docs/mpi.md`). The version comes from `pyproject.toml` (single source; CMake reads it).
The prefix picks the backend; never hard-code an arch. `pip install .` is the canonical install.

## Tests (three standalone CMake projects + Python scripts)

Each is its own `project()` configured against the same prefix (`-DCMAKE_PREFIX_PATH=...`), with the
header-only core found via `-DTPX_DIR=<suite>/core` (default: the sibling checkout):

| project | what | run |
|---|---|---|
| `tests/kokkos` | 8 kernel unit tests vs serial references (contact preprocessing, narrow-phase, velocity/position/friction solves, integration, periodicity, thermostat) | `cmake -S tests/kokkos -B build_kokkos ...; cmake --build build_kokkos -j8; OMP_NUM_THREADS=4 OMP_PROC_BIND=false ctest --test-dir build_kokkos --output-on-failure` |
| `tests/arborx` | ArborX broad-phase + full pipeline (2 tests) | same recipe with `tests/arborx` / `build_arborx` |
| `tests/kokkos_mpi` | distributed step / migration / rebalance, np=1,2,4 (24 ctests, host + CUDA) | add `-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun` (FindMPI may otherwise pick ParaView's) |

Python entry points (`PYTHONPATH=build python ...`): demos `verify_packing_spheres.py`,
`verify_packing_hollow_cylinders.py`, `verify_collision_{spheres,hollow_cylinders}.py`,
`verify_stacking{,_nofric,_inelastic}.py`, `verify_precession.py`, `verify_thermostat.py`;
assert-bearing checks `test_hertz.py`, `test_hertz_shapes.py`, `test_colored_gs.py`,
`test_pair_materials.py`, `test_cone_friction.py`, `test_statics_battery.py`, `test_bounce{,_gravity}.py`,
`tests/verify_{sdf_particle,restitution,rotating_drum}.py`; distributed `mpi/validate_exact.py`,
`mpi/validate_periodic.py`, `mpi/verify_distributed.py` (`mpirun -np N python mpi/...`).

**OMP trap:** every prefix carries an OpenMP host backend; bound the pool (`OMP_NUM_THREADS=4..8
OMP_PROC_BIND=false`) for any battery — an unbounded pool on the 48-core host is an hour-long stall.

## Call-order requirements (confirmed in `src/sim.hpp`)

- `initialize_shape(...)` / `set_sdf_shape(...)` RESET the shape registry to one shape; `add_shape` /
  `add_sdf_shape` / `add_scene_shape` append and return the index. Call the single-shape entry point
  BEFORE `add_shape` (it would wipe the mixture) and BEFORE `set_positions` (which stamps shape 0's
  unit-mass inverse inertia on every particle).
- `set_positions(xyz)` (re)sets the particle count and resets every particle's quaternion, scale,
  inverse mass, shape id (to 0), velocities and gid. So `set_shape_ids`, `set_inv_mass`,
  `set_inv_inertia`, `set_scales`, `set_velocities` all go AFTER it; `set_shape_ids` also requires
  exactly one id per particle.
- `set_domain(...)` (either form) and `set_global_scale(s)` BOTH reset the broad-phase skin to
  `0.1 * globalScale` — call `set_global_scale` first, then `set_domain`, if you rely on the skin.
- `step(dt)`: `dt > 0` advances; `dt == 0` (the default) is a dynamics-free relaxation substep
  (overlap removal only). `set_dt` only stores dt; `step` overwrites it with its argument.
- The default body-body material is FRICTIONLESS (`set_material_params`); walls carry their own.

## Environment variables read in `src/` (retiring them is QUALITY_PLAN package E; do not add more)

Constructor (`sim.hpp`, startup overrides): `PECLET_DEM_REST_MODEL` (newton|poisson restitution model),
`PECLET_DEM_SLEEP` (1/0 island sleeping), `PECLET_DEM_SLEEP_SCALE`, `PECLET_DEM_SLEEP_K`,
`PECLET_DEM_WAKE_SCALE`, `PECLET_DEM_SLEEP_WAKELOST`, `PECLET_DEM_SLEEP_INVMASS_FRAC` (sleeping tuning),
`PECLET_DEM_VERLET_SKIN` (Verlet-cached broad-phase skin fraction).
Solve driver (`solve_driver.hpp`, read once per process): `PECLET_DEM_NO_GRAPH` (disable CUDA-graph replay),
`PECLET_DEM_NO_INCR_COLOR` (full recolouring instead of incremental), `PECLET_DEM_REST_NEWTON_OFF`,
`PECLET_DEM_REST_ONESIDED` (restitution A/B ablations), `PECLET_DEM_ML_GATES` (multilevel gate mask).
Fused sweeps (`solver_fused.hpp`, CUDA): `PECLET_DEM_FUSED` / `PECLET_DEM_NO_FUSED` (force the path),
`PECLET_DEM_FUSED_GRID` (block-count cap). Force driver: `PECLET_DEM_HERTZ_PROFILE` (timing print).
`PECLET_DEM_STAB_MODE` / `PECLET_DEM_SYMMETRIC_PGS` appear in comments only — not read.

## Git

Commit inside this repo, then bump the pointer in the umbrella. Stage named paths only.
