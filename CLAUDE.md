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

## Tests (one tree per backend: `-DPECLET_DEM_BUILD_TESTS=ON`)

The C++ suites and the Python tests are registered by the ROOT CMake under one option (OFF by
default so a wheel build is unaffected; ON in CI):

```bash
cmake -S . -B build_dev -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp" \
      -DPECLET_DEM_BUILD_TESTS=ON -DPECLET_DEM_MPI=ON -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
cmake --build build_dev -j8
OMP_NUM_THREADS=2 OMP_PROC_BIND=false PYTHONPATH=<core-python-build> \
    ctest --test-dir build_dev --output-on-failure -j1        # 47 tests (35 without PECLET_DEM_MPI)
```

| suite | what | ctests |
|---|---|---|
| `tests/kokkos` | kernel unit tests vs serial references (contact preprocessing, narrow-phase, velocity/position/friction solves, integration, periodicity, thermostat) | 8 |
| `tests/arborx` | ArborX broad-phase vs an O(N^2) oracle + the full single-rank pipeline | 2 |
| `tests/kokkos_mpi` (needs `PECLET_DEM_MPI`) | distributed step (XPBD + Hertz engines, closed + periodic, mid-run rebalance) / migration / rebalance vs single-rank, np=1,2,4; label `mpi` | 24 |
| `tests/python` | `python_tests` = `pytest tests/python` on the module in the build tree: Hertz + non-spherical Hertz, cone friction (Walton), pair materials, coloured GS (binary exactness, conservation, Enskog cooling, colouring invariant), statics battery, bounce, restitution, SDF particles, hollow-cylinder overlap, growth packing, rotating drum; label `python` | 1 |
| `tests/python/mpi` (needs `PECLET_DEM_MPI`) | `python_mpi_<name>_np{1,2,4}`: exact step vs serial, periodic wrap, cross-rank observables, MPI rotating drum — launched through `mpirun`, on core's `peclet.core.mpi` + mpi4py (put a built `core/python` tree on `PYTHONPATH`; exit 77 = ctest SKIP when that stack is missing); labels `python;mpi` | 12 |

Each `tests/<suite>/CMakeLists.txt` still configures standalone (`cmake -S tests/kokkos -B build_kokkos
-DCMAKE_PREFIX_PATH=... [-DPECLET_CORE_DIR=<suite>/core]`) for a quick single-suite loop. `-DMPIEXEC_PREFLAGS=--oversubscribe`
lets the np=4 rung run on a 4-core machine. Nothing is labelled `bench` today; CI runs `ctest -LE bench`, i.e. everything.

`examples/` holds the demos (`verify_packing_{spheres,hollow_cylinders}.py`, `verify_collision_*.py`,
`verify_stacking*.py`, `verify_precession.py`, `verify_thermostat.py`, the `pack.py` / `pack_meter.py`
packing protocol + meter, `generate_*.py` shape / packing generators, and the distributed
`driver_distributed.py` skeleton + `bench_step.py` microbenchmark, `mpirun -np N python examples/...`).
They are run with the build tree on `PYTHONPATH`; none of them hard-codes a build directory.

**OMP trap:** every prefix carries an OpenMP host backend; bound the pool (`OMP_NUM_THREADS=2..8
OMP_PROC_BIND=false`) for any battery — an unbounded pool on the 48-core host is an hour-long stall.

**Defaults the legacy scripts tripped over (all confirmed in `src/`):** `velocityIterations` defaults
to 0 (no velocity solve, so no restitution — call `set_solver_iterations(pos, vel)`), gravity defaults
to zero, there is no implicit floor at the domain minimum (`add_plane`), and an `(N,4)` positions
array's 4th column is the INVERSE mass (`w=0` = fixed body).

**Single-rank periodic wrap contacts are asymmetric** (`sim.hpp` `demStep`, `ghostBand = maxRad`): only
grains within one radius of a periodic face get an image, so a wrap pair whose farther partner sits
beyond that band is detected from one side and the whole overlap correction lands on that partner
(measured 2026-09-08). The distributed step resolves the same pair symmetrically; the Python MPI
periodic test keeps its straddlers symmetric about the face for that reason.

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

## CI (`.github/workflows`)

`ci.yml`: `single-rank` (host OpenMP, `OMP_NUM_THREADS=2`, `ctest -LE bench` over kokkos + arborx +
python, then a configure + one-TU compile against the DEFAULT `PECLET_CORE_TAG` as the stale-pin
check) and `mpi` (core as the sibling checkout, `peclet.core.mpi` built, `-L mpi` at np=1,2,4
oversubscribed, `OMP_NUM_THREADS=1`). `quality.yml`: ruff critical errors + a BLOCKING clang-format
over `src/` and `tests/` (`.clang-format`; `.clang-tidy` is voro's, informational). Pinned inputs:
Kokkos 5.1.1 / ArborX v2.1 (cached), nanobind 2.13.0. Watch a push with
`gh run watch -R computational-chemical-engineering/peclet-dem <id> --exit-status`.

## Git

Commit inside this repo, then bump the pointer in the umbrella. Stage named paths only.
