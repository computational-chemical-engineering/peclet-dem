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
    ctest --test-dir build_dev --output-on-failure -j1        # 47 tests (11 without PECLET_DEM_MPI)
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

**1.0.0 API traps (packages E + F, 2026-09-08).** A stepper called before `set_dt` raises
`RuntimeError` — there is no default time step, and `relax(n)` is the only exception. Every
per-particle setter raises if its row count is not `num_particles`, and `set_positions` raises past
`capacity`, where all four used to corrupt or silently truncate (`set_velocities` read an `(N,4)`
input as `(N,3)` and mis-indexed every row after the first). `get_sdf_grid((rx, ry, rz))` returns a
**Fortran-order** `(rx, ry, rz)` array indexed `[x, y, z]` — it used to hand back a C-order array
over x-fastest data, silently transposing the axes on a non-cubic grid; the setters
(`set_sdf_shape` / `add_sdf_shape` / `add_sdf_wall`) take that same 3-D array and convert a C-order
input implicitly. `max_overlap` is the position loop's last-iteration residual and under-reports;
`compute_overlaps()` re-measures the committed state. Shape and mode arguments are strings whose
error message lists the accepted set. No environment variable changes what the module computes.

**Single-rank periodic wrap contacts are asymmetric** (`sim.hpp` `demStep`, `ghostBand = maxRad`): only
grains within one radius of a periodic face get an image, so a wrap pair whose farther partner sits
beyond that band is detected from one side and the whole overlap correction lands on that partner
(measured 2026-09-08). The distributed step resolves the same pair symmetrically; the Python MPI
periodic test keeps its straddlers symmetric about the face for that reason.

## The two API tiers (QUALITY_PLAN §3.F, D2 — landed 2026-09-08)

`Simulation` carries only what a user needs to set up, run and read out a run; every instrument,
ablation and execution-policy switch lives on `sim.diagnostics` (a `Diagnostics` view holding a
reference to the simulation, no state of its own): `set_stabilization` with the two measurement
modes `'escalate'`/`'ordered'`, `set_velocity_solver('gauss_seidel'|'jacobi')` (the Jacobi A/B —
CHANGES RESULTS), `set_cuda_graphs` / `set_fused_sweeps` (bit-identical GPU submission policies),
`coloring_conflicts()`, `rest_orphan_stats()` / `rest_bank_stats()`, `wall_sdf_at()`,
`profiling_info()`, and under MPI `mpi_rebuilds` / `mpi_gathers`. `set_incremental_coloring` stays
public because it changes the Gauss-Seidel order. Conventions (suite `docs/NAMING.md`): stored scalars
are properties (`num_particles`, `num_contacts`, `max_overlap`, `dt`, `gravity`, `growth_factor`,
`stabilization`, `rank`, ...), computed scalars are bare methods (`compute_overlaps()`), array copies
are `get_*`; every triple is a 3-sequence (`set_gravity((0, 0, -9.8))`, `add_plane(point, normal)`);
every grid SDF is ONE `(nx, ny, nz)` array indexed `[x, y, z]` (`set_sdf_shape`, `add_sdf_shape`,
`add_sdf_wall` take it, `get_sdf_grid` returns it — a C-order input converts implicitly); shapes are
strings (`initialize_shape('sphere'|'hollow_cylinder'|'box', radius, ...)`, `set_sphere_shape` is
gone); `set_dt(dt)` + `step(n)` / `step_hertz(substeps)` / `step_mpi(n)` / `step_hertz_mpi(substeps)`
— no stepper takes `dt`, and a step before `set_dt` RAISES; `relax(n)` is the dynamics-free
overlap-removal substep that `step(0.0)` used to be.

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
- `set_dt(dt)` (dt > 0) BEFORE any of `step(n)`, `step_hertz(substeps)`, `step_mpi(n)`,
  `step_hertz_mpi(substeps)` — each raises `RuntimeError` otherwise (no default time step).
  `relax(n)` (overlap removal only) needs no dt. `set_positions` raises if N exceeds `capacity`,
  and every per-particle setter raises if its row count is not `num_particles`.
- The default body-body material is FRICTIONLESS (`set_material_params`); walls carry their own.

## Environment variables (QUALITY_PLAN package E is DONE — do not add more)

**No environment variable changes what `peclet.dem` computes.** The 16 `PECLET_DEM_*` runtime reads
were retired at 1.0.0: every knob that selected an algorithm became a `Simulation` setter with the
old behaviour as its default, and the four pure A/B ablations were deleted together with the code
they gated. `getenv` in `src/` is now exactly one call — `PECLET_DEM_HERTZ_PROFILE` (a timing
print in `solve_driver_force.hpp`).

| retired variable | replacement |
|---|---|
| `PECLET_DEM_REST_MODEL` | `set_restitution_model('newton'\|'poisson')` — default `'newton'` |
| `PECLET_DEM_SLEEP` | `set_sleeping(enabled)` — default `True` |
| `PECLET_DEM_SLEEP_SCALE` / `_K` / `PECLET_DEM_WAKE_SCALE` | `set_sleeping(threshold_scale=2.0, consecutive=64, wake_scale=40.0)` |
| `PECLET_DEM_SLEEP_WAKELOST` | `set_sleeping(wake_on_lost_contact=False)` |
| `PECLET_DEM_SLEEP_INVMASS_FRAC` | `set_sleeping(immovable_frac=0.01)` |
| `PECLET_DEM_VERLET_SKIN` | `set_verlet_skin(skin_frac)` — default `0.0` (off) |
| `PECLET_DEM_NO_GRAPH` | `diagnostics.set_cuda_graphs(enabled)` — default `True` |
| `PECLET_DEM_FUSED` / `PECLET_DEM_NO_FUSED` | `diagnostics.set_fused_sweeps('auto'\|'on'\|'off')` — default `'auto'` |
| `PECLET_DEM_NO_INCR_COLOR` | `set_incremental_coloring(enabled)` — default `True` |
| `PECLET_DEM_FUSED_GRID` | deleted (tuning knob; uncapped measured best at every size) |
| `PECLET_DEM_ML_GATES` | deleted (A/B over the multilevel gate mask; `kGateSlip` ships) |
| `PECLET_DEM_REST_NEWTON_OFF`, `PECLET_DEM_REST_ONESIDED` | deleted with their kernels (both measured worse) |

The stored scalars read back as properties: `sleeping`, `verlet_skin`, `incremental_coloring` on
`Simulation`; `cuda_graphs`, `fused_sweeps` on `diagnostics`. `set_cuda_graphs` / `set_fused_sweeps`
only choose how the same arithmetic is submitted to a GPU (bit-identical results, inert on non-CUDA
backends), which is why they are diagnostics; `set_incremental_coloring(False)` DOES change results —
the colouring fixes the Gauss-Seidel sweep order. `PECLET_DEM_STAB_MODE` and `PECLET_DEM_SYMMETRIC_PGS`
were never read by any version of the code (they survived only in comments);
`set_stabilization('off')` is what those scripts meant.

`PECLET_DEM_MPI` / `PECLET_DEM_MPI_HALO_HPP` are compile-time macros, not environment variables.

## CI (`.github/workflows`)

`ci.yml`: `single-rank` (host OpenMP, `OMP_NUM_THREADS=2`, `ctest -LE bench` over kokkos + arborx +
python, then a configure + one-TU compile against the DEFAULT `PECLET_CORE_TAG` as the stale-pin
check) and `mpi` (core as the sibling checkout, `peclet.core.mpi` built, `-L mpi` at np=1,2,4
oversubscribed, `OMP_NUM_THREADS=1`). `quality.yml`: ruff critical errors + a BLOCKING clang-format
over `src/` and `tests/` (`.clang-format`; `.clang-tidy` is voro's, informational). Pinned inputs:
Kokkos 5.1.1 / ArborX v2.1 (cached), nanobind 2.13.0. Watch a push with
`gh run watch -R computational-chemical-engineering/peclet-dem <id> --exit-status`.

## Docs (`docs/` vs `docs/archive/`)

`docs/` holds only what describes the code that ships — `solver_details.md` (both engines, the
shared solve driver, what changes results and what does not), `mpi.md` (the distributed step,
`sync_every`, periodicity/capacity rules, what is validated), `multi_gpu_testing.md` (device
binding, launch recipes, profiling, the optimisation backlog), `visualization.md` and `Doxyfile`.
Dated campaign records go to `docs/archive/` with a row in `docs/archive/README.md` — never
deleted, never maintained (QUALITY_PLAN D7). Currently archived: `packing_investigation.md` (the
five-phase RCP investigation) and `velocity_solver_algorithm.md` (pre-PGS summary of the velocity
solve). `src/dem_bindings.cpp` and `examples/pack.py` still cite `docs/packing_investigation.md`
as plain text — add the `archive/` prefix when those files are next touched.

## Git

Commit inside this repo, then bump the pointer in the umbrella. Stage named paths only.
