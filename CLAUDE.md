# CLAUDE.md — dem (`peclet.dem`)

Kokkos + ArborX XPBD Discrete Element Method, header-only C++20 under `src/` behind one nanobind
class (`Simulation`, `src/dem_bindings.cpp`; the facade is `src/sim.hpp` on the `ShapeRegistry` base of
`src/shape_registry.hpp`, the step drivers are the free functions of `src/step_solve.hpp` and
`src/step_solve_mpi.hpp`). Drive it from Python; there is no C++ main. Read the
umbrella `../CLAUDE.md` first (one venv, prefixes, `docs/NAMING.md`); `README.md` here has the folder map.

## Build

```bash
source ../.venv/bin/activate                      # the ONE suite venv (nanobind is found through it)
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp"   # or nvidia-cuda (nvcc on PATH)
cmake --build build -j8                           # -> build/peclet/dem/_dem.*.so ; PYTHONPATH=build; import peclet.dem
```
`-DPECLET_DEM_MPI=ON` links MPI and exposes `init_mpi` / `enable_mpi_step` / `step_mpi` / `step_hertz_mpi`
/ `rebalance` / `migrate_to_weights` (see `docs/mpi.md`). The version comes from `pyproject.toml` (single source; CMake reads it).
The prefix picks the backend; never hard-code an arch. `pip install .` is the canonical install.

## Settled decisions — do not reverse silently

Chosen *against* the obvious or textbook alternative, on measured evidence. Full entries with
verbatim quotes and provenance in [`../docs/decisions/dem.md`](../docs/decisions/dem.md); the index is
[`../docs/DECISIONS.md`](../docs/DECISIONS.md). Reversing one takes a new recorded decision, not a
judgement call in the moment.

- **Contacts use the staged symmetric + one-sided solver** (Guendelman staging: Phase A all-symmetric,
  Phase B stabilization only if the post-loop residual exceeds 2·g·dt). The naive per-pair ballistic
  gate **failed** — any one-sided contact at the moving/static interface is a momentum sink.
- **Wall SDF sign convention is `val − residual`** (container convention), never `val + residual`.
- **Particle data layout is plain Kokkos SoA Views** with the backend-default layout — not `float4`,
  not a CaSoA variant.
- **A body touched from several places is solved through copies** (`docs/contact_solve_framework.md`):
  mass split (`m/k`, exact active count, mean reconciliation) for PGS and the overlap projection;
  exclusive holding (one rank per sync interval) for the `g = 0` one-shot restitution sweep. Never
  raw-sum ghost increments in the one-shot, never mass-split the one-shot (it compounds
  restitution), never force a colour. **Superseded** (2026-09-25, §12 S2 / WO-2): the old bullet
  here was "the velocity solve uses the over-relaxed `min(1, 2/count)` average, not a raw Jacobi
  sum" — that was the `'jacobi'` diagnostic's count-averaging, and it is now mass-split Jacobi:
  conservative, and at a fixed iteration budget more dissipative than the form it replaced.
- **The overlap projection sweeps whole contact pairs**, and **colourings never exceed 64
  colours**: hubs get local copies.
- **The PGS friction bound comes from the converged normal accumulator**, never from the live
  approach value.
- **Sleeping is an `invMassEff` swap around the solve call**, not a per-manifold mechanism.
- **Radius and halo sizing derive from `baseRadius*scale*globalScale`** — all three factors.
- **Distributed contacts are owner-exclusive with reverse accumulation**, not solved redundantly
  on both owners (the redundant solve was momentum-non-conserving under Gauss–Seidel).
- **Never use `get_max_overlap()` as the sole packing-quality gate.**
- Shipped constants that look arbitrary and are not: stabilization cap **K=64** (not 16), multilevel
  slip gate at **8·g·dt** (not ungated, not 2·g·dt).

## Tests (one tree per backend: `-DPECLET_DEM_BUILD_TESTS=ON`)

The C++ suites and the Python tests are registered by the ROOT CMake under one option (OFF by
default so a wheel build is unaffected; ON in CI):

```bash
cmake -S . -B build_dev -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp" \
      -DPECLET_DEM_BUILD_TESTS=ON -DPECLET_DEM_MPI=ON -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
cmake --build build_dev -j8
OMP_NUM_THREADS=2 OMP_PROC_BIND=false PYTHONPATH=<core-python-build> \
    ctest --test-dir build_dev --output-on-failure -j1        # 78 tests (11 without PECLET_DEM_MPI)
```

| suite | what | ctests |
|---|---|---|
| `tests/kokkos` | kernel unit tests vs serial references (contact preprocessing, narrow-phase, velocity/position/friction solves, integration, periodicity, thermostat) | 8 |
| `tests/arborx` | ArborX broad-phase vs an O(N^2) oracle + the full single-rank pipeline | 2 |
| `tests/kokkos_mpi` (needs `PECLET_DEM_MPI`) | distributed step (XPBD + Hertz engines, closed + periodic, mid-run rebalance) / migration / rebalance vs single-rank, and the collective schedule under rank-divergent layouts (`halo_schedule_*`: one-sided halo, divergent Verlet-skin rebuild, skin reuse across a reordering migration or with empty ranks; a hang = TIMEOUT 120 s) and the ghost band (`ghost_band_*`: a cross-face pair just inside the contact reach vs `MPI_COMM_SELF`), np=1,2,4, and `migrate_to_weights(w, align)` vs flow's aligned partition (`align_*`, np=1,2,4,8); label `mpi` | 55 |
| `tests/python` | `python_tests` = `pytest tests/python` on the module in the build tree: Hertz + non-spherical Hertz, cone friction (Walton), pair materials, coloured GS (binary exactness, conservation, Enskog cooling, colouring invariant), statics battery, bounce, restitution, SDF particles, hollow-cylinder overlap, growth packing, rotating drum, periodic wrap symmetry; label `python` | 1 |
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

**Single-rank periodic wrap contacts are symmetric** (`step_solve.hpp` `demStep`, `ghostBand = 2 *
maxRad + margin`, since 2026-09-10): every partner of every pair the narrow phase can report gets a
periodic image, so a wrap pair is resolved exactly like a pair inside the box — each body carries
half the overlap — and single-rank matches the distributed step (`tests/python/test_periodic_wrap_symmetry.py`,
`tests/python/mpi/test_validate_periodic.py` with asymmetric straddlers). The old band of one radius
still DETECTED every wrap pair (the nearer partner is always within one radius of the face) but a
pair with `max(a, b) > R_max` moved only its far partner, by its own half. Results of any periodic
single-rank run with such pairs changed at that commit; `coupling` sees it through `step()` only.

**The distributed step's MPI schedule must never branch on rank-local state** (2026-09-24). Every
rank has to make the same sequence of point-to-point and collective calls, and the particle halo is
NOT symmetric: a rank whose particles sit near a shared face owes ghosts to a neighbour that owes it
none. `ParticleHalo` (`src/mpi_halo.hpp`) used to skip every forward when `numGhost_ == 0`, so such a
rank skipped its sends and the step deadlocked (neighbour in `MPI_Waitall`, it in the next
`MPI_Allreduce`) — coupling's `test_mpi_moving_suspension` at np=4, a particle row on the y = 16 block
face. A forward may be skipped only when the rank neither sends nor receives (`exchanges()`), and a
decision that gates a collective (the Verlet-skin topology rebuild, an NBX round) is reduced across
ranks first; adaptive stops are Allreduce-MAXed for the same reason.
`tests/kokkos_mpi/test_halo_schedule_mpi.cpp` holds the layouts that exposed both; add a mode there
for any new one.

**`migrate_to_weights(w, align=1)` must build the partition the coupled flow solver owns**
(2026-09-24, S5 of `../amr/docs/amr_mg_core_boundary.md`). flow's `rebalance_by_weights(w)` builds
the ALIGNED weighted ORB (split planes on multiples of `2^a`, `a` from its 1.05 imbalance budget, for
its pressure multigrid) and returns `2^a`; coupling passes it here, and dem builds the same partition
through core's coarse-first `init(n, G, w, {align, ...})`. `align=1` is the plain weighted ORB, bit
for bit the pre-`align` call. dem's own `rebalance()` / `rebalance_every` stay UNALIGNED — alignment
is a multigrid concern and dem alone has none; do not "harmonise" them. `align` is validated (power
of two, divides the ORB grid, at least np aligned boxes, weights cover the grid → `ValueError`);
`tests/kokkos_mpi/test_align_mpi.cpp` (np = 1, 2, 4, 8) pins dem's partition to flow's call cell for
cell.

**Reproducibility trap — local order must never be MPI arrival order** (2026-09-25). core's NBX
rounds deliver messages in ARRIVAL order: `ParticleMigrator::migrate` appends migrants per message
and `ParticleHaloTopology::build` numbers the ghost blocks per message. The local slot order feeds
the manifold order (sorted by slot-pair key) → `colorKey` → colouring → Gauss–Seidel order, so an
arrival-ordered layout made two runs of one build diverge at np ≥ 4 (np = 8: 6 of 8 fingerprint
scenarios; the first divergent quantity was always the ghost order, positions follow 1–2 substeps
later). `mpi_halo.hpp` therefore unpacks ghost block g into `no + ghostSlot_(g)` (ascending source
rank) and stably re-orders migrants by `MigratePack::srcRank`; np ≤ 2 are bit-identical to before,
np 4/8 bitwise stable over 5 runs (OMP_NUM_THREADS=1). Any new receive path into the SoA must keep
this. **Threads are a second, separate source:** the narrow phase appends contacts in thread order,
so runs are bitwise reproducible only at `OMP_NUM_THREADS=1` (or Serial). Conservation does
not depend on it (see below).

**Every contact is owned by exactly one rank, and ghost increments are reverse-accumulated**
(2026-09-25, `docs/mpi_momentum_conservation.md`). `ContactOwnership` (`mpi_halo.hpp`):
the only owner that sees the pair, else the lower-gid body's owner. `demStepMpi` partitions
the contacts owned-first and reduces manifolds owned-first (key bit 63), and
`demSolveContacts` sweeps only `[0, ncOwned)` / `[0, nmOwned)`. Only the grounded/height
levels and the warm-ledger match read the visible range. Every `syncVelocities` /
`syncPositions` is **reverse, then forward**. The position phase opens with
`publishPositions` (forward only: the integration of a ghost is not an interaction). NEVER
re-introduce a redundant two-owner solve, never skip a reverse on rank-local grounds (it is
collective), and any new kernel that writes body state during the solve must write ghost
slots only as the partner half of an impulse pair: the reverse delivers exactly what is
there. `ghost_band_*` no longer pins one thread.

Since 2026-09-25 (`docs/contact_solve_framework.md`):
- `openVelocityPhase` exchanges the active-copy counts `k` and holder masks with the warm-start
  increments;
- every later sync applies the mean for `k > 1` in projection phases;
- the `g = 0` one-shot fires a rank-shared body's contacts only on its holder rank (a `gate` per
  manifold);
- the velocity phase maps every ghost image of a body to one slot (`realIndices`, set in
  `demStepMpi`, not in `gather`).

**The distributed XPBD ghost band is `max(rcut, 2.1 R_max + 0.25 R_max + P)`** (`P` = the Verlet
skin, else the substep's predicted displacement; `step_solve_mpi.hpp` `demStepMpi`,
`docs/contact_solve_framework.md` §5.1), and the step migrates to the current blocks when any body
is more than `0.25 R_max` outside its block (a vote folded into the existing radius `Allreduce`, so
no extra message). Both owners then see every pair. The narrow phase reports a pair while the gap
is below the margin `0.1 R_max`, so a cross-face partner can sit `2.1 R_max` from the face before
any allowance for motion between gathers; `rcut` is only a LOWER bound (default 0 = exactly the
reach), the margin is global (a rank-local one made polydisperse ranks disagree on a pair), and a
band or a migration rebuilds the ghost lists. `tests/kokkos_mpi/test_ghost_band_mpi.cpp` and
`test_missed_drift_*` hold the layouts.

**Ownership is not fixed across a run — the drift vote can migrate a body mid-run** (WO-7,
2026-09-26). A body more than `0.25 R_max` outside its owner's block triggers `migrateToBlocks`
(above), so a body's owning rank, and its slot, can change between steps. **Tests and any code
that pairs a body's state across a step must identify it by global id (gid), never by the setup's
owned-list index** (§12 S22) — the fixed-ownership assumption is gone. **Cross-rank Hertz pair
lists are canonically oriented, lower gid first** (§12 S21): the Mindlin spring's sign depends on
pair order, and a gid-keyed history carry through migration used to hand one rank the wrong sign
after a drift migration (measured dP 9.1e-4 with friction until fixed; at np 1, gid equals slot
and the broad phase already emits lower-first, so it is byte-identical there). Ghost selection and
the drift vote measure distance on the **domain-clamped** ownership coordinate on non-periodic
axes, not the raw position (§12 S20), so a body outside an unwalled boundary is still ghosted
across the nearest block face instead of silently dropped.

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

## Call-order requirements (confirmed in `src/sim.hpp` + `src/shape_registry.hpp`)

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
