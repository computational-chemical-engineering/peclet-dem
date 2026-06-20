# dem — MPI integration (transport-core)

MPI block-parallelism for the DEM/XPBD solver on the shared `transport-core` library (sibling repo
`../../transport-core`), mirroring the approach validated for `sdflow`. The Lagrangian counterpart to
the Eulerian grid halo is **particle migration** (reassign particles to their owning rank) + **ghost
particles** (copies within one interaction radius of a block boundary, so each rank's ArborX broadphase
runs locally).

Standalone (decoupled from the main `demgpu` CUDA/pybind build — host C++ + MPI + header-only
transport-core):
```bash
cmake -S mpi -B mpi/build -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
cmake --build mpi/build -j
ctest --test-dir mpi/build --output-on-failure          # particle_migration_np{1,2,4}
```
Force `-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun` (FindMPI may otherwise pick ParaView's bundled mpiexec,
which launches OpenMPI binaries as singletons).

## Status

- [x] **Step 1 — migration (done):** `mpi/test_particle_migration.cpp` decomposes the periodic domain
      (`tpx::decomp::BlockDecomposer` over a cell grid covering `[domain_min, domain_min+size)`), builds
      a `tpx::halo::DomainMap` from packing's domain + periodicity, and migrates packing-style particles
      with `tpx::halo::ParticleMigrator`. Each particle's full per-particle SoA record (`PackingParticle`
      = pos/vel/quat/ang_vel/inv_inertia/scale/shape_id + id) is the opaque payload; its position drives
      ownership. Validated over 6 random-walk steps: count conserved, every particle on its owning rank,
      id multiset preserved, np=1,2,4.
- [x] **Step 2 — ghost particles (done):** the same test then calls
      `ParticleMigrator::gatherGhosts(rcut)` to collect copies of particles within one interaction
      radius of the block boundary (periodic images handled), the input to a local cuBQL broadphase.
      Each received ghost lies within `rcut` of this rank's block; rigorous correctness (vs a
      brute-force reference) is covered by transport-core's `test_ghost_particles_mpi`.

- [x] **Step 3 — distributed per-step loop (done, standalone):** `mpi/test_dem_step.cpp` runs the full
      Lagrangian loop the real solver will follow — `migrate → gatherGhosts(rcut) → forces on owned
      from owned+ghost → integrate` — each step, with a soft-sphere repulsion in place of XPBD/cuBQL so
      it builds with just MPI + transport-core. Validated against a serial N-body reference: every
      owned particle's position+velocity matches cell-for-cell over 25 steps, np=1,2,4 (bit-exact via
      min-image separations + id-ordered force summation). This is the distributed algorithm structure;
      what remains is swapping the toy force for packing's actual broadphase/narrowphase/XPBD.

## Next — wire into the real `Simulation`

The loop above must drive packing's actual GPU solver. Two prerequisites are currently blocked in this
environment:
- The prebuilt `demgpu` module **fails to import** (`CUDA Error: invalid argument`, stale build path)
  → needs a fresh rebuild for the sm_120 GPU.
- `mpi4py` is present in system Python but **not in packing's `.venv`**.

### The three communication schemes (validated as standalone DEM drivers)
All built on transport-core's persistent **`ParticleHalo`** (`build` + field-agnostic `forward` /
`reverse(sum)`); the communication machinery lives in the core, only the toy force + integrator are
here. Each matches a serial reference cell-for-cell, np=1,2,4:
- [x] **A — replicate / frozen ghosts** (`test_dem_step.cpp`): import ghost state, compute boundary
      pairs twice, integrate owned.
- [x] **B — Newton-on** (`test_dem_scheme_b.cpp`): import ghost state, compute each pair once,
      `reverse(force, sum)` the ghost reactions to owners, integrate owned.
- [x] **C — force-accumulation, local ghost integration** (`test_dem_scheme_c.cpp`): compute each pair
      once, `reverse(force)` to owners, **`forward(totalForce)` to ghosts, integrate owned + ghosts
      locally** (no state import between rebuilds). Also checks the ghost copies stay consistent with
      their owners. This is the scheme that maps to Voronoi conservative-flux exchange.

### Unblocked + the integration harness (done)
- [x] **demgpu builds + runs** on the sm_120 GPU with **`-DDEMGPU_ENABLE_MPI=ON`** (the default; the
      prebuilt `.so` was stale from another checkout). Correct call order is `initialize()` *before*
      `set_positions()`.
- [x] **`tpx_mpi` Python shim** (in `transport-core/python/`): exposes `ParticleMigrator` (migrate +
      gather_ghosts) to Python/mpi4py over numpy arrays. Validated np=1,2,4.
- [x] **mpi4py driver skeleton** `mpi/driver_distributed.py`: each rank runs a demgpu `Simulation`;
      per step `get → migrate (tpx_mpi) → gather ghosts → set owned+ghost → step → keep owned`. Runs
      end-to-end, particles conserved, np≥2 — the **plumbing** (demgpu + tpx_mpi + mpi4py) works.

### FROZEN scheme — implemented
- [x] **`set_inv_mass` added to demgpu** (`Simulation::set_inv_mass`, `get_inv_mass`): per-particle
      inverse mass, allowing `0` (= fixed/infinite-mass). `set_positions` forces `w=1`, so call
      `set_inv_mass` after it. Verified a frozen particle stays put under gravity + collisions.
- [x] **`mpi/driver_distributed.py` runs the FROZEN scheme**: gathered ghosts get `inv_mass=0` + zero
      velocity (fixed collision obstacles), the per-block solver is non-periodic (MPI ghosts provide
      periodicity) with a padded domain. Ghosts verified to stay frozen; particles conserved, np≥2.
      Because demgpu detects collisions once per substep and iterates a fixed contact set, "ghosts
      fixed during iterations" matches its *serial* behaviour — FROZEN only approximates the boundary
      **mass split** (the owned particle takes the full correction instead of its `w_i/(w_i+w_j)` share).

### EXACT scheme — implemented + validated (MPI-aware `step()` in C++)
- [x] **transport-core wired into demgpu** (`-DDEMGPU_HAVE_TPX`, header-only sibling repo).
      `src/mpi/mpi_halo.h` (`MpiParticleHalo`) wraps `tpx` BlockDecomposer + ParticleMigrator +
      ParticleHalo with host-staged `forward_positions` (xyz + periodic image shift, `.w`=inv_mass) /
      `forward4` / `forward_float` / `forward_int`.
- [x] **`Simulation::step_mpi()` — the EXACT pipeline.** Enabled via `mpi_init(origin,size,gsize,
      periodic)` + `enable_mpi_step(rcut)`. Each substep: predict velocity (owned) → **gather ghosts
      carrying REAL mass** (`mpi_gather_ghosts`: build halo over owned, forward full state into the
      ghost slots `[num_real, num_real+num_ghost)`, mark `d_vel.w=1`, self-map `d_real_indices`) →
      broadphase/narrowphase over owned+ghost → velocity solve **with an owner→ghost `forward4` after
      every iteration** → apply-velocity/predict-position + forward → position solve **with an
      owner→ghost `forward_positions`+`forward4(quat)` after every iteration** → commit (owned kept).
      Each owned particle therefore sees *all* its neighbours every iteration and computes its
      **complete serial XPBD delta locally** — no constraint-delta reverse, no double-count filtering.
- [x] **Validated vs serial** (`mpi/validate_exact.py`, same IC run serially on rank 0 and distributed):
      np=1 **bit-exact** (`0.0`); np=2/4 agree to **mean ~5e-5, max ~4e-3** over 15 settling steps
      (Jacobi atomic-ordering float noise at the split, not a physics error).
### Perf pass (`mpi/bench_step.py`)
The per-step cost is dominated by the **host-staged owner→ghost exchange** (CUDA-aware MPI is
unavailable on this box). Done:
- [x] **Persistent direct exchange** in `tpx::halo::ParticleHalo` — after `build()` the neighbour set
      and message sizes are fixed, so `forward`/`forwardPositions`/`reverse` now use plain
      `Irecv`/`Isend`/`Waitall` instead of an NBX consensus round every call (matters at scale /
      multi-node; the 19 transport-core ctests still pass).
- [x] **Combined gather** — the substep ghost gather packs all non-position fields into one record
      (`MpiParticleHalo::GatherPack`), so it is **3 MPI exchanges/step instead of 9** (`d_pos`,
      `d_pos_pred`, the pack). NB: `d_pos_pred` *must* be forwarded, not copied from `d_pos` — see the
      energy-leak fix below.
- [x] **`sync_every` (M) knob** — `enable_mpi_step(rcut, sync_every=M)`. M=1 is EXACT (mean err ~2e-5);
      M=4 is ~36% faster at a mean boundary error ~1.7e-3 (`np=2`, 15 settling steps).
- [x] **`forward_rotation=False`** — skips the ghost quaternion forward; **exact for spheres**
      (bit-identical validation) and ~12% faster.

Indicative `np=2`, N=4000 spheres on the single dev GPU: EXACT 10.3 -> **9.0 ms/step**; M=4 **6.6 ms/step**.
NB: on **one** GPU, `np>1` shares a single device (CUDA contexts serialise), so absolute numbers here
are contention-bound and *not* representative of 1-rank-per-GPU scaling. The remaining lever is a
**device-resident pack** (gather/scatter kernels to cut the ~18 synchronous `cudaMemcpy`/step to ~2),
best measured on real multi-GPU.

- [x] **Cross-rank physics** (`mpi/verify_distributed.py`): observables the `verify_*` scripts report,
      for contacts straddling the rank split, serial vs distributed (the serial reference uses the
      same per-step rebuild as the distributed driver, isolating the genuine boundary effect). Adds
      `Simulation.set_cuda_device`/`cuda_device_count` for one-rank-per-GPU binding.
  - **Validated — quasi-static settling (the DEM packing regime):** max pair overlap, mean z, min z
    match serial to **~1e-5** (np=1/2/4); np=1 (0 ghosts) is bit-exact.
  - **Fixed — rotational-state migration bug (was mis-attributed to the boundary):** the example
    drivers carried only `[vel,id]` through migration, dropping each particle's quaternion + angular
    velocity. Frictionless spheres still pick up spin from off-centre collisions, so resetting it to
    identity every step **discarded rotational energy (~1.4% KE at np=1)**. Carrying the full state
    (`set_quaternions`/`set_angular_velocities`, and forwarding `d_ang_vel_pred` in `step_mpi`'s
    velocity loop when `forward_rotation`) restores **np=1 to 0.0% leak** (KE 0.9966 = serial,
    bit-exact). `verify_distributed.py` now carries full state; so should any real driver.
  - **FIXED — fast-elastic boundary leak (was a stale ghost `d_pos_pred`):** a non-overlapping walled
    elastic (e=1) gas now conserves KE **identically to serial at every np** (np=1/2/4 all 0.9966 vs
    serial 0.9966; boundary diff < 0.01%). Root cause, found by per-contact impulse instrumentation:
    the substep gather filled the ghost **predicted** position by *copying the committed* `d_pos`
    (`ghost d_pos_pred = ghost d_pos`) on the assumption `pos_pred == pos` at gather time. But
    `predict_velocity` already advances `d_pos_pred = d_pos + v·dt`, so ghosts entered the solve **one
    predict-step stale** (~`v·dt`), giving boundary contacts a slightly wrong normal/relative velocity
    and hence a wrong impulse — small per contact, systematically dissipative, accumulating, and
    growing with the boundary (ghost) fraction → with np. Invisible in clean 2-/4-body collisions
    (exact) and in quasi-static settling (~1e-5); only dense+fast clusters exposed it. Fix:
    `mpi_gather_ghosts` forwards `d_pos_pred` with the periodic shift, like `d_pos` (one extra
    exchange/step, ~0.2 ms). This was a regression from the step7c "combined gather" optimisation.
    `verify_distributed.py`'s fast-elastic KE is now a hard gate (passes np=1/2/4).
- [x] **Multi-GPU testing & profiling guide:** [multi_gpu_testing.md](multi_gpu_testing.md) — device
      binding, launch recipes, scaling/comm-fraction metrics, the Nsight/MPI profiling toolchain, and
      the ranked optimisation backlog (device-resident pack first).
  - **Periodic-wrap validated** (`mpi/validate_periodic.py`): a periodic axis only works distributed
    if it is split across ≥2 ranks (a rank never ghosts to itself), so the test makes exactly the
    ORB-split axes periodic via the halo (X at np=2; X+Y at np=4) and walls the rest, comparing to a
    serial reference that uses demgpu's *internal* periodicity. Deterministic **2-body** (single-axis,
    0.6→0.8) and **corner** (diagonal X+Y, 0.707→0.8) wrap collisions resolve **exactly** (serial =
    dist = analytic diameter); the N-body matches serial to mean ~6e-5 with periodicity adding **zero
    error over the non-periodic baseline** (max identical at np=1/2/4). `withinRcutOfBlock` picks the
    best periodic image per axis independently, so corner/diagonal images are handled. (Note: a clean
    per-particle comparison needs a non-overlapping IC — stiff initial overlaps are chaotically
    sensitive and mask the wrap signal.)
- [ ] Remaining: device-resident pack + reuse a fixed-capacity `Simulation` instead of rebuilding each
      step (both best measured on real multi-GPU — see the guide).

Note: packing already has its own MPI scaffolding (`src/mpi/communicator.cpp`, `domain.cpp`); the
transport-core approach above can complement or supersede it.

See `../../docs/ROADMAP.md` (Phase 4) and `../../sdflow/doc/mpi_parallelization_status.md` for the
Eulerian precedent.
