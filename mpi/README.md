# packing-gpu — MPI integration (transport-core)

MPI block-parallelism for the DEM/XPBD solver on the shared `transport-core` library (sibling repo
`../../transport-core`), mirroring the approach validated for `cfd-gpu`. The Lagrangian counterpart to
cfd's grid halo is **particle migration** (reassign particles to their owning rank) + **ghost
particles** (copies within one interaction radius of a block boundary, so each rank's cuBQL broadphase
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

### EXACT scheme + remaining
- [ ] **EXACT (physics-unchanged):** compute each cross contact once, **reverse-accumulate the ghost
      constraint deltas to their owners** (via `ParticleHalo::reverse`) so each owner gets its
      mass-weighted share, with a sync every `M` iterations (M-deep halo). demgpu already accumulates
      *periodic*-ghost deltas to reals via `d_real_indices` + `d_delta_pos`; the change is to route the
      MPI-ghost deltas cross-rank. Needs the solver to expose / cross-communicate the per-particle delta
      accumulators inside the position-iteration loop.
- [ ] Validate `verify_*` (packing fraction, restitution) match single-rank across ranks.
- [ ] Perf: reuse a fixed-capacity `Simulation` + active-count instead of rebuilding each step.

Note: packing already has its own MPI scaffolding (`src/mpi/communicator.cpp`, `domain.cpp`); the
transport-core approach above can complement or supersede it.

See `../../docs/ROADMAP.md` (Phase 4) and `../../cfd-gpu/doc/mpi_parallelization_status.md` for the
Eulerian precedent.
