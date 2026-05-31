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

## Next

- [ ] **Per-step loop** in the actual solver: predict → migrate (ownership) → gather ghosts → local
      broadphase/narrowphase + XPBD solve. Download/migrate/upload at the host boundary (migration is
      infrequent; CUDA-aware MPI is unavailable on this box — host-staged, as in cfd).
- [ ] Validate the `verify_*` scripts (packing fraction, restitution) match single-rank across ranks.

See `../../docs/ROADMAP.md` (Phase 4) and `../../cfd-gpu/doc/mpi_parallelization_status.md` for the
Eulerian precedent.
