# Plan: distributed (MPI) Kokkos demStep for demgpu_kokkos

The packing MPI is the only remaining piece of the suite Kokkos migration. The cfd MPI is **done** (the
whole `sdflow` solver runs multi-rank — `cfd-gpu/tests/kokkos_mpi/test_sdflow_mpi.cpp`, bit-exact np=1,2,4).
This note is the actionable plan for the packing equivalent. It is gated work (`DEMGPU_KOKKOS_MPI`, mirroring
the CUDA `DEMGPU_ENABLE_MPI`); the single-GPU `demgpu_kokkos` module must stay byte-identical when off.

## What already exists (no new primitives needed)

- **transport-core Kokkos particle halo** (validated np=1,2,4):
  - `tpx::halo::ParticleHalo<3>` (host, `include/tpx/halo/particle_halo.hpp`): `build(std::vector<Vec<3>> pos,
    double rcut)` → ghost topology; `numGhost()`; `forwardPositions(owned, ghost)` (host, applies the periodic
    image shift); `shift_` (per-ghost periodic shift, `flatten()` exposes it); `flatten()` device-friendly topo.
  - `tpx::halo::DeviceParticleHaloKokkos<3>` (`particle_halo_kokkos.hpp`): `init(const ParticleHalo<3>&)`,
    `forward<T>(View<T> owned, View<T> ghost)` (verbatim owner→ghost), `reverse<T>` (atomic ghost→owner, NOT
    needed here — the EXACT scheme uses only forwards), `numGhost()`. **`forward<T>` takes 1-D `View<T*>`** —
    so pack the per-particle state into a POD struct and `forward<GatherPack>` (the 2-D `View<float*[3/4]>`
    SoA fields are not directly forwardable).
- `tpx::halo::ParticleMigrator<3>` (`particle_migrator.hpp`) for re-assigning particles to owning ranks
  (packing barely migrates — particles stay in the box — so the per-step work is the GHOST GATHER, not
  migration; build the halo over the current owned positions each step like the CUDA does).
- The Kokkos `Particles` SoA (`src/particles.hpp`) already has **ghost slots** (`capacity >
  numReal`); `demStep` already runs broad/narrow/solve over `numParticles = numReal + numGhost`. The
  single-GPU periodic ghost generation (`generateGhostsKokkos`, `periodicity_kokkos.hpp`) fills those slots —
  **the MPI gather REPLACES that call**.

## The CUDA template (port faithfully)

`packing-gpu/src/simulation.cpp::step_mpi` (≈ lines 679–757) + `mpi_gather_ghosts` (≈946–1018) +
`mpi_forward_positions`/`mpi_forward4`, driven by `src/mpi/mpi_halo.h` (`MpiParticleHalo`). The CUDA
`GatherPack` (mpi_halo.h:112) = `{float4 vel, vel_pred, quat, quat_pred, ang_vel, ang_vel_pred, inv_inertia;
float scale; int shape;}` (positions forwarded separately with the shift; `.w`=inv_mass carried verbatim).

EXACT distributed XPBD substep (each owned particle sees ALL its neighbours — owned or ghost, both carrying
REAL mass — so it computes its full serial delta locally; ghosts are refreshed from owners each iteration):
1. predict velocity (owned only; `num_particles = num_real`).
2. **gather ghosts**: `halo.build(owned positions, rcut)`; forward the full owned state into ghost slots
   `[num_real, num_real+num_ghost)`; positions via the shifted forward, everything else verbatim; ghosts get
   `vel.w = 1` (ghost-type marker) and self-mapped `realIndices` (their deltas are discarded). Set
   `num_particles = num_real + num_ghost`.
3. broad/narrow phase + manifold reduction over owned+ghosts (the existing kernels — they already loop to
   `numParticles`).
4. velocity-solve loop; every `mpi_sync_every` iters (and the last), forward `velPred` (+`angVelPred` if
   rotating) owner→ghost.
5. apply velocity & predict position; forward `posPred` (shifted) (+`quatPred` if rotating).
6. position-solve loop; every `mpi_sync_every` iters (and last), forward `posPred`/`quatPred`.
7. final commit (owned results kept; ghosts discarded, re-gathered next substep). Restore `num_particles =
   num_real`.

`rcut` = the periodic ghost band = `1.0 * globalScale` (the skin; same as `sim_kokkos.hpp`'s `ghostBand`).

## Kokkos integration (the concrete edits)

- New gated header `src/mpi_halo.hpp`: a `KokkosParticleHalo` wrapping `ParticleHalo<3>` +
  `DeviceParticleHaloKokkos<3>`, with: `init(origin,size,gsize,periodic,comm)`; `gather(Particles&)` that
  downloads owned positions to host, `halo.build`, `dev.init`, packs owned `Particles` fields → `View<GatherPack>`,
  `dev.forward`, unpacks into ghost slots; `forwardPositions(V3 posPred)` = `dev.forward` of the position +
  a device kernel adding the per-ghost `shift_` (upload `shift_` to a device `View<F3>` once per build);
  `forward(V3/V4 field)` verbatim for vel/quat/etc. (one `forward<F3>`/`forward<F4>` each, slicing
  `[0,numReal)` owned and `[numReal,..)` ghost via `Kokkos::subview`).
- `sim_kokkos.hpp`: a gated `demStepMpi(Particles&, KokkosParticleHalo&)` mirroring the CUDA `step_mpi` order
  above, reusing all the existing kernels; replace step (2)’s `generateGhostsKokkos` with `halo.gather`; add
  the per-iteration forwards in the velocity/position loops. `KokkosSim` gets `initMpi(...)` + `stepMpi(n)`.
- `kokkos_module/binding.cpp`: gate `init_mpi`/`step_mpi` behind `DEMGPU_KOKKOS_MPI` (the default module never
  defines it → no MPI link, byte-identical). New CMake target `demgpu_kokkos_mpi` (find MPI + transport-core),
  like cfd’s `tests/kokkos_mpi`.

## Validation (weaker than cfd — plan for it)

The XPBD solver accumulates per-particle deltas via **atomics**, so the ghost ordering differs from the
single-block periodic gen and the atomic sum order differs → **NOT bit-exact, even at np=1** (unlike the cfd
solver, which has a unique steady fixed point). Validate:
1. a **few-step** position comparison (fixed seed config) distributed vs single-rank → agree to ~1e-4,
   growing slowly (the substep is physically the same — each owned particle’s full neighbourhood is present);
2. and/or **physical** equivalence over a full pack (same packing fraction / coordination statistics).
Per-process only (CUDA `demgpu` and `demgpu_kokkos` both own Kokkos init → can’t co-import; and the parity is
vs the single-rank Kokkos run, run as a size-1 communicator using the SAME code, like the cfd tests do with
`MPI_COMM_SELF`). Test file: `packing-gpu/tests/kokkos_mpi/test_demstep_mpi.cpp` (build pattern = cfd’s
`tests/kokkos_mpi/CMakeLists.txt`: `find_package(Kokkos+ArborX+MPI)` + header-only transport-core).

## Gotchas

- Ghosts carry **REAL mass** (the EXACT scheme) — do NOT set them infinite-mass; mark `vel.w=1` so velocity
  deltas landing on ghost slots are discarded (owner is remote).
- `posPred` AND committed `pos` are both forwarded (predict-velocity already advanced `posPred`; forwarding
  only `pos` leaves ghosts a predict-step stale → corrupts boundary contact normals; see CUDA comment).
- The position forward needs the **periodic image shift** (per ghost) — verbatim forward is wrong for
  positions across a periodic face.
- Build pattern + gating exactly mirror `cfd-gpu/tests/kokkos_mpi` and the `CFD_KOKKOS_MPI` gating in
  `mac_cutcell_mg_kokkos.hpp` / `sdflow_ibm_kokkos.hpp` (gated includes + `using tpx::halo::...` + members).
