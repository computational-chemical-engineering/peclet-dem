# Cross-Rank Contact Momentum Conservation — Reconnaissance Digest

**Target:** dem-momentum worktree, architecture for conserving momentum across rank boundaries when solving distributed contacts.

---

## 1. demSolveContacts Phase Sequence (`src/solve_driver.hpp:239`)

The shared contact solver runs the same modern sequence on both single-GPU and MPI paths.

### Signature and entry
```cpp
template <class Hooks>
inline void demSolveContacts(Particles& P, int nc, int nm, int nBodies,
                             Kokkos::View<const int*, CpMem> keyIdx, const Hooks& hooks)
```
- `nc`: contact count; `nm`: manifold count; `nBodies = numParticles` (owned + ghosts)
- `keyIdx`: body-slot → identity for pair-key construction (realIdx on single-GPU, gid under MPI)
- `hooks`: `MpiSolveHooks::distributed = true` gates MPI-specific paths

### Full phase skeleton (lines 239–919)

1. **Setup** (l.243–333): friction flag, persistent contact gate (gravity-dependent), incremental colouring policy, sleeping mask, legacy friction accumulator, Poisson restitution setup.

2. **Manifold colouring** (l.267–300): Graph-colour manifold topology (body-disjoint colour classes). Incremental carry on single-GPU with gravity (`incrColor`), full recolour on MPI or `g=0`. Leftover contacts get Jacobi fallback.

3. **Warm-started PGS setup** (l.319–356): If gravity present: load previous substep's converged normal impulse by pair key, record pre-solve approach, apply impulses up front. Orphan account settle (Poisson event-level restitution, MPI dedup via gid↔slot map).

4. **Persistent manifold marking** (l.396–398): Mark which manifolds are loaded from last substep (e-value gate). Update grounded levels (Guendelman shock propagation).

5. **Velocity phase** (l.400–505):
   - Side-flags zeroed (fully symmetric main sweeps; one-sided only in stabilization)
   - Approach velocity recorded (`computeVn0Kokkos`)
   - Warm impulses applied (if PGS)
   - **Hook call:** `hooks.syncVelocities(P)` if MPI, to refresh ghosts after warm start
   - Coloured-GS (or Jacobi) velocity sweeps, per-iteration residual max-reduced (`hooks.allMax`)
   - **Adaptive stop:** break if residual ≤ `0.02 * vRest` (PGS) or `vRest` (plain GS)
   - Leftover (colour-unsaturable) contacts treated with Jacobi fallback
   - **MPI sync:** `hooks.syncPoint(it)` every `syncEvery` iterations, final refresh

6. **Stabilization pass** (l.508–759): If residual remains > 2·g·dt, apply one-sided/multilevel/ordered sweeps (mode-gated). Each mode re-syncs boundaries and uses Allreduce-MAXed QS residual for adaptive stop.

7. **Poisson bookkeeping** (l.763–767): Bank kinetic compression for next substep (once per step on FINAL velocity state).

8. **Friction** (l.780–788): Legacy path (plane pre-computed normal load); applies velocity deltas, syncs.

9. **Position phase** (l.790–917):
   - Velocity→position integration (Euler on owned, ghost discarded)
   - **Hook call:** `hooks.syncPositions(P)` after predict
   - Contact colouring (graph-colour; incremental on single-GPU with carry)
   - Coloured-GS overlap projection loop, per-iteration residual max-reduced
   - **Adaptive stop:** break if residual < 1e-4 * baseRadius * globalScale
   - Jacobi fallback for leftover contacts
   - **MPI sync:** `hooks.syncPoint(it)`, final refresh
   - Position-channel impulse carry (friction Coulomb bound)

---

## 2. MpiSolveHooks and demStepMpi (`src/step_solve_mpi.hpp`)

### MpiSolveHooks structure (l.32–53)
```cpp
struct MpiSolveHooks {
  static constexpr bool distributed = true;
  ParticleHalo& halo;
  int syncEvery;
  bool forwardRotation;  // false for spheres (no quaternion forward)
  float allMax(float v) const { 
    float g = v;
    MPI_Allreduce(&v, &g, 1, MPI_FLOAT, MPI_MAX, halo.comm());
    return g;
  }
  bool syncPoint(int it) const { return (it + 1) % syncEvery == 0; }
  void syncVelocities(Particles& P) const {
    halo.forward(P.velPred);          // owner→ghost forward
    if (forwardRotation)
      halo.forward(P.angVelPred);
  }
  void syncPositions(Particles& P) const {
    halo.forwardPositions(P.posPred);  // with periodic image shift
    if (forwardRotation)
      halo.forward4(P.quatPred);
  }
};
```

### demStepMpi substep sequence (l.93–151)

1. **Growth update** (l.101–107): Scale particles if `growthFactor` active.

2. **Velocity predict on owned** (l.110–115): No ghosts yet; `P.numParticles = P.numReal`.

3. **Halo gather** (l.117–120): Fetch ghosts over band; sets `P.numParticles = numReal + numGhost`, self-maps `realIndices`.

4. **Narrow phase** (l.122–135): Broad/narrow phase + manifold reduction (owned + ghost bodies; `contactSlot` map for friction Coulomb carry).

5. **Solve** (l.137–141): Call `demSolveContacts` with `MpiSolveHooks`, which gates all sync/allMax calls.

6. **Commit** (l.143–144): Keep owned deltas; discard ghosts; thermostat if active.

7. **Restore** (l.150): `P.numParticles = P.numReal` for getters.

### MpiForceHooks (l.159–188)
Hertz–Mindlin driver hooks; halo topology rebuild at every pair-list rebuild (band = `2 R_max_global + skin`), state-forward between rebuilds. Allreduce gates the collective rebuild schedule.

### demStepHertzMpi (l.193–202)
Band = `(2.0 + skinFrac) * R_max_global` (Allreduce MAX over ranks). Calls `demStepForce` with `MpiForceHooks`.

---

## 3. Impulse Application — Velocity and Position Kernels

### Velocity impulse kernel sketch (`solver_velocity.hpp:55–225`)

**Kernel:** `solveVelocityKokkos`, one thread per manifold.

```cpp
// Per-manifold normal impulse (simplified):
const int idA = m.bodyA, idB = m.bodyB;
const int realA = realIdx(idA);     // Map to real index (own or ghost)
int realB = idB;
if (idB >= 0) {
  realB = realIdx(idB);
  if (realA > realB) return;         // Periodic dedup: canonical twin only
}

// Extract mass/inertia from both bodies (real indices)
const float invMassA = invMass(realA);
const float invMassB = (idB >= 0) ? invMass(realB) : 0.0f;  // Wall: zero
const F3 invIA = ld3(invInertia, realA);
const F3 invIB = (idB >= 0) ? ld3(invInertia, realB) : F3{0, 0, 0};

// Approach velocity (both bodies' vels read from SoA by real index)
float vn = dot3(vA, Nsum) + dot3(wA, TauA) + 
           dot3(vB, -Nsum) + dot3(wB, TauB);

// Lambda = impulse magnitude (symmetric formula; cancels on both bodies)
const float lambda = (-restitution * vn - vn) / (wA_n + wB_n);

// Linear impulse on A (atomic, can write to ghost slot)
Kokkos::atomic_add(&deltaVel(realA, 0), Jlin.x * invMassA);  // +J/m_A
Kokkos::atomic_add(&deltaVel(realA, 1), Jlin.y * invMassA);
Kokkos::atomic_add(&deltaVel(realA, 2), Jlin.z * invMassA);
// Angular: dw_world = R (invI_local * (R^T Jang))
{
  const F3 Jl = invRotateVector(qA, JangA);
  const F3 dwl{Jl.x * invIA.x, Jl.y * invIA.y, Jl.z * invIA.z};
  const F3 dww = rotateVector(qA, dwl);
  Kokkos::atomic_add(&deltaAngVel(realA, 0), dww.x);
  // ... y, z
}

// Impulse on B (opposite sign; Newton's third law, atomic)
if (idB >= 0 && applyB) {
  Kokkos::atomic_add(&deltaVel(realB, 0), -Jlin.x * invMassB);  // -J/m_B
  Kokkos::atomic_add(&deltaVel(realB, 1), -Jlin.y * invMassB);
  Kokkos::atomic_add(&deltaVel(realB, 2), -Jlin.z * invMassB);
  // Angular (opposite sign)
  const F3 Jl = invRotateVector(qB, JangB);
  const F3 dwl{Jl.x * invIB.x, Jl.y * invIB.y, Jl.z * invIB.z};
  const F3 dww = rotateVector(qB, dwl);
  Kokkos::atomic_add(&deltaAngVel(realB, 0), -dww.x);  // Note: opposite
  // ... y, z
}
Kokkos::atomic_add(&velCounts(realA), 1);
if (idB >= 0) Kokkos::atomic_add(&velCounts(realB), 1);
```

**Averaged apply** (`applyVelocityDeltasAveragedKokkos`, l.233–253):
```cpp
const float f = Kokkos::fmin(1.0f, 2.0f / static_cast<float>(count));
velPred(i, c) += deltaVel(i, c) * f;  // Over-relaxed: f ∈ [1, 2/count]
```

**Cross-rank write behavior:** Kernel atomically writes both bodies' deltas to their respective real indices (whether owner or ghost). Ghost slots' deltas are discarded at the next forward refresh; only owned rows are kept and integrated.

### Position impulse kernel (`solver_position.hpp:36–133`)

**Kernel:** `solvePositionKokkos`, one thread per contact (not manifold).

```cpp
const int idA = c.bodyA, idB = c.bodyB;  // Raw body slots
const float invMassA = invMass(idA);
const float invMassB = (idB >= 0) ? invMass(idB) : 0.0f;

const F3 pA = ldF3(posPred, idA);        // Predicted positions from body slots
const F3 pB = (idB >= 0) ? ldF3(posPred, idB) : F3{0, 0, 0};

// Penetration constraint
const F3 pAc = add3(pA, rA);             // Contact point
const F3 pBc = add3(pB, rB);
const float C = dot3(sub3(pAc, pBc), n);  // Negative = penetrated

if (C >= 0.0f) return;  // No overlap

// Compliance matrix element (inertia-weighted effective mass)
const float wTotal = computeW(rA, n, invMassA, invIA) + 
                     computeW(rB, n, invMassB, invIB);
const float dLambda = -C / wTotal;

// Linear correction: moves A and B apart
Kokkos::atomic_add(&deltaPos(idA, 0), n.x * dLambda * invMassA);  // +n·dL/m_A
Kokkos::atomic_add(&deltaPos(idA, 1), n.y * dLambda * invMassA);
Kokkos::atomic_add(&deltaPos(idA, 2), n.z * dLambda * invMassA);

if (idB >= 0) {
  Kokkos::atomic_add(&deltaPos(idB, 0), -n.x * dLambda * invMassB);  // -n·dL/m_B
  Kokkos::atomic_add(&deltaPos(idB, 1), -n.y * dLambda * invMassB);
  Kokkos::atomic_add(&deltaPos(idB, 2), -n.z * dLambda * invMassB);
}
```

**Ghost bodies:** Position kernel writes to BOTH owned AND ghost slots (identified by raw body index `idA`, `idB`). Ghost deltas are discarded by `applyUpdatesKokkos` after the colour sweep (only owned rows ∈ [0, numReal) are kept).

---

## 4. Colouring and Ghost Body Participation

### Manifold colouring (`solver_velocity.hpp`, implicit in `demSolveContacts`)

**Graph:** Vertices = body slots (owned + ghosts); edges = manifolds (bodyA, bodyB pairs).

**Colour assignment:** Greedy round-based max-index arbitration. No two manifolds sharing a body get the same colour. Colour class $C_i$ is an independent set → within a colour, RMW on sweeps is race-free.

**Incremental carry** (single-GPU only, gravity-gated): Surviving pairs carry colour across substeps; new manifolds arbitrate against them. Full recolour on MPI or first substep.

**Leftover contacts** (colour-mask saturation, degree > 62): Applied via Jacobi fallback with count-averaging.

**Ghost participation:** Ghosts participate in the colouring symmetrically. Their manifolds are coloured and swept; their deltas are atomically accumulated; deltas are then DISCARDED by `selfMapReals` (ghost slots map to themselves in `realIndices`, so updates land in ghost rows which are re-overwritten at the next gather/forward).

---

## 5. ParticleHalo Message Pattern (`src/mpi_halo.hpp`)

### Forward operations (l.854–876)

```cpp
void forward(V3 field) {
  if (!exchanges()) return;  // Gate: skip only if neither sends nor receives
  haloPackF3(field, ownedF3_, numReal_);
  dev_.forward(ownedF3_, ghostF3_);  // NBX or persistent neighbourhood
  haloUnpackF3(field, ghostF3_, shiftDev_, ghostSlot_, numReal_, numGhost_, /*doShift=*/false);
}

void forwardPositions(V3 field) {
  if (!exchanges()) return;
  haloPackF3(field, ownedF3_, numReal_);
  dev_.forward(ownedF3_, ghostF3_);
  haloUnpackF3(field, ghostF3_, shiftDev_, ghostSlot_, numReal_, numGhost_, /*doShift=*/true);
}

void forward4(V4 field) {  // Quaternions
  if (!exchanges()) return;
  haloPackF4(field, ownedF4_, numReal_);
  dev_.forward(ownedF4_, ghostF4_);
  haloUnpackF4(field, ghostF4_, ghostSlot_, numReal_, numGhost_);
}
```

**Message pattern:** Uses core's `ParticleHalo<3>::dev_.forward()` — **NBX (Neighborhood Collective MPI) on topology rebuild, persistent neighbourhood collective on state forwards**. One forward per field type (F3 vs F4).

**NO REVERSE** (ghost→owner accumulate) operations in dem. Core's `ParticleMigrator::migrate` and `ParticleHalo::gather` are one-way (owner→ghost). Force/velocity deltas computed on ghost slots are DISCARDED; only owned rows are integrated.

---

## 6. Hertz–Mindlin Force Driver (`src/solve_driver_force.hpp`)

### Pair force computation

**Signature:** `hertzPairForcesKokkos(pairs, np, pos, vel, angVel, rad, invMass, matId, pairMaterials, …, hertzE, hertzNu, dt, hertzXi, deltaVel, deltaAngVel)`

**Per-pair:** Load positions, velocities, radii from the pairs array (both bodies by slot index). Compute Hertz contact stiffness (elastic moduli, contact radius). Apply normal spring force + Mindlin tangential history xi. Scatter forces (sign opposite) to both bodies' slots.

**Hertz history:** Keyed by **gid-pair** (`pairKeyFromGids(gid_i, gid_j)`, l.76). Each owner tracks its half of paired history. On migration, `MigratePack` carries each particle's slice of the ledger (up to 14 pairs per particle in `hertzPairEntry[kWarmCarryMax]`).

**Single-rank behaviour:** Local slot indices ≡ gids (identity mapping) → history carries transparently. Under MPI, gid carries across halo rebuilds and ownership changes.

**Ghost participation:** Pair kernels atomically write forces to both bodies' deltaVel/deltaAngVel (owned or ghost slots). Ghost forces are cleared by `zeroForceScratchKokkos` (l.180–192) after integration; only owned rows are kept.

---

## 7. Test: test_ghost_band_mpi "margin" mode (~l.186)

**Scene:**
- Closed cube L=16, split at x=8 (2 blocks in x, 1 in y,z)
- Three bodies: p (x=7.7, r=1.0, scale=2.0), q (x=9.27, r=0.5), s (x=9.97, r=0.5)
- p on left block, q and s on right block
- Gap p-q: 0.07 (< contact reach); p-q pair discovered at different gaps by each block's margin

**The issue:** 
- p's block (large grains) has margin 0.1 * R_max = 0.1 (r=1.0) → reports p-q at gap ≥ -0.07
- q's block (small grains) has margin 0.1 * R_max = 0.05 (r=0.5) → drops p-q because gap < 0.07 violates its margin
- One-sided resolution: p's block moves p, q's block doesn't see the contact

**Validation:** Run 4 substeps distributed vs MPI_COMM_SELF (global particle set, no halo). Compare per-particle positions by global id; tolerance 1e-4.

**CMakeLists pinning** (`tests/kokkos_mpi/CMakeLists.txt:83`):
```cmake
set_tests_properties(ghost_band_${mode}_np${np} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1)
```

---

## 8. docs/mpi.md Lines 60–100 (Verbatim)

"and the `MPI_Allreduce(MAX)` on every adaptive-stop residual (a rank-local break would desynchronise the collective refreshes and deadlock). Persistent-contact pair keys are built from **global ids**, so the ledger survives halo rebuilds and ownership migration, and `MigratePack` carries each particle's slice of it across a rebalance. The force-based engine has the same shape in `src/solve_driver_force.hpp` (`SoloForceHooks` / `MpiForceHooks`, Mindlin history keyed by gid).
- **Periodicity:** cross-rank ghosts supply the wrap on *decomposed* axes; local periodic self-ghosts (the halo built with `includePeriodicSelf`) supply it on *undecomposed* periodic axes.

### The EXACT scheme + the `sync_every` (M) knob
`sync_every=1` is **EXACT**: every owned particle has all its neighbours refreshed every iteration, so it reproduces the serial XPBD delta at np=1; at np=2/4 the rank-local sweep order of the colored PGS differs from single-rank, so agreement is statistical rather than bit-exact (numbers under *What is validated* below). `sync_every=M>1` is an approximation — boundary error grows with M in exchange for fewer halo exchanges per step. `forward_rotation=False` skips the ghost quaternion forward and is **exact for spheres**.

### What is validated
- `tests/kokkos_mpi/` — the distributed Kokkos `demStep`/`rebalance` ctests, run under `mpirun` at **np=1,2,4**, in both a closed (non-periodic) box and a fully-periodic lattice (the periodic case exercises the local periodic self-ghosts on undecomposed axes). Build/run (from the root tree):
  ```bash
  cmake -S . -B build_dev -DCMAKE_PREFIX_PATH="<suite>/extern/install/<backend>" \
        -DPECLET_DEM_MPI=ON -DPECLET_DEM_BUILD_TESTS=ON -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
  cmake --build build_dev -j && OMP_NUM_THREADS=1 ctest --test-dir build_dev -L mpi --output-on-failure
  ```
  (`tests/kokkos_mpi` also still configures standalone.)
- `tests/python/mpi/` — the Python drivers on core's `peclet.core.mpi` + mpi4py, registered as the `python_mpi_*_np{1,2,4}` ctests (exit 77 = SKIP when that stack is missing): `test_validate_exact` (per-particle vs serial), `test_validate_periodic` (wrap through the split axes: 2-body, corner, N-body with resting straddlers), `test_verify_distributed` (elastic energy + settling-pack observables) and `test_verify_rotating_drum_mpi` (moving SDF wall + rebalancing).
- np=1 agrees with the single-rank step to float noise (max 1e-4 over 15 steps). At np=2/4 the modern stack (processor-block Gauss–Seidel with rank-local colouring) sweeps the finite-iteration PGS in a different order than single-rank, so per-particle agreement on a stiff, randomly overlapping IC is *statistical* (measured 2026-09-08, N=200, 15 steps: mean 5e-3, 95 % quantile 4e-2, max 0.11 = a quarter diameter); resting wrap contacts agree exactly and the aggregate observables (energy, overlap, pile geometry) match to their tolerances. The C++ `demstep_*` ctests are the tolerance-based statement of the same thing."

---

## 9. Ghost Band / Skin Logic

### Band calculation (`step_solve_mpi.hpp:93–108`)

```cpp
inline void demStepMpi(Particles& P, ParticleHalo& halo, double rcut, int syncEvery, ...) {
  const float margin = 0.1f * globalMaxRadius(P, halo.comm());  // 0.1 * R_max_global
  const double band = std::max(rcut, xpbdContactReach(globalMaxRadius(P, halo.comm())));
  // where xpbdContactReach(rMax) = 2.1 * rMax = 2 * rMax + 0.1 * rMax
}
```

**Reach:** Narrow phase reports pair at gap < margin, i.e. centre distance < r_i + r_j + margin. A body across the face can be ≤ 2 R_max + 0.1 R_max from the other. Both owners must see it.

**Band:** `max(caller's rcut, 2.1 R_max_global)` — if rcut < 2.1 R_max, it is widened.

**Visibility:** A cross-rank pair is visible to **BOTH owners** by design: the halo gathers ghosts in a band wide enough that both the owned and its cross-face partner sit within it. Each owner computes its body's delta locally; ghost updates are discarded.

**Rebuild trigger:** Ghost lists are rebuilt whenever the band changes (e.g., growth updates radius, or caller sets a larger rcut). `tests/kokkos_mpi/test_ghost_band_mpi.cpp:band_change` exercises this.

---

## Files Involved (Line Counts)

| File | Lines | Role |
|------|-------|------|
| `src/solve_driver.hpp` | 923 | Main velocity + position solve driver; all phase hooks |
| `src/step_solve_mpi.hpp` | 207 | MPI substep choreography; MpiSolveHooks, MpiForceHooks |
| `src/mpi_halo.hpp` | 1009 | ParticleHalo gather, forward, topology, MPI schedule |
| `src/solver_velocity.hpp` | 1733 | Manifold velocity solves (Jacobi, coloured GS, PGS); impulse apply |
| `src/solver_position.hpp` | 562 | Contact position solve (overlap projection); impulse apply |
| `src/contact_preprocessing.hpp` | 611 | Contact→manifold reduction; pair key, persistent detect |
| `src/solve_driver_force.hpp` | 317 | Force-based DEM driver (Hertz–Mindlin); gid-keyed history |
| `src/particles.hpp` | 547 | Particle SoA structure; manifold/contact arrays |
| `tests/kokkos_mpi/test_ghost_band_mpi.cpp` | (see file) | Ghost band validation: 4 modes (np=1,2,4,8) |
| `docs/mpi.md` | (see file) | Distributed step specification & validation summary |

**Total production:** ~5909 lines C++; tests + docs additional.

