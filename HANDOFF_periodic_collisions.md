# Handoff: fix periodic collision detection in peclet.dem

**Bug:** in the single-GPU `step()` path, particle–particle collisions **across a
periodic boundary** are neither detected nor resolved. Interior collisions work
fine. Discovered via the `random-packed-bed` gallery example (its g(r) had weight at
r < d — real overlap). Full context: `../peclet-examples/ISSUES.md` ("dem: periodic
collisions across a boundary…").

Paste the block below as the first message of a new session started in this
directory (`cd .../suite/dem`).

---

Fix a confirmed bug in peclet.dem: periodic collisions across a domain boundary
are not detected or resolved in the single-GPU step() path.

## The bug (minimal 2-particle repro)
Box L=6 ([-3,3]), radius 0.5, fully periodic. Two spheres overlapping ACROSS a
face are invisible to the engine; an identical overlap in the interior resolves
fine. Repro (PYTHONPATH=$PWD/build_mpi, or a fresh build):

    import numpy as np
    from peclet import dem
    def test(name, ax, bx, L=6.0):
        s=dem.Simulation(2); s.initialize(shape_type=1,radius=0.5)
        h=L/2; s.set_domain((-h,-h,-h),(h,h,h)); s.enable_periodicity(True,True,True)
        s.set_gravity(0,0,0); s.set_material_params(1.,1.,0.); s.set_solver_iterations(50,50)
        s.set_positions(np.array([[ax,0,0,1],[bx,0,0,1]],np.float32))
        s.set_velocities(np.zeros((2,4),np.float32)); s.set_scales(np.ones(2,np.float32)); s.set_dt(0.01)
        ov=float(s.compute_overlaps())
        for _ in range(300): s.step(0.01)
        p=s.get_positions()[:,:3]; d=p[1]-p[0]; d-=L*np.round(d/L)
        print(f"{name}: detected_overlap={ov:.3f} final_dist={np.linalg.norm(d):.3f}")
    test("interior (0.0, 0.6)", 0.0, 0.6)     # -> detected 0.400, final 1.000  WORKS
    test("boundary (2.7, -2.7)", 2.7, -2.7)   # -> detected 0.000, final 0.600  BROKEN

The fix is done when the boundary pair reports detected_overlap≈0.4 and relaxes to
final_dist≈1.0, identically to the interior pair.

## What is already localized (don't re-derive)
- step() -> demStep(P) (src/sim.hpp:619 -> :46); demStep calls generateGhostsKokkos
  at src/sim.hpp:75. computeOverlapsKokkos (src/sim.hpp:151) also ghosts at :165.
- Ghost emission is NOT the size issue: ghostBand = 1.0f*globalScale (sim.hpp:74);
  forcing set_global_scale(2.0) (which widens the band to definitely cover these
  particles, and also grows radius = scale*globalScale*baseRadius) does NOT fix it.
- generateGhostsKokkos (src/periodicity.hpp:22) emits shifted ghosts and maps each
  ghost back to its real owner via realIndices "for momentum conservation".
- Broadphase src/broadphase_arborx.hpp (findCollisionsArborX) queries each REAL
  particle against the BVH of real+ghost. Narrowphase src/narrowphase.hpp.
- The MPI path (demStepMpi, sim.hpp:210) deliberately DISCARDS ghost deltas
  ("ghost deltas land on self-mapped slots and are discarded", sim.hpp:~198) because
  each rank recomputes its own. If the single-GPU demStep shares that discard logic,
  a real–ghost contact's correction on the real particle would be lost — a strong
  candidate for the root cause.

## Two hypotheses to distinguish first
- H1: generateGhostsKokkos emits nothing across the boundary (emit condition / atomic
  slot alloc / capacity). Test: print readInt(P.topGhost) - P.numReal right after the
  generate call in demStep for the 2-particle boundary case; expect >0.
- H2: ghosts ARE emitted and the contact is found, but the XPBD velocity/position
  solve doesn't apply the correction to the real particle (delta discarded, or the
  ghost->owner scatter via realIndices is missing/atomic-clobbered).
Instrument num contacts / manifolds mid-step (get_num_contacts/get_num_manifolds)
for the boundary case to see whether a contact is even registered.

## Constraints
- This is a validated engine — match the CUDA semantics (the headers cite the CUDA
  originals). Do NOT change the distributed step_mpi path (it handles periodicity via
  the cross-rank halo and is separately validated; keep it byte-identical).
- Build (see dem section of ../CLAUDE.md): the OpenMP build used by the repro lives
  in build_mpi; rebuild with
    cmake -S . -B build_mpi -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp" -DDEM_MPI=ON
    cmake --build build_mpi -j
  (or the nvidia-cuda prefix for CUDA; put nvcc on PATH).

## Validation before declaring done
1. The 2-particle repro above: boundary pair resolves to ~1.0, detected ~0.4.
2. Also test contacts across CORNERS/EDGES (two/three periodic shifts at once).
3. Regenerate a packing (the LS protocol in ../peclet-examples/examples/random-packed-
   bed/index.qmd, function pack_bed) and check: min pairwise gap (periodic min-image)
   >= ~-0.005, dem get_max_overlap ~0, and g(r)=0 for r<d with first peak at r=d.
4. Run the dem test suite for regressions:
   ctest --test-dir build_mpi --output-on-failure   (and the kokkos_mpi tests).
5. Commit in dem, bump the umbrella pointer in ../ (push per the repo's usual flow).

## Downstream (after the fix)
Regenerate the sibling gallery example (../peclet-examples/examples/random-packed-bed):
with periodic collisions fixed, drop phi_ref back to ~0.62-0.64 (RCP — the 0.66 was a
mistaken attempt to raise Z that only added overlaps), re-render (PECLET_LOCAL_BUILD or
pip peclet), verify Z and g(r) look physical, and update the ISSUES.md entry status to
resolved. Re-add the example to the nav if it was pulled.
