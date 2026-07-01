"""Cross-rank physics validation for the MPI-aware dem step.

Where validate_exact.py checks per-particle agreement, this checks the *physical observables* the
project's verify_*.py scripts care about, for scenarios whose contacts straddle the inter-rank split.
Each scenario runs two ways -- serial reference on rank 0 over all N; distributed across all ranks
(migration + step_mpi) -- and the aggregate observables are compared.

  A. Elastic energy conservation. Particles start on a NON-overlapping grid (so there is no
     overlap-resolution transient injecting energy) inside a fully-walled box with restitution=1
     (elastic walls + elastic collisions). Total KE must stay ~constant, and the distributed run must
     conserve it the same way the serial run does -- i.e. collisions resolved across ranks neither
     gain nor lose energy.
  B. Settling pack. Gravity + floor, run long enough to actually land (min_z -> radius) and form
     load-bearing contacts. Compares the penetration metric verify_packing_* reports (max pair
     overlap) and the pile geometry (mean/min z).

Aggregate observables match even though per-particle positions differ by Jacobi atomic-ordering noise.
Run: PYTHONPATH=build_sm120:../transport-core/python/build mpirun -np 2 python3 mpi/verify_distributed.py
"""
import sys
import numpy as np
from mpi4py import MPI
from peclet import dem
from peclet.core import mpi as tpx_mpi

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

_loc = comm.Split_type(MPI.COMM_TYPE_SHARED)
_ndev = dem.Simulation.cuda_device_count()
if _ndev > 0:
    dem.Simulation.set_cuda_device(_loc.rank % _ndev)

dmin = [0.0, 0.0, 0.0]
L = [8.0, 8.0, 8.0]
radius = 0.4
rcut = 2.0 * radius
dt = 0.002

# Six global walls (floor/ceiling + 4 sides), inward normals, at the [0,L] domain boundary.
WALLS = [((0, 0, 0), (0, 0, 1)), ((0, 0, L[2]), (0, 0, -1)),
         ((0, 0, 0), (1, 0, 0)), ((L[0], 0, 0), (-1, 0, 0)),
         ((0, 0, 0), (0, 1, 0)), ((0, L[1], 0), (0, -1, 0))]


def grid_positions(n, seed):
    # jittered cubic grid, spacing 1.0 > diameter 0.8 (+/-0.08 jitter keeps min separation 0.84) so
    # the initial state has zero overlap -- no energy-injecting overlap-resolution transient.
    ax = np.arange(0.6, L[0] - 0.5, 1.0)
    pts = np.array([[x, y, z] for x in ax for y in ax for z in ax], dtype=np.float64)
    assert pts.shape[0] >= n, f"grid has {pts.shape[0]} < {n} sites"
    pts = pts[:n] + np.random.RandomState(seed).uniform(-0.08, 0.08, (n, 3))
    return pts


def make_sim(n, restitution, friction, grav, planes):
    s = dem.Simulation(num_particles=int(n))
    m = rcut + 0.5
    s.set_domain((dmin[0] - m, dmin[1] - m, dmin[2] - m), (L[0] + m, L[1] + m, L[2] + m))
    s.enable_periodicity(False, False, False)
    s.initialize(shape_type=1, radius=radius)
    s.set_material_params(restitution, 0.0, friction)
    s.set_solver_iterations(8, 20)
    s.set_gravity(*grav)
    for pt, nrm in planes:
        s.add_plane(pt, nrm)
    return s


def run_serial(g_pos, g_vel, nsteps, **kw):
    # Rebuild the Simulation every step, mirroring the distributed driver (which must rebuild because
    # migration changes the owned set), so the comparison isolates the cross-rank effect. CRUCIAL:
    # carry the FULL per-particle state (quaternion + angular velocity) across the rebuild -- spheres
    # pick up spin from off-centre collisions, and dropping it discards rotational energy (~1.4% KE).
    pos, vel = g_pos.copy(), g_vel.copy()
    quat = ang = None
    for _ in range(nsteps):
        s = make_sim(pos.shape[0], **kw)
        s.set_positions(pos.astype(np.float32))
        s.set_velocities(vel.astype(np.float32))
        if quat is not None:
            s.set_quaternions(quat.astype(np.float32))
            s.set_angular_velocities(ang.astype(np.float32))
        s.step(dt)
        pos = np.array(s.get_positions(False)).astype(np.float64)
        vel = np.array(s.get_velocities()).astype(np.float64)
        quat = np.array(s.get_quaternions()).astype(np.float64)
        ang = np.array(s.get_angular_velocities()).astype(np.float64)
    return pos, vel


def run_distributed(g_pos, g_vel, nsteps, **kw):
    N = g_pos.shape[0]
    ids = np.arange(N)
    mig = tpx_mpi.Migrator(origin=dmin, size=L, gsize=[16, 16, 16], periodic=[False, False, False])
    own = np.array([mig.owner_of(tuple(p)) for p in g_pos])
    mine = np.where(own == rank)[0]
    pos, vel, idd = g_pos[mine].copy(), g_vel[mine].copy(), ids[mine].astype(np.float64)
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (mine.size, 1))
    ang = np.zeros((mine.size, 3))
    for _ in range(nsteps):
        # carry FULL per-particle state through migration (vel + quaternion + angular velocity + id),
        # otherwise rotational energy is discarded each step (see run_serial).
        pay = (np.column_stack([vel, quat, ang, idd]) if pos.shape[0] else np.zeros((0, 11)))
        pos, pay = mig.migrate(pos, pay)
        vel, quat, ang, idd = pay[:, 0:3].copy(), pay[:, 3:7].copy(), pay[:, 7:10].copy(), pay[:, 10].copy()
        n = pos.shape[0]
        if n == 0:
            raise RuntimeError(f"rank {rank} owns 0 particles -- pick N/np so every rank is non-empty")
        s = make_sim(n, **kw)
        s.set_positions(pos.astype(np.float32))
        s.set_velocities(vel.astype(np.float32))
        s.set_quaternions(quat.astype(np.float32))
        s.set_angular_velocities(ang.astype(np.float32))
        s.mpi_init(origin=tuple(dmin), size=tuple(L), gsize=(16, 16, 16),
                   periodic=(False, False, False))
        s.enable_mpi_step(rcut, sync_every=1, forward_rotation=True)
        s.step(dt)
        pos = np.array(s.get_positions(False))[:n].astype(np.float64)
        vel = np.array(s.get_velocities())[:n].astype(np.float64)
        quat = np.array(s.get_quaternions())[:n].astype(np.float64)
        ang = np.array(s.get_angular_velocities())[:n].astype(np.float64)
    allp, allv, alli = comm.gather(pos, 0), comm.gather(vel, 0), comm.gather(idd, 0)
    if rank != 0:
        return None, None
    P, V = np.full((N, 3), np.nan), np.full((N, 3), np.nan)
    for p, v, i in zip(allp, allv, alli):
        for k, idv in enumerate(i):
            P[int(idv)], V[int(idv)] = p[k], v[k]
    assert np.isfinite(P).all(), "some ids missing"
    return P, V


def max_overlap(pos):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, 1e9)
    return float(np.maximum(0.0, 2.0 * radius - d).max())


def report(name, ref, dist, tol):
    ok = abs(ref - dist) <= tol
    print(f"  [{'OK' if ok else 'FAIL'}] {name:<14} serial={ref:+.5f}  dist={dist:+.5f}  "
          f"|d|={abs(ref - dist):.2e} (tol {tol:.1e})")
    return ok


# ----- Test A: elastic energy conservation (non-overlapping start, walled box) -----
NA = 240
A_pos = grid_positions(NA, seed=3)
A_vel = np.random.RandomState(7).normal(0.0, 4.0, size=(NA, 3))
kwA = dict(restitution=1.0, friction=0.0, grav=(0, 0, 0), planes=WALLS)
rpA, rvA = (run_serial(A_pos, A_vel, 80, **kwA) if rank == 0 else (None, None))
dpA, dvA = run_distributed(A_pos, A_vel, 80, **kwA)

# ----- Test B: settling pack, run long enough to actually land -----
NB = 240
B_pos = grid_positions(NB, seed=5)
B_vel = np.zeros((NB, 3))
kwB = dict(restitution=0.3, friction=0.5, grav=(0, 0, -9.8), planes=WALLS)
rpB, rvB = (run_serial(B_pos, B_vel, 300, **kwB) if rank == 0 else (None, None))
dpB, dvB = run_distributed(B_pos, B_vel, 300, **kwB)

if rank == 0:
    ke = lambda v: 0.5 * float((v ** 2).sum())
    ke0 = ke(A_vel)
    print(f"np={size}: cross-rank physics vs serial reference (serial uses the same per-step rebuild "
          f"as the distributed driver, so the comparison isolates the ghost/boundary effect).\n")

    gate = []
    # --- A: fast elastic energy conservation across the boundary (no leak at any np) ---
    print("[VALIDATION] fast elastic, walled box -- energy conserved across the rank boundary:")
    print(f"  serial conserves KE to {ke(rvA) / ke0:.4f}; distributed to {ke(dvA) / ke0:.4f} "
          f"(boundary diff = {100 * (ke(rvA) - ke(dvA)) / ke0:+.2f}% of KE_init)")
    gate.append(report("elastic KE", ke(rvA), ke(dvA), tol=0.01 * ke0))
    # --- B: dissipative settling (the DEM packing regime) ---
    print("[VALIDATION] settling pack -- penetration & pile geometry (the DEM use case):")
    gate.append(report("max overlap", max_overlap(rpB), max_overlap(dpB), tol=0.05 * radius))
    gate.append(report("mean z", float(rpB[:, 2].mean()), float(dpB[:, 2].mean()), tol=0.02))
    gate.append(report("min z", float(rpB[:, 2].min()), float(dpB[:, 2].min()), tol=0.02))

    ok = all(gate)
    print(f"\n{'OK' if ok else 'FAIL'} (np={size}): {sum(gate)}/{len(gate)} cross-rank observables "
          f"match serial (fast-elastic energy + quasi-static settling)")
    sys.exit(0 if ok else 1)
