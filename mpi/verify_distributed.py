"""Cross-rank physics validation for the MPI-aware demgpu step.

Where validate_exact.py checks per-particle agreement, this checks the *physical observables* the
project's verify_*.py scripts care about -- and does so for scenarios whose contacts straddle the
inter-rank split, so it exercises cross-rank collision resolution. Each scenario is run two ways
(serial reference on rank 0 over all N; distributed across all ranks with migration + step_mpi) and
the aggregate observables are compared:

  A. Elastic granular gas (restitution=1, no gravity): total kinetic energy. Elastic XPBD should
     conserve KE; the distributed run must conserve it the *same way* the serial run does.
  B. Gravity settling onto a floor (restitution<1): packing observables -- max pair overlap (the
     penetration metric verify_packing_* reports), centre-of-mass height, floor contact height.

Observables match to tolerance even though per-particle positions differ by Jacobi atomic-ordering
noise. Run:
  PYTHONPATH=build_sm120:../transport-core/python/build mpirun -np 2 python3 mpi/verify_distributed.py
"""
import sys
import numpy as np
from mpi4py import MPI
import demgpu
import tpx_mpi

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

# one rank per GPU (no-op on a single device)
_loc = comm.Split_type(MPI.COMM_TYPE_SHARED)
_ndev = demgpu.Simulation.cuda_device_count()
if _ndev > 0:
    demgpu.Simulation.set_cuda_device(_loc.rank % _ndev)

dmin = [0.0, 0.0, 0.0]
L = [8.0, 8.0, 8.0]
radius = 0.4
rcut = 2.0 * radius
dt = 0.002


def make_sim(n, restitution, friction, grav, floor):
    s = demgpu.Simulation(num_particles=int(n))
    m = rcut + 0.5
    s.set_domain((dmin[0] - m, dmin[1] - m, dmin[2] - m), (L[0] + m, L[1] + m, L[2] + m))
    s.enable_periodicity(False, False, False)
    s.initialize(shape_type=1, radius=radius)  # sphere
    s.set_material_params(restitution, 0.0, friction)
    s.set_solver_iterations(8, 20)
    s.set_gravity(*grav)
    if floor:
        s.add_plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    return s


def run_serial(g_pos, g_vel, nsteps, **kw):
    s = make_sim(g_pos.shape[0], **kw)
    s.set_positions(g_pos.astype(np.float32))
    s.set_velocities(g_vel.astype(np.float32))
    for _ in range(nsteps):
        s.step(dt)
    return np.array(s.get_positions(False)), np.array(s.get_velocities())


def run_distributed(g_pos, g_vel, nsteps, **kw):
    N = g_pos.shape[0]
    ids = np.arange(N)
    mig = tpx_mpi.Migrator(origin=dmin, size=L, gsize=[16, 16, 16], periodic=[False, False, False])
    own = np.array([mig.owner_of(tuple(p)) for p in g_pos])
    mine = np.where(own == rank)[0]
    pos, vel, idd = g_pos[mine].copy(), g_vel[mine].copy(), ids[mine].astype(np.float64)
    for _ in range(nsteps):
        pay = np.column_stack([vel, idd]) if pos.shape[0] else np.zeros((0, 4))
        pos, pay = mig.migrate(pos, pay)
        vel, idd = pay[:, 0:3].copy(), pay[:, 3].copy()
        n = pos.shape[0]
        if n == 0:
            raise RuntimeError(f"rank {rank} owns 0 particles -- pick N/np so every rank is non-empty")
        s = make_sim(n, **kw)
        s.set_positions(pos.astype(np.float32))
        s.set_velocities(vel.astype(np.float32))
        s.mpi_init(origin=tuple(dmin), size=tuple(L), gsize=(16, 16, 16),
                   periodic=(False, False, False))
        s.enable_mpi_step(rcut, sync_every=1, forward_rotation=False)  # EXACT, spheres
        s.step(dt)
        pos = np.array(s.get_positions(False))[:n].astype(np.float64)
        vel = np.array(s.get_velocities())[:n].astype(np.float64)
    allp = comm.gather(pos, root=0)
    allv = comm.gather(vel, root=0)
    alli = comm.gather(idd, root=0)
    if rank != 0:
        return None, None
    P = np.full((N, 3), np.nan)
    V = np.full((N, 3), np.nan)
    for p, v, i in zip(allp, allv, alli):
        for k, idv in enumerate(i):
            P[int(idv)] = p[k]
            V[int(idv)] = v[k]
    assert np.isfinite(P).all(), "some ids missing"
    return P, V


def max_overlap(pos):
    n = pos.shape[0]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, 1e9)
    return float(np.maximum(0.0, 2.0 * radius - d).max())


def report(name, ref, dist, tol):
    ok = abs(ref - dist) <= tol + 1e-9 + 0.02 * abs(ref)
    flag = "OK" if ok else "FAIL"
    print(f"  [{flag}] {name:<22} serial={ref:+.5f}  dist={dist:+.5f}  |d|={abs(ref - dist):.2e}")
    return ok


# ----- Test A: elastic granular gas, KE conservation across ranks -----
rng = np.random.RandomState(7)
NA = 240
A_pos = rng.uniform(1.0, 7.0, size=(NA, 3))
A_vel = rng.normal(0.0, 3.0, size=(NA, 3))
kwA = dict(restitution=1.0, friction=0.0, grav=(0, 0, 0), floor=False)
rp, rv = (run_serial(A_pos, A_vel, 30, **kwA) if rank == 0 else (None, None))
dp, dv = run_distributed(A_pos, A_vel, 30, **kwA)

# ----- Test B: gravity settling onto a floor, packing observables -----
NB = 240
B_pos = rng.uniform(1.0, 7.0, size=(NB, 3))
B_vel = np.zeros((NB, 3))
kwB = dict(restitution=0.3, friction=0.5, grav=(0, 0, -9.8), floor=True)
rpB, rvB = (run_serial(B_pos, B_vel, 40, **kwB) if rank == 0 else (None, None))
dpB, dvB = run_distributed(B_pos, B_vel, 40, **kwB)

if rank == 0:
    oks = []
    print(f"np={size}: cross-rank physics vs serial reference")
    print("Test A -- elastic gas (KE conservation):")
    keR = 0.5 * float((rv ** 2).sum())
    keD = 0.5 * float((dv ** 2).sum())
    ke0 = 0.5 * float((A_vel ** 2).sum())
    print(f"  (KE_init={ke0:.4f}; serial conserves to {keR / ke0:.3f}, dist to {keD / ke0:.3f})")
    oks.append(report("total KE", keR, keD, tol=0.05 * ke0))
    print("Test B -- settling pack (penetration / heights):")
    oks.append(report("max overlap", max_overlap(rpB), max_overlap(dpB), tol=0.02 * radius))
    oks.append(report("mean z", float(rpB[:, 2].mean()), float(dpB[:, 2].mean()), tol=0.05))
    oks.append(report("min z", float(rpB[:, 2].min()), float(dpB[:, 2].min()), tol=0.05))
    allok = all(oks)
    print(f"{'OK' if allok else 'FAIL'} (np={size}): {sum(oks)}/{len(oks)} observables match serial")
    sys.exit(0 if allok else 1)
