"""Validate the EXACT MPI-aware dem step against a serial reference.

Same initial condition is run two ways and compared by particle id:
  * reference : rank 0 runs all N particles in one Simulation with the ordinary serial step().
  * distributed: every rank owns a subset (migrated each step) and runs the MPI-aware step()
                 (gather ghosts with real mass + per-iteration owner->ghost forward).
If the EXACT scheme is correct, the two agree to within Jacobi/atomic-ordering float noise.

Non-periodic domain with a ground plane (gravity settling) so the only ghosts are at the
inter-rank split -- this isolates MPI-ghost correctness from periodic-wrap handling.

Run:  PYTHONPATH=build_sm120:../transport-core/python/build mpirun -np 2 python3 mpi/validate_exact.py
"""
import os
import sys
import numpy as np
from mpi4py import MPI
import dem
import tpx_mpi

# M = sync_every (1 = EXACT), R = forward_rotation (1 = forward ghost quaternions). Defaults = EXACT.
SYNC_EVERY = int(os.environ.get("M", "1"))
FWD_ROT = bool(int(os.environ.get("R", "1")))

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

dmin = [0.0, 0.0, 0.0]
L = [8.0, 8.0, 8.0]
radius = 0.4
rcut = 2.0 * radius
N = 200
nsteps = 15
dt = 0.002

rng = np.random.RandomState(1)
g_pos = rng.uniform(1.0, 7.0, size=(N, 3)).astype(np.float64)
g_vel = np.zeros((N, 3))


def make_sim(n):
    s = dem.Simulation(num_particles=int(n))
    m = rcut + 0.5
    s.set_domain((dmin[0] - m, dmin[1] - m, dmin[2] - m), (L[0] + m, L[1] + m, L[2] + m))
    s.enable_periodicity(False, False, False)
    s.initialize(shape_type=0, radius=radius)
    s.add_plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))  # ground at z=0
    s.set_solver_iterations(8, 4)
    s.set_gravity(0.0, 0.0, -9.8)
    return s


# --- reference: all N particles, serial step (rank 0 only) ---
ref_pos = None
if rank == 0:
    ref = make_sim(N)
    ref.set_positions(g_pos.astype(np.float32))
    ref.set_velocities(g_vel.astype(np.float32))
    for _ in range(nsteps):
        ref.step(dt)
    ref_pos = np.array(ref.get_positions(False))

# --- distributed: round-robin ownership, migrate + MPI-aware step each step ---
mig = tpx_mpi.Migrator(origin=dmin, size=L, gsize=[16, 16, 16], periodic=[False, False, False])
mine = np.arange(rank, N, size)
pos = g_pos[mine].copy()
vel = g_vel[mine].copy()
ids = mine.astype(np.float64)

for step in range(nsteps):
    pay = np.column_stack([vel, ids]) if pos.shape[0] else np.zeros((0, 4))
    pos, pay = mig.migrate(pos, pay)
    vel, ids = pay[:, 0:3].copy(), pay[:, 3].copy()
    n = pos.shape[0]
    if n == 0:
        raise RuntimeError(f"rank {rank} owns 0 particles -- pick N/np so every rank is non-empty")

    s = make_sim(n)
    s.set_positions(pos.astype(np.float32))
    s.set_velocities(vel.astype(np.float32))
    s.mpi_init(origin=tuple(dmin), size=tuple(L), gsize=(16, 16, 16), periodic=(False, False, False))
    s.enable_mpi_step(rcut, sync_every=SYNC_EVERY, forward_rotation=FWD_ROT)
    s.step(dt)
    pos = np.array(s.get_positions(False))[:n].astype(np.float64)
    vel = np.array(s.get_velocities())[:n].astype(np.float64)

# gather distributed final state to rank 0, reorder by id, compare
allpos = comm.gather(pos, root=0)
allids = comm.gather(ids, root=0)
if rank == 0:
    D = np.full((N, 3), np.nan)
    for p, i in zip(allpos, allids):
        for k, idv in enumerate(i):
            D[int(idv)] = p[k]
    assert np.isfinite(D).all(), "some ids missing from distributed gather"
    err = np.abs(D - ref_pos)
    maxerr = float(err.max())
    meanerr = float(err.mean())
    ok = maxerr < 5e-2
    print(f"np={size}: max|dist-serial|={maxerr:.3e}  mean={meanerr:.3e}  "
          f"({'OK' if ok else 'CHECK'}) over {nsteps} steps, N={N}")
    sys.exit(0 if ok else 1)
