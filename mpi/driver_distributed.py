"""Distributed packing-gpu driver skeleton: dem Simulation per rank, orchestrated by mpi4py +
transport-core's tpx_mpi (block decomposition + particle migration/ghosts).

Per step, each rank:
    get owned particles from its Simulation -> migrate (reassign ownership) -> gather ghosts within
    one interaction radius -> rebuild a Simulation over owned+ghost -> step -> keep owned.

Run (system python3 has mpi4py + numpy; build both modules first):
    cmake -S . -B build_sm120 -DDEMGPU_ENABLE_MPI=ON && cmake --build build_sm120 --target dem -j
    cmake -S ../transport-core/python -B ../transport-core/python/build && \
        cmake --build ../transport-core/python/build -j
    PYTHONPATH=build_sm120:../transport-core/python/build \
        mpirun -np 4 python3 mpi/driver_distributed.py

STATUS: implements the **FROZEN** scheme. Ghosts are set to inv_mass=0 (via the new dem
set_inv_mass) with zero velocity, so the solver treats them as fixed collision obstacles -- they push
owned particles but are never integrated. dem detects collisions once per substep and runs its
Jacobi iterations on the fixed contact set, so "ghosts fixed during the iterations" matches its serial
behaviour; FROZEN only approximates the boundary mass-split (owned takes the full correction).
The EXACT scheme (reverse-accumulate ghost constraint deltas to owners via ParticleHalo, so each
owner gets its mass-weighted share) is the next step -- see mpi/README.md.
"""
import sys
import numpy as np
from mpi4py import MPI
from peclet import dem
from peclet.core import mpi as tpx_mpi

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

# domain + decomposition
dmin = [0.0, 0.0, 0.0]
L = [8.0, 8.0, 8.0]
radius = 0.5
rcut = 2.0 * radius
mig = tpx_mpi.Migrator(origin=dmin, size=L, gsize=[32, 32, 32], periodic=[True, True, True])

# initial global particles (id 0..N-1), each rank seeds those it initially owns by id % size
N = 256
rng = np.random.RandomState(0)
all_pos = rng.uniform(0.5, 7.5, size=(N, 3))
mine = np.arange(rank, N, size)
pos = all_pos[mine].astype(np.float64)
vel = np.zeros((pos.shape[0], 3), dtype=np.float64)
ids = mine.astype(np.float64)


def build_sim(n):
    # Per-block solver: NON-periodic (periodicity is provided by the MPI ghosts), domain padded by
    # the interaction radius so ghost images just outside the box aren't clipped/wrapped.
    s = dem.Simulation(num_particles=int(n))
    m = rcut + 0.5
    s.set_domain((dmin[0] - m, dmin[1] - m, dmin[2] - m),
                 (dmin[0] + L[0] + m, dmin[1] + L[1] + m, dmin[2] + L[2] + m))
    s.enable_periodicity(False, False, False)
    s.initialize(shape_type=0, radius=radius)  # spheres
    s.set_solver_iterations(8, 0)
    return s


for step in range(5):
    # payload columns: vx, vy, vz, id. NB: a physically-complete driver must also carry the
    # quaternion + angular velocity through migration (see mpi/verify_distributed.py) -- dropping
    # rotational state discards spin energy each step. This FROZEN plumbing demo omits it for brevity.
    pay = np.column_stack([vel, ids]) if pos.shape[0] else np.zeros((0, 4))

    # 1) reassign ownership
    pos, pay = mig.migrate(pos, pay)
    vel, ids = pay[:, 0:3].copy(), pay[:, 3].copy()
    n_owned = pos.shape[0]

    # 2) gather ghosts (copies within rcut of this block; owned by other ranks)
    gpos, gpay = mig.gather_ghosts(pos, pay, rcut)
    n_ghost = gpos.shape[0]

    # 3) local Simulation over owned + ghost (ghosts appended after owned).
    #    FROZEN scheme: ghosts are infinite-mass (inv_mass=0) with zero velocity, so the solver
    #    treats them as fixed collision obstacles -- they push owned particles but are never
    #    integrated. (Physically an approximation at the boundary; the EXACT scheme reverse-
    #    accumulates ghost deltas to owners -- see README.)
    combined_pos = np.vstack([pos, gpos]) if n_ghost else pos
    ghost_vel = np.zeros((n_ghost, 3))
    combined_vel = np.vstack([vel, ghost_vel]) if n_ghost else vel
    inv_mass = np.concatenate([np.ones(n_owned), np.zeros(n_ghost)]).astype(np.float32)
    sim = build_sim(combined_pos.shape[0])
    sim.set_positions(combined_pos.astype(np.float32))
    sim.set_inv_mass(inv_mass)  # freeze the ghosts (must come after set_positions)
    sim.set_velocities(combined_vel.astype(np.float32))

    sim.step(0.002)

    # 4) keep only owned particles' updated state; verify the ghosts stayed frozen
    out_pos = np.array(sim.get_positions(False))
    if n_ghost:
        ghost_drift = float(np.max(np.abs(out_pos[n_owned:] - gpos)))
        assert ghost_drift < 1e-5, f"ghosts moved ({ghost_drift}) -- not frozen"
    pos = out_pos[:n_owned].astype(np.float64)
    vel = np.array(sim.get_velocities())[:n_owned].astype(np.float64)

    n_global = comm.allreduce(n_owned, MPI.SUM)
    g_ghost = comm.allreduce(n_ghost, MPI.SUM)
    if rank == 0:
        print(f"step {step}: owned(total)={n_global}/{N}  ghosts(total)={g_ghost}  "
              f"finite={bool(np.isfinite(out_pos).all())}")

n_global = comm.allreduce(pos.shape[0], MPI.SUM)
if rank == 0:
    ok = (n_global == N)
    print(f"{'OK' if ok else 'FAIL'} (np={size}): distributed FROZEN-ghost XPBD ran end-to-end, "
          f"{n_global}/{N} particles conserved, ghosts stayed frozen")
    sys.exit(0 if ok else 1)
