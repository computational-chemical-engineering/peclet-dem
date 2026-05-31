"""Distributed packing-gpu driver skeleton: demgpu Simulation per rank, orchestrated by mpi4py +
transport-core's tpx_mpi (block decomposition + particle migration/ghosts).

Per step, each rank:
    get owned particles from its Simulation -> migrate (reassign ownership) -> gather ghosts within
    one interaction radius -> rebuild a Simulation over owned+ghost -> step -> keep owned.

Run (system python3 has mpi4py + numpy; build both modules first):
    cmake -S . -B build_sm120 -DDEMGPU_ENABLE_MPI=ON && cmake --build build_sm120 --target demgpu -j
    cmake -S ../transport-core/python -B ../transport-core/python/build && \
        cmake --build ../transport-core/python/build -j
    PYTHONPATH=build_sm120:../transport-core/python/build \
        mpirun -np 4 python3 mpi/driver_distributed.py

STATUS: this validates the *plumbing* — demgpu + tpx_mpi + mpi4py interoperate and the
get->migrate->ghost->set->step->extract loop runs end-to-end across ranks. It is NOT yet physically
correct distributed packing, because ghost particles must be treated as **fixed (infinite mass)**
during the local XPBD constraint solve so each contact is resolved consistently across ranks. demgpu
stores inv_mass in d_pos.w and the Python API has no per-particle inv_mass setter, so the remaining
work is a small demgpu change: a "fixed/ghost" flag (or set_inv_mass) so gathered ghosts participate
in collision detection but are not integrated. See mpi/README.md.
"""
import sys
import numpy as np
from mpi4py import MPI
import demgpu
import tpx_mpi

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
    s = demgpu.Simulation(num_particles=int(n))
    s.set_domain((dmin[0], dmin[1], dmin[2]), (dmin[0] + L[0], dmin[1] + L[1], dmin[2] + L[2]))
    s.enable_periodicity(True, True, True)
    s.initialize(shape_type=0, radius=radius)  # spheres
    s.set_solver_iterations(8, 0)
    return s


for step in range(5):
    # payload columns: vx, vy, vz, id
    pay = np.column_stack([vel, ids]) if pos.shape[0] else np.zeros((0, 4))

    # 1) reassign ownership
    pos, pay = mig.migrate(pos, pay)
    vel, ids = pay[:, 0:3].copy(), pay[:, 3].copy()
    n_owned = pos.shape[0]

    # 2) gather ghosts (copies within rcut of this block; owned by other ranks)
    gpos, gpay = mig.gather_ghosts(pos, pay, rcut)
    n_ghost = gpos.shape[0]

    # 3) local Simulation over owned + ghost; ghosts appended after owned
    combined_pos = np.vstack([pos, gpos]) if n_ghost else pos
    combined_vel = np.vstack([vel, gpay[:, 0:3]]) if n_ghost else vel
    sim = build_sim(combined_pos.shape[0])
    sim.set_positions(combined_pos.astype(np.float32))
    sim.set_velocities(combined_vel.astype(np.float32))

    # 4) step  (NOTE: ghosts are currently integrated too — see module docstring / README)
    sim.step(0.002)

    # 5) keep only owned particles' updated state
    out_pos = np.array(sim.get_positions(False))[:n_owned].astype(np.float64)
    out_vel = np.array(sim.get_velocities())[:n_owned].astype(np.float64)
    pos, vel = out_pos, out_vel

    n_global = comm.allreduce(n_owned, MPI.SUM)
    g_ghost = comm.allreduce(n_ghost, MPI.SUM)
    if rank == 0:
        print(f"step {step}: owned(total)={n_global}/{N}  ghosts(total)={g_ghost}  "
              f"finite={bool(np.isfinite(out_pos).all())}")

n_global = comm.allreduce(pos.shape[0], MPI.SUM)
if rank == 0:
    ok = (n_global == N)
    print(f"{'OK' if ok else 'FAIL'} (np={size}): orchestration ran end-to-end, "
          f"{n_global}/{N} particles conserved (plumbing; ghost-XPBD physics pending)")
    sys.exit(0 if ok else 1)
