"""Microbenchmark for the MPI-aware dem step. Times steady-state ms/step at the current rank count
(rebuild-free: one Simulation, fixed ownership, no migration -- isolates the step cost).

Env knobs:  PI/VI = position/velocity iterations; M = sync_every (1=EXACT); R = forward_rotation (0/1).

Run:  PYTHONPATH=build_sm120:../core/python/build mpirun -np 2 python3 mpi/bench_step.py
      M=4 R=0 ... mpirun ...   # spheres, refresh ghosts every 4 iters
"""
import os
import time
import numpy as np
from mpi4py import MPI
from peclet from peclet import dem

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

dmin = [0.0, 0.0, 0.0]
L = [8.0, 8.0, 8.0]
radius = 0.4
rcut = 2.0 * radius
N = 4000
warmup, nsteps, dt = 5, 60, 0.002

rng = np.random.RandomState(rank + 1)
mine = np.arange(rank, N, size)
pos = rng.uniform(0.5, 7.5, size=(mine.size, 3)).astype(np.float32)
vel = np.zeros((mine.size, 3), dtype=np.float32)

m = rcut + 0.5
s = dem.Simulation(num_particles=int(mine.size))
s.set_domain((dmin[0] - m, dmin[1] - m, dmin[2] - m), (L[0] + m, L[1] + m, L[2] + m))
s.enable_periodicity(False, False, False)
s.initialize(shape_type=0, radius=radius)
s.add_plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
s.set_solver_iterations(int(os.environ.get('PI', '8')), int(os.environ.get('VI', '4')))
s.set_gravity(0.0, 0.0, -9.8)
s.set_positions(pos)
s.set_velocities(vel)
s.mpi_init(origin=tuple(dmin), size=tuple(L), gsize=(16, 16, 16), periodic=(False, False, False))
s.enable_mpi_step(rcut, sync_every=int(os.environ.get('M','1')), forward_rotation=bool(int(os.environ.get('R','1'))))

for _ in range(warmup):
    s.step(dt)
comm.Barrier()
t0 = time.perf_counter()
for _ in range(nsteps):
    s.step(dt)
comm.Barrier()
dt_ms = (time.perf_counter() - t0) / nsteps * 1e3
ng = s.num_particles(True) - s.num_particles(False)
ng_tot = comm.reduce(ng, op=MPI.SUM, root=0)
if rank == 0:
    print(f"np={size} N={N}: {dt_ms:.3f} ms/step  (ghosts/step total ~{ng_tot})")
