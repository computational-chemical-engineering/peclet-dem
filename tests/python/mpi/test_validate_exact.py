"""Validate the EXACT MPI-aware dem step against a serial reference.

Same initial condition is run two ways and compared by particle id:
  * reference : rank 0 runs all N particles in one Simulation with the ordinary serial step().
  * distributed: every rank owns a subset (migrated each step on core's ParticleMigrator) and runs
                 the distributed step_mpi (gather ghosts with real mass + per-iteration
                 owner->ghost forward).
At np=1 the two agree to float noise; at np>=2 the processor-block Gauss-Seidel sweeps the
finite-iteration PGS in a different order, so per-particle agreement on this randomly overlapping
IC is statistical (bounded mean / 95 % quantile; no particle off by a diameter).

Non-periodic domain with a ground plane (gravity settling) so the only ghosts are at the
inter-rank split -- this isolates MPI-ghost correctness from periodic-wrap handling.

Run:  PYTHONPATH=<dem-build>:<core-python-build> mpirun -np 2 python tests/python/mpi/test_validate_exact.py
      (also collected by pytest; skipped unless the MPI Python stack is importable)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mpi_common import SKIP, script_main, skip_unless_mpi  # noqa: E402

if SKIP is None:
    from mpi4py import MPI
    from peclet import dem
    from peclet.core import mpi as core_mpi

# M = sync_every (1 = EXACT), R = forward_rotation (1 = forward ghost quaternions). Defaults = EXACT.
SYNC_EVERY = int(os.environ.get("M", "1"))
FWD_ROT = bool(int(os.environ.get("R", "1")))

dmin = [0.0, 0.0, 0.0]
L = [8.0, 8.0, 8.0]
radius = 0.4
rcut = 2.0 * radius
N = 200
nsteps = 15
dt = 0.002


def make_sim(n):
    s = dem.Simulation(2 * int(n) + 64)   # capacity = owned + ghost slots
    m = rcut + 0.5
    s.set_domain((dmin[0] - m, dmin[1] - m, dmin[2] - m), (L[0] + m, L[1] + m, L[2] + m))
    s.set_periodic(False, False, False)
    s.initialize_shape('sphere', radius=radius)
    s.add_plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))  # ground at z=0
    s.set_solver_iterations(8, 4)
    s.set_gravity((0.0, 0.0, -9.8))
    s.set_dt(dt)
    return s


@skip_unless_mpi
def test_exact_step_matches_serial():
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    rng = np.random.RandomState(1)
    g_pos = rng.uniform(1.0, 7.0, size=(N, 3)).astype(np.float64)
    g_vel = np.zeros((N, 3))

    # --- reference: all N particles, serial step (rank 0 only) ---
    ref_pos = None
    if rank == 0:
        ref = make_sim(N)
        ref.set_positions(g_pos.astype(np.float32))
        ref.set_velocities(g_vel.astype(np.float32))
        ref.set_dt(dt)
        for _ in range(nsteps):
            ref.step()
        ref_pos = np.array(ref.get_positions())

    # --- distributed: round-robin ownership, migrate + distributed step each step ---
    mig = core_mpi.ParticleMigrator(origin=dmin, extent=L, cells=[16, 16, 16],
                                    periodic=[False, False, False])
    mine = np.arange(rank, N, size)
    pos = g_pos[mine].copy()
    vel = g_vel[mine].copy()
    ids = mine.astype(np.float64)
    for _ in range(nsteps):
        pay = np.column_stack([vel, ids]) if pos.shape[0] else np.zeros((0, 4))
        pos, pay = mig.migrate(pos, pay)
        vel, ids = pay[:, 0:3].copy(), pay[:, 3].copy()
        n = pos.shape[0]
        assert n > 0, f"rank {rank} owns 0 particles -- pick N/np so every rank is non-empty"
        s = make_sim(n)
        s.set_positions(pos.astype(np.float32))
        s.set_velocities(vel.astype(np.float32))
        s.init_mpi(origin=tuple(dmin), extent=tuple(L), cells=(16, 16, 16),
                   periodic=(False, False, False))
        s.enable_mpi_step(rcut, sync_every=SYNC_EVERY, forward_rotation=FWD_ROT)
        s.step_mpi(1)
        pos = np.array(s.get_positions())[:n].astype(np.float64)
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
        maxerr, meanerr = float(err.max()), float(err.mean())
        pe = err.max(axis=1)
        q95, nbad = float(np.quantile(pe, 0.95)), int((pe > 2e-2).sum())
        print(f"np={size}: max|dist-serial|={maxerr:.3e}  mean={meanerr:.3e}  q95={q95:.3e}  "
              f"particles>2e-2: {nbad}/{N} over {nsteps} steps, N={N}")
        # np=1 on a deterministic backend (one OpenMP thread; device atomics are not ordered)
        # agrees to float noise (~1e-4). At np>=2 the modern MPI stack (processor-block
        # Gauss-Seidel, rank-local colouring) sweeps the finite-iteration PGS in a different order
        # than single-rank -- and on this randomly overlapping IC any ordering change is amplified
        # chaotically (2 OpenMP threads alone give max 0.08 at np=1) -- so agreement is statistical:
        # measured 2026-09-08 mean 5e-3, q95 4e-2, max 0.11 at np=2,4.
        deterministic = (os.environ.get("OMP_NUM_THREADS") == "1"
                         and not any(k in dem.execution_space for k in ("Cuda", "HIP")))
        if size == 1 and deterministic:
            assert maxerr < 1e-3
        assert meanerr < 2e-2 and q95 < 1e-1
        assert maxerr < 0.4, "a particle is off by ~a diameter: a lost ghost contact"


if __name__ == "__main__":
    script_main(test_exact_step_matches_serial)
