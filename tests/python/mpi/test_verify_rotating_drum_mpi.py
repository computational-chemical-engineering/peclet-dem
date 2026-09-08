"""The moving-wall rotating drum under the distributed (MPI) step.

The drum barrel + end caps are a static SDF wall replicated on every rank; a rigid-body wall
velocity spins it. Wall friction (a boundary contact touches only the owned grain, so it is exact
across ranks) drags the bed up the rising side under step_mpi exactly as single-rank. We fill the
bed once (deterministic, broadcast to all ranks), distribute grains by ORB ownership, spin under
step_mpi with periodic load rebalancing, then gather and check confinement + the bed tilt.

A CLOSED drum (end caps, non-periodic) is used so the domain is decomposed across ranks with
cross-rank ghosts only -- the physics matches single-rank. (A z-periodic barrel relies on
undecomposed-axis self-ghosts, whose interaction with a fast moving wall is a separate MPI
limitation; use a closed or axis-decomposed drum under MPI.)

Run:  PYTHONPATH=<dem-build>:<core-python-build> mpirun -np 2 python tests/python/mpi/test_verify_rotating_drum_mpi.py
      (also -np 1 -- the SAME step_mpi code on one block == the native run)
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
    from peclet.dem import build_wall_sdf

cx, cy, R, Lz = 20.0, 20.0, 15.0, 8.0
lo, hi = (0.0, 0.0, 0.0), (40.0, 40.0, Lz)
G, omega, dt = 20.0, 0.6, 0.004
gsize, periodic, rcut = (16, 16, 4), (False, False, False), 3.0


def drum_sdf(p, rad=2.5):
    # closed drum with ROUNDED interior edges (no sharp barrel/cap corner where a
    # trilinear-smoothed SDF corner would leak grains)
    dr = np.maximum(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - (R - rad), 0.0)
    dz = np.maximum(np.abs(p[:, 2] - Lz / 2) - (Lz / 2 - rad), 0.0)
    return rad - np.hypot(dr, dz)  # + void, - wall


def configure(sim):
    sim.initialize_shape('sphere', 1.0)
    sim.set_domain(lo, hi)
    sim.set_periodic(*periodic)
    wid = build_wall_sdf(drum_sdf, (lo, hi), resolution=96).add_to(sim, restitution=0.1, friction=0.6)
    sim.set_gravity((0.0, -G, 0.0))
    sim.set_material_params(0.1, 0.0, 0.4)
    sim.set_solver_iterations(24, 6)
    sim.set_dt(dt)
    return wid


def bed_tilt(pos):
    return np.degrees(np.arctan2(pos[:, 0].mean() - cx, -(pos[:, 1].mean() - cy)))


@skip_unless_mpi
def test_rotating_drum_mpi():
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    # --- fill once on rank 0 (single-rank step), broadcast the packed state ---
    packed = None
    if rank == 0:
        sim = dem.Simulation(400)
        configure(sim)
        g = np.arange(cx - R + 1.5, cx + R - 1.5, 2.4)
        zs = np.arange(lo[2] + 0.6, hi[2] - 0.6, 2.2)
        pts = np.array([(x, y, z) for z in zs for x in g for y in g
                        if (x - cx)**2 + (y - cy)**2 < (R - 2.0)**2 and y < cy - R + 0.33 * 2 * R],
                       np.float32)
        p = np.zeros((len(pts), 4), np.float32)
        p[:, :3] = pts
        p[:, 3] = 1.0
        sim.set_positions(p)
        sim.set_scales(np.ones(len(pts), np.float32))
        sim.set_growth_params(1.0, 0.15)
        for _ in range(1500):
            grow = sim.max_overlap < 0.06 and float(sim.get_scales().mean()) < 0.999
            sim.set_growth_params(1.0 if grow else 0.0, sim.growth_factor)
            sim.step()
        sim.set_growth_params(0.0, sim.growth_factor)
        for _ in range(1000):
            sim.step()
        packed = sim.get_positions().reshape(-1, 3).astype(np.float64)
    packed = comm.bcast(packed, root=0)

    # --- distribute grains by ORB ownership, spin under step_mpi ---
    extent = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
    mig = core_mpi.ParticleMigrator(origin=lo, extent=extent, cells=gsize, periodic=periodic)
    mine = np.where(np.array([mig.owner_of(tuple(x)) for x in packed]) == rank)[0]
    p = np.zeros((len(mine), 4), np.float32)
    p[:, :3] = packed[mine]
    p[:, 3] = 1.0
    sim = dem.Simulation(max(400, 4 * len(mine)))
    wid = configure(sim)
    sim.set_positions(p)
    sim.init_mpi(origin=lo, extent=extent, cells=gsize, periodic=periodic)
    sim.enable_mpi_step(rcut, sync_every=1, forward_rotation=True, rebalance_every=15)
    sim.set_wall_velocity(wid, lin_vel=(0, 0, 0), ang_vel=(0, 0, omega), center=(cx, cy, 0))
    for _ in range(90):
        sim.step_mpi(29)

    mypos = sim.get_positions().reshape(-1, 3)[:sim.num_particles].astype(np.float64)
    allpos = comm.gather(mypos, root=0)
    if rank == 0:
        P = np.concatenate(allpos, axis=0)
        assert len(P) == len(packed), "grains lost or duplicated across the ranks"
        esc = int((np.hypot(P[:, 0] - cx, P[:, 1] - cy) > R + 1.0).sum())
        tilt = bed_tilt(P)
        print(f"np={size}: grains={len(P)}  escaped={esc}  bed tilt={tilt:+.1f} deg  "
              f"[wall friction drags the bed up the rising side]")
        assert esc <= 1, "grains escaped the drum"
        assert tilt > 8.0, "bed not dragged up the rising side"


if __name__ == "__main__":
    script_main(test_rotating_drum_mpi)
