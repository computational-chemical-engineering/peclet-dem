"""S6 byte-identity scenes (docs/contact_solve_framework.md §12 S6): spheres with 2-3 simultaneous
wall contacts -- a box corner (6 analytic planes) and a rotating drum (an SDF barrel + two analytic
end-cap planes). Single-rank sim.step(); dumps the final state to <out>/<case>.npz. Compare two
builds with np.array_equal (the --compare mode).

  OMP_NUM_THREADS=1 PYTHONPATH=<build> python s6_wall_dumps.py dump <outdir>
  python s6_wall_dumps.py compare <dirA> <dirB>
"""
import os
import sys

import numpy as np


def corner(dem, g0=False, incremental=True):
    sim = dem.Simulation(512)
    sim.initialize_shape("sphere", 0.5)
    sim.set_domain((0.0, 0.0, 0.0), (8.0, 8.0, 8.0))
    for p, n in [((0, 0, 0), (1, 0, 0)), ((8, 0, 0), (-1, 0, 0)), ((0, 0, 0), (0, 1, 0)),
                 ((0, 8, 0), (0, -1, 0)), ((0, 0, 0), (0, 0, 1)), ((0, 0, 8), (0, 0, -1))]:
        sim.add_plane(p, n)
    rng = np.random.default_rng(5)
    g = np.arange(0.6, 4.0, 1.05)
    pts = np.array([(x, y, z) for x in g for y in g for z in g], np.float32)
    pts += rng.uniform(-0.03, 0.03, pts.shape).astype(np.float32)
    p = np.zeros((len(pts), 4), np.float32)
    p[:, :3] = pts
    p[:, 3] = 1.0
    sim.set_positions(p)
    v = rng.normal(0.0, 0.5, (len(pts), 3)).astype(np.float32) - 1.0  # toward the corner
    sim.set_velocities(v)
    sim.set_material_params(0.5, 0.0, 0.3)
    sim.set_solver_iterations(20, 8)
    if not g0:
        sim.set_gravity((-3.0, -3.0, -9.8))
    sim.set_incremental_coloring(incremental)
    sim.set_dt(5e-3)
    return sim


def drum(dem, incremental=True):
    from peclet.dem import build_wall_sdf
    cx, cy, R, Lz = 6.0, 6.0, 5.0, 4.0
    lo, hi = (0.0, 0.0, 0.0), (12.0, 12.0, Lz)
    sim = dem.Simulation(512)
    sim.initialize_shape("sphere", 0.5)
    sim.set_domain(lo, hi)
    barrel = lambda q: R - np.hypot(q[:, 0] - cx, q[:, 1] - cy)  # + void, - wall
    wid = build_wall_sdf(barrel, (lo, hi), resolution=64).add_to(sim, restitution=0.2, friction=0.6)
    sim.add_plane((0, 0, 0), (0, 0, 1))
    sim.add_plane((0, 0, Lz), (0, 0, -1))
    g = np.arange(cx - R + 0.6, cx + R - 0.6, 1.02)
    zs = np.arange(0.55, Lz - 0.5, 1.02)
    pts = np.array([(x, y, z) for z in zs for x in g for y in g
                    if np.hypot(x - cx, y - cy) < R - 0.55 and y < cy], np.float32)
    p = np.zeros((len(pts), 4), np.float32)
    p[:, :3] = pts
    p[:, 3] = 1.0
    sim.set_positions(p)
    sim.set_gravity((0.0, -9.8, 0.0))
    sim.set_material_params(0.3, 0.0, 0.4)
    sim.set_solver_iterations(20, 8)
    sim.set_wall_velocity(wid, lin_vel=(0, 0, 0), ang_vel=(0, 0, 0.8), center=(cx, cy, 0))
    sim.set_incremental_coloring(incremental)
    sim.set_dt(4e-3)
    return sim


CASES = {
    "corner_pgs": lambda d: corner(d),
    "corner_pgs_full": lambda d: corner(d, incremental=False),
    "corner_g0": lambda d: corner(d, g0=True),
    "drum": lambda d: drum(d),
    "drum_full": lambda d: drum(d, incremental=False),
}


def main():
    if sys.argv[1] == "dump":
        from peclet import dem
        out = sys.argv[2]
        os.makedirs(out, exist_ok=True)
        for name, make in CASES.items():
            sim = make(dem)
            maxnc = 0
            for _ in range(150):
                sim.step()
                maxnc = max(maxnc, sim.num_contacts)
            np.savez(os.path.join(out, name + ".npz"), x=sim.get_positions(), v=sim.get_velocities(),
                     w=sim.get_angular_velocities(), q=sim.get_quaternions())
            print(f"{name}: N={sim.num_particles} max contacts={maxnc} "
                  f"conflicts={sim.diagnostics.coloring_conflicts()}")
    else:
        a, b = sys.argv[2], sys.argv[3]
        for name in CASES:
            A, B = np.load(os.path.join(a, name + ".npz")), np.load(os.path.join(b, name + ".npz"))
            same = all(np.array_equal(A[k], B[k]) for k in A.files)
            print(f"{'IDENT' if same else 'DIFF'} {name}")


if __name__ == "__main__":
    main()
