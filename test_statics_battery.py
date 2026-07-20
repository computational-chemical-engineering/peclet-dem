#!/usr/bin/env python
"""Gravity-statics battery (recreation of the solver session's probes, scaled).

A. COLUMN: a deep column (~96k grains, 60 layers) dropped as a slightly-loose
   lattice must ground and come to rest: mean |vz| drains to ~0, the median
   nearest-neighbour spacing stays ~1.0 d_p (no crush), and the bed height
   matches the lattice (no collective sink through the floor).

B. POUR-COLLAPSE: the same inventory poured violently (released from height,
   impact speed ~ sqrt(2 g H) ~ 20 d_p/s) must relax to a physical bed:
   z95 ~ lattice-bed height (the old count-averaged solver jammed at ~55-65%
   of it with deep overlaps), max pair overlap -> ~0.

Run with the default solver and with PECLET_DEM_SYMMETRIC_PGS=1 to compare the
one-sided branch against pure symmetric warm-started PGS + cone friction.
"""
import argparse
import os
import time

import numpy as np

from peclet import dem
from peclet.dem import build_wall_sdf

R = 0.5
D = 1.0
NX, NY = 40, 40
LAYERS = 60


def make_sim(n, lz=140.0):
    s = dem.Simulation(n)
    s.set_sphere_shape(R)
    lo, hi = (0.0, 0.0, -1.0), (NX * 1.05 + 1.0, NY * 1.05 + 1.0, lz)
    s.set_domain(lo, hi)
    s.enable_periodicity(False, False, False)
    wall = build_wall_sdf(
        lambda p: np.minimum.reduce([p[:, 2], p[:, 0] - lo[0], hi[0] - p[:, 0],
                                     p[:, 1] - lo[1], hi[1] - p[:, 1]]),
        (lo, hi), resolution=(96, 96, 160))
    wall.add_to(s, restitution=0.2, friction=0.3)
    s.set_gravity(0.0, 0.0, -10.0)
    s.set_material_params(0.5, 0.0, 0.3)
    s.set_thermostat(0, 0)
    s.set_solver_iterations(12, 8)
    return s


def lattice(z0, spacing=1.05):
    pts = [(0.55 + spacing * i, 0.55 + spacing * j, z0 + spacing * k)
           for k in range(LAYERS) for j in range(NY) for i in range(NX)]
    return np.array(pts, np.float32)


def metrics(s, n):
    p = s.get_positions()
    v = s.get_velocities()
    z = p[:, 2]
    z95 = float(np.quantile(z, 0.95))
    vz = float(np.abs(v[:, 2]).mean())
    ov = float(s.get_max_overlap())
    # nn spacing on a subsample
    idx = np.random.default_rng(0).choice(n, size=min(4000, n), replace=False)
    from scipy.spatial import cKDTree
    t = cKDTree(p)
    dd, _ = t.query(p[idx], k=2)
    nn = float(np.median(dd[:, 1]))
    return z95, vz, nn, ov


def run(mode, steps=4000, dt=0.01):
    n = NX * NY * LAYERS
    s = make_sim(n)
    if mode == "column":
        s.set_positions(lattice(0.55))
        s.set_velocities(np.zeros((n, 3), np.float32))
    else:  # pour: released high, hits at ~ sqrt(2*10*20) ~ 20 d/s
        s.set_positions(lattice(20.0))
        v = np.zeros((n, 3), np.float32)
        v[:, 2] = -10.0
        s.set_velocities(v)
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(n, np.float32))
    s.set_inv_inertia(np.full((n, 3), 1.0 / (0.4 * R * R), np.float32))
    t0 = time.perf_counter()
    for i in range(steps):
        s.step(dt)
    wall = time.perf_counter() - t0
    z95, vz, nn, ov = metrics(s, n)
    # expected dense-bed z95: solid volume / (area * phi~0.6) ~ height
    area = (NX * 1.05) * (NY * 1.05)
    h_rcp = n * (4 / 3 * np.pi * R**3) / (area * 0.60)
    print(f"[{mode:7s}] z95 {z95:6.2f} (lattice-bed ~{LAYERS*1.05*0.95:.0f}, "
          f"phi0.6 bed ~{h_rcp:.1f})  mean|vz| {vz:.4f}  nn {nn:.3f} d_p  "
          f"max_overlap {ov:.4f}  wall {wall:.0f}s")
    return z95, vz, nn, ov


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["column", "pour", "both"], default="both")
    args = ap.parse_args()
    tag = "SYM" if os.environ.get("PECLET_DEM_SYMMETRIC_PGS") else "DEFAULT"
    print(f"== statics battery ({tag}) ==")
    if args.mode in ("column", "both"):
        run("column")
    if args.mode in ("pour", "both"):
        run("pour")
