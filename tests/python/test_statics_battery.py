"""Gravity-statics battery (the solver session's probes, scaled to a CI-sized bed).

A. COLUMN: a column of grains dropped as a slightly-loose lattice must ground and come to rest:
   mean |vz| drains to ~0, the median nearest-neighbour spacing stays ~1.0 d_p (no crush), and
   the bed height matches the settled lattice (no collective sink through the floor).

B. POUR-COLLAPSE: the same inventory poured violently (released from height, impact speed
   ~ sqrt(2 g H) ~ 20 d_p/s) must relax to a physical bed: z95 ~ the lattice-bed height (the old
   count-averaged solver jammed at ~55-65 % of it with deep overlaps), max pair overlap -> ~0.

The original probes ran 96k grains (40 x 40 x 60 layers) for 4000 steps; the pytest bed is
12 x 12 x 12 layers (1728 grains), which keeps every observable the asserts check (the OMP host
runs it in well under a minute).
"""
import time

import numpy as np
import pytest
from scipy.spatial import cKDTree

from peclet import dem
from peclet.dem import build_wall_sdf

R = 0.5
NX, NY = 12, 12
LAYERS = 12
SPACING = 1.05


def make_sim(n, lz=60.0):
    s = dem.Simulation(n)
    s.set_sphere_shape(R)
    lo, hi = (0.0, 0.0, -1.0), (NX * SPACING + 1.0, NY * SPACING + 1.0, lz)
    s.set_domain(lo, hi)
    s.set_periodic(False, False, False)
    wall = build_wall_sdf(
        lambda p: np.minimum.reduce([p[:, 2], p[:, 0] - lo[0], hi[0] - p[:, 0],
                                     p[:, 1] - lo[1], hi[1] - p[:, 1]]),
        (lo, hi), resolution=(48, 48, 96))
    wall.add_to(s, restitution=0.2, friction=0.3)
    s.set_gravity(0.0, 0.0, -10.0)
    s.set_material_params(0.5, 0.0, 0.3)
    s.set_thermostat(0, 0)
    s.set_solver_iterations(12, 8)
    return s


def lattice(z0, spacing=SPACING):
    pts = [(0.55 + spacing * i, 0.55 + spacing * j, z0 + spacing * k)
           for k in range(LAYERS) for j in range(NY) for i in range(NX)]
    return np.array(pts, np.float32)


def metrics(s, n):
    p = s.get_positions()
    v = s.get_velocities()
    z95 = float(np.quantile(p[:, 2], 0.95))
    vz = float(np.abs(v[:, 2]).mean())
    ov = float(s.max_overlap())
    idx = np.random.default_rng(0).choice(n, size=min(4000, n), replace=False)
    dd, _ = cKDTree(p).query(p[idx], k=2)
    nn = float(np.median(dd[:, 1]))
    return z95, vz, nn, ov, float(p[:, 2].min())


def run(mode, steps, dt=0.01):
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
    for _ in range(steps):
        s.step(dt)
    wall = time.perf_counter() - t0
    z95, vz, nn, ov, zmin = metrics(s, n)
    area = (NX * SPACING) * (NY * SPACING)
    h_rcp = n * (4 / 3 * np.pi * R**3) / (area * 0.60)
    print(f"[{mode:7s}] z95 {z95:6.2f} (lattice-bed ~{LAYERS * SPACING * 0.95:.1f}, phi0.6 bed ~{h_rcp:.1f})  "
          f"mean|vz| {vz:.4f}  nn {nn:.3f} d_p  max_overlap {ov:.4f}  zmin {zmin:.3f}  wall {wall:.0f}s")
    return z95, vz, nn, ov, zmin


# Bed-height window for z95 (the 95 % quantile of grain centres): a dense bed between the crystal
# limit (phi = 0.74, the collapsed cubic lattice crystallises) and the loose lattice it started as.
_N = NX * NY * LAYERS
_AREA = (NX * SPACING) * (NY * SPACING)
_H_CRYSTAL = _N * (4 / 3 * np.pi * R**3) / (_AREA * 0.74)
_H_LATTICE = LAYERS * SPACING
Z95_LO, Z95_HI = 0.9 * _H_CRYSTAL - R, _H_LATTICE


def test_column_comes_to_rest():
    z95, vz, nn, ov, zmin = run("column", steps=800)
    assert vz < 0.02, "column still moving"
    assert 0.95 < nn < 1.10, "lattice crushed or blown apart"
    assert Z95_LO < z95 < Z95_HI, f"bed height {z95:.2f} outside ({Z95_LO:.1f}, {Z95_HI:.1f}): sink or fluff"
    assert zmin > 0.3, "grains sank through the floor"
    assert ov < 0.05, "deep overlaps in the resting bed"


def test_pour_collapse_relaxes():
    z95, vz, nn, ov, zmin = run("pour", steps=1500)
    assert vz < 0.05, "poured bed still moving"
    assert 0.9 < nn < 1.10, "grains crushed or blown apart"
    assert Z95_LO < z95 < Z95_HI, f"poured bed {z95:.2f} outside ({Z95_LO:.1f}, {Z95_HI:.1f}): jammed short / too tall"
    assert zmin > 0.3, "grains sank through the floor"
    assert ov < 0.05, "deep overlaps in the poured bed"
