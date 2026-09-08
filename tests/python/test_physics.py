"""Overlap removal + stable growth packing (the XPBD position solve end to end).

1. Two overlapping spheres are pushed apart to contact in one relaxation step.
2. Particles grown from 10 % to full size on a cubic grid at phi = 0.40 and 0.50 (a simple cubic
   lattice holds at most phi = 0.524) end with no pair overlap deeper than 0.1 r and stay inside
   the box (no explosion).
"""
import math

import numpy as np
import pytest

from peclet import dem


def test_two_particles_pushed_apart():
    sim = dem.Simulation(2)
    sim.set_domain(np.array([-10, -10, -10], dtype=np.float32), np.array([10, 10, 10], dtype=np.float32))
    sim.initialize_shape(0, radius=0.5)
    sim.set_solver_iterations(10, 0)
    pos = np.array([[0.0, 0.0, 0.0, 1.0], [0.7, 0.0, 0.0, 1.0]], dtype=np.float32)  # overlap 0.3 (r=0.5); w = inverse mass
    sim.set_positions(pos)
    sim.set_velocities(np.zeros_like(pos))
    sim.set_scales(np.ones(2, dtype=np.float32))
    sim.set_global_scale(1.0)
    sim.step(0.01)
    p = sim.get_positions()
    d = float(np.linalg.norm(p[0][:3] - p[1][:3]))
    print(f"final distance {d:.4f}")
    assert d > 0.95, "still overlapping (contact distance 1.0)"


def _max_pair_overlap(pos, diameter):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.maximum(diameter - d, 0.0).max())


@pytest.mark.parametrize("num_particles,density_target", [(125, 0.40), (216, 0.50)])
def test_growth_packing(num_particles, density_target, steps=500):
    r = 1.0
    vol_box = num_particles * (4.0 / 3.0 * math.pi * r**3) / density_target
    L = vol_box ** (1.0 / 3.0)
    print(f"phi={density_target} N={num_particles} L={L:.3f}")
    sim = dem.Simulation(num_particles)
    sim.set_domain(np.array([-L / 2] * 3, dtype=np.float32), np.array([L / 2] * 3, dtype=np.float32))
    sim.initialize_shape(0, radius=0.5)
    sim.set_solver_iterations(10, 4)
    k = int(np.ceil(num_particles ** (1 / 3)))
    spacing = L / k
    offset = -L / 2 + spacing / 2
    grid = [[offset + x * spacing, offset + y * spacing, offset + z * spacing, 1.0]
            for x in range(k) for y in range(k) for z in range(k)][:num_particles]
    sim.set_positions(np.array(grid, dtype=np.float32))
    sim.set_velocities(np.zeros((num_particles, 4), dtype=np.float32))
    sim.set_global_scale(0.1)
    for i in range(steps):
        sim.set_global_scale(min(1.0, 0.1 + 0.9 * i / (steps * 0.8)))   # linear growth to full size
        sim.step(0.01)
    pos = sim.get_positions()[:, :3]
    max_overlap = _max_pair_overlap(pos, 2.0 * r)
    max_dist = float(np.max(np.abs(pos)))
    print(f"max overlap {max_overlap:.4f}  max |x| {max_dist:.3f} (L/2 = {L / 2:.3f})")
    assert np.isfinite(pos).all()
    assert max_dist <= 1.5 * L / 2, "explosion: particles flew out of the domain"
    assert max_overlap <= 0.1, "high overlap after growth"
