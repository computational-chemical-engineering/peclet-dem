"""Validation for per-pair materials + the PGS friction bound.

1. Binary impact: pair restitution overrides the global material exactly (and the
   global stays in effect for untabled pairs).
2. Friction through the pair table: a rotation-locked slab on a frictional SDF floor under
   tilted gravity sticks at mu=0.9 and slides at mu=0.05 — the bound-from-lambdaAcc fix
   makes friction ACT on warm contacts (it was measured inert before).
"""
import numpy as np
from peclet import dem


def binary(e_global, use_pair, pair_e=0.7):
    s = dem.Simulation(4)
    s.initialize_shape('sphere', 0.5)
    s.set_domain((0, 0, 0), (20, 20, 20))
    s.set_periodic(False, False, False)
    s.set_positions(np.array([[8, 10, 10], [12, 10, 10]], np.float32))
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(2, np.float32))
    s.set_inv_inertia(np.full((2, 3), 1.0, np.float32))
    s.set_velocities(np.array([[1, 0, 0], [-1, 0, 0]], np.float32))
    s.set_gravity((0, 0, 0))
    s.set_thermostat(0, 0)
    s.set_material_params(e_global, 0.0, 0.0)
    if use_pair:
        s.set_material_ids([0, 1])
        s.set_pair_material(0, 1, pair_e, 0.0)
    s.set_solver_iterations(8, 4)
    s.set_dt(0.01)
    for _ in range(400):
        s.step()
    v = s.get_velocities()
    return (v[1, 0] - v[0, 0]) / 2.0


def slab_slide(mu, steps=1500):
    """A small block of rotation-LOCKED grains on a tilted-gravity plane (tan(theta)=0.5):
    with mu well above 0.5 it must stick; with mu ~0 it slides far. Uses an SDF floor so the
    wall material is exercised through the pair table. (Free-rotating spheres would ROLL down
    the incline at any mu -- measured mu-independent -- which is physics, not friction.)"""
    from peclet.dem import build_wall_sdf

    n_side = 4
    r = 0.5
    pts = [(4 + 1.9 * i, 4 + 1.9 * j, 0.55 + 1.9 * k)
           for i in range(n_side) for j in range(n_side) for k in range(2)]
    pts = np.array(pts, np.float32)
    n = len(pts)
    s = dem.Simulation(n)
    s.initialize_shape('sphere', r)
    lo, hi = (0, 0, -1.0), (40, 12, 8)
    s.set_domain(lo, hi)
    s.set_periodic(False, False, False)
    wall = build_wall_sdf(lambda p: p[:, 2], (lo, hi), resolution=(64, 24, 24))
    wid = wall.add_to(s, restitution=0.0, friction=mu)
    s.set_positions(pts)
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(n, np.float32))
    s.set_inv_inertia(np.zeros((n, 3), np.float32))  # rotation locked: pure slide
    s.set_velocities(np.zeros((n, 3), np.float32))
    s.set_gravity((5.0, 0.0, -10.0))  # tan(theta) = 0.5
    s.set_material_params(0.0, 0.0, mu)
    s.set_thermostat(0, 0)
    s.set_solver_iterations(12, 8)
    x0 = pts[:, 0].mean()
    s.set_dt(0.01)
    for _ in range(steps):
        s.step()
    return float(s.get_positions()[:, 0].mean() - x0)


def test_pair_restitution_overrides_global():
    e1 = binary(0.2, False)
    e2 = binary(0.2, True)
    e3 = binary(0.9, True)
    print(f"binary e: global-only {e1:.3f} (want 0.2), pair {e2:.3f} (want 0.7), "
          f"pair-over-0.9 {e3:.3f} (want 0.7)")
    assert abs(e1 - 0.2) < 0.02 and abs(e2 - 0.7) < 0.02 and abs(e3 - 0.7) < 0.02


def test_wall_friction_acts_through_pair_table():
    """The wall's pair-table friction must ACT on the warm contacts of a settled slab: mu=0.9
    holds it on the tan(theta)=0.5 incline, mu=0.05 lets it slide (the bound-from-lambdaAcc
    fix; friction was measured inert on warm contacts before it)."""
    d_lo = slab_slide(0.05)
    d_hi = slab_slide(0.9)
    print(f"slab slide over 15 s: mu=0.05 -> {d_lo:.2f}, mu=0.9 -> {d_hi:.2f}")
    assert d_lo > 5.0, "a mu=0.05 slab must slide"
    assert abs(d_hi) < 0.1 * d_lo, "a mu=0.9 slab must stick"
