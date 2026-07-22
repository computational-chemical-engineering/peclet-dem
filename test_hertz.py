"""Acceptance tests for the soft-sphere Hertz-Mindlin engine (step_hertz).

Same physics battery as the impulse solver's cone tests, run through the explicit
force model: binary restitution (damping formula), wall bounce, static stick on an
incline (rotation locked), kinetic Coulomb slide distances, and slide -> roll 5/7.
Soft stiffness (E = 1e7) keeps the Rayleigh time step at 1e-4 for fast tests.
"""
import numpy as np
from peclet import dem
from peclet.dem import build_wall_sdf

E0, NU0 = 1.0e7, 0.25
DT = 5e-5


def base(n, lo, hi, e, mu, wall_f=None, wall_res=64):
    s = dem.Simulation(n)
    s.set_sphere_shape(0.5)
    s.set_domain(lo, hi)
    s.enable_periodicity(False, False, False)
    s.set_material_params(e, 0.0, mu)
    s.set_hertz_material(0, E0, NU0)
    s.set_thermostat(0, 0)
    return s


def binary(e):
    s = base(2, (0, 0, 0), (20, 20, 20), e, 0.0)
    s.set_positions(np.array([[8, 10, 10], [12, 10, 10]], np.float32))
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(2, np.float32))
    s.set_inv_inertia(np.full((2, 3), 1.0, np.float32))
    s.set_velocities(np.array([[1, 0, 0], [-1, 0, 0]], np.float32))
    s.set_gravity(0, 0, 0)
    s.step_hertz(DT, 60000)
    v = s.get_velocities()
    return (v[1, 0] - v[0, 0]) / 2.0


def slab_slide(mu, lock_rotation=True, steps=200000):
    n_side = 4
    pts = [(4 + 1.9 * i, 4 + 1.9 * j, 0.55 + 1.9 * k)
           for i in range(n_side) for j in range(n_side) for k in range(2)]
    pts = np.array(pts, np.float32)
    n = len(pts)
    s = base(n, (0, 0, -1.0), (60, 12, 8), 0.2, mu)
    wall = build_wall_sdf(lambda p: p[:, 2], ((0, 0, -1.0), (60, 12, 8)), resolution=(96, 24, 24))
    wall.add_to(s, restitution=0.2, friction=mu)
    s.set_positions(pts)
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(n, np.float32))
    inv_i = 0.0 if lock_rotation else 1.0 / (0.4 * 0.25)
    s.set_inv_inertia(np.full((n, 3), inv_i, np.float32))
    s.set_velocities(np.zeros((n, 3), np.float32))
    s.set_gravity(5.0, 0.0, -10.0)
    x0 = pts[:, 0].mean()
    s.step_hertz(DT, steps)
    return float(s.get_positions()[:, 0].mean() - x0)


def single_slide(mu, gx=8.0, steps=160000):
    s = base(1, (0, 0, -1.0), (200, 8, 6), 0.0, mu)
    wall = build_wall_sdf(lambda p: p[:, 2], ((0, 0, -1.0), (200, 8, 6)), resolution=(128, 16, 16))
    wall.add_to(s, restitution=0.0, friction=mu)
    s.set_positions(np.array([[5, 4, 0.5]], np.float32))
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(1, np.float32))
    s.set_inv_inertia(np.zeros((1, 3), np.float32))
    s.set_velocities(np.zeros((1, 3), np.float32))
    s.set_gravity(gx, 0.0, -10.0)
    s.step_hertz(DT, steps)
    return float(s.get_positions()[0, 0] - 5.0)


def roll_ratio(v0=4.0, mu=0.5, steps=120000):
    s = base(1, (0, 0, -1.0), (200, 8, 6), 0.0, mu)
    wall = build_wall_sdf(lambda p: p[:, 2], ((0, 0, -1.0), (200, 8, 6)), resolution=(128, 16, 16))
    wall.add_to(s, restitution=0.0, friction=mu)
    s.set_positions(np.array([[5, 4, 0.5]], np.float32))
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(1, np.float32))
    s.set_inv_inertia(np.full((1, 3), 1.0 / (0.4 * 0.25), np.float32))
    s.set_velocities(np.array([[v0, 0, 0]], np.float32))
    s.set_gravity(0.0, 0.0, -10.0)
    s.step_hertz(DT, steps)
    v = s.get_velocities()[0]
    w = s.get_angular_velocities()[0]
    return float(v[0]) / v0, float(w[1] * 0.5) / v0


print("== hertz 1. binary restitution (damping formula) ==")
for e in (0.2, 0.5, 0.8):
    ee = binary(e)
    print(f"   e={e}: e_eff={ee:.3f}")
    assert abs(ee - e) < 0.04

print("== hertz 2. slab stick (rotation locked, tan=0.5) ==")
d_hi = slab_slide(0.9)
d_lo = slab_slide(0.05)
print(f"   mu=0.9 slide {d_hi:+.2f}  |  mu=0.05 slide {d_lo:+.2f}")
assert abs(d_hi) < 1.0 and d_lo > 5.0

print("== hertz 3. kinetic slide (gx=8, gz=10) ==")
d3 = single_slide(0.3)
d6 = single_slide(0.6)
t = 160000 * DT
print(f"   mu=0.3 dist {d3:.1f} (theory {0.5*5*t*t:.1f})  mu=0.6 dist {d6:.1f} "
      f"(theory {0.5*2*t*t:.1f})")
assert abs(d3 / (0.5 * 5 * t * t) - 1) < 0.1
assert abs(d6 / (0.5 * 2 * t * t) - 1) < 0.15

print("== hertz 4. slide -> roll 5/7 ==")
rv, rw = roll_ratio()
print(f"   v/v0 = {rv:.3f}  wr/v0 = {rw:.3f}  (theory {5/7:.3f})")
assert abs(rv - 5 / 7) < 0.03 and abs(rw - 5 / 7) < 0.05

print("ALL HERTZ ACCEPTANCE TESTS PASS")
