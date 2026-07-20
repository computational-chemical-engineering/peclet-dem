#!/usr/bin/env python
"""Acceptance tests for the friction-cone (sequential tangential impulse) PGS.

1. STICK: a mu=0.9 grain slab on a tan(theta)=0.5 effective incline must NOT slide
   (the count-averaged sweep slid ~34 units); a mu=0.05 slab must slide freely.
2. KINETIC: single sphere, incline steeper than the friction angle -> slides with
   a = g(sin - mu cos); measured deceleration ratio must track mu.
3. ROLL: sliding sphere on a frictional floor converts to rolling, v -> 5/7 v0.
4. Binary normal restitution unchanged (cone must not touch head-on impacts).
"""
import numpy as np
from peclet import dem
from peclet.dem import build_wall_sdf


def slab_slide(mu, steps=1500):
    """Rotation-LOCKED sphere slab on a tan(theta)=0.5 incline: pure sliding, so
    static friction is the only thing that can hold it. (Free-rotating spheres
    ROLL down any incline without rolling resistance -- that is physics, not a
    friction defect, so they make no stick test.)"""
    n_side = 4
    r = 0.5
    pts = [(4 + 1.9 * i, 4 + 1.9 * j, 0.55 + 1.9 * k)
           for i in range(n_side) for j in range(n_side) for k in range(2)]
    pts = np.array(pts, np.float32)
    n = len(pts)
    s = dem.Simulation(n)
    s.set_sphere_shape(r)
    lo, hi = (0, 0, -1.0), (60, 12, 8)
    s.set_domain(lo, hi)
    s.enable_periodicity(False, False, False)
    wall = build_wall_sdf(lambda p: p[:, 2], (lo, hi), resolution=(96, 24, 24))
    wall.add_to(s, restitution=0.0, friction=mu)
    s.set_positions(pts)
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(n, np.float32))
    s.set_inv_inertia(np.zeros((n, 3), np.float32))  # rotation locked: pure slide
    s.set_velocities(np.zeros((n, 3), np.float32))
    s.set_gravity(5.0, 0.0, -10.0)  # tan(theta) = 0.5
    s.set_material_params(0.0, 0.0, mu)
    s.set_thermostat(0, 0)
    s.set_solver_iterations(12, 8)
    x0 = pts[:, 0].mean()
    for _ in range(steps):
        s.step(0.01)
    return float(s.get_positions()[:, 0].mean() - x0)


def single_slide(mu, gx=8.0, steps=800):
    """One sphere with rotation LOCKED (inv_inertia=0): pure sliding, no rolling.
    a = gx - mu*gz -> distance = 0.5*a*t^2."""
    r = 0.5
    s = dem.Simulation(2)
    s.set_sphere_shape(r)
    lo, hi = (0, 0, -1.0), (200, 8, 6)
    s.set_domain(lo, hi)
    s.enable_periodicity(False, False, False)
    wall = build_wall_sdf(lambda p: p[:, 2], (lo, hi), resolution=(128, 16, 16))
    wall.add_to(s, restitution=0.0, friction=mu)
    s.set_positions(np.array([[5, 4, 0.5]], np.float32))
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(1, np.float32))
    s.set_inv_inertia(np.zeros((1, 3), np.float32))  # no rotation: pure slide
    s.set_velocities(np.zeros((1, 3), np.float32))
    s.set_gravity(gx, 0.0, -10.0)
    s.set_material_params(0.0, 0.0, mu)
    s.set_thermostat(0, 0)
    s.set_solver_iterations(12, 8)
    for _ in range(steps):
        s.step(0.01)
    return float(s.get_positions()[0, 0] - 5.0)


def roll_ratio(v0=4.0, mu=0.5, steps=1200):
    """Sphere sliding at v0 on a flat frictional floor, free rotation: friction
    torques it up to rolling; terminal v = 5/7 v0 (classic)."""
    r = 0.5
    s = dem.Simulation(2)
    s.set_sphere_shape(r)
    lo, hi = (0, 0, -1.0), (200, 8, 6)
    s.set_domain(lo, hi)
    s.enable_periodicity(False, False, False)
    wall = build_wall_sdf(lambda p: p[:, 2], (lo, hi), resolution=(128, 16, 16))
    wall.add_to(s, restitution=0.0, friction=mu)
    s.set_positions(np.array([[5, 4, 0.5]], np.float32))
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(1, np.float32))
    s.set_inv_inertia(np.full((1, 3), 1.0 / (0.4 * 1.0 * r * r), np.float32))
    s.set_velocities(np.array([[v0, 0, 0]], np.float32))
    s.set_gravity(0.0, 0.0, -10.0)
    s.set_material_params(0.0, 0.0, mu)
    s.set_thermostat(0, 0)
    s.set_solver_iterations(12, 8)
    for _ in range(steps):
        s.step(0.005)
    v = s.get_velocities()[0]
    w = s.get_angular_velocities()[0]
    # rolling without slipping in +x: contact velocity v_x - w_y*r = 0 -> w_y = +v_x/r
    return float(v[0]) / v0, float(w[1] * r) / v0


def binary_e(e):
    s = dem.Simulation(4)
    s.set_sphere_shape(0.5)
    s.set_domain((0, 0, 0), (20, 20, 20))
    s.enable_periodicity(False, False, False)
    s.set_positions(np.array([[8, 10, 10], [12, 10, 10]], np.float32))
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(2, np.float32))
    s.set_inv_inertia(np.full((2, 3), 1.0, np.float32))
    s.set_velocities(np.array([[1, 0, 0], [-1, 0, 0]], np.float32))
    s.set_gravity(0, 0, -1e-6)  # tiny g so the PGS path is exercised
    s.set_thermostat(0, 0)
    s.set_material_params(e, 0.0, 0.5)
    s.set_solver_iterations(8, 4)
    for _ in range(400):
        s.step(0.01)
    v = s.get_velocities()
    return (v[1, 0] - v[0, 0]) / 2.0


print("== 1. slab stick ==")
d_hi = slab_slide(0.9)
d_lo = slab_slide(0.05)
print(f"   mu=0.9 slide {d_hi:+.2f}  |  mu=0.05 slide {d_lo:+.2f}")
assert abs(d_hi) < 1.0, f"mu=0.9 slab must stick (slid {d_hi})"
assert d_lo > 5.0, "mu=0.05 slab must slide"

print("== 2. kinetic slide (rotation locked, gx=8, gz=10) ==")
d3 = single_slide(0.3)   # a = 8 - 3 = 5
d6 = single_slide(0.6)   # a = 8 - 6 = 2
print(f"   mu=0.3 dist {d3:.1f} (theory ~{0.5*5*8**2:.0f})  "
      f"mu=0.6 dist {d6:.1f} (theory ~{0.5*2*8**2:.0f})")
assert abs(d3 / (0.5 * 5 * 64) - 1) < 0.15
assert abs(d6 / (0.5 * 2 * 64) - 1) < 0.25

print("== 3. slide -> roll 5/7 ==")
rv, rw = roll_ratio()
print(f"   v/v0 = {rv:.3f}  wr/v0 = {rw:.3f}  (theory 5/7 = {5/7:.3f})")
assert abs(rv - 5 / 7) < 0.03 and abs(rw - 5 / 7) < 0.05

print("== 4. binary restitution with friction active ==")
for e in (0.2, 0.8):
    ee = binary_e(e)
    print(f"   e={e}: e_eff={ee:.3f}")
    assert abs(ee - e) < 0.02

print("ALL CONE-FRICTION ACCEPTANCE TESTS PASS")
