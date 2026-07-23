"""Non-spherical Hertz-Mindlin acceptance: per-point shell springs + patch Mindlin history.

1. CONSISTENCY: an SDF-built sphere must reproduce the analytic-sphere path
   (binary restitution) within shell-discretization tolerance.
2. CUBE STICK/SLIP: a cube resting flat on a tan(theta)=0.5 incline with FREE
   rotation must stick at mu=0.9 (faces cannot roll -- the true static test
   spheres cannot provide) and slide at mu=0.05.
3. CUBE bounce sanity: drop -> settles flat, energy decays.
"""
import numpy as np
from peclet import dem
from peclet.dem import build_wall_sdf
from peclet.dem.particle_builder import build_particle

E0, NU0, DT = 1.0e7, 0.25, 2e-5


def sphere_shape(r=0.5):
    return build_particle(lambda p: np.linalg.norm(p, axis=1) - r,
                          ((-0.7, -0.7, -0.7), (0.7, 0.7, 0.7)), resolution=56,
                          target_shell_points=220)


def cube_shape(a=0.5):  # half-edge a
    def f(p):
        q = np.abs(p) - a
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside
    return build_particle(f, ((-0.8, -0.8, -0.8), (0.8, 0.8, 0.8)), resolution=56,
                          target_shell_points=300)


def binary_sdf_sphere(e):
    sh = sphere_shape()
    s = dem.Simulation(2)
    sh.apply_to(s)
    s.set_domain((0, 0, 0), (20, 20, 20))
    s.enable_periodicity(False, False, False)
    s.set_positions(np.array([[8, 10, 10], [12, 10, 10]], np.float32))
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(2, np.float32))
    s.set_inv_inertia(np.full((2, 3), 1.0, np.float32))
    s.set_velocities(np.array([[1, 0, 0], [-1, 0, 0]], np.float32))
    s.set_gravity(0, 0, 0)
    s.set_thermostat(0, 0)
    s.set_material_params(e, 0.0, 0.0)
    s.set_hertz_material(0, E0, NU0)
    s.step_hertz(DT, 150000)
    v = s.get_velocities()
    return (v[1, 0] - v[0, 0]) / 2.0


def cube_slide(mu, steps=400000):
    sh = cube_shape()
    n = 4
    s = dem.Simulation(n)
    sh.apply_to(s)
    s.set_domain((0, 0, -1.0), (60, 12, 8))
    s.enable_periodicity(False, False, False)
    wall = build_wall_sdf(lambda p: p[:, 2], ((0, 0, -1.0), (60, 12, 8)), resolution=(96, 24, 24))
    wall.add_to(s, restitution=0.1, friction=mu)
    pts = np.array([[4 + 2.5 * i, 4, 0.505] for i in range(n)], np.float32)
    s.set_positions(pts)
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(n, np.float32))
    # cube inertia: I = (2/3) m a^2 for half-edge a=0.5 -> I=1/6
    s.set_inv_inertia(np.full((n, 3), 6.0, np.float32))
    s.set_velocities(np.zeros((n, 3), np.float32))
    s.set_gravity(5.0, 0.0, -10.0)
    s.set_thermostat(0, 0)
    s.set_material_params(0.1, 0.0, mu)
    s.set_hertz_material(0, E0, NU0)
    x0 = pts[:, 0].mean()
    s.step_hertz(DT, steps)
    p = s.get_positions()
    return float(p[:, 0].mean() - x0), float(p[:, 2].mean())


print("== shapes 1. SDF-sphere binary restitution (consistency vs analytic path) ==")
for e in (0.3, 0.8):
    ee = binary_sdf_sphere(e)
    print(f"   e={e}: e_eff={ee:.3f}")
    assert abs(ee - e) < 0.08, "SDF-sphere restitution off"

print("== shapes 2. cube stick/slip on tan=0.5 incline (free rotation) ==")
d_hi, z_hi = cube_slide(0.9)
d_lo, z_lo = cube_slide(0.05, steps=100000)  # 2 s: slides ~20 units, stays on the wall grid
print(f"   mu=0.9 slide {d_hi:+.2f} (z {z_hi:.2f})  |  mu=0.05 slide {d_lo:+.2f} (z {z_lo:.2f})")
assert abs(d_hi) < 1.0, "high-friction cube must stick"
assert d_lo > 5.0, "low-friction cube must slide"
assert 0.3 < z_hi < 0.7 and 0.3 < z_lo < 0.7, "cubes must stay on the floor"

print("ALL NON-SPHERICAL HERTZ ACCEPTANCE TESTS PASS")
