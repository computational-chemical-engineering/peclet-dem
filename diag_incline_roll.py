"""Diagnostic: a single sphere rolling on a plane (tilted gravity = inclined plane).

Measures the rolling rate and energy conservation to compare integration schemes.
Analytic (solid sphere, I=2/5 m r^2, rolling without slipping above mu_c=(2/7)tan th):
  a = (5/7) g sin th,  v(t) = a t,  omega*r = v (no-slip contact vel = 0),
  KE = (7/10) m v^2 = PE released  =>  KE/PE = 1 exactly.
Frictionless: a = g sin th, no spin, KE/PE = 1 (all translational).
"""
import math, sys
import numpy as np
sys.path.insert(0, "build")
from peclet import dem

g, theta = 9.8, math.radians(20.0)
gx, gz = g * math.sin(theta), -g * math.cos(theta)
R, T, dt = 0.5, 0.6, 0.002
nsteps = int(T / dt)


def run(mu):
    s = dem.Simulation(1)
    s.initialize(shape_type=1, radius=R)
    s.set_domain((-50, -50, -2), (50, 50, 50))
    s.enable_periodicity(False, False, False)
    s.set_gravity(gx, 0.0, gz)
    s.add_plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))     # floor z=0, normal +z
    s.set_material_params(0.0, 0.0, mu)                # inelastic normal; tangential target = stop sliding (Coulomb-bounded by mu)
    s.set_solver_iterations(20, 20)
    pos = np.array([[0.0, 0.0, R, 1.0]], np.float32)  # resting on the floor
    s.set_positions(pos)
    s.set_velocities(np.zeros((1, 3), np.float32))
    s.set_angular_velocities(np.zeros((1, 3), np.float32))
    s.set_scales(np.ones(1, np.float32))
    x0 = float(s.get_positions()[0, 0])
    for _ in range(nsteps):
        s.step(dt)
    p = s.get_positions()[0]
    v = s.get_velocities()[0]
    w = s.get_angular_velocities()[0]
    vt = float(v[0])
    wy = float(w[1])                                  # spin about +y rolls in +x
    v_contact = vt - wy * R                           # contact-point tangential vel = v_x + (omega x r_contact)_x, r_contact=(0,0,-R), omega=(0,wy,0) => vt - wy*R
    dx = float(p[0]) - x0
    KE = 0.5 * (vt * vt + float(v[1])**2 + float(v[2])**2) + 0.5 * (2.0/5.0) * R * R * (wy*wy + float(w[0])**2 + float(w[2])**2)
    PE = g * math.sin(theta) * dx                     # m=1; released PE = g sin th * along-incline displacement
    return vt, wy, v_contact, KE, PE, dx


print(f"  incline 20deg, dt={dt}, t={T}s   (rolling: a=(5/7)g sin th => v={ (5/7)*g*math.sin(theta)*T:.3f}; slide v={g*math.sin(theta)*T:.3f})")
print(f"  {'mu':>5} {'v_t':>8} {'w_y*r':>8} {'v_cont':>8} {'KE':>7} {'PE':>7} {'KE/PE':>7}")
for mu in (0.0, 0.05, 0.20, 0.50):
    vt, wy, vc, KE, PE, dx = run(mu)
    print(f"  {mu:5.2f} {vt:8.4f} {wy*R:8.4f} {vc:8.4f} {KE:7.3f} {PE:7.3f} {KE/PE:7.4f}")
