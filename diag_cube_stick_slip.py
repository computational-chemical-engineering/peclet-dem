"""Stick-slip transition of a CUBE on an inclined plane (a non-rolling grain).

Unlike a sphere (which rolls), a cube resting on a face either holds (static friction)
or slides as a rigid block. Coulomb's law predicts the transition at the friction angle
mu = tan(theta):
  * mu >= tan(theta): STICK  -> displacement ~ 0
  * mu <  tan(theta): SLIDE  -> a = g(sin theta - mu cos theta), x(T) = 1/2 a T^2
(Below theta < 45 deg a cube does not topple, so the test is a clean stick/slide.)

Realised here with tilted gravity on a horizontal plane (equivalent to an incline).
The cube rests flat; friction comes from Fix A (collisional, each step the block
re-approaches the plane under gravity -> normal impulse -> mu-bounded tangential impulse).
"""
import math, sys
import numpy as np
sys.path.insert(0, "build")
import dem

g, theta = 9.8, math.radians(20.0)
tan_t = math.tan(theta)
gx, gz = g * math.sin(theta), -g * math.cos(theta)
H = 0.5           # half-extent (cube side = 1.0)
T, dt = 0.5, 0.002
nsteps = int(T / dt)


def run(mu):
    s = dem.Simulation(1)
    s.initialize(shape_type=3, radius=H)           # cube, half-extent H
    s.set_domain((-50, -50, -2), (50, 50, 50))
    s.enable_periodicity(False, False, False)
    s.set_gravity(gx, 0.0, gz)
    s.add_plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))  # floor z=0, normal +z
    s.set_material_params(0.0, 0.0, mu)            # inelastic normal; tangential target = stop sliding
    s.set_solver_iterations(20, 20)
    s.set_positions(np.array([[0.0, 0.0, H, 1.0]], np.float32))   # bottom face on the floor
    s.set_velocities(np.zeros((1, 3), np.float32))
    s.set_quaternions(np.array([[0, 0, 0, 1]], np.float32))
    s.set_angular_velocities(np.zeros((1, 3), np.float32))
    s.set_scales(np.ones(1, np.float32))
    x0 = float(s.get_positions()[0, 0])
    for _ in range(nsteps):
        s.step(dt)
    p = s.get_positions()[0]
    q = s.get_quaternions()[0]
    dx = float(p[0]) - x0
    vx = float(s.get_velocities()[0, 0])
    # tilt = angle of the cube's body z-axis from world z (toppling/rotation indicator)
    qx, qy, qz, qw = [float(v) for v in q]
    zb_z = 1.0 - 2.0 * (qx * qx + qy * qy)         # world-z component of body z-axis
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, zb_z))))
    return dx, vx, tilt


print(f"  cube on a {math.degrees(theta):.0f} deg incline, friction angle tan(theta)={tan_t:.3f}; t={T}s")
print(f"  {'mu':>5} {'dx':>9} {'dx_slide':>9} {'vx':>8} {'tilt deg':>9}  regime")
for mu in (0.10, 0.20, 0.30, tan_t, 0.45, 0.60):
    dx, vx, tilt = run(mu)
    a = g * (math.sin(theta) - mu * math.cos(theta))
    dx_slide = 0.5 * a * T * T if a > 0 else 0.0   # analytic block slide from rest
    regime = "STICK" if mu >= tan_t else "slide"
    print(f"  {mu:5.3f} {dx:9.4f} {dx_slide:9.4f} {vx:8.4f} {tilt:9.4f}  expect {regime}")
