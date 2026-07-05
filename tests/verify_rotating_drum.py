"""Verify static SDF walls + moving-wall velocity: a rotating drum.

Confines spheres inside a cylindrical drum (a static SDF wall, axis = z), settles them under gravity,
then spins the wall about its axis. Checks (1) confinement — no grain escapes the barrel — and (2) the
dynamic response — a spinning drum drags the bed up its rising side, so the bed's centre of mass
shifts sideways (perpendicular to gravity) in the spin direction, versus a static drum whose bed sits
symmetric about the bottom.

Units are grain radii (grain radius = 1, global_scale = 1): the sphere–sphere narrow-phase assumes
global_scale == 1, so we size the *drum* in grain radii rather than shrinking the grains.
"""
import numpy as np
from peclet import dem
from peclet.dem import build_wall_sdf

print("backend:", dem.execution_space)

# --- drum geometry (axis along z), in grain-radius units ---
cx, cy = 20.0, 20.0
R = 15.0           # barrel radius (15 grain radii)
Lz = 10.0          # drum length
lo = (0.0, 0.0, 0.0)
hi = (40.0, 40.0, Lz)


def drum_sdf(p):
    # positive in the void (inside the barrel, between the end caps), negative in the solid wall
    r = np.sqrt((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2)
    barrel = R - r
    caps = np.minimum(p[:, 2] - lo[2], hi[2] - p[:, 2])
    return np.minimum(barrel, caps)


def build(seed=0):
    sim = dem.Simulation(400)
    sim.set_sphere_shape(1.0)          # grain radius 1, global_scale 1 (default)
    sim.set_domain(lo, hi)
    sim.enable_periodicity(False, False, False)
    wall = build_wall_sdf(drum_sdf, (lo, hi), resolution=96)
    wid = wall.add_to(sim, restitution=0.1, friction=0.7)
    sim.set_gravity(0.0, -12.0, 0.0)
    sim.set_material_params(0.1, 0.0, 0.4)
    sim.set_solver_iterations(24, 6)

    # seed grain centres on a loose lattice in the lower half of the drum, then GROW them from small
    # to full size (Lubachevsky–Stillinger) so they never start exactly touching (a stack of grains
    # placed surface-to-surface is an unstable initial condition for the geometric solver).
    step = 2.4
    g = np.arange(cx - R + 1.5, cx + R - 1.5, step)
    zs = np.arange(lo[2] + 1.2, hi[2] - 1.2, step)
    pts = [(x, y, z) for z in zs for x in g for y in g
           if (x - cx) ** 2 + (y - cy) ** 2 < (R - 2.0) ** 2 and y < cy + 0.35 * R]
    pts = np.array(pts, np.float32)
    N = len(pts)
    p = np.zeros((N, 4), np.float32)
    p[:, :3] = pts
    p[:, 3] = 1.0
    sim.set_positions(p)
    sim.set_scales(np.ones(N, np.float32))
    sim.set_growth_params(1.0, 0.15)        # start at 15% size
    sim.set_thermostat(0.0, 0.0)
    return sim, wid, N


def settle_and_spin(sim, wid, omega, spin_steps):
    dt = 0.004
    crit = 0.06
    # grow to full size, gated on overlap
    for _ in range(1500):
        grow = sim.get_max_overlap() < crit and float(sim.get_scales().mean()) < 0.999
        sim.set_growth_params(1.0 if grow else 0.0, sim.get_growth_factor())
        sim.step(dt)
    sim.set_growth_params(0.0, sim.get_growth_factor())
    # settle the packed bed (drum still)
    for _ in range(1200):
        sim.step(dt)
    # spin the drum about its z-axis through the axis point (cx, cy)
    sim.set_wall_velocity(wid, (0.0, 0.0, 0.0), (0.0, 0.0, omega), (cx, cy, 0.0))
    for _ in range(spin_steps):
        sim.step(dt)
    pos = sim.get_positions().reshape(-1, 3)
    r = np.sqrt((pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2)
    return float((r - R).max()), float(pos[:, 0].mean() - cx), float(pos[:, 1].mean() - cy)


print("\n-- static drum (omega = 0) --")
sim, wid, N = build()
pen0, cx0, cy0 = settle_and_spin(sim, wid, 0.0, 2500)
print(f"N grains = {N}")
print(f"max radial poke past barrel = {pen0:+.3f} (grain radius = 1)")
print(f"bed COM offset: x = {cx0:+.3f}, y = {cy0:+.3f}")

print("\n-- spinning drum (omega = +2 rad/s, CCW) --")
sim, wid, N = build()
penP, cxP, cyP = settle_and_spin(sim, wid, 2.0, 2500)
print(f"max radial poke past barrel = {penP:+.3f}")
print(f"bed COM offset: x = {cxP:+.3f}, y = {cyP:+.3f}")

print("\n-- checks --")
confined = penP < 1.5 and pen0 < 1.5
# CCW spin (omega_z > 0) with gravity in -y drags the bottom bed toward +x: COM x should move +.
dragged = (cxP - cx0) > 0.5
print(f"confinement (no tunneling)   : {'PASS' if confined else 'FAIL'}  "
      f"(worst poke static={pen0:+.3f}, spin={penP:+.3f})")
print(f"bed dragged in spin direction: {'PASS' if dragged else 'FAIL'}  "
      f"(dCOM_x = {cxP - cx0:+.3f})")
