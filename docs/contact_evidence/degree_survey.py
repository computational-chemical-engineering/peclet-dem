"""Defect 1 survey: do production-like packings reach the colour overflow?

Reports, for a growth-packed periodic bed, the colouring invariant violations of the last substep
(sim.diagnostics.coloring_conflicts(): (velocity/manifold graph, position/contact graph)), the
contacts and manifolds per particle, and the largest per-body degree of both graphs estimated from
the committed state (spheres: exact from the positions and radii within the contact reach
r_i + r_j + 0.1 R_max; other shapes: not available from Python, conflicts only).

Usage (host build on PYTHONPATH, bounded OpenMP pool):
  OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=build_ct python docs/contact_evidence/degree_survey.py [case ...]
Cases: mono, poly10, bi2, bi3, bi4, bi6, ring (the ring case is slow on the host; ring_progress.py
prints the same diagnostics as it goes and is what survey.txt quotes for rings)
"""
import math
import sys

import numpy as np

from peclet import dem


def sphere_degrees(sim, radius, box):
    x = np.asarray(sim.get_positions())[:, :3].astype(np.float64)
    r = radius * np.asarray(sim.get_scales()).astype(np.float64)
    rmax = r.max()
    n = len(r)
    deg = np.zeros(n, int)
    for i in range(n):
        d = x - x[i]
        d -= box * np.round(d / box)
        dist = np.sqrt((d * d).sum(1))
        m = dist < r + r[i] + 0.1 * rmax
        m[i] = False
        deg[i] = m.sum()
    return deg


def grow(sim, steps, dt, criterion, label):
    growth_rate = sim.growth_rate if hasattr(sim, "growth_rate") else 1.0
    worst = (0, 0)
    for i in range(steps):
        sim.step()
        ov = sim.max_overlap
        if ov > criterion:
            for _ in range(8):
                sim.relax()
            if sim.max_overlap > criterion:
                gf = sim.growth_factor * math.exp(-growth_rate * dt)
                growth_rate *= 0.95
                sim.set_growth_params(growth_rate, gf)
        else:
            growth_rate = min(growth_rate * 1.02, 2.0)
            sim.set_growth_params(growth_rate, sim.growth_factor)
        c = sim.diagnostics.coloring_conflicts()
        worst = (max(worst[0], c[0]), max(worst[1], c[1]))
    return worst


def run_spheres(label, scales_fn, N=2000, phi=0.62, steps=3000, dt=2e-3, iters=30):
    radius = 0.5
    rng = np.random.default_rng(3)
    s_t = scales_fn(rng, N).astype(np.float32)
    vol = (4 / 3) * math.pi * radius ** 3 * np.sum(s_t ** 3)
    box = (vol / phi) ** (1 / 3)
    sim = dem.Simulation(N)
    sim.initialize_shape("sphere", radius=radius)
    sim.set_domain((0, 0, 0), (box, box, box))
    sim.set_periodic(True, True, True)
    sim.set_gravity((0.0, 0.0, 0.0))
    sim.set_material_params(0.8, 1.0, 0.0)
    sim.set_solver_iterations(iters, iters)
    pos = rng.uniform(0, box, (N, 4)).astype(np.float32)
    pos[:, 3] = 1.0
    sim.set_positions(pos)
    sim.set_scales(s_t)
    sim.set_velocities(rng.normal(0, 1, (N, 3)).astype(np.float32))
    sim.set_growth_params(1.0, 0.05)
    sim.set_thermostat(1.0, dt)
    sim.set_dt(dt)
    worst = grow(sim, steps, dt, 5e-3, label)
    sim.set_thermostat(0.0, 1e4 * dt)
    sim.set_material_params(0.3, 1.0, 0.0)
    w2 = grow(sim, steps // 3, dt, 5e-3, label)
    worst = (max(worst[0], w2[0]), max(worst[1], w2[1]))
    s = np.asarray(sim.get_scales())
    phi_now = (4 / 3) * math.pi * radius ** 3 * np.sum(s ** 3) / box ** 3
    deg = sphere_degrees(sim, radius, box)
    big = s_t >= s_t.max() * 0.999
    print(f"SURVEY case={label} N={N} phi={phi_now:.3f} size_ratio={s_t.max() / s_t.min():.2f} "
          f"contacts/particle={2 * sim.num_contacts / N:.2f} manifolds/particle={2 * sim.num_manifolds / N:.2f} "
          f"maxDeg={deg.max()} meanDeg={deg.mean():.2f} maxDeg(largest grains)={deg[big].max()} "
          f"conflicts_last(vel,pos)={sim.diagnostics.coloring_conflicts()} conflicts_max_over_run(vel,pos)={worst}",
          flush=True)


def run_ring(N=80, steps=4000, dt=1e-3, iters=26):
    # RingBed-CFD-Surrogate configs/packing_hollow_example.json: D = 1, H/D = 1.5, wall 0.18
    radius, height, thick = 0.5, 0.75, 0.18
    vol_p = math.pi * height * (2 * radius * thick - thick ** 2)
    phi = 0.55
    box = (N * vol_p / phi) ** (1 / 3)
    rng = np.random.default_rng(42)
    sim = dem.Simulation(N)
    sim.initialize_shape("hollow_cylinder", radius=radius, height=height, thickness=thick)
    sim.set_domain((0, 0, 0), (box, box, box))
    sim.set_periodic(True, True, True)
    sim.set_gravity((0.0, 0.0, 0.0))
    sim.set_material_params(1.0, 1.0, 0.02)
    sim.set_solver_iterations(iters, iters)
    pos = rng.uniform(0, box, (N, 4)).astype(np.float32)
    pos[:, 3] = 1.0
    sim.set_positions(pos)
    q = rng.normal(0, 1, (N, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sim.set_quaternions(q)
    sim.set_velocities(rng.normal(0, math.sqrt(0.75), (N, 3)).astype(np.float32))
    sim.set_growth_params(2.7, 0.05)
    sim.set_thermostat(0.75, dt)
    sim.set_dt(dt)
    worst = grow(sim, steps, dt, 2.2e-3, "ring")
    s = np.asarray(sim.get_scales())
    phi_now = phi * np.mean(s ** 3)
    print(f"SURVEY case=ring N={N} phi={phi_now:.3f} contacts/particle={2 * sim.num_contacts / N:.1f} "
          f"manifolds/particle={2 * sim.num_manifolds / N:.2f} "
          f"conflicts_last(vel,pos)={sim.diagnostics.coloring_conflicts()} conflicts_max_over_run(vel,pos)={worst}",
          flush=True)


def bidisperse(q, frac_big_vol=0.5):
    def f(rng, N):
        # number fraction of big grains for a volume fraction frac_big_vol
        nb = max(1, int(round(N * frac_big_vol / (frac_big_vol + (1 - frac_big_vol) * q ** 3))))
        s = np.ones(N)
        s[:nb] = q
        rng.shuffle(s)
        return s
    return f


CASES = {
    "mono": lambda: run_spheres("mono", lambda rng, N: np.ones(N)),
    "poly10": lambda: run_spheres("poly10", lambda rng, N: 1.0 + 0.1 * rng.uniform(-1, 1, N)),
    "bi2": lambda: run_spheres("bi2", bidisperse(2.0)),
    "bi3": lambda: run_spheres("bi3", bidisperse(3.0)),
    "bi4": lambda: run_spheres("bi4", bidisperse(4.0)),
    "bi6": lambda: run_spheres("bi6", bidisperse(6.0), N=4000),
    "ring": run_ring,
}

if __name__ == "__main__":
    for c in (sys.argv[1:] or list(CASES)):
        CASES[c]()
