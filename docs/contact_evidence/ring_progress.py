"""Defect 1 survey, non-spherical: a RingBed-style hollow-cylinder growth packing (D = 1, H/D = 1.5,
wall 0.18, N = 80, periodic, friction 0.02, g = 0, 26 + 26 iterations, dt 1e-3 -- the
RingBed-CFD-Surrogate configs/packing_hollow_example.json geometry), printing every 250 steps the
contact / manifold counts and sim.diagnostics.coloring_conflicts() = (velocity/manifold graph,
position/contact-point graph) same-colour pairs of the last substep, and the worst so far.
Slow on the host past phi ~ 0.35 (minutes per 250 steps); ring_progress.txt was stopped at step 1500.

  OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=build_ct python docs/contact_evidence/ring_progress.py 80
"""
import math, sys, time
import numpy as np
from peclet import dem
N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
radius, height, thick = 0.5, 0.75, 0.18
vol_p = math.pi * height * (2 * radius * thick - thick ** 2)
phi = 0.55; dt = 1e-3
box = (N * vol_p / phi) ** (1 / 3)
rng = np.random.default_rng(42)
sim = dem.Simulation(N)
sim.initialize_shape("hollow_cylinder", radius=radius, height=height, thickness=thick)
sim.set_domain((0, 0, 0), (box, box, box)); sim.set_periodic(True, True, True)
sim.set_gravity((0.0, 0.0, 0.0)); sim.set_material_params(1.0, 1.0, 0.02); sim.set_solver_iterations(26, 26)
pos = rng.uniform(0, box, (N, 4)).astype(np.float32); pos[:, 3] = 1.0; sim.set_positions(pos)
q = rng.normal(0, 1, (N, 4)).astype(np.float32); q /= np.linalg.norm(q, axis=1, keepdims=True); sim.set_quaternions(q)
sim.set_velocities(rng.normal(0, math.sqrt(0.75), (N, 3)).astype(np.float32))
gr = 2.7; sim.set_growth_params(gr, 0.05); sim.set_thermostat(0.75, dt); sim.set_dt(dt)
worst = (0, 0); t0 = time.time()
for i in range(6000):
    sim.step()
    if sim.max_overlap > 2.2e-3:
        for _ in range(4): sim.relax()
        if sim.max_overlap > 2.2e-3:
            gf = sim.growth_factor * math.exp(-gr * dt); gr *= 0.99; sim.set_growth_params(gr, gf)
    else:
        gr = min(gr * 1.02, 2.7); sim.set_growth_params(gr, sim.growth_factor)
    c = sim.diagnostics.coloring_conflicts(); worst = (max(worst[0], c[0]), max(worst[1], c[1]))
    if i % 250 == 0 or i == 5999:
        s = np.asarray(sim.get_scales())
        print(f"RING step={i} t={time.time()-t0:.0f}s phi={phi*np.mean(s**3):.3f} contacts={sim.num_contacts} "
              f"per_particle={2*sim.num_contacts/N:.1f} manifolds={sim.num_manifolds} conflicts(vel,pos)={c} worst={worst}", flush=True)
