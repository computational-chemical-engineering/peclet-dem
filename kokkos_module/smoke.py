#!/usr/bin/env python3
# Python smoke test for the Kokkos-backed demgpu module (the flip's Python surface).
# Drops a grid of spheres onto a ground plane under gravity and checks the pipeline runs from Python,
# returns positions, and produces sensible physics (no NaNs/escape; particles settle above the floor).
import sys
import numpy as np
import demgpu_kokkos as dem

print("execution space:", dem.execution_space)

G, spacing, L = 5, 1.9, 12.0
xs = [(0.5 + ix * spacing, 0.5 + iy * spacing, 2.0 + iz * spacing)
      for iz in range(G) for iy in range(G) for ix in range(G)]
pos = np.array(xs, dtype=np.float32)
n = len(pos)

sim = dem.Simulation(capacity=n * 8)
sim.set_sphere_shape(1.0)
sim.set_domain(L, L, L, px=True, py=True, pz=False)   # ground plane in z
sim.set_global_scale(1.0)
sim.set_dt(0.004)
sim.set_gravity(0.0, 0.0, -9.8)
sim.set_solver_iterations(pos=12, vel=4)
sim.set_material_params(restitution=0.1, friction=0.4)
sim.add_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)            # floor at z=0, normal +z
sim.set_positions(pos)

z0 = sim.get_positions()[:, 2].mean()
sim.step(200)
p = sim.get_positions()
z1 = p[:, 2].mean()

print(f"particles={sim.num_particles()} contacts(last)={sim.num_contacts()} "
      f"max_overlap={sim.max_overlap():.4f}")
print(f"mean z: {z0:.3f} -> {z1:.3f}")

ok = True
if not np.all(np.isfinite(p)):
    print("FAIL: non-finite positions"); ok = False
if p[:, 2].min() < -1.5:
    print("FAIL: particles fell through the floor"); ok = False
if z1 >= z0:
    print("FAIL: particles did not settle under gravity"); ok = False
print("PASS" if ok else "FAIL")

# Release the simulation (frees its Kokkos Views) before the atexit Kokkos::finalize runs.
del sim
import gc
gc.collect()
sys.exit(0 if ok else 1)
