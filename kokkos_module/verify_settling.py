#!/usr/bin/env python3
# Physical validation of the Kokkos DEM solver: non-overlapping spheres settle in a box under gravity
# and the position solver resolves contact overlaps to a small residual (the substance of a packing
# verification, self-contained — no CUDA reference needed).
import sys, gc
import numpy as np
import demgpu_kokkos as dem

print("execution space:", dem.execution_space)

# Non-overlapping initial grid (spacing 2.2 > diameter 2) so overlaps come only from settling.
G, spacing, R, L = 5, 2.2, 1.0, 14.0
xs = [(1.5 + ix * spacing, 1.5 + iy * spacing, 2.0 + iz * spacing)
      for iz in range(G) for iy in range(G) for ix in range(G)]
pos = np.array(xs, dtype=np.float32)
n = len(pos)

sim = dem.Simulation(capacity=n * 8)
sim.set_sphere_shape(R)
sim.set_domain(L, L, L, px=False, py=False, pz=False)  # closed box
sim.set_global_scale(1.0)
sim.set_dt(0.003)
sim.set_gravity(0.0, 0.0, -9.8)
sim.set_solver_iterations(pos=20, vel=4)
sim.set_material_params(restitution=0.0, friction=0.5)
# six walls of the box (inward normals)
sim.add_plane(0, 0, 0, 0, 0, 1)      # floor
sim.add_plane(0, 0, L, 0, 0, -1)     # ceiling
sim.add_plane(0, 0, 0, 1, 0, 0)
sim.add_plane(L, 0, 0, -1, 0, 0)
sim.add_plane(0, 0, 0, 0, 1, 0)
sim.add_plane(0, L, 0, 0, -1, 0)
sim.set_positions(pos)

ov_hist = []
for k in range(20):
    sim.step(40)
    p = sim.get_positions()
    ov_hist.append(sim.max_overlap())
    if not np.all(np.isfinite(p)):
        print("FAIL: non-finite at block", k); sys.exit(1)

p = sim.get_positions()
final_ov = ov_hist[-1]
zmin = p[:, 2].min()
# kinetic-energy proxy: positions barely move over the last block
sim.step(40); p2 = sim.get_positions()
drift = float(np.sqrt(((p2 - p) ** 2).sum(axis=1)).mean())

print(f"particles={sim.num_particles()} contacts={sim.num_contacts()}")
print(f"overlap trajectory (every 40 steps): {[f'{o:.3f}' for o in ov_hist]}")
print(f"final max_overlap={final_ov:.4f} (radius={R})  zmin={zmin:.3f}  settle drift={drift:.4e}")

ok = True
if zmin < -0.2 * R:
    print("FAIL: particles below floor"); ok = False
if final_ov > 0.25 * R:
    print(f"FAIL: overlaps not resolved (max_overlap {final_ov:.3f} > {0.25*R:.3f})"); ok = False
if drift > 5e-3:
    print(f"FAIL: system not settling (drift {drift:.4e})"); ok = False
print("PASS" if ok else "FAIL")

del sim
gc.collect()
sys.exit(0 if ok else 1)
