"""Phase 0 reproduction: what density does the CURRENT engine actually reach for periodic monodisperse
spheres? Grows N frictionless spheres (the shipped single-phase protocol) into periodic boxes sized for a
range of design phi, settles, and measures the TRUE density / coordination / residual overlap with
pack_meter. The achievable random-close-packing density is the largest design phi whose overlap still
resolves (max_overlap below criterion) after settling; the reference is phi ~ 0.64, Z ~ 6.
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, "build")
import dem  # noqa: E402

import pack_meter  # noqa: E402

BASE_R = 0.5
VOL_P = (4.0 / 3.0) * math.pi * BASE_R ** 3


def run(phi_design, N=2000, dt=0.01, rate=2.0, iters=50, settle=600, seed=42, restitution=0.0):
    vol_total = N * VOL_P
    side = (vol_total / phi_design) ** (1.0 / 3.0)
    half = side / 2.0
    sim = dem.Simulation(N)
    sim.initialize(shape_type=1, radius=BASE_R)
    sim.set_domain((-half, -half, -half), (half, half, half))
    sim.enable_periodicity(True, True, True)
    sim.set_gravity(0, 0, 0)
    sim.set_material_params(restitution, restitution, 0.0)  # frictionless
    sim.set_solver_iterations(iters, iters)

    rng = np.random.default_rng(seed)
    pos = rng.uniform(-half, half, (N, 4)).astype(np.float32)
    pos[:, 3] = 1.0
    sim.set_positions(pos)
    sim.set_velocities(np.zeros((N, 4), np.float32))
    sim.set_scales(np.full(N, 1.0, np.float32))   # target full size
    sim.set_growth_params(rate, 0.05)             # grow factor 0.05 -> 1.0

    grow_steps = int(np.ceil(math.log(1.0 / 0.05) / (rate * dt))) + 5
    t0 = time.time()
    for _ in range(grow_steps + settle):
        sim.step(dt)

    pos = sim.get_positions()[:, :3]
    scales = sim.get_scales().ravel()
    radii = BASE_R * scales  # global_scale = 1
    vel = sim.get_velocities()[:, :3]
    dmin = np.array(sim.get_domain_min()); dmax = np.array(sim.get_domain_max())
    m = pack_meter.measure(pos, radii, dmin, dmax)
    ke = float(np.mean(np.sum(vel ** 2, axis=1)))
    return dict(phi_design=phi_design, side=side, secs=time.time() - t0, ke=ke,
                growth_factor=sim.get_growth_factor(), engine_overlap=sim.get_max_overlap(), **m)


def main():
    print("Phase 0 -- periodic monodisperse sphere packing, current engine (frictionless single-phase growth)")
    print(f"{'phi_des':>7} {'phi_naive':>9} {'phi_corr':>9} {'maxOv':>8} {'engOv':>8} {'Z':>5} "
          f"{'ratt':>5} {'KE':>9} {'gf':>5} {'s':>5}")
    for phi_d in (0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68):
        r = run(phi_d)
        print(f"{r['phi_design']:7.2f} {r['phi_naive']:9.3f} {r['phi_corrected']:9.3f} "
              f"{r['max_overlap']:8.1e} {r['engine_overlap']:8.1e} {r['coordination']:5.2f} "
              f"{r['rattlers']:5d} {r['ke']:9.2e} {r['growth_factor']:5.2f} {r['secs']:5.0f}")


if __name__ == "__main__":
    main()
