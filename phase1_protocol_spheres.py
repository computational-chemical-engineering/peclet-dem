"""Phase 1 decisive test: run the RingBed-style annealing protocol (Maxwell-Boltzmann seed + elastic
restitution + thermostat agitation during adaptive growth, then a dissipative cooling phase) on the
CURRENT engine for periodic monodisperse spheres, and measure the real packing with pack_meter. If this
reaches phi ~ 0.64 with Z ~ 6 and overlap below criterion, the engine is sound and Phase 0's frozen result
was purely the missing agitation/adaptive-growth machinery; if it still fails, the defect is in the engine.
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, "build")
sys.path.insert(0, ".")
import dem  # noqa: E402

import pack_meter  # noqa: E402

BASE = 0.5
VOLP = (4.0 / 3.0) * math.pi * BASE ** 3


def run(phi_ref=0.72, N=1000, T=1.0, dt=0.005, limit_time=6.0, iters=30, scale_init=0.05,
        rest_pre=1.0, rest_post=0.5, friction=0.0, criterion=1e-3, cooling_time=4.0,
        settle_patience=6, growth_accel=1.02, growth_decay=0.95, seed=1, verbose=True):
    side = (N * VOLP / phi_ref) ** (1.0 / 3.0)
    half = side / 2.0
    growth_rate_init = -math.log(scale_init)
    growth_rate = growth_rate_init
    cooling_step = min(int(cooling_time / dt), int(limit_time / dt))

    rng = np.random.default_rng(seed)
    s = dem.Simulation(N)
    s.initialize(shape_type=1, radius=BASE)
    s.set_domain((-half, -half, -half), (half, half, half))
    s.enable_periodicity(True, True, True)
    s.set_gravity(0.0, 0.0, 0.0)
    s.set_material_params(rest_pre, 1.0, friction)
    s.set_solver_iterations(iters, iters)

    pos = rng.uniform(-half, half, (N, 4)).astype(np.float32); pos[:, 3] = 1.0
    s.set_positions(pos)
    s.set_velocities(rng.normal(0.0, math.sqrt(T), (N, 3)).astype(np.float32))
    s.set_scales(np.full(N, 1.0, np.float32))
    s.set_growth_params(growth_rate, scale_init)
    s.set_thermostat(T, 1.0 * dt)

    t0 = time.time()
    for step in range(int(limit_time / dt)):
        if step == cooling_step:
            s.set_material_params(rest_post, 1.0, friction)
            s.set_thermostat(0.0, 1.0e4 * dt)
        s.step(dt)
        mo = float(s.get_max_overlap())
        if mo > criterion:
            it = 0
            while True:
                s.step(0.0); it += 1
                mn = float(s.get_max_overlap())
                if mn >= 0.95 * mo and it > settle_patience:
                    break
                mo = mn
            if mo > criterion:
                gf = float(s.get_growth_factor()) * math.exp(-growth_rate * dt)
                growth_rate *= growth_decay
                s.set_growth_params(growth_rate, gf)
        else:
            growth_rate = min(growth_rate * growth_accel, growth_rate_init)
            s.set_growth_params(growth_rate, float(s.get_growth_factor()))
        if verbose and step % 200 == 0:
            sc = s.get_scales().ravel()
            print(f"  step {step:5d} phi~{phi_ref*np.mean(sc**3):.3f} gf={s.get_growth_factor():.3f} "
                  f"maxOv={mo:.2e}", flush=True)

    sc = s.get_scales().ravel()
    pos = s.get_positions()[:, :3]
    radii = BASE * sc
    m = pack_meter.measure(pos, radii, np.array(s.get_domain_min()), np.array(s.get_domain_max()), gofr=False)
    phi_protocol = phi_ref * float(np.mean(sc ** 3))
    ke = float(np.mean(np.sum(s.get_velocities()[:, :3] ** 2, axis=1)))
    return dict(phi_ref=phi_ref, phi_protocol=phi_protocol, scale_mean=float(sc.mean()),
                ke=ke, secs=time.time() - t0, **m)


def main():
    print("Phase 1 -- RingBed annealing protocol on the current engine (periodic monodisperse spheres)")
    r = run()
    print(f"\nRESULT: phi_protocol={r['phi_protocol']:.3f}  phi_corrected={r['phi_corrected']:.3f}  "
          f"Z={r['coordination']:.2f}  rattlers={r['rattlers']}/{r['N']}  max_overlap={r['max_overlap']:.2e}  "
          f"KE={r['ke']:.1e}  ({r['secs']:.0f}s)")
    ok = r['coordination'] > 5.5 and r['max_overlap'] < 5e-3 and r['phi_corrected'] > 0.62
    print(f"VERDICT: {'ENGINE OK (clean jammed packing)' if ok else 'STILL FAILS -> engine defect'}")


if __name__ == "__main__":
    main()
