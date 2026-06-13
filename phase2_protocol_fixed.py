"""Phase 2 fix test: the RingBed annealing protocol driven by a CORRECT overlap signal.

Phase 1 showed the engine's get_max_overlap() under-reports (position-solver residual) and compute_overlaps()
over-reports (periodic-ghost artifact); both defeat the protocol's overlap-criterion feedback. The position
solver itself is correct (resolves isolated overlaps exactly). So we drive the adaptive growth off a
brute-force-verified min-image overlap computed in Python (pack_meter) instead of the broken engine query,
and check whether the engine then reaches a clean random close packing (phi ~ 0.64, Z ~ 6, overlap << 1).
"""
import math
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "build")
sys.path.insert(0, ".")
import demgpu  # noqa: E402

import pack_meter  # noqa: E402

BASE = 0.5
VOLP = (4.0 / 3.0) * math.pi * BASE ** 3


def true_max_overlap(sim):
    """Brute-force-correct max pair overlap (fraction of contact distance) on the committed state."""
    pos = sim.get_positions()[:, :3].astype(np.float64)
    r = (BASE * sim.get_scales().ravel()).astype(np.float64)
    dmin = np.array(sim.get_domain_min()); box = np.array(sim.get_domain_max()) - dmin
    wp = np.mod(pos - dmin, box)
    tree = cKDTree(wp, boxsize=box)
    pairs = tree.query_pairs(r=2.0 * r.max(), output_type="ndarray")
    if not len(pairs):
        return 0.0
    i, j = pairs[:, 0], pairs[:, 1]
    d = wp[i] - wp[j]; d -= box * np.round(d / box)
    dist = np.linalg.norm(d, axis=1)
    ov = (r[i] + r[j]) - dist
    return float(np.max(ov / (r[i] + r[j])))


def run(phi_ref=0.68, N=800, T=1.0, dt=0.002, limit_time=10.0, iters=50, scale_init=0.05,
        rest_pre=1.0, rest_post=0.5, friction=0.0, criterion=5e-3, cooling_time=7.0,
        settle_patience=6, growth_accel=1.02, growth_decay=0.85, growth_rate_init=0.5,
        seed=1, verbose=True):
    side = (N * VOLP / phi_ref) ** (1.0 / 3.0); half = side / 2.0
    growth_rate = growth_rate_init
    cooling_step = min(int(cooling_time / dt), int(limit_time / dt))
    rng = np.random.default_rng(seed)
    s = demgpu.Simulation(N)
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
        mo = true_max_overlap(s)            # <-- CORRECT signal (was the broken sim.get_max_overlap())
        if mo > criterion:
            it = 0
            while True:
                s.step(0.0); it += 1
                mn = true_max_overlap(s)
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
        if verbose and step % 100 == 0:
            sc = s.get_scales().ravel()
            print(f"  step {step:5d} phi~{phi_ref*np.mean(sc**3):.3f} gf={s.get_growth_factor():.3f} "
                  f"trueOv={mo:.2e}", flush=True)

    # Final quench: stop growth, fully dissipative, thermostat off -> settle into the rigid contact network.
    s.set_material_params(0.0, 0.0, friction)
    s.set_thermostat(0.0, 10.0 * dt)
    for _ in range(2000):
        s.step(dt)
    sc = s.get_scales().ravel(); pos = s.get_positions()[:, :3]
    m = pack_meter.measure(pos, BASE * sc, np.array(s.get_domain_min()), np.array(s.get_domain_max()))
    ke = float(np.mean(np.sum(s.get_velocities()[:, :3] ** 2, axis=1)))
    return dict(phi_protocol=phi_ref * float(np.mean(sc ** 3)), ke=ke, secs=time.time() - t0,
                _pos=pos.copy(), _radii=(BASE * sc).copy(), _dmin=np.array(s.get_domain_min()),
                _dmax=np.array(s.get_domain_max()), **m)


def coordination_at(pos, radii, dmin, dmax, gap):
    """Mean contacts per particle counting pairs with centre distance < (r_i+r_j)(1+gap) -- gap>0 allows a
    small surface gap (RCP contacts are ~touching, so a strict compression test undercounts)."""
    pos = np.asarray(pos)[:, :3].astype(np.float64); radii = np.asarray(radii, np.float64)
    dmin = np.asarray(dmin, np.float64); box = np.asarray(dmax, np.float64) - dmin
    wp = np.mod(pos - dmin, box); tree = cKDTree(wp, boxsize=box)
    pairs = tree.query_pairs(r=2 * radii.max() * (1 + max(gap, 0)), output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = wp[i] - wp[j]; d -= box * np.round(d / box); dist = np.linalg.norm(d, axis=1)
    rsum = radii[i] + radii[j]
    con = dist < rsum * (1 + gap)
    deg = np.bincount(np.concatenate([i[con], j[con]]), minlength=len(pos))
    nz = deg[deg > 0]
    return (nz.mean() if len(nz) else 0.0), int(np.sum(deg == 0))


def main():
    print("Phase 2 -- annealing protocol driven by a CORRECT overlap signal (periodic spheres)")
    r = run()
    pos, radii, dmin, dmax = r["_pos"], r["_radii"], r["_dmin"], r["_dmax"]
    print(f"\nRESULT: phi={r['phi_corrected']:.3f}  max_overlap={r['max_overlap']:.2e}  KE={r['ke']:.1e}  "
          f"({r['secs']:.0f}s)")
    print("  coordination Z (rattlers) at increasing contact gap tolerance:")
    for gap in (-0.001, 0.0, 0.002, 0.005, 0.01, 0.02):
        Z, ratt = coordination_at(pos, radii, dmin, dmax, gap)
        print(f"    gap={gap:+.3f}: Z={Z:.2f}  rattlers={ratt}/{r['N']}")
    rr, gg = pack_meter._gofr(np.mod(pos[:, :3] - dmin, dmax - dmin), np.asarray(dmax) - np.asarray(dmin),
                              radii, 200)
    peak = rr[1:][np.argmax(gg[1:])]
    print(f"  g(r): contact peak at r/D={peak:.3f} (RCP -> ~1.0, with a split second peak near 1.7-2.0)")


if __name__ == "__main__":
    main()
