"""Reusable dense-random-packing protocol for the GPU DEM engine (replaces the broken
verify_packing_spheres.py).

A Lubachevsky-Stillinger-style growth + annealing protocol that converges to the jamming density:

  * particles start small and grow toward a target that fills the box to ``phi_ref``;
  * during growth the system is agitated (a thermostat keeps it fluid) and collisions are elastic, so
    particles rearrange instead of freezing;
  * growth is FEEDBACK-CONTROLLED by the true committed overlap (the engine's fixed ``compute_overlaps()``):
    grow while the overlap is below ``criterion``, back the growth off when it exceeds it -> the size
    converges to the jamming point;
  * then a dissipative cooling phase + a final quench settle the packing into its rigid contact network.

Validated for periodic monodisperse spheres: phi ~ 0.635 (random close packing ~ 0.64), isostatic
coordination Z ~ 6 at the contact gap, g(r) contact peak at one diameter. See docs/packing_investigation.md.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "build")
import demgpu  # noqa: E402

import pack_meter  # noqa: E402


def sphere_volume(radius):
    return (4.0 / 3.0) * math.pi * radius ** 3


def pack_spheres(N=800, phi_ref=0.68, radius=0.5, friction=0.0, temperature=1.0, dt=0.002,
                 limit_time=10.0, iters=50, scale_init=0.05, rest_pre=1.0, rest_post=0.5,
                 criterion=5e-3, cooling_time=7.0, quench_steps=2000, settle_patience=6,
                 growth_accel=1.02, growth_decay=0.85, growth_rate_init=0.5, seed=1, verbose=False):
    """Pack N monodisperse spheres in a periodic cubic box and return (sim, info).

    Drives the annealing protocol off the engine's committed overlap (``compute_overlaps()``), normalised
    by the current contact distance. ``friction`` > 0 lowers the jamming density toward random loose
    packing. Returns the Simulation and a dict with the achieved phi and runtime info; call
    ``analyze_spheres`` for the microstructure (Z, g(r)).
    """
    volp = sphere_volume(radius)
    side = (N * volp / phi_ref) ** (1.0 / 3.0)
    half = side / 2.0
    growth_rate = growth_rate_init
    cooling_step = min(int(cooling_time / dt), int(limit_time / dt))
    rng = np.random.default_rng(seed)

    s = demgpu.Simulation(N)
    s.initialize(shape_type=1, radius=radius)
    s.set_domain((-half, -half, -half), (half, half, half))
    s.enable_periodicity(True, True, True)
    s.set_gravity(0.0, 0.0, 0.0)
    s.set_material_params(rest_pre, 1.0, friction)
    s.set_solver_iterations(iters, iters)
    pos = rng.uniform(-half, half, (N, 4)).astype(np.float32); pos[:, 3] = 1.0
    s.set_positions(pos)
    s.set_velocities(rng.normal(0.0, math.sqrt(temperature), (N, 3)).astype(np.float32))
    s.set_scales(np.full(N, 1.0, np.float32))
    s.set_growth_params(growth_rate, scale_init)
    s.set_thermostat(temperature, 1.0 * dt)

    def overlap_frac():
        contact = 2.0 * radius * float(s.get_scales().ravel().mean())
        return float(s.compute_overlaps()) / max(contact, 1e-9)

    for step in range(int(limit_time / dt)):
        if step == cooling_step:
            s.set_material_params(rest_post, 1.0, friction)
            s.set_thermostat(0.0, 1.0e4 * dt)
        s.step(dt)
        mo = overlap_frac()
        if mo > criterion:
            it = 0
            while True:
                s.step(0.0); it += 1
                mn = overlap_frac()
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
            print(f"  step {step:5d} phi~{phi_ref*np.mean(sc**3):.3f} overlap={mo:.2e}", flush=True)

    # final quench: stop growth, fully dissipative, thermostat off -> settle into the rigid network
    s.set_material_params(0.0, 0.0, friction)
    s.set_thermostat(0.0, 10.0 * dt)
    for _ in range(quench_steps):
        s.step(dt)

    sc = s.get_scales().ravel()
    return s, dict(N=N, phi_ref=phi_ref, friction=friction, phi=phi_ref * float(np.mean(sc ** 3)),
                   ke=float(np.mean(np.sum(s.get_velocities()[:, :3] ** 2, axis=1))))


def analyze_spheres(sim, radius=0.5, gaps=(0.0, 0.002, 0.005)):
    """Microstructure of a sphere packing: overlap-corrected phi, coordination Z at each contact gap,
    rattler fraction and the g(r) contact-peak position."""
    pos = sim.get_positions()[:, :3]
    r = radius * sim.get_scales().ravel()
    dmin = np.array(sim.get_domain_min()); dmax = np.array(sim.get_domain_max())
    m = pack_meter.measure(pos, r, dmin, dmax, gofr=True)
    box = dmax - dmin
    wp = np.mod(np.asarray(pos)[:, :3].astype(np.float64) - dmin, box)
    rr, gg = pack_meter._gofr(wp, box, r.astype(np.float64), 200)
    Z = {}
    from scipy.spatial import cKDTree
    tree = cKDTree(wp, boxsize=box)
    for gap in gaps:
        pairs = tree.query_pairs(r=2 * r.max() * (1 + max(gap, 0)), output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        d = wp[i] - wp[j]; d -= box * np.round(d / box)
        con = np.linalg.norm(d, axis=1) < (r[i] + r[j]) * (1 + gap)
        deg = np.bincount(np.concatenate([i[con], j[con]]), minlength=len(r))
        Z[gap] = (float(deg[deg > 0].mean()) if (deg > 0).any() else 0.0, int(np.sum(deg == 0)))
    return dict(phi_corrected=m["phi_corrected"], max_overlap=m["max_overlap"],
                Z=Z, gofr_peak=float(rr[1:][np.argmax(gg[1:])]))


def random_quaternions(rng, n):
    q = rng.normal(0.0, 1.0, (n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def ring_volume(radius, height, thickness):
    r_in = radius - thickness
    return math.pi * (radius ** 2 - r_in ** 2) * height


def pack_rings(N=400, phi_ref=0.55, radius=0.5, height=1.0, thickness=0.15, temperature=1.0, dt=0.002,
               limit_time=12.0, iters=50, scale_init=0.05, rest_pre=1.0, rest_post=0.5, criterion=2e-3,
               cooling_time=8.0, quench_steps=2000, settle_patience=6, growth_accel=1.02,
               growth_decay=0.85, growth_rate_init=0.5, seed=1, verbose=False):
    """Pack N hollow-cylinder (ring) particles in a periodic box -- the same annealing protocol as
    pack_spheres but with orientations and the SDF-based overlap signal. Returns (sim, info) with the
    achieved phi (= phi_ref*mean(scale^3), the RingBed convention). Reference: squat-thick rings reach
    phi ~ 0.30-0.33 (rings pack far looser than spheres)."""
    volp = ring_volume(radius, height, thickness)
    side = (N * volp / phi_ref) ** (1.0 / 3.0)
    half = side / 2.0
    growth_rate = growth_rate_init
    cooling_step = min(int(cooling_time / dt), int(limit_time / dt))
    rng = np.random.default_rng(seed)

    s = demgpu.Simulation(N)
    s.initialize(shape_type=2, radius=radius, height=height, thickness=thickness)
    s.set_domain((-half, -half, -half), (half, half, half))
    s.enable_periodicity(True, True, True)
    s.set_gravity(0.0, 0.0, 0.0)
    s.set_material_params(rest_pre, 1.0, 0.0)
    s.set_solver_iterations(iters, iters)
    pos = rng.uniform(-half, half, (N, 4)).astype(np.float32); pos[:, 3] = 1.0
    s.set_positions(pos)
    s.set_velocities(rng.normal(0.0, math.sqrt(temperature), (N, 3)).astype(np.float32))
    s.set_quaternions(random_quaternions(rng, N))
    s.set_angular_velocities(np.zeros((N, 3), np.float32))
    s.set_scales(np.full(N, 1.0, np.float32))
    s.set_growth_params(growth_rate, scale_init)
    s.set_thermostat(temperature, 1.0 * dt)

    for step in range(int(limit_time / dt)):
        if step == cooling_step:
            s.set_material_params(rest_post, 1.0, 0.0)
            s.set_thermostat(0.0, 1.0e4 * dt)
        s.step(dt)
        mo = float(s.compute_overlaps())            # absolute SDF penetration (engine fixed)
        if mo > criterion:
            it = 0
            while True:
                s.step(0.0); it += 1
                mn = float(s.compute_overlaps())
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
            print(f"  step {step:5d} phi~{phi_ref*np.mean(sc**3):.3f} overlap={mo:.2e}", flush=True)

    s.set_material_params(0.0, 0.0, 0.0); s.set_thermostat(0.0, 10.0 * dt)
    for _ in range(quench_steps):
        s.step(dt)
    sc = s.get_scales().ravel()
    return s, dict(N=N, phi_ref=phi_ref, phi=phi_ref * float(np.mean(sc ** 3)),
                   max_overlap=float(s.compute_overlaps()),
                   ke=float(np.mean(np.sum(s.get_velocities()[:, :3] ** 2, axis=1))))


def main():
    print("Random close packing of periodic monodisperse spheres (frictionless)")
    sim, info = pack_spheres(verbose=True)
    a = analyze_spheres(sim)
    print(f"\n  phi = {a['phi_corrected']:.3f}   max_overlap = {a['max_overlap']:.2e}   KE = {info['ke']:.1e}")
    for gap, (Z, ratt) in a["Z"].items():
        print(f"  Z(gap={gap:+.3f}) = {Z:.2f}   rattlers = {ratt}/{info['N']}")
    print(f"  g(r) contact peak at r/D = {a['gofr_peak']:.3f}  (RCP -> ~1.0)")
    ok = a["Z"][0.002][0] > 5.8 and a["phi_corrected"] > 0.61 and a["max_overlap"] < 5e-2
    print(f"  result: {'PASS -- genuine random close packing' if ok else 'off'}")


if __name__ == "__main__":
    main()
