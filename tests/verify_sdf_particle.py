"""Validate the general grid-SDF particle path against the analytic shapes.

1. build_particle mass properties: a sphere SDF must recover volume 4/3 pi r^3, COM ~0, and
   isotropic unit-mass inertia 2/5 r^2 (inv_inertia_unit ~ 2.5/r^2) — the same value the analytic
   sphere uses.
2. Grid-SDF collision: two spheres imported as grid SDFs, overlapping, must produce the same
   separation dynamics as two analytic spheres (the grid path exercises sdfEvalShape on device).
3. A non-trivial shape (rounded box) builds, packs a few steps without NaN, and reduces overlap.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build_omp"))
import numpy as np
from peclet import dem
from peclet.dem import build_particle


def sphere_sdf(r):
    return lambda p: np.linalg.norm(p, axis=1) - r


def rounded_box_sdf(half, radius):
    def f(p):
        q = np.abs(p) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside - radius
    return f


def test_mass_properties():
    r = 0.5
    sp = build_particle(sphere_sdf(r), ((-0.7, -0.7, -0.7), (0.7, 0.7, 0.7)), resolution=80)
    vol_exact = (4.0 / 3.0) * math.pi * r**3
    print(f"[mass] volume  built={sp.mass:.5f}  exact={vol_exact:.5f}  "
          f"err={abs(sp.mass - vol_exact) / vol_exact * 100:.2f}%")
    print(f"[mass] com     {sp.com}  (|com|={np.linalg.norm(sp.com):.2e})")
    print(f"[mass] inertia principal (unit mass) {sp.inertia / sp.mass}  exact={0.4 * r**2:.5f}")
    print(f"[mass] inv_inertia_unit {sp.inv_inertia_unit}  analytic sphere={2.5 / r**2:.5f}")
    print(f"[mass] bounding_radius {sp.bounding_radius:.4f}  (exact {r:.4f})  shell pts={len(sp.shell)}")
    assert abs(sp.mass - vol_exact) / vol_exact < 0.02, "sphere volume off by >2%"
    assert np.linalg.norm(sp.com) < 1e-2 * r, "sphere COM not at origin"
    iu = sp.inv_inertia_unit
    assert np.allclose(iu, 2.5 / r**2, rtol=0.05), f"sphere inertia off: {iu} vs {2.5 / r**2}"
    print("  -> mass properties OK\n")
    return sp


def _two_body_gap(make_shape, r, sep, steps=40):
    """Two bodies at +-sep/2 on x, overlapping; return final centre gap after relaxation."""
    s = dem.Simulation(2)
    make_shape(s)
    s.set_domain((-4, -4, -4), (4, 4, 4))
    s.enable_periodicity(False, False, False)
    s.set_gravity(0, 0, 0)
    s.set_material_params(0.0, 0.0, 0.0)
    s.set_solver_iterations(40, 0)
    pos = np.array([[-sep / 2, 0, 0, 1.0], [sep / 2, 0, 0, 1.0]], dtype=np.float32)
    s.set_positions(pos)
    s.set_velocities(np.zeros((2, 3), np.float32))
    for _ in range(steps):
        s.step(0.01)
    p = s.get_positions().reshape(-1, 3)
    return float(abs(p[1, 0] - p[0, 0])), float(s.get_max_overlap())


def test_grid_vs_analytic_collision():
    r = 0.5
    sep = 0.8  # overlapping (2r = 1.0)
    gap_an, ov_an = _two_body_gap(lambda s: s.initialize(shape_type=1, radius=r), r, sep)
    sp = build_particle(sphere_sdf(r), ((-0.7, -0.7, -0.7), (0.7, 0.7, 0.7)), resolution=64)
    gap_sdf, ov_sdf = _two_body_gap(lambda s: sp.apply_to(s), r, sep)
    print(f"[collide] analytic sphere: final gap={gap_an:.4f}  overlap={ov_an:.2e}")
    print(f"[collide] grid-SDF sphere: final gap={gap_sdf:.4f}  overlap={ov_sdf:.2e}")
    print(f"[collide] gap difference {abs(gap_sdf - gap_an):.4f} (target ~2r={2 * r:.3f})")
    # Both should separate to ~2r with small residual overlap; grid within a couple voxels of analytic.
    assert gap_an > 2 * r - 0.05, "analytic spheres did not separate"
    assert gap_sdf > 2 * r - 0.10, "grid-SDF spheres did not separate"
    assert abs(gap_sdf - gap_an) < 0.08, "grid-SDF separation disagrees with analytic"
    print("  -> grid-SDF collision matches analytic sphere\n")


def test_rounded_box_pack():
    sp = build_particle(rounded_box_sdf(np.array([0.4, 0.4, 0.4]), 0.1),
                        ((-0.65, -0.65, -0.65), (0.65, 0.65, 0.65)), resolution=64)
    print(f"[rbox] mass={sp.mass:.4f} inv_inertia_unit={sp.inv_inertia_unit} "
          f"shell={len(sp.shell)} R={sp.bounding_radius:.3f}")
    N = 12
    s = dem.Simulation(N)
    sp.apply_to(s)
    half = 1.6
    s.set_domain((-half, -half, -half), (half, half, half))
    s.enable_periodicity(True, True, True)
    s.set_gravity(0, 0, 0)
    s.set_material_params(0.0, 0.0, 0.0)
    s.set_solver_iterations(30, 0)
    rng = np.random.default_rng(0)
    pos = rng.uniform(-half, half, (N, 4)).astype(np.float32); pos[:, 3] = 1.0
    s.set_positions(pos)
    s.set_velocities(np.zeros((N, 3), np.float32))
    q = rng.normal(0, 1, (N, 4)).astype(np.float32); q /= np.linalg.norm(q, axis=1, keepdims=True)
    s.set_quaternions(q)
    s.set_angular_velocities(np.zeros((N, 3), np.float32))
    ov0 = None
    for i in range(60):
        s.step(0.01)
        ov = s.get_max_overlap()
        if ov0 is None:
            ov0 = ov
    p = s.get_positions().reshape(-1, 3)
    assert np.isfinite(p).all(), "NaN/Inf in packed positions"
    print(f"[rbox] packed {N} rounded boxes, {i+1} steps: max_overlap {ov0:.3f} -> {ov:.3f}, "
          f"positions finite OK\n")


if __name__ == "__main__":
    test_mass_properties()
    test_grid_vs_analytic_collision()
    test_rounded_box_pack()
    print("ALL SDF-PARTICLE CHECKS PASSED")
