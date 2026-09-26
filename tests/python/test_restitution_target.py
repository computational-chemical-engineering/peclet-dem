#!/usr/bin/env python3
"""The PGS restitution target law as a diagnostics A/B (docs/contact_physics_followups.md §4, §6
G-C1, WO-C1).

On the g != 0 path the velocity solve is a projected Gauss-Seidel over every closed contact. The
restitution TARGET of a contact is either

  * 'newton' (default): -e x the pre-solve approach on a contact that approaches, 0 on one that is
    already separating; or
  * 'moreau': -e x the pre-solve normal velocity on EVERY closed contact above the resting
    threshold.

For a converged solve with a uniform e the kinetic-energy change of one substep is
dT = -1/2 (1-e)/(1+e) lambda^T W lambda <= 0 under Moreau (= 0 at e = 1), while Newton adds
e/(1+e) lambda_c u_c for every pre-separating contact that ends up loaded, so a dense kinetic cluster
GAINS energy at e >= 0.9 (§4.1). This test measures that on one converged step of a dense,
frictionless, polydisperse cluster in free fall (gravity only switches the PGS path on: the relative
dynamics are those of g = 0). The gravity is small so that the adaptive stop (0.02 x 2 dt |g|) and
the resting threshold (2 dt |g|) are far below the contact velocities: the solve is converged and
almost every contact carries the material e.
"""
import numpy as np
import pytest

import peclet.dem as dem


def _np(a):
    return a.get() if hasattr(a, "get") and type(a).__module__.startswith("cupy") else np.asarray(a)


def _cluster(seed=1, side=7, spacing=0.95):
    """The dense cluster of the note's model A.4: a side^3 lattice at `spacing` with N(0, 0.04)
    jitter, radii 0.5 (1 +- 0.1), m ~ r^3, velocities N(0, 1)."""
    rng = np.random.default_rng(seed)
    g = np.arange(side) * spacing
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    X = X - X.mean(0) + rng.normal(0.0, 0.04, X.shape)
    scale = 1.0 + 0.1 * rng.uniform(-1.0, 1.0, len(X))
    V = rng.normal(0.0, 1.0, X.shape)
    return X.astype(np.float32), scale.astype(np.float32), V.astype(np.float32)


def one_step_ke_ratio(e, target):
    X, scale, V = _cluster()
    n = len(X)
    m = scale.astype(np.float64) ** 3
    s = dem.Simulation(n + 64)
    s.initialize_shape("sphere", radius=0.5)
    s.set_domain((-40, -40, -40), (40, 40, 40))
    s.set_gravity((0.0, 0.0, -0.01))          # switches the PGS path on; vRest = 2 dt |g| = 2e-4
    s.set_material_params(e, 0.0, 0.0)          # frictionless
    s.set_solver_iterations(20, 2000)
    s.set_stabilization("off")
    s.set_sleeping(False)
    s.diagnostics.set_restitution_target(target)
    s.set_dt(0.01)
    s.set_positions(np.c_[X, (1.0 / m).astype(np.float32)])
    s.set_scales(scale)
    s.set_velocities(V)

    def ke():
        v = _np(s.get_velocities())[:n].astype(np.float64)
        p = (m[:, None] * v).sum(0)
        return 0.5 * (m[:, None] * v * v).sum() - 0.5 * (p @ p) / m.sum()

    ke0 = ke()
    s.step()
    return ke() / ke0, s.num_contacts


@pytest.mark.parametrize("e", [0.5, 0.9, 1.0])
def test_restitution_target_energy(e):
    """Moreau never creates kinetic energy in the converged dense step (and conserves it at e = 1
    up to the sub-threshold e = 0 contacts); Newton's ratio is printed for the A/B."""
    r_m, nc = one_step_ke_ratio(e, "moreau")
    r_n, _ = one_step_ke_ratio(e, "newton")
    print(f"   e={e:.1f}  contacts={nc}  KE1/KE0: newton={r_n:.6f}  moreau={r_m:.6f}")
    assert nc > 300, "the cluster must be dense (hundreds of closed contacts)"
    assert r_m <= 1.0 + 1e-5
    if e == 1.0:
        assert r_m >= 1.0 - 2e-3


def test_restitution_target_setter():
    """Default 'newton'; unknown names and the Moreau + Poisson combination raise (R-C4)."""
    s = dem.Simulation(4)
    assert s.diagnostics.restitution_target == "newton"
    s.diagnostics.set_restitution_target("moreau")
    assert s.diagnostics.restitution_target == "moreau"
    with pytest.raises(ValueError):
        s.set_restitution_model("poisson")
    assert s.restitution_model == "newton"
    s.diagnostics.set_restitution_target("newton")
    s.set_restitution_model("poisson")
    with pytest.raises(ValueError):
        s.diagnostics.set_restitution_target("moreau")
    assert s.diagnostics.restitution_target == "newton"
    with pytest.raises(ValueError):
        s.diagnostics.set_restitution_target("poisson")
