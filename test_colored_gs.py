#!/usr/bin/env python3
"""A-priori validation of the colored Gauss–Seidel (GS) restitution solve vs the legacy count-
averaged Jacobi solve. Answers, with measurements rather than argument, the three claims made about
the scheme:

  A. BINARY EXACTNESS  — a lone (count==1) collision must realise the prescribed restitution e
     exactly. Head-on equal-mass spheres: e_eff = sqrt(KE_after/KE_before) must equal e.
  B. CONSERVATION      — the impulse is equal-and-opposite at a single contact point, so a collision
     must conserve BOTH linear and (orbital+spin) angular momentum. GS applies J directly to the two
     bodies -> exact; the count-averaged Jacobi scales each body's summed impulse by its own contact
     count, so in multi-contact it breaks per-contact momentum conservation. We measure the drift for
     both. Cylinders (off-centre contacts) exercise the ANGULAR update / spin.
  C. DISSIPATION       — in a dense granular gas the multi-contact under-convergence of the averaged
     Jacobi shows up as too-little cooling (effective e too high). We measure the cooling slope
     against the Enskog/Haff prediction for GS vs Jacobi.

    PECLET_LOCAL_BUILD=.../dem/build_cuda_mphys python test_colored_gs.py
"""
import os, sys, time
import numpy as np

_local = os.environ.get("PECLET_LOCAL_BUILD")
if _local:
    for p in _local.split(os.pathsep):
        sys.path.insert(0, p)
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import peclet.dem as dem

def _np(a):
    return a.get() if hasattr(a, "get") and type(a).__module__.startswith("cupy") else np.asarray(a)

def Rmat(q):  # (N,4) xyzw -> (N,3,3)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

# ----------------------------------------------------------------------------- A. binary exactness
def test_binary(e, use_gs=True):
    r = 0.5; u = 1.0; dt = 0.005
    s = dem.Simulation(2); s.initialize(shape_type=1, radius=r); s.set_sphere_shape(r)
    s.set_domain((-10, -10, -10), (10, 10, 10)); s.set_gravity(0, 0, 0)
    s.set_material_params(e, 0.0, 0.0); s.set_solver_iterations(10, 12); s.set_dt(dt)
    s.set_velocity_use_gs(use_gs)
    P = np.array([[-2.0, 0, 0, 1.0], [2.0, 0, 0, 1.0]], np.float32)   # .w = invMass = 1
    V = np.array([[+u, 0, 0], [-u, 0, 0]], np.float32)
    s.set_positions(P); s.set_velocities(V)
    ke0 = 0.5 * (V ** 2).sum()
    for i in range(1200):
        s.step(dt)
    Vf = _np(s.get_velocities())[:2]
    ke1 = 0.5 * (Vf ** 2).sum()
    p_drift = np.abs(Vf.sum(0)).max()          # total momentum should stay 0
    return np.sqrt(max(ke1, 0) / ke0), p_drift

# ------------------------------------------------------- B. linear + angular momentum conservation
def test_conservation(use_gs=True, seed=0):
    """A cloud of cylinders fired inward in free space: no gravity/walls, so total linear momentum P
    and total angular momentum L (orbital m x×v + spin R I_local Rᵀ ω) are exactly conserved by any
    momentum-conserving contact law. Returns the worst-case relative drift of |P| and |L|."""
    N = 24; dt = 0.004
    s = dem.Simulation(N)
    s.initialize(shape_type=2, radius=0.5, height=1.0, thickness=0.2)   # cylinders -> off-centre hits
    s.set_domain((-20, -20, -20), (20, 20, 20)); s.set_gravity(0, 0, 0)
    s.set_material_params(0.7, 0.0, 0.0); s.set_solver_iterations(8, 10); s.set_dt(dt)
    s.set_velocity_use_gs(use_gs)
    rng = np.random.default_rng(seed)
    P0 = rng.uniform(-3, 3, (N, 3)).astype(np.float32)
    pos = np.c_[P0, np.ones(N, np.float32)]                            # invMass = 1 -> mass = 1
    vel = rng.normal(0, 2.0, (N, 3)).astype(np.float32)               # isotropic -> nonzero net L
    q = np.tile([0, 0, 0, 1], (N, 1)).astype(np.float32)
    s.set_positions(pos); s.set_velocities(vel); s.set_quaternions(q)
    s.set_angular_velocities(np.zeros((N, 3), np.float32)); s.set_scales(np.ones(N, np.float32))
    invI = _np(s.get_inv_inertia())[:N, :3]
    Iloc = np.where(invI > 0, 1.0 / np.maximum(invI, 1e-30), 0.0)      # local principal inertias
    m = 1.0

    def PL():
        X = _np(s.get_positions())[:N, :3].astype(np.float64)
        V = _np(s.get_velocities())[:N].astype(np.float64)
        W = _np(s.get_angular_velocities())[:N].astype(np.float64)
        Q = _np(s.get_quaternions())[:N].astype(np.float64)
        R = Rmat(Q)
        wl = np.einsum('nji,nj->ni', R, W)                  # Rᵀ ω  (body frame)
        Lspin = np.einsum('nij,nj->ni', R, Iloc * wl)       # R (I_local · Rᵀω)
        Lorb = m * np.cross(X, V)
        Li = Lorb + Lspin
        # characteristic angular-momentum "content" (scale to normalise the net-drift against)
        scale = np.linalg.norm(Li, axis=1).sum()
        return (m * V).sum(0), Li.sum(0), scale

    P_init, L_init, _ = PL()
    Pscale = np.abs(m * vel).sum()
    spun = 0.0; Lscale = 1e-30
    for i in range(500):
        s.step(dt)
        if i == 250:
            spun = np.abs(_np(s.get_angular_velocities())[:N]).max()   # confirm spin was imparted
        _, _, sc = PL(); Lscale = max(Lscale, sc)
    P_fin, L_fin, _ = PL()
    Pdrift = np.linalg.norm(P_fin - P_init) / max(Pscale, 1e-30)
    Ldrift = np.linalg.norm(L_fin - L_init) / Lscale        # net drift vs total L content
    return Pdrift, Ldrift, spun

# ------------------------------------------------------------- C. dense-gas cooling vs Enskog/Haff
def test_cooling(use_gs=True):
    """Homogeneous cooling of a moderately dense granular gas (φ≈0.20). Haff/Enskog fixes the cooling
    slope from (e, density); the count-averaged solve under-dissipates in multi-contact -> shallower
    slope (effective e too high). Returns measured_slope / enskog_slope over the early window."""
    L = 20.0; rp = 0.5; e = 0.8; T0 = 1.0; dt = 0.02
    Vp = (4 / 3) * np.pi * rp ** 3
    N = int(round(0.35 * L ** 3 / Vp)); phi = N * Vp / L ** 3
    n_den = N / L ** 3; g0 = (1 - phi / 2) / (1 - phi) ** 3
    enskog = (1 - e ** 2) / 3 * 2 * np.sqrt(np.pi) * n_den * (2 * rp) ** 2 * g0 * np.sqrt(T0)
    rng = np.random.default_rng(3)
    Pp = rng.uniform(0, L, (N, 3)).astype(np.float32)
    v = rng.normal(0, np.sqrt(T0), (N, 3)).astype(np.float32); v -= v.mean(0)
    s = dem.Simulation(N + 64); s.initialize(shape_type=1, radius=rp); s.set_sphere_shape(rp)
    s.set_domain((0, 0, 0), (L, L, L)); s.enable_periodicity(True, True, True)
    s.set_gravity(0, 0, 0); s.set_material_params(e, 0.0, 0.0); s.set_solver_iterations(6, 8)
    s.set_dt(dt); s.set_velocity_use_gs(use_gs)
    s.set_positions(np.c_[Pp, np.ones(N, np.float32)]); s.set_velocities(v)
    def gT(V): vp = V - V.mean(0); return float((vp * vp).sum(1).mean() / 3.0)
    ts, Tr = [0.0], [1.0]
    for i in range(400):
        s.step(dt); t = (i + 1) * dt
        if (i + 1) % 10 == 0:
            Tr.append(gT(_np(s.get_velocities())[:N]) / T0); ts.append(t)
    ts, Tr = np.array(ts), np.array(Tr)
    # Haff: 1/sqrt(T/T0) = 1 + (slope/2) t  -> slope = 2 * d(1/sqrt(Tr))/dt over the linear window
    y = 1.0 / np.sqrt(np.maximum(Tr, 1e-9)); w = ts < 3.0
    meas = 2.0 * np.polyfit(ts[w], y[w], 1)[0]
    return meas / enskog, phi, N

# ---------------------------------------------------------------------------------------- run all
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 78)
    print("A. BINARY RESTITUTION EXACTNESS (colored GS, count==1 must be exact)")
    ok = True
    for e in [0.2, 0.5, 0.8, 0.95]:
        eff, pd = test_binary(e, use_gs=True)
        err = abs(eff - e)
        ok &= err < 0.03 and pd < 1e-4
        print(f"   e={e:.2f}:  e_eff={eff:.4f}   |err|={err:.4f}   momentum_drift={pd:.2e}   "
              f"{'OK' if err < 0.03 else 'FAIL'}")

    print("\nB. MOMENTUM + ANGULAR-MOMENTUM CONSERVATION (free cylinder cloud)")
    for lbl, gs in [("colored GS ", True), ("Jacobi(avg)", False)]:
        pd, ld, spun = test_conservation(use_gs=gs)
        print(f"   {lbl}:  |ΔP|/|P|={pd:.2e}   |ΔL|/|L|={ld:.2e}   max|ω| imparted={spun:.2f}")

    print("\nC. DENSE-GAS COOLING SLOPE / ENSKOG  (1.0 = correct; <1 = under-dissipating)")
    for lbl, gs in [("colored GS ", True), ("Jacobi(avg)", False)]:
        ratio, phi, N = test_cooling(use_gs=gs)
        print(f"   {lbl}:  measured/Enskog = {ratio:.3f}    (φ={phi:.2f}, N={N})")

    print("=" * 78)
    print(f"binary-exactness {'PASS' if ok else 'FAIL'}   ({time.time()-t0:.0f}s)")
