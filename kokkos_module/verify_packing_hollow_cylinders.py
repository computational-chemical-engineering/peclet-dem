#!/usr/bin/env python3
# Hollow-cylinder (ring) shapes for the Kokkos DEM port: validates analytic-shape shell generation +
# inertia + narrow-phase contacts, and checks Kokkos-vs-CUDA parity.
#
# The CUDA demgpu and Kokkos demgpu_kokkos modules each own Kokkos/CUDA init and CANNOT co-import, so
# parity is run per-process: this script re-execs itself as `--backend cuda` and `--backend kokkos`
# subprocesses, each running the IDENTICAL deterministic ring scenario and emitting metrics as JSON;
# the driver then compares them. A faithful port must match to roundoff (positions/overlap), with the
# integer contact/manifold counts identical.
#
# Run:  PYTHONPATH=build_module:build  python kokkos_module/verify_packing_hollow_cylinders.py
import sys, os, json, subprocess, gc
import numpy as np

# Ring geometry (outer radius 1 so global_scale==base radius), domain interior, periodicity OFF so the
# absolute coordinates need no shared origin between the two domain conventions.
RADIUS, HEIGHT, THICK = 1.0, 0.6, 0.3
N_SIDE = 4                       # 4^3 = 64 rings
SPACING = 1.7                    # < 2*RADIUS so neighbours overlap -> real contacts
STEPS = 8
DT = 1e-3
POS_ITERS, VEL_ITERS = 10, 4
RESTITUTION, FRICTION = 1.0, 0.0


def make_scenario():
    """Deterministic initial state shared by both backends (positions, normalized quaternions)."""
    rng = np.random.default_rng(7)
    xs = np.arange(N_SIDE) * SPACING
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    pos += rng.normal(0.0, 0.05, pos.shape).astype(np.float32)  # small jitter
    quat = rng.normal(0.0, 1.0, (pos.shape[0], 4)).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return pos, quat


def run_cuda():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
    import demgpu
    pos, quat = make_scenario()
    n = pos.shape[0]
    sim = demgpu.Simulation(n)
    sim.initialize(shape_type=2, radius=RADIUS, height=HEIGHT, thickness=THICK)
    L = N_SIDE * SPACING + 10.0
    sim.set_domain((-L, -L, -L), (L, L, L))
    sim.enable_periodicity(False, False, False)
    sim.set_gravity(0, 0, 0)
    sim.set_global_scale(1.0)
    sim.set_material_params(RESTITUTION, RESTITUTION, FRICTION)
    sim.set_solver_iterations(POS_ITERS, VEL_ITERS)
    pos4 = np.concatenate([pos, np.ones((n, 1), np.float32)], axis=1)
    sim.set_positions(pos4)
    sim.set_quaternions(quat)
    for _ in range(STEPS):
        sim.step(DT)
    out = collect(sim.get_positions(), sim.get_num_contacts(), sim.get_num_manifolds(),
                  sim.get_max_overlap(), sim.get_inv_inertia(), pos)
    return out


def run_kokkos():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build_module"))
    import demgpu_kokkos as dem
    pos, quat = make_scenario()
    n = pos.shape[0]
    sim = dem.Simulation(n * 8)
    sim.initialize_shape(2, RADIUS, HEIGHT, THICK)
    L = N_SIDE * SPACING + 10.0
    sim.set_domain(L, L, L, False, False, False)
    sim.set_gravity(0, 0, 0)
    sim.set_global_scale(1.0)
    sim.set_dt(DT)
    sim.set_material_params(RESTITUTION, RESTITUTION, FRICTION)
    sim.set_solver_iterations(POS_ITERS, VEL_ITERS)
    sim.set_positions(pos)
    sim.set_quaternions(quat)
    sim.step(STEPS)
    out = collect(sim.get_positions(), sim.num_contacts(), sim.num_manifolds(),
                  sim.max_overlap(), sim.get_inv_inertia(), pos)
    del sim
    gc.collect()
    return out


def collect(final_pos, ncontacts, nmanifolds, max_ov, inv_inertia, start_pos):
    final_pos = np.asarray(final_pos, np.float32)
    disp = np.linalg.norm(final_pos - start_pos, axis=1)
    return {
        "n": int(final_pos.shape[0]),
        "num_contacts": int(ncontacts),
        "num_manifolds": int(nmanifolds),
        "max_overlap": float(max_ov),
        "inv_inertia0": [float(v) for v in np.asarray(inv_inertia)[0]],
        "mean_disp": float(disp.mean()),
        "max_disp": float(disp.max()),
        "final_pos": final_pos.tolist(),
    }


def driver():
    here = os.path.abspath(__file__)
    res = {}
    for backend in ("cuda", "kokkos"):
        print(f"[driver] running {backend} subprocess ...")
        p = subprocess.run([sys.executable, here, "--backend", backend],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(p.stdout); print(p.stderr, file=sys.stderr)
            print(f"FAIL: {backend} subprocess produced no result"); return 1
        res[backend] = json.loads(line[len("RESULT "):])

    c, k = res["cuda"], res["kokkos"]
    print("\n  metric            CUDA            Kokkos")
    print(f"  num_contacts      {c['num_contacts']:<15} {k['num_contacts']}")
    print(f"  num_manifolds     {c['num_manifolds']:<15} {k['num_manifolds']}")
    print(f"  max_overlap       {c['max_overlap']:<15.6f} {k['max_overlap']:.6f}")
    print(f"  inv_inertia[0]    {c['inv_inertia0']}  {k['inv_inertia0']}")
    print(f"  mean_disp         {c['mean_disp']:<15.6e} {k['mean_disp']:.6e}")
    print(f"  max_disp          {c['max_disp']:<15.6e} {k['max_disp']:.6e}")

    cp, kp = np.array(c["final_pos"], np.float64), np.array(k["final_pos"], np.float64)
    pos_l2 = np.linalg.norm(cp - kp, axis=1)
    print(f"  per-particle |Δpos| vs CUDA: mean={pos_l2.mean():.3e}  max={pos_l2.max():.3e}")

    ok = True
    # Inertia is closed-form -> must be exact.
    if not np.allclose(c["inv_inertia0"], k["inv_inertia0"], rtol=1e-5, atol=1e-6):
        print("FAIL: inverse inertia mismatch"); ok = False
    # The shells/broadphase produce the same geometry -> identical contact & manifold counts.
    if c["num_contacts"] != k["num_contacts"]:
        print("FAIL: contact count mismatch"); ok = False
    if c["num_manifolds"] != k["num_manifolds"]:
        print("FAIL: manifold count mismatch"); ok = False
    # Positions/overlap: faithful port, roundoff + atomic-ordering only.
    if abs(c["max_overlap"] - k["max_overlap"]) > 2e-3:
        print("FAIL: max_overlap differs beyond roundoff"); ok = False
    if pos_l2.max() > 5e-3:
        print("FAIL: final positions diverge beyond roundoff"); ok = False
    # Physical sanity: overlapping rings actually moved and contacts were found.
    if k["num_contacts"] == 0 or k["mean_disp"] < 1e-4:
        print("FAIL: scenario is trivial (no contacts / no motion)"); ok = False

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--backend" in sys.argv:
        backend = sys.argv[sys.argv.index("--backend") + 1]
        out = run_cuda() if backend == "cuda" else run_kokkos()
        print("RESULT " + json.dumps(out))
        sys.exit(0)
    sys.exit(driver())
