"""Byte-exactness gate for structural refactors: SHA-256 of the final state of fixed-seed runs.

One case per public entry path of ``peclet.dem`` (QUALITY_PLAN §3.G): XPBD ``step`` with island
sleeping on and off (periodic box with wrap contacts, a floor plane, gravity, friction), ``relax``,
``step_hertz`` (Hertz-Mindlin with Mindlin history against an SDF floor), a non-spherical grid-SDF
shape, and -- when mpi4py + ``peclet.core.mpi`` are importable -- ``step_mpi`` and
``step_hertz_mpi`` at np=1 and np=2 (launched through ``mpirun``, the state gathered to rank 0 and
sorted by the carried particle id). Every hash is over the contiguous float64 bytes of the final
positions, velocities and quaternions.

Run at ONE thread: dem's XPBD path is not run-to-run reproducible at 4 threads (float atomics in
the delta accumulation), it is at 1. The script pins ``OMP_NUM_THREADS=1`` before Kokkos starts.

    PYTHONPATH=<dem-build>:<core-python-build> python tests/regression/state_hash.py --save ref/
    ... refactor, rebuild ...
    PYTHONPATH=<dem-build>:<core-python-build> python tests/regression/state_hash.py --check ref/

``--save DIR`` writes ``DIR/<case>.npz`` (the arrays) and ``DIR/hashes.json``; ``--check DIR``
re-runs every case and exits 1 if any hash differs from ``hashes.json`` (cases missing from the
reference are reported, not failed). ``--cases a,b`` restricts the run; ``--no-mpi`` skips the
mpirun cases. Run twice with ``--save`` into two directories and diff the JSON to confirm a case is
reproducible before trusting it as a gate.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

os.environ["OMP_NUM_THREADS"] = "1"  # before Kokkos initialises (module import)
os.environ.setdefault("OMP_PROC_BIND", "false")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SINGLE_CASES = ["xpbd_sleep_on", "xpbd_sleep_off", "relax", "hertz", "sdf_shape"]
MPI_CASES = ["mpi_xpbd", "mpi_hertz"]
MPI_NP = [1, 2]

L = 8.0
RADIUS = 0.5


def state_hash(pos, vel, quat):
    h = hashlib.sha256()
    for a in (pos, vel, quat):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def _final(s, n=None):
    pos = np.asarray(s.get_positions(), dtype=np.float64).reshape(-1, 3)
    vel = np.asarray(s.get_velocities(), dtype=np.float64).reshape(-1, 3)
    quat = np.asarray(s.get_quaternions(), dtype=np.float64).reshape(-1, 4)
    if n is not None:
        pos, vel, quat = pos[:n], vel[:n], quat[:n]
    return pos, vel, quat


def _jittered_grid(seed, lo=0.3, hi=None, spacing=1.05, jitter=0.08, zmax=None):
    """Cubic grid starting AT lo so the first column straddles the x=0/y=0 faces (wrap contacts),
    jittered so the wrap pairs are asymmetric about the face."""
    hi = L - 0.2 if hi is None else hi
    ax = np.arange(lo, hi, spacing)
    az = ax if zmax is None else ax[ax < zmax]
    pts = np.array([[x, y, z] for x in ax for y in ax for z in az], dtype=np.float64)
    rng = np.random.RandomState(seed)
    return pts + rng.uniform(-jitter, jitter, pts.shape)


def _hertz_bed(seed):
    """Two layers resting on the floor, the upper in the hollows of the lower: persistent pair
    contacts with friction, so the Mindlin history is carried across the Verlet rebuilds that the
    settling triggers (that carry is what the gid-keyed ledger search does)."""
    ax = np.arange(1.0, L - 0.9, 1.0)
    lower = np.array([[x, y, RADIUS] for x in ax for y in ax])
    upper = np.array([[x + 0.5, y + 0.5, RADIUS + 0.76] for x in ax[:-1] for y in ax[:-1]])
    pos = np.vstack([lower, upper])
    vel = np.random.RandomState(seed).normal(0.0, 0.2, pos.shape)
    return pos, vel


def _xpbd_periodic_bed(sleeping):
    from peclet import dem
    pos = _jittered_grid(seed=11, zmax=4.5)
    n = pos.shape[0]
    s = dem.Simulation(n)
    s.initialize_shape('sphere', radius=RADIUS)
    s.set_domain((0.0, 0.0, 0.0), (L, L, L))
    s.set_periodic(True, True, False)
    s.set_gravity((0.0, 0.0, -9.81))
    s.set_material_params(0.2, 0.0, 0.3)
    s.set_solver_iterations(8, 4)
    s.set_sleeping(sleeping)
    s.add_plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    s.set_dt(0.002)
    s.set_positions(pos.astype(np.float32))
    s.set_velocities(np.random.RandomState(12).normal(0.0, 0.5, (n, 3)).astype(np.float32))
    s.step(600)
    return _final(s)


def case_xpbd_sleep_on():
    return _xpbd_periodic_bed(True)


def case_xpbd_sleep_off():
    return _xpbd_periodic_bed(False)


def case_relax():
    from peclet import dem
    pos = _jittered_grid(seed=21, spacing=0.9, jitter=0.15)
    n = pos.shape[0]
    s = dem.Simulation(n)
    s.initialize_shape('sphere', radius=RADIUS)
    s.set_domain((0.0, 0.0, 0.0), (L, L, L))
    s.set_periodic(True, True, True)
    s.set_gravity((0.0, 0.0, 0.0))
    s.set_material_params(0.0, 0.0, 0.0)
    s.set_solver_iterations(12, 0)
    s.set_positions(pos.astype(np.float32))
    s.relax(40)
    return _final(s)


def case_hertz():
    from peclet import dem
    from peclet.dem import build_wall_sdf
    pos, vel = _hertz_bed(seed=31)
    n = pos.shape[0]
    s = dem.Simulation(n)
    s.initialize_shape('sphere', RADIUS)
    s.set_domain((0.0, 0.0, -1.0), (L, L, L))
    s.set_periodic(False, False, False)
    s.set_material_params(0.3, 0.0, 0.4)
    s.set_hertz_material(0, 1.0e7, 0.25)
    s.set_thermostat(0, 0)
    wall = build_wall_sdf(lambda p: p[:, 2], ((0, 0, -1.0), (L, L, L)), resolution=(48, 48, 48))
    wall.add_to(s, restitution=0.3, friction=0.4)
    s.set_positions(pos.astype(np.float32))
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(n, np.float32))
    s.set_inv_inertia(np.full((n, 3), 1.0 / (0.4 * RADIUS * RADIUS), np.float32))
    s.set_velocities(vel.astype(np.float32))
    s.set_gravity((0.0, 0.0, -9.81))
    s.set_dt(5e-5)
    s.step_hertz(6000)
    return _final(s)


def case_sdf_shape():
    from peclet import dem
    from peclet.dem import build_particle

    def rounded_box(p, half=0.4, r=0.1):
        q = np.abs(p) - half
        return (np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)
                - r)

    sp = build_particle(rounded_box, ((-0.65, -0.65, -0.65), (0.65, 0.65, 0.65)), resolution=48)
    N = 24
    s = dem.Simulation(N)
    sp.apply_to(s)
    half = 2.0
    s.set_domain((-half, -half, -half), (half, half, half))
    s.set_periodic(True, True, True)
    s.set_gravity((0.0, 0.0, 0.0))
    s.set_material_params(0.0, 0.0, 0.0)
    s.set_solver_iterations(20, 0)
    rng = np.random.RandomState(41)
    pos = rng.uniform(-half, half, (N, 3)).astype(np.float32)
    s.set_positions(pos)
    s.set_velocities(np.zeros((N, 3), np.float32))
    q = rng.normal(0, 1, (N, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    s.set_quaternions(q)
    s.set_angular_velocities(np.zeros((N, 3), np.float32))
    s.set_dt(0.01)
    s.step(40)
    return _final(s)


# ---- MPI cases (run inside `mpirun -np N python state_hash.py --mpi-case NAME`) -----------------

def _mpi_distribute(comm, g_pos, g_vel, periodic):
    from peclet.core import mpi as core_mpi
    mig = core_mpi.ParticleMigrator(origin=[0.0] * 3, extent=[L] * 3, cells=[16] * 3,
                                    periodic=list(periodic))
    own = np.array([mig.owner_of(tuple(p)) for p in g_pos])
    mine = np.where(own == comm.rank)[0]
    assert mine.size > 0, f"rank {comm.rank} owns no particles"
    return g_pos[mine].copy(), g_vel[mine].copy(), mine.astype(np.int64)


def _mpi_gather_sorted(comm, s, n, ids):
    pos, vel, quat = _final(s, n)
    allp, allv, allq, alli = (comm.gather(pos, 0), comm.gather(vel, 0), comm.gather(quat, 0),
                              comm.gather(ids, 0))
    if comm.rank != 0:
        return None
    P, V, Q, I = (np.vstack(allp), np.vstack(allv), np.vstack(allq), np.concatenate(alli))
    order = np.argsort(I, kind="stable")
    assert np.array_equal(I[order], np.arange(I.size)), "ids are not a permutation"
    return P[order], V[order], Q[order]


def case_mpi_xpbd(comm):
    """Periodic on x (the axis ORB splits at np=2; local self-ghosts at np=1), walls on y/z."""
    from peclet import dem
    periodic = (True, False, False)
    g_pos = _jittered_grid(seed=51, lo=0.35, hi=L - 0.5, spacing=1.05, jitter=0.1)
    g_pos[:, 1:] = np.clip(g_pos[:, 1:], RADIUS + 0.05, L - RADIUS - 0.05)
    g_vel = np.random.RandomState(52).normal(0.0, 3.0, g_pos.shape)
    pos, vel, ids = _mpi_distribute(comm, g_pos, g_vel, periodic)
    n = pos.shape[0]
    rcut = 2.0 * RADIUS
    s = dem.Simulation(2 * n + 64)
    m = rcut + 0.5
    s.set_domain((-m, -m, -m), (L + m, L + m, L + m))
    s.set_periodic(False, False, False)
    s.initialize_shape('sphere', radius=RADIUS)
    s.set_material_params(0.5, 0.0, 0.2)
    s.set_solver_iterations(8, 4)
    s.set_gravity((0.0, 0.0, 0.0))
    s.set_dt(0.002)
    for ax in (1, 2):
        n0 = [0.0, 0.0, 0.0]
        n0[ax] = 1.0
        s.add_plane((0.0, 0.0, 0.0), tuple(n0))
        nL = [0.0, 0.0, 0.0]
        nL[ax] = -1.0
        pt = [0.0, 0.0, 0.0]
        pt[ax] = L
        s.add_plane(tuple(pt), tuple(nL))
    s.set_positions(pos.astype(np.float32))
    s.set_velocities(vel.astype(np.float32))
    s.init_mpi(origin=(0.0,) * 3, extent=(L,) * 3, cells=(16,) * 3, periodic=periodic)
    s.enable_mpi_step(rcut, 1, False)
    s.step_mpi(20)
    return _mpi_gather_sorted(comm, s, n, ids)


def case_mpi_hertz(comm):
    """Closed box (the force engine is non-periodic), SDF floor, gravity, friction."""
    from peclet import dem
    from peclet.dem import build_wall_sdf
    periodic = (False, False, False)
    g_pos, g_vel = _hertz_bed(seed=61)
    pos, vel, ids = _mpi_distribute(comm, g_pos, g_vel, periodic)
    n = pos.shape[0]
    s = dem.Simulation(2 * n + 64)
    s.set_domain((-1.0, -1.0, -1.0), (L + 1.0, L + 1.0, L + 1.0))
    s.set_periodic(False, False, False)
    s.initialize_shape('sphere', RADIUS)
    s.set_material_params(0.3, 0.0, 0.4)
    s.set_hertz_material(0, 1.0e7, 0.25)
    s.set_thermostat(0, 0)
    wall = build_wall_sdf(lambda p: p[:, 2], ((-1.0, -1.0, -1.0), (L + 1, L + 1, L + 1)),
                          resolution=(48, 48, 48))
    wall.add_to(s, restitution=0.3, friction=0.4)
    s.set_positions(pos.astype(np.float32))
    s.set_scales_uniform(1.0)
    s.set_inv_mass(np.ones(n, np.float32))
    s.set_inv_inertia(np.full((n, 3), 1.0 / (0.4 * RADIUS * RADIUS), np.float32))
    s.set_velocities(vel.astype(np.float32))
    s.set_gravity((0.0, 0.0, -9.81))
    s.set_dt(5e-5)
    s.init_mpi(origin=(0.0,) * 3, extent=(L,) * 3, cells=(16,) * 3, periodic=periodic)
    s.enable_mpi_step(2.0 * RADIUS, 1, True)
    for _ in range(12):
        s.step_hertz_mpi(400, skin_frac=0.1)
    return _mpi_gather_sorted(comm, s, n, ids)


def _run_mpi_child(name):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    out = globals()["case_" + name](comm)
    if comm.rank == 0:
        pos, vel, quat = out
        np.savez(os.environ["STATE_HASH_OUT"], pos=pos, vel=vel, quat=quat)
        print("STATE_HASH", state_hash(pos, vel, quat), pos.shape[0])
    return 0


def _mpi_available():
    try:
        import mpi4py  # noqa: F401
        from peclet import dem
        from peclet.core import mpi as core_mpi  # noqa: F401
        return hasattr(dem.Simulation, "init_mpi")
    except ImportError:
        return False


def run_mpi_case(name, np_, outdir, mpirun):
    outfile = os.path.join(outdir, f"{name}_np{np_}.npz")
    env = dict(os.environ, STATE_HASH_OUT=outfile, OMP_NUM_THREADS="1", OMP_PROC_BIND="false")
    cmd = [mpirun, "-np", str(np_), sys.executable, os.path.abspath(__file__), "--mpi-case", name]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError(f"{name} np={np_} failed (exit {r.returncode})")
    for line in r.stdout.splitlines():
        if line.startswith("STATE_HASH "):
            _, h, n = line.split()
            return h, int(n)
    raise RuntimeError(f"{name} np={np_}: no hash line in output:\n{r.stdout}{r.stderr}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--save", metavar="DIR", help="write <case>.npz + hashes.json into DIR")
    ap.add_argument("--check", metavar="DIR", help="compare against DIR/hashes.json")
    ap.add_argument("--cases", help="comma-separated subset of the cases")
    ap.add_argument("--no-mpi", action="store_true", help="skip the mpirun cases")
    ap.add_argument("--mpirun", default=os.environ.get("MPIRUN", "mpirun"))
    ap.add_argument("--mpi-case", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.mpi_case:
        return _run_mpi_child(args.mpi_case)

    outdir = args.save or args.check or os.path.join(HERE, "state_hash_out")
    os.makedirs(outdir, exist_ok=True)
    wanted = set(args.cases.split(",")) if args.cases else None
    hashes = {}
    for name in SINGLE_CASES:
        if wanted and name not in wanted:
            continue
        pos, vel, quat = globals()["case_" + name]()
        np.savez(os.path.join(outdir, name + ".npz"), pos=pos, vel=vel, quat=quat)
        hashes[name] = state_hash(pos, vel, quat)
        print(f"{name:22s} n={pos.shape[0]:5d}  {hashes[name]}")
    if not args.no_mpi:
        if _mpi_available():
            for name in MPI_CASES:
                for np_ in MPI_NP:
                    key = f"{name}_np{np_}"
                    if wanted and key not in wanted and name not in wanted:
                        continue
                    hashes[key], n = run_mpi_case(name, np_, outdir, args.mpirun)
                    print(f"{key:22s} n={n:5d}  {hashes[key]}")
        else:
            print("mpi cases skipped: mpi4py + peclet.core.mpi + a PECLET_DEM_MPI build needed")

    if args.check:
        with open(os.path.join(outdir, "hashes.json")) as f:
            ref = json.load(f)
        bad, missing = [], []
        for k, v in hashes.items():
            if k not in ref:
                missing.append(k)
            elif ref[k] != v:
                bad.append(k)
        for k in bad:
            print(f"MISMATCH {k}: reference {ref[k]} != {hashes[k]}")
        for k in missing:
            print(f"(no reference for {k})")
        print("state_hash: " + ("ALL IDENTICAL" if not bad else f"{len(bad)} DIFFER"))
        return 1 if bad else 0
    with open(os.path.join(outdir, "hashes.json"), "w") as f:
        json.dump(hashes, f, indent=1, sort_keys=True)
    print(f"wrote {os.path.join(outdir, 'hashes.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
