"""Validate periodic-wrap handling of the MPI-aware dem step.

A periodic axis only works distributed if it is split across >=2 ranks: a rank never ghosts to
itself, so a single rank spanning a periodic axis loses the wrap. ORB splits axis 0 at np=2 and
axes 0,1 at np=4, so this test makes exactly the split axes periodic (via the transport-core halo)
and walls the rest, then compares against a serial reference that uses dem's *internal*
periodicity. Agreement means the halo's periodic image ghosts reproduce true periodic interactions.

  serial      : one Simulation, enable_periodicity(*split) + walls on the unsplit axes.
  distributed : per-block non-periodic + walls; mpi_init(periodic=split) so the halo supplies the
                wrap-ghosts; migration wraps positions on the periodic axes.

Run: PYTHONPATH=build_sm120:../transport-core/python/build mpirun -np 2 python3 mpi/validate_periodic.py
"""
import os
import sys
import numpy as np
from mpi4py import MPI
import dem
import tpx_mpi

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size
_loc = comm.Split_type(MPI.COMM_TYPE_SHARED)
_nd = dem.Simulation.cuda_device_count()
if _nd > 0:
    dem.Simulation.set_cuda_device(_loc.rank % _nd)

dmin = [0.0, 0.0, 0.0]
L = [8.0, 8.0, 8.0]
radius = 0.4
rcut = 2.0 * radius
dt = 0.002
gs = [16, 16, 16]
N = 200
nsteps = 15

# which axes does ORB split? -> those are the periodic ones (others get walls).
_m = tpx_mpi.Migrator(origin=dmin, size=L, gsize=gs, periodic=[True, True, True])
split = []
for ax in range(3):
    lo = [4.0, 4.0, 4.0]; lo[ax] = 1.0
    hi = [4.0, 4.0, 4.0]; hi[ax] = 7.0
    if _m.owner_of(tuple(lo)) != _m.owner_of(tuple(hi)):
        split.append(ax)
periodic = tuple(ax in split for ax in range(3))
wall_axes = [ax for ax in range(3) if ax not in split]


def wall_planes():
    planes = []
    for ax in wall_axes:
        n0 = [0.0, 0.0, 0.0]; n0[ax] = 1.0
        planes.append(((0.0, 0.0, 0.0), tuple(n0)))
        nL = [0.0, 0.0, 0.0]; nL[ax] = -1.0
        pt = [0.0, 0.0, 0.0]; pt[ax] = L[ax]
        planes.append((tuple(pt), tuple(nL)))
    return planes


def make_sim(n, dist):
    s = dem.Simulation(num_particles=int(n))
    if dist:
        # per-block solver: non-periodic, domain padded so wrap-image ghosts (just outside [0,L]) fit.
        m = rcut + 0.5
        s.set_domain((dmin[0] - m, dmin[1] - m, dmin[2] - m), (L[0] + m, L[1] + m, L[2] + m))
        s.enable_periodicity(False, False, False)
    else:
        # serial reference: TRUE [0,L] domain so dem's internal periodicity wraps at the right box.
        s.set_domain((dmin[0], dmin[1], dmin[2]), (L[0], L[1], L[2]))
        s.enable_periodicity(*periodic)
    s.initialize(shape_type=1, radius=radius)
    s.set_material_params(1.0, 0.0, 0.0)
    s.set_solver_iterations(8, 4)
    s.set_gravity(0.0, 0.0, 0.0)
    for pt, nr in wall_planes():
        s.add_plane(pt, nr)
    if dist:
        s.mpi_init(origin=tuple(dmin), size=tuple(L), gsize=tuple(gs), periodic=periodic)
        s.enable_mpi_step(rcut, 1, False)
    return s


def pdiff(A, B):
    d = A - B
    for ax in range(3):
        if periodic[ax]:
            d[:, ax] = (d[:, ax] + L[ax] / 2) % L[ax] - L[ax] / 2
    return np.abs(d)


def run(g_pos, g_vel, nsteps):
    """Run one IC serially (rank 0, all N) and distributed; return (serial_pos, dist_pos) on rank 0."""
    ref = None
    if rank == 0:
        s = make_sim(g_pos.shape[0], dist=False)
        s.set_positions(g_pos.astype(np.float32))
        s.set_velocities(g_vel.astype(np.float32))
        for _ in range(nsteps):
            s.step(dt)
        ref = np.array(s.get_positions(False))
    mig = tpx_mpi.Migrator(origin=dmin, size=L, gsize=gs, periodic=list(periodic))
    ids = np.arange(g_pos.shape[0])
    own = np.array([mig.owner_of(tuple(p)) for p in g_pos])
    mine = np.where(own == rank)[0]
    pos, vel, idd = g_pos[mine].copy(), g_vel[mine].copy(), ids[mine].astype(np.float64)
    for _ in range(nsteps):
        pay = np.column_stack([vel, idd]) if pos.shape[0] else np.zeros((0, 4))
        pos, pay = mig.migrate(pos, pay)
        vel, idd = pay[:, 0:3].copy(), pay[:, 3].copy()
        n = pos.shape[0]
        if n == 0:
            raise RuntimeError(f"rank {rank} owns 0 particles -- adjust N/np")
        s = make_sim(n, dist=True)
        s.set_positions(pos.astype(np.float32))
        s.set_velocities(vel.astype(np.float32))
        s.step(dt)
        pos = np.array(s.get_positions(False))[:n].astype(np.float64)
        vel = np.array(s.get_velocities())[:n].astype(np.float64)
    allp, alli = comm.gather(pos, 0), comm.gather(idd, 0)
    if rank != 0:
        return None, None
    D = np.full((g_pos.shape[0], 3), np.nan)
    for p, i in zip(allp, alli):
        for k, idv in enumerate(i):
            D[int(idv)] = p[k]
    assert np.isfinite(D).all(), "missing ids"
    return ref, D


# ---- Test 1: deterministic 2-body through the FIRST periodic axis (decisive wrap proof) ----
# Needs <=2 ranks so both particles land on (different) non-empty ranks; skipped for np>2.
ax = split[0] if split else 0
if split and size <= 2:
    a, b = [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]
    a[ax], b[ax] = 0.3, L[ax] - 0.3  # 0.6 apart through the wrap (< diameter 0.8 => must repel)
    tp = np.array([a, b])
    tv = np.zeros((2, 3))
    rs, rd = run(tp, tv, 25)
    if rank == 0:
        def wrap_sep(P):
            d = P[0] - P[1]
            d[ax] = (d[ax] + L[ax] / 2) % L[ax] - L[ax] / 2
            return float(np.linalg.norm(d))
        sep_serial = wrap_sep(rs)
        sep_dist = wrap_sep(rd)
        # both should push apart to ~diameter (0.8); if the distributed wrap were broken they'd stay ~0.6
        ok2 = (sep_dist > 0.7) and (abs(sep_serial - sep_dist) < 0.05)
        print(f"np={size}: [2-body wrap, axis {'xyz'[ax]}] start sep 0.6 -> serial {sep_serial:.4f}, "
              f"dist {sep_dist:.4f} (diameter 0.8)  {'OK' if ok2 else 'FAIL (wrap not resolved!)'}")

# ---- Test 1.5: deterministic CORNER (diagonal two-axis wrap). Needs the 2x2 split (np==4); the two
# anchors keep the other two quadrant-ranks non-empty. ----
if len(split) >= 2 and size == 4:
    a0, a1 = split[0], split[1]
    P = np.array([[4.0, 4.0, 4.0] for _ in range(4)])
    P[0, a0], P[0, a1] = 0.25, 0.25                 # corner (lo,lo)
    P[1, a0], P[1, a1] = L[a0] - 0.25, L[a1] - 0.25  # corner (hi,hi): diag-wrap partner of P0, 0.707 apart
    P[2, a0], P[2, a1] = L[a0] * 0.25, L[a1] * 0.75  # anchor (lo,hi), isolated
    P[3, a0], P[3, a1] = L[a0] * 0.75, L[a1] * 0.25  # anchor (hi,lo), isolated
    rs, rd = run(P, np.zeros((4, 3)), 25)
    if rank == 0:
        def diag_sep(Q):
            d = Q[0] - Q[1]
            for a in (a0, a1):
                d[a] = (d[a] + L[a] / 2) % L[a] - L[a] / 2
            return float(np.linalg.norm(d))
        ss, sd = diag_sep(rs), diag_sep(rd)
        okc = (sd > 0.7) and abs(ss - sd) < 0.05
        print(f"np={size}: [corner wrap, axes {'xyz'[a0]}{'xyz'[a1]}] start diag sep 0.707 -> "
              f"serial {ss:.4f}, dist {sd:.4f} (diameter 0.8)  {'OK' if okc else 'FAIL'}")

rng = np.random.RandomState(2)
# NON-overlapping jittered grid bulk (spacing 1.0 > diameter 0.8): no stiff initial overlaps, so the
# per-particle serial-vs-distributed comparison is clean (stiff overlaps are chaotically sensitive and
# would mask the wrap signal). Velocities let bulk particles collide during the run.
_g = np.arange(0.7, 7.4, 1.0)
bulk = np.array([[x, y, z] for x in _g for y in _g for z in _g]) + rng.uniform(-0.06, 0.06, (len(_g) ** 3, 3))
bvel = rng.normal(0.0, 6.0, size=(bulk.shape[0], 3))  # fast enough to actually collide during the run


def straddler_pairs(axp):
    # 4 pairs that overlap ONLY through the periodic boundary on axis `axp` (0.5 apart through the
    # wrap < diameter 0.8 => must repel through the wrap). Spread on the other axes so pairs don't
    # collide with each other; if the distributed wrap is broken these disagree strongly with serial.
    others = [a for a in range(3) if a != axp]
    out = []
    for i, g in enumerate(np.linspace(1.2, 6.8, 4)):
        lo = [0.0, 0.0, 0.0]; hi = [0.0, 0.0, 0.0]
        lo[axp], hi[axp] = 0.30, L[axp] - 0.45  # 0.75 apart through the wrap: gentle overlap (0.05)
        lo[others[0]] = hi[others[0]] = g
        lo[others[1]] = hi[others[1]] = 1.5 + (i % 3) * 2.2
        out += [lo, hi]
    return np.array(out)


# ... plus deterministic straddler pairs on every periodic axis (gentle wrap contacts). Drop any bulk
# grid point within a diameter of a straddler so the straddlers don't create spurious stiff overlaps.
extra = [straddler_pairs(a) for a in split]
if extra:
    strad = np.vstack(extra)
    far = np.all(np.linalg.norm(bulk[:, None, :] - strad[None, :, :], axis=-1) > 0.85, axis=1)
    bulk, bvel = bulk[far], bvel[far]
    g_pos = np.vstack([bulk, strad])
    g_vel = np.vstack([bvel, np.zeros((strad.shape[0], 3))])
else:
    g_pos, g_vel = bulk, bvel
N = g_pos.shape[0]

nsteps = int(os.environ.get("NSTEPS", "20"))
ref, D = run(g_pos, g_vel, nsteps)

if rank == 0:
    # sanity: confirm the periodic wrap is actually exercised -- count pairs that are within rcut
    # ONLY through the periodic wrap (min-image separation uses the wrap, not the direct gap).
    # Checked on the INITIAL config (the straddlers separate to ~rcut by the final frame).
    wrap_pairs = 0
    if split:
        P = g_pos
        for i in range(N):
            for j in range(i + 1, N):
                d = P[i] - P[j]
                dm = d.copy()
                for a in range(3):
                    if periodic[a]:
                        dm[a] = (dm[a] + L[a] / 2) % L[a] - L[a] / 2
                if np.linalg.norm(dm) < rcut and np.linalg.norm(d) >= rcut:
                    wrap_pairs += 1
    err = pdiff(D, ref)
    maxe, meane = float(err.max()), float(err.mean())
    ok = (maxe < 5e-2) and (not split or wrap_pairs > 0)
    pax = "".join("xyz"[a] for a in split) or "(none)"
    print(f"np={size}: [N-body] periodic={pax} walls={['xyz'[a] for a in wall_axes]}  "
          f"wrap-only contact pairs={wrap_pairs}  max|dist-serial|={maxe:.3e} mean={meane:.3e}  "
          f"({'OK' if ok else 'CHECK'}) over {nsteps} steps, N={N}")
    sys.exit(0 if ok else 1)
