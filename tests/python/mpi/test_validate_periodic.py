"""Validate periodic-wrap handling of the distributed dem step.

A periodic axis only works distributed if it is split across >=2 ranks: a rank never ghosts to
itself, so a single rank spanning a periodic axis loses the wrap. ORB splits axis 0 at np=2 and
axes 0,1 at np=4, so this test makes exactly the split axes periodic (via the core halo) and walls
the rest, then compares against a serial reference that uses dem's *internal* periodicity.
Agreement means the halo's periodic image ghosts reproduce true periodic interactions.

  serial      : one Simulation, set_periodic(*split) + walls on the unsplit axes.
  distributed : per-block non-periodic + walls; init_mpi(periodic=split) so the halo supplies the
                wrap-ghosts; migration wraps positions on the periodic axes.

Run: PYTHONPATH=<dem-build>:<core-python-build> mpirun -np 2 python tests/python/mpi/test_validate_periodic.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mpi_common import SKIP, script_main, skip_unless_mpi  # noqa: E402

if SKIP is None:
    from mpi4py import MPI
    from peclet import dem
    from peclet.core import mpi as core_mpi

dmin = [0.0, 0.0, 0.0]
L = [8.0, 8.0, 8.0]
radius = 0.4
rcut = 2.0 * radius
dt = 0.002
gs = [16, 16, 16]


class Setup:
    """Which axes ORB splits at this rank count -> those are the periodic ones (others get walls)."""

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank, self.size = self.comm.rank, self.comm.size
        m = core_mpi.ParticleMigrator(origin=dmin, extent=L, cells=gs, periodic=[True, True, True])
        self.split = []
        for ax in range(3):
            lo = [4.0, 4.0, 4.0]; lo[ax] = 1.0
            hi = [4.0, 4.0, 4.0]; hi[ax] = 7.0
            if m.owner_of(tuple(lo)) != m.owner_of(tuple(hi)):
                self.split.append(ax)
        self.periodic = tuple(ax in self.split for ax in range(3))
        self.wall_axes = [ax for ax in range(3) if ax not in self.split]

    def wall_planes(self):
        planes = []
        for ax in self.wall_axes:
            n0 = [0.0, 0.0, 0.0]; n0[ax] = 1.0
            planes.append(((0.0, 0.0, 0.0), tuple(n0)))
            nL = [0.0, 0.0, 0.0]; nL[ax] = -1.0
            pt = [0.0, 0.0, 0.0]; pt[ax] = L[ax]
            planes.append((tuple(pt), tuple(nL)))
        return planes

    def make_sim(self, n, dist):
        s = dem.Simulation(2 * int(n) + 64 if dist else int(n))   # ghost slots on the blocks
        if dist:
            # per-block solver: non-periodic, domain padded so wrap-image ghosts (just outside
            # [0,L]) fit.
            m = rcut + 0.5
            s.set_domain((dmin[0] - m, dmin[1] - m, dmin[2] - m), (L[0] + m, L[1] + m, L[2] + m))
            s.set_periodic(False, False, False)
        else:
            # serial reference: TRUE [0,L] domain so dem's internal periodicity wraps at the box.
            s.set_domain((dmin[0], dmin[1], dmin[2]), (L[0], L[1], L[2]))
            s.set_periodic(*self.periodic)
        s.initialize_shape('sphere', radius=radius)
        s.set_material_params(1.0, 0.0, 0.0)
        s.set_solver_iterations(8, 4)
        s.set_gravity((0.0, 0.0, 0.0))
        s.set_dt(dt)
        for pt, nr in self.wall_planes():
            s.add_plane(pt, nr)
        if dist:
            s.init_mpi(origin=tuple(dmin), extent=tuple(L), cells=tuple(gs), periodic=self.periodic)
            s.enable_mpi_step(rcut, 1, False)
        return s

    def pdiff(self, A, B):
        d = A - B
        for ax in range(3):
            if self.periodic[ax]:
                d[:, ax] = (d[:, ax] + L[ax] / 2) % L[ax] - L[ax] / 2
        return np.abs(d)

    def run(self, g_pos, g_vel, nsteps):
        """Run one IC serially (rank 0, all N) and distributed; (serial_pos, dist_pos) on rank 0."""
        ref = None
        if self.rank == 0:
            s = self.make_sim(g_pos.shape[0], dist=False)
            s.set_positions(g_pos.astype(np.float32))
            s.set_velocities(g_vel.astype(np.float32))
            for _ in range(nsteps):
                s.step()
            ref = np.array(s.get_positions())
        mig = core_mpi.ParticleMigrator(origin=dmin, extent=L, cells=gs, periodic=list(self.periodic))
        ids = np.arange(g_pos.shape[0])
        own = np.array([mig.owner_of(tuple(p)) for p in g_pos])
        mine = np.where(own == self.rank)[0]
        pos, vel, idd = g_pos[mine].copy(), g_vel[mine].copy(), ids[mine].astype(np.float64)
        for _ in range(nsteps):
            pay = np.column_stack([vel, idd]) if pos.shape[0] else np.zeros((0, 4))
            pos, pay = mig.migrate(pos, pay)
            vel, idd = pay[:, 0:3].copy(), pay[:, 3].copy()
            n = pos.shape[0]
            assert n > 0, f"rank {self.rank} owns 0 particles -- adjust N/np"
            s = self.make_sim(n, dist=True)
            s.set_positions(pos.astype(np.float32))
            s.set_velocities(vel.astype(np.float32))
            s.step_mpi(1)
            pos = np.array(s.get_positions())[:n].astype(np.float64)
            vel = np.array(s.get_velocities())[:n].astype(np.float64)
        allp, alli = self.comm.gather(pos, 0), self.comm.gather(idd, 0)
        if self.rank != 0:
            return None, None
        D = np.full((g_pos.shape[0], 3), np.nan)
        for p, i in zip(allp, alli):
            for k, idv in enumerate(i):
                D[int(idv)] = p[k]
        assert np.isfinite(D).all(), "missing ids"
        return ref, D


def _wrap_sep(P, axes):
    d = P[0] - P[1]
    for a in axes:
        d[a] = (d[a] + L[a] / 2) % L[a] - L[a] / 2
    return float(np.linalg.norm(d))


@skip_unless_mpi
def test_two_body_wrap():
    """Deterministic 2-body pair through the FIRST periodic axis (decisive wrap proof). Needs <=2
    ranks so both particles land on (different) non-empty ranks; a no-op at np>2 or np=1."""
    st = Setup()
    if not st.split or st.size > 2:
        print(f"np={st.size}: [2-body wrap] not applicable (split axes {st.split})")
        return
    ax = st.split[0]
    a, b = [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]
    a[ax], b[ax] = 0.3, L[ax] - 0.3  # 0.6 apart through the wrap (< diameter 0.8 => must repel)
    rs, rd = st.run(np.array([a, b]), np.zeros((2, 3)), 25)
    if st.rank == 0:
        sep_serial, sep_dist = _wrap_sep(rs, [ax]), _wrap_sep(rd, [ax])
        print(f"np={st.size}: [2-body wrap, axis {'xyz'[ax]}] start sep 0.6 -> serial {sep_serial:.4f}, "
              f"dist {sep_dist:.4f} (diameter 0.8)")
        # both push apart to ~diameter (0.8); a broken distributed wrap would leave them at ~0.6
        assert sep_dist > 0.7, "wrap not resolved"
        assert abs(sep_serial - sep_dist) < 0.05


@skip_unless_mpi
def test_corner_wrap():
    """Deterministic CORNER (diagonal two-axis wrap). Needs the 2x2 split (np==4); the two anchors
    keep the other two quadrant-ranks non-empty."""
    st = Setup()
    if len(st.split) < 2 or st.size != 4:
        print(f"np={st.size}: [corner wrap] not applicable (split axes {st.split})")
        return
    a0, a1 = st.split[0], st.split[1]
    P = np.array([[4.0, 4.0, 4.0] for _ in range(4)])
    P[0, a0], P[0, a1] = 0.25, 0.25                  # corner (lo,lo)
    P[1, a0], P[1, a1] = L[a0] - 0.25, L[a1] - 0.25  # corner (hi,hi): diag-wrap partner, 0.707 apart
    P[2, a0], P[2, a1] = L[a0] * 0.25, L[a1] * 0.75  # anchor (lo,hi), isolated
    P[3, a0], P[3, a1] = L[a0] * 0.75, L[a1] * 0.25  # anchor (hi,lo), isolated
    rs, rd = st.run(P, np.zeros((4, 3)), 25)
    if st.rank == 0:
        ss, sd = _wrap_sep(rs, (a0, a1)), _wrap_sep(rd, (a0, a1))
        print(f"np={st.size}: [corner wrap, axes {'xyz'[a0]}{'xyz'[a1]}] start diag sep 0.707 -> "
              f"serial {ss:.4f}, dist {sd:.4f} (diameter 0.8)")
        assert sd > 0.7 and abs(ss - sd) < 0.05


def _straddler_pairs(axp):
    # 4 pairs that overlap ONLY through the periodic boundary on axis `axp` (0.75 apart through the
    # wrap < diameter 0.8 => gentle overlap 0.05), SYMMETRIC about the face (both centres within one
    # radius of it). Spread on the other axes so pairs don't collide with each other; if the
    # distributed wrap is broken these disagree strongly with serial.
    # NB the single-rank step ghosts only particles within ONE radius of a periodic face
    # (sim.hpp demStep, `ghostBand = maxRad`): a wrap pair whose farther partner sits beyond that
    # band is still detected, but the whole overlap correction lands on that partner alone
    # (measured 2026-09-08: (0.30, L-0.45) -> serial moves only the L-0.45 body, by 0.05; the
    # distributed step moves both by 0.025). Keep the straddlers symmetric so the reference is the
    # symmetric answer; the asymmetry is a single-rank periodicity limitation, not an MPI one.
    others = [a for a in range(3) if a != axp]
    out = []
    for i, g in enumerate(np.linspace(1.2, 6.8, 4)):
        lo = [0.0, 0.0, 0.0]; hi = [0.0, 0.0, 0.0]
        lo[axp], hi[axp] = 0.375, L[axp] - 0.375
        lo[others[0]] = hi[others[0]] = g
        lo[others[1]] = hi[others[1]] = 1.5 + (i % 3) * 2.2
        out += [lo, hi]
    return np.array(out)


@skip_unless_mpi
def test_nbody_periodic():
    """Jittered non-overlapping bulk (spacing 1.0 > diameter 0.8, so no stiff initial overlaps
    whose chaotic sensitivity would mask the wrap signal) with colliding velocities, plus
    deterministic straddler pairs on every periodic axis: per-particle agreement with serial."""
    st = Setup()
    rng = np.random.RandomState(2)
    _g = np.arange(0.7, 7.4, 1.0)
    bulk = (np.array([[x, y, z] for x in _g for y in _g for z in _g])
            + rng.uniform(-0.06, 0.06, (len(_g) ** 3, 3)))
    bvel = rng.normal(0.0, 6.0, size=(bulk.shape[0], 3))
    extra = [_straddler_pairs(a) for a in st.split]
    if extra:
        strad = np.vstack(extra)
        # keep the bulk > 1.3 from any straddler: at ~6 diameters/s the bulk travels ~0.25 in 20
        # steps, so no bulk grain can reach a straddler (contact at 0.8) and perturb its wrap contact
        far = np.all(np.linalg.norm(bulk[:, None, :] - strad[None, :, :], axis=-1) > 1.3, axis=1)
        bulk, bvel = bulk[far], bvel[far]
        g_pos = np.vstack([bulk, strad])
        g_vel = np.vstack([bvel, np.zeros((strad.shape[0], 3))])
    else:
        g_pos, g_vel = bulk, bvel
    N = g_pos.shape[0]
    nsteps = int(os.environ.get("NSTEPS", "20"))
    ref, D = st.run(g_pos, g_vel, nsteps)
    if st.rank == 0:
        # sanity: the periodic wrap is actually exercised -- count INITIAL pairs within rcut ONLY
        # through the wrap (the straddlers separate to ~rcut by the final frame).
        wrap_pairs = 0
        if st.split:
            for i in range(N):
                for j in range(i + 1, N):
                    d = g_pos[i] - g_pos[j]
                    dm = d.copy()
                    for a in range(3):
                        if st.periodic[a]:
                            dm[a] = (dm[a] + L[a] / 2) % L[a] - L[a] / 2
                    if np.linalg.norm(dm) < rcut and np.linalg.norm(d) >= rcut:
                        wrap_pairs += 1
        err = st.pdiff(D, ref)
        maxe, meane = float(err.max()), float(err.mean())
        pe = err.max(axis=1)
        nstrad = sum(len(e) for e in extra)
        q95 = float(np.quantile(pe, 0.95))
        strad_err = float(pe[-nstrad:].max()) if nstrad else 0.0
        print(f"   per-particle err: q95={q95:.3e}  >2e-2: {int((pe > 2e-2).sum())}/{N}  "
              f"straddlers(last {nstrad}) max={strad_err:.3e}")
        pax = "".join("xyz"[a] for a in st.split) or "(none)"
        print(f"np={st.size}: [N-body] periodic={pax} walls={['xyz'[a] for a in st.wall_axes]}  "
              f"wrap-only contact pairs={wrap_pairs}  max|dist-serial|={maxe:.3e} mean={meane:.3e} "
              f"over {nsteps} steps, N={N}")
        # The bulk collides at ~6 diameters/s through the wrap and inside the box; the finite-
        # iteration PGS is sweep-order dependent (rank-local colouring), so per-particle agreement
        # is statistical: the bulk agrees on average, the resting straddlers -- the decisive wrap
        # contacts -- agree to float noise, and nothing is off by a diameter (a lost wrap contact).
        assert not st.split or wrap_pairs > 0, "the wrap was not exercised"
        assert strad_err < 5e-3, "wrap-contact straddlers disagree with serial"
        assert meane < 5e-3 and q95 < 5e-2
        assert maxe < 0.4, "a particle is off by ~a diameter: a lost contact"


if __name__ == "__main__":
    script_main(test_two_body_wrap, test_corner_wrap, test_nbody_periodic)
