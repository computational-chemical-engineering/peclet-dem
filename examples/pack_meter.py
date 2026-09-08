"""Trustworthy packing-fraction + microstructure meter for the DEM packing engine.

The shipped verify scripts compute phi from the *base* radius and a fixed domain, ignoring the grown
per-particle ``scale`` -- so they report the design phi by construction. This module measures the packing
from the ACTUAL state: effective radii ``r_i = base_radius * scale_i * global_scale`` in the true periodic
box, with three independent diagnostics:

  * ``phi_naive``      -- sum of sphere volumes / box volume (over-counts where spheres overlap).
  * ``phi_corrected``  -- phi_naive minus the pairwise lens (overlap) volumes (minimum-image), i.e. the
                          solid fraction actually occupied; the meaningful density once overlaps are small.
  * ``coordination``   -- mean number of touching/compressed neighbours Z (frictionless RCP -> Z ~ 6).
  * ``max_overlap``    -- deepest pair interpenetration as a fraction of the contact distance.
  * ``gofr``           -- radial distribution function g(r) (the RCP split-second-peak signature).

Neighbour queries use a periodic scipy.spatial.cKDTree, so everything respects the periodic wrap.
Reference: monodisperse frictionless random close packing phi ~ 0.64, Z ~ 6.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def _wrap_to_box(pos, dmin, box):
    """Shift positions into [0, box) per axis (cKDTree's periodic boxsize convention)."""
    return np.mod(pos - dmin, box)


def _lens_volume(d, a, b):
    """Volume of the intersection of two spheres (radii a, b, centre distance d). Zero if disjoint."""
    out = np.zeros_like(d)
    hit = d < (a + b)
    dd, aa, bb = d[hit], a[hit], b[hit]
    # also guard full containment (d < |a-b|): contained sphere volume
    contained = dd <= np.abs(aa - bb)
    rmin = np.minimum(aa, bb)
    vol = np.empty_like(dd)
    vol[contained] = (4.0 / 3.0) * math.pi * rmin[contained] ** 3
    nc = ~contained
    dc, ac, bc = dd[nc], aa[nc], bb[nc]
    vol[nc] = (math.pi * (ac + bc - dc) ** 2 *
               (dc * dc + 2 * dc * (ac + bc) - 3 * (ac - bc) ** 2)) / (12.0 * dc)
    out[hit] = vol
    return out


def measure(pos, radii, dmin, dmax, contact_tol=1e-3, gofr=False, nbins=120):
    """Measure the packing of spheres at ``pos`` with ``radii`` in the periodic box [dmin, dmax].

    pos     : (N,3) centres.  radii : (N,) effective radii.  dmin/dmax : (3,) box corners.
    contact_tol : a pair counts as a contact when d < r_i + r_j - contact_tol*(r_i+r_j) (slight
                  compression), i.e. a genuine load-bearing contact rather than a grazing touch.
    Returns a dict with phi_naive, phi_corrected, coordination, max_overlap, rattlers, n_contacts and
    (optionally) the g(r) arrays.
    """
    pos = np.asarray(pos, dtype=np.float64)[:, :3]
    radii = np.asarray(radii, dtype=np.float64)
    dmin = np.asarray(dmin, dtype=np.float64)
    box = np.asarray(dmax, dtype=np.float64) - dmin
    V = float(np.prod(box))
    N = len(pos)

    wp = _wrap_to_box(pos, dmin, box)
    tree = cKDTree(wp, boxsize=box)
    rmax = float(radii.max())
    pairs = tree.query_pairs(r=2.0 * rmax, output_type="ndarray")  # all potentially-overlapping pairs

    phi_naive = float(np.sum((4.0 / 3.0) * math.pi * radii ** 3) / V)
    res = dict(N=N, V=V, phi_naive=phi_naive, phi_corrected=phi_naive,
               coordination=0.0, max_overlap=0.0, rattlers=N, n_contacts=0)
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        delta = wp[i] - wp[j]
        delta -= box * np.round(delta / box)  # minimum image
        d = np.linalg.norm(delta, axis=1)
        rsum = radii[i] + radii[j]
        overlap = rsum - d  # >0 => interpenetration
        # overlap-corrected solid fraction
        lens = _lens_volume(d, radii[i], radii[j])
        res["phi_corrected"] = float(phi_naive - np.sum(lens) / V)
        res["max_overlap"] = float(np.max(overlap / rsum)) if len(overlap) else 0.0
        # genuine contacts (slight compression) -> coordination number, excluding rattlers
        contact = overlap > contact_tol * rsum
        ci, cj = i[contact], j[contact]
        res["n_contacts"] = int(contact.sum())
        deg = np.bincount(np.concatenate([ci, cj]), minlength=N)
        load_bearing = deg >= 1
        res["rattlers"] = int(np.sum(deg == 0))
        res["coordination"] = float(deg[load_bearing].mean()) if load_bearing.any() else 0.0
    if gofr:
        res["gofr_r"], res["gofr_g"] = _gofr(wp, box, radii, nbins)
    return res


def _gofr(wp, box, radii, nbins):
    """Radial distribution function g(r) out to half the smallest box side, normalised by ideal gas."""
    N = len(wp)
    rho = N / float(np.prod(box))
    rmax = 0.5 * float(box.min())
    tree = cKDTree(wp, boxsize=box)
    counts = np.zeros(nbins)
    edges = np.linspace(0.0, rmax, nbins + 1)
    for k in range(N):
        nb = tree.query_ball_point(wp[k], rmax)
        if len(nb) <= 1:
            continue
        delta = wp[nb] - wp[k]
        delta -= box * np.round(delta / box)
        d = np.linalg.norm(delta, axis=1)
        d = d[d > 1e-9]
        counts += np.histogram(d, bins=edges)[0]
    shell = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = counts / (N * rho * shell)
    centers = 0.5 * (edges[1:] + edges[:-1])
    # report r in units of the mean diameter (contact at r/D = 1)
    D = 2.0 * float(np.mean(radii))
    return centers / D, g
