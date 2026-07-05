"""peclet.dem.particle_builder — build a general DEM particle from an implicit solid (SDF).

The :class:`peclet.dem.Simulation` collision model is *point-shell vs signed-distance field*: each
particle carries a set of surface points and a canonical-space SDF, and a contact is a shell point of
one body penetrating the SDF of another. For the analytic shapes (sphere / cylinder / box) both are
generated in C++. This module builds the same two ingredients — **surface point shell** and **grid
SDF** — plus the **mass properties** for an *arbitrary* shape described by an implicit function, so a
user can author a particle with any implicit-solid-modeling tool and pack it.

Give it a signed-distance function ``f(points) -> distance`` (negative inside; ``points`` is an
``(N,3)`` array) — e.g. hand-written NumPy CSG, or a callable from an implicit-modeling library such
as `fogleman/sdf` — over a bounding box, and :func:`build_particle` returns a :class:`ParticleShape`
holding everything :meth:`Simulation.set_sdf_shape` needs:

* a **grid SDF** sampled on a regular lattice (the collision field, evaluated on-device),
* a **surface point shell** from marching cubes on that grid (vertices or triangle centroids),
* **mass, centre of mass and the full/principal inertia tensor** by voxel integration.

The shape is returned in its **principal-axis body frame** (recentred on the centre of mass and
rotated onto the principal axes of inertia), so the solver's diagonal inverse-inertia model is
*exact* — no off-diagonal inertia is dropped. The principal frame is obtained by exact re-evaluation
of the SDF callable on the rotated lattice (no resampling error).

Only NumPy and scikit-image (marching cubes) are required.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["ParticleShape", "build_particle", "WallSDF", "build_wall_sdf"]


@dataclass
class ParticleShape:
    """A general DEM particle built from an SDF, in its principal-axis body frame.

    Attributes
    ----------
    grid : np.ndarray
        ``(nx, ny, nz)`` signed-distance samples (negative inside) at lattice nodes
        ``origin + (i, j, k) * spacing``.
    origin, spacing : np.ndarray
        ``(3,)`` canonical-space coordinate of node ``(0,0,0)`` and the per-axis node spacing.
    shell : np.ndarray
        ``(M, 3)`` surface points (the collision probes), in canonical body-frame coordinates.
    mass : float
        Mass at the requested density (``density * enclosed volume``).
    com : np.ndarray
        Centre of mass in the *input* frame (the offset that was removed). In the returned body
        frame the COM is at the origin.
    inertia : np.ndarray
        ``(3,)`` principal moments of inertia about the COM at the requested density.
    inv_inertia_unit : np.ndarray
        ``(3,)`` diagonal inverse inertia for a **unit-mass** body (``mass / inertia``) — what
        :meth:`Simulation.set_sdf_shape` expects, matching the unit-mass ``inv_mass = 1`` packing
        convention (a solid unit sphere gives ``2.5 / r**2``).
    principal_rotation : np.ndarray
        ``(3, 3)`` rotation whose columns are the principal axes in the input frame
        (``p_input = com + R @ p_body``).
    bounding_radius : float
        Canonical radius enclosing the surface (broad-phase / SDF-export bound).
    vertices, faces : np.ndarray
        The marching-cubes surface mesh in the body frame: ``vertices`` is ``(V, 3)`` and ``faces``
        is ``(F, 3)`` of vertex indices. Handy for rendering (transform per particle) or STL export
        (:meth:`to_stl`).
    """

    grid: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    shell: np.ndarray
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    inv_inertia_unit: np.ndarray
    principal_rotation: np.ndarray
    bounding_radius: float
    vertices: np.ndarray
    faces: np.ndarray

    def apply_to(self, sim, *, unit_mass: bool = True) -> None:
        """Upload this shape onto a :class:`peclet.dem.Simulation` (replaces shape 0 for all
        particles). With ``unit_mass=True`` (the packing default) particles keep ``inv_mass = 1`` and
        get this shape's unit-mass inertia; pass ``unit_mass=False`` then override per particle with
        ``sim.set_inv_mass``/``sim.set_inv_inertia`` for the real density."""
        nx, ny, nz = self.grid.shape
        grid_flat = np.asarray(self.grid, dtype=np.float32).ravel(order="F")  # x-fastest
        inv_i = self.inv_inertia_unit if unit_mass else (1.0 / self.inertia)
        sim.set_sdf_shape(
            grid_flat,
            int(nx),
            int(ny),
            int(nz),
            tuple(float(v) for v in self.origin),
            tuple(float(v) for v in self.spacing),
            np.ascontiguousarray(self.shell, dtype=np.float32),
            tuple(float(v) for v in inv_i),
            float(self.bounding_radius),
        )

    def to_stl(self, path, *, binary: bool = True, scale: float = 1.0) -> str:
        """Write the particle's surface mesh to an STL file (in the body frame) and return ``path``.

        Opens straight into ParaView/Blender/Ovito or any mesh tool. ``scale`` multiplies the
        coordinates (the shape is unit-scale; pass a physical size if you want). ``binary=False``
        writes ASCII STL. Facet normals are computed and oriented outward from the mesh centroid."""
        v = np.asarray(self.vertices, dtype=np.float64) * float(scale)
        f = np.asarray(self.faces)
        tris = v[f]  # (F, 3, 3)
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
        # orient outward from the centroid (the COM sits at the body-frame origin)
        flip = np.sum(n * (tris.mean(axis=1) - v.mean(axis=0)), axis=1) < 0
        n[flip] *= -1
        if binary:
            import struct

            with open(path, "wb") as fh:
                fh.write(b"peclet.dem build_particle".ljust(80, b"\0"))
                fh.write(struct.pack("<I", len(f)))
                rec = np.zeros(len(f), dtype=np.dtype([("d", "<f4", (12,)), ("a", "<u2")]))
                rec["d"][:, 0:3] = n
                rec["d"][:, 3:6] = tris[:, 0]
                rec["d"][:, 6:9] = tris[:, 1]
                rec["d"][:, 9:12] = tris[:, 2]
                fh.write(rec.tobytes())
        else:
            with open(path, "w") as fh:
                fh.write("solid particle\n")
                for i in range(len(f)):
                    fh.write(f"facet normal {n[i, 0]:.6e} {n[i, 1]:.6e} {n[i, 2]:.6e}\n outer loop\n")
                    for j in range(3):
                        fh.write(f"  vertex {tris[i, j, 0]:.6e} {tris[i, j, 1]:.6e} {tris[i, j, 2]:.6e}\n")
                    fh.write(" endloop\nendfacet\n")
                fh.write("endsolid particle\n")
        return path


def _sample(f, lo, hi, res):
    """Sample ``f`` on an ``(res,res,res)`` node lattice over ``[lo, hi]``; return
    ``(phi, origin, spacing, coords)`` with ``phi`` shaped ``(nx, ny, nz)``."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if np.isscalar(res):
        res = (int(res), int(res), int(res))
    nx, ny, nz = res
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    zs = np.linspace(lo[2], hi[2], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    phi = np.asarray(f(pts), float).reshape(nx, ny, nz)
    spacing = np.array(
        [
            (hi[0] - lo[0]) / (nx - 1),
            (hi[1] - lo[1]) / (ny - 1),
            (hi[2] - lo[2]) / (nz - 1),
        ]
    )
    coords = (xs, ys, zs)
    return phi, lo.copy(), spacing, coords


def _triangle_areas(verts, faces):
    a = verts[faces[:, 1]] - verts[faces[:, 0]]
    b = verts[faces[:, 2]] - verts[faces[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def _voxel_downsample(points, spacing):
    """Keep one (mean) representative per occupied cell of a `spacing`-sized voxel grid — a uniform
    thinning of the marching-cubes point cloud so the collision shell has a controlled density
    (comparable to the analytic shapes' few-hundred-point shells) rather than thousands of points."""
    if spacing <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / spacing).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    sums = np.zeros((n, 3))
    np.add.at(sums, inv, points)
    counts = np.bincount(inv, minlength=n).reshape(-1, 1)
    return sums / counts


def _mass_properties(phi, coords, density):
    """Mass, COM and full inertia tensor by voxel integration over the interior (``phi < 0``).

    Uses a linear-fraction interior weight per node (a smooth approximation to the occupied volume)
    so the properties converge with resolution rather than jumping voxel-by-voxel."""
    xs, ys, zs = coords
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]
    cell = dx * dy * dz
    # Smooth interior indicator: 1 well inside, 0 well outside, linear across one node spacing.
    h = 0.5 * (dx + dy + dz)
    w = np.clip(0.5 - phi / h, 0.0, 1.0)  # (nx,ny,nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    m_vox = density * cell * w
    mass = float(m_vox.sum())
    if mass <= 0.0:
        raise ValueError(
            "build_particle: the SDF encloses no volume on this grid — check the sign "
            "convention (negative inside) and that the bounds contain the solid."
        )
    cx = float((m_vox * X).sum() / mass)
    cy = float((m_vox * Y).sum() / mass)
    cz = float((m_vox * Z).sum() / mass)
    com = np.array([cx, cy, cz])
    rx, ry, rz = X - cx, Y - cy, Z - cz
    Ixx = float((m_vox * (ry * ry + rz * rz)).sum())
    Iyy = float((m_vox * (rx * rx + rz * rz)).sum())
    Izz = float((m_vox * (rx * rx + ry * ry)).sum())
    Ixy = float((-m_vox * rx * ry).sum())
    Ixz = float((-m_vox * rx * rz).sum())
    Iyz = float((-m_vox * ry * rz).sum())
    tensor = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    return mass, com, tensor


def build_particle(
    f,
    bounds,
    resolution=64,
    *,
    density=1.0,
    shell="vertices",
    target_shell_points=400,
    shell_spacing=None,
    align_principal=True,
    margin=0.1,
):
    """Build a :class:`ParticleShape` from a signed-distance function.

    Parameters
    ----------
    f : callable
        ``f(points) -> distances``: ``points`` is ``(N, 3)``, returns ``(N,)`` signed distances,
        **negative inside** the solid. Any implicit-modeling front-end that can evaluate on point
        arrays works (hand-written NumPy CSG, `fogleman/sdf`, ...).
    bounds : ((xlo, ylo, zlo), (xhi, yhi, zhi))
        Axis-aligned box enclosing the solid, with a little slack so the zero level set is interior.
    resolution : int or (nx, ny, nz)
        Lattice resolution. 48–96 is a good range; higher = smoother shell/SDF but larger on-device
        grid.
    density : float
        Material density; scales ``mass`` and ``inertia`` (not ``inv_inertia_unit``).
    shell : {"vertices", "centroids"}
        Place surface points at marching-cubes vertices or triangle centroids, then thin them to a
        controlled density (see ``target_shell_points``).
    target_shell_points : int
        Approximate number of surface points to keep. The dense marching-cubes cloud is voxel-thinned
        to about this many — a few hundred is a good collision-shell size (comparable to the analytic
        shapes). Ignored if ``shell_spacing`` is given.
    shell_spacing : float or None
        Explicit thinning cell size (canonical units); overrides ``target_shell_points``. ``None``
        derives it from the surface area and the target count.
    align_principal : bool
        Recentre on the COM and rotate onto the principal axes so the stored diagonal inertia is
        exact. Strongly recommended (the solver inertia is diagonal). Requires ``f`` callable.
    margin : float
        Extra half-width (as a fraction of the box size) added around ``bounds`` when re-sampling in
        the principal frame, so the rotated shape still fits.

    Returns
    -------
    ParticleShape
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "build_particle needs scikit-image for marching cubes: pip install scikit-image"
        ) from e

    if not callable(f):
        raise TypeError("f must be a callable f(points)->distance (negative inside)")
    (lo0, hi0) = (np.asarray(bounds[0], float), np.asarray(bounds[1], float))

    # 1. Coarse sample to locate the COM and principal axes.
    phi0, origin0, spacing0, coords0 = _sample(f, lo0, hi0, resolution)
    mass0, com, tensor = _mass_properties(phi0, coords0, density)
    evals, evecs = np.linalg.eigh(tensor)  # ascending eigenvalues; columns = principal axes
    R = evecs
    if np.linalg.det(R) < 0:  # keep a proper (right-handed) rotation
        R[:, 0] = -R[:, 0]

    if align_principal:
        # 2. Re-sample in the principal body frame: node q -> input point com + R @ q. Choose a
        #    symmetric box big enough to hold the rotated shape (project the input corners).
        corners = np.array([[x, y, z] for x in (lo0[0], hi0[0]) for y in (lo0[1], hi0[1]) for z in (lo0[2], hi0[2])])
        body_corners = (corners - com) @ R  # input -> body
        half = np.abs(body_corners).max(axis=0) * (1.0 + margin)
        lo, hi = -half, half

        def f_body(q):
            return f(com + q @ R.T)  # body -> input, then evaluate

        phi, origin, spacing, coords = _sample(f_body, lo, hi, resolution)
        # Recompute the (now nearly diagonal) mass properties in the body frame.
        mass, com_b, tensor_b = _mass_properties(phi, coords, density)
        inertia = np.diag(tensor_b).copy()
        off = np.abs(tensor_b - np.diag(inertia)).max()
        if off > 0.02 * max(inertia.max(), 1e-12):
            warnings.warn(
                f"principal alignment residual off-diagonal inertia {off:.3g} is large relative to "
                "the principal moments; increase resolution.",
                stacklevel=2,
            )
    else:
        phi, origin, spacing, coords = phi0, origin0, spacing0, coords0
        mass = mass0
        inertia = np.diag(tensor).copy()
        off = np.abs(tensor - np.diag(inertia)).max()
        if off > 0.02 * max(inertia.max(), 1e-12):
            warnings.warn(
                "align_principal=False but the inertia tensor has significant off-diagonal terms; "
                "the solver's diagonal-inertia model will be inexact for this orientation.",
                stacklevel=2,
            )

    inertia = np.maximum(inertia, 1e-12)
    inv_inertia_unit = mass / inertia  # unit-mass diagonal inverse inertia

    # 3. Surface point shell from marching cubes on the (body-frame) grid, thinned to a controlled
    #    density (thousands of raw MC vertices would swamp the contact solver).
    verts, faces, _normals, _vals = marching_cubes(phi, level=0.0, spacing=tuple(spacing))
    verts = verts + origin  # index+spacing -> canonical coords
    if shell == "centroids":
        pts = verts[faces].mean(axis=1)
    elif shell == "vertices":
        pts = verts
    else:
        raise ValueError("shell must be 'vertices' or 'centroids'")
    if shell_spacing is None:
        area = float(_triangle_areas(verts, faces).sum())
        shell_spacing = math.sqrt(area / max(target_shell_points, 1)) if area > 0 else 0.0
    pts = _voxel_downsample(pts, shell_spacing)
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    bounding_radius = float(np.linalg.norm(pts, axis=1).max()) if len(pts) else float(np.abs(half).max())

    return ParticleShape(
        grid=np.ascontiguousarray(phi, dtype=np.float32),
        origin=np.asarray(origin, float),
        spacing=np.asarray(spacing, float),
        shell=pts,
        mass=float(mass),
        com=np.asarray(com, float),
        inertia=np.asarray(inertia, float),
        inv_inertia_unit=np.asarray(inv_inertia_unit, float),
        principal_rotation=np.asarray(R, float),
        bounding_radius=bounding_radius,
        vertices=np.ascontiguousarray(verts, dtype=np.float64),
        faces=np.ascontiguousarray(faces, dtype=np.int64),
    )


@dataclass
class WallSDF:
    """A static, world-space SDF wall/container the grains collide against (drum, hopper, tray).

    Attributes
    ----------
    grid : np.ndarray
        ``(nx, ny, nz)`` signed-distance samples in **world** coordinates at nodes
        ``origin + (i, j, k) * spacing`` — **positive in the void** where the grains live, **negative
        inside the solid wall** (the opposite sign convention to :func:`build_particle`, whose solid is
        the particle).
    origin, spacing : np.ndarray
        ``(3,)`` world coordinate of node ``(0,0,0)`` and the per-axis node spacing.
    """

    grid: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray

    def add_to(self, sim, *, restitution: float = 0.0, friction: float = 0.0) -> int:
        """Upload this wall onto a :class:`peclet.dem.Simulation` with the given binary particle–wall
        material. Returns the wall index (pass it to ``sim.set_wall_velocity`` for a moving wall)."""
        nx, ny, nz = self.grid.shape
        grid_flat = np.asarray(self.grid, dtype=np.float32).ravel(order="F")  # x-fastest
        return sim.add_sdf_wall(
            grid_flat,
            int(nx),
            int(ny),
            int(nz),
            tuple(float(v) for v in self.origin),
            tuple(float(v) for v in self.spacing),
            float(restitution),
            float(friction),
        )


def build_wall_sdf(f, bounds, resolution=64):
    """Sample a static container/wall SDF onto a regular world-space lattice.

    Give it ``f(points) -> distance`` (``points`` is ``(N, 3)`` world coordinates) that is **positive
    in the void where the grains live and negative inside the solid wall**, over an axis-aligned box
    ``bounds`` that spans the whole simulation domain (the grid must cover wherever a grain can reach).
    Returns a :class:`WallSDF` ready for :meth:`WallSDF.add_to`.

    Parameters
    ----------
    f : callable
        ``f(points) -> distances``; positive in the void, negative in the wall.
    bounds : ((xlo, ylo, zlo), (xhi, yhi, zhi))
        Axis-aligned box covering the domain (typically the full ``set_domain`` box).
    resolution : int or (nx, ny, nz)
        Lattice resolution. 64–128 resolves a smooth curved wall well.
    """
    if not callable(f):
        raise TypeError("f must be a callable f(points)->distance (positive in the void)")
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    phi, origin, spacing, _coords = _sample(f, lo, hi, resolution)
    return WallSDF(
        grid=np.ascontiguousarray(phi, dtype=np.float32),
        origin=np.asarray(origin, float),
        spacing=np.asarray(spacing, float),
    )
