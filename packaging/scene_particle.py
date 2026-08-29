"""peclet.dem.scene_particle — a COMPOSED analytic SDF shape as a DEM particle, in one call.

The pipeline the analytic-SDF campaign builds toward: author a particle as a CSG tree in
:mod:`peclet.core.geom` (leaves + union/intersection/difference + transforms), and hand the SAME
tree to every consumer —

* **dem** collides against the exact tree (``Simulation.add_scene_shape``: the collision field is
  ``evalTree`` in canonical body space, no sampled-grid approximation), with a surface point shell
  generated from a lattice bake (the point-shell model still needs probes; their spacing, not the
  field, is then the contact resolution),
* **mass properties** come from implicit quadrature (:func:`peclet.core.geom` ``body_properties``):
  mass, COM, full inertia tensor, principal moments + quaternion — sign-exact bracketing, so
  bound-only leaves (ellipsoid, superquadric, CSG) carry no systematic bias,
* the tree is **re-expressed in its principal body frame exactly** (``principal_frame``: one
  composed transform node, no resampling), which is what dem's diagonal-inertia rotational update
  requires — the answer to "my shape's reference frame is not its principal frame",
* **flow** (resolved CFD-DEM) can consume the same reframed tree analytically via
  ``SceneBuilder.encode()`` + ``flow.set_scene``.

Example::

    from peclet.core import geom
    from peclet.dem import scene_particle

    b = geom.SceneBuilder()
    core = b.add_leaf("box", [0.3, 0.18, 0.12])
    arm  = b.add_leaf("capsule", [0.08, 0.25], translation=[0.25, 0.1, 0.0])
    grain = b.add_union(core, arm)
    sp = scene_particle.build(b, grain, bounds=([-0.8, -0.6, -0.5], [0.9, 0.7, 0.5]))
    shape_id = sp.register(sim)          # -> Simulation.add_scene_shape
    # sp.home_root is the principal-frame tree; sp.quat maps body -> input frame
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SceneParticle:
    """A composed analytic particle, principal-frame, ready for dem (and flow)."""

    builder: object          # the peclet.core.geom.SceneBuilder (owns the nodes)
    home_root: int           # root of the PRINCIPAL-FRAME copy of the tree
    input_root: int          # the tree as authored (untouched)
    node_ints: np.ndarray    # flat encoding of the whole builder (add_scene_shape / set_scene)
    node_reals: np.ndarray
    shell: np.ndarray        # (M,3) float32 surface probes, principal body frame
    mass: float
    volume: float
    com: np.ndarray          # COM of the AUTHORED tree, in its input frame
    principal: np.ndarray    # (3,) principal moments at the given density
    quat: np.ndarray         # (4,) x,y,z,w: p_input = com + rotate(quat, p_body)
    inv_inertia_unit: tuple  # unit-mass diagonal inverse inertia (what dem's convention wants)
    bounding_radius: float

    def register(self, sim, shell=None):
        """Register with a :class:`peclet.dem.Simulation`; returns the shape id.

        REMEMBER dem's gotcha: ``set_positions`` resets every particle to shape 0, so call
        ``set_shape_ids`` AFTER ``set_positions``.
        """
        return sim.add_scene_shape(
            np.ascontiguousarray(self.node_ints, dtype=np.int32),
            np.ascontiguousarray(self.node_reals, dtype=np.float32),
            self.home_root,
            np.ascontiguousarray(self.shell if shell is None else shell, dtype=np.float32),
            tuple(self.inv_inertia_unit),
            float(self.bounding_radius),
        )


def build(builder, root, bounds, *, density=1.0, n=40, order=5, nseg=8,
          shell_resolution=96, target_shell_points=400, margin=0.08):
    """Measure, reframe and shell a composed tree; returns a :class:`SceneParticle`.

    ``bounds = (lo, hi)`` must contain the AUTHORED solid (use the leaf bounding radii; too-large
    bounds only cost accuracy per cell). ``shell_resolution`` is the bake lattice for the
    marching-cubes probe shell — it bounds CONTACT resolution only; the collision field itself is
    the exact tree.
    """
    from skimage import measure

    lo, hi = (list(map(float, bounds[0])), list(map(float, bounds[1])))
    props = builder.body_properties(root, lo=lo, hi=hi, n=n, order=order, nseg=nseg,
                                    density=density)
    home = builder.principal_frame(root, lo=lo, hi=hi, n=n, order=order, nseg=nseg)

    # principal-frame bounding half-box: project the input box corners through the reframe
    com = np.asarray(props["com"])
    R = np.asarray(props["rotation"]).reshape(3, 3)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    body = (corners - com) @ R
    half = float(np.abs(body).max() * (1.0 + margin))

    nb_ = int(shell_resolution)
    sp = 2.0 * half / (nb_ - 1)
    grid = np.asarray(builder.bake(home, origin=[-half] * 3, spacing=[sp] * 3,
                                   dims=[nb_, nb_, nb_]))
    g3 = np.ascontiguousarray(grid.reshape(nb_, nb_, nb_, order="F"))
    verts, faces, _, _ = measure.marching_cubes(g3, level=0.0, spacing=(sp, sp, sp))
    verts += -half
    step = max(1, len(verts) // int(target_shell_points))
    shell = np.ascontiguousarray(verts[::step], dtype=np.float32)

    ni, nr, ii, ir = builder.encode()
    principal = np.asarray(props["principal"], dtype=float)
    inv_unit = tuple(props["mass"] / np.maximum(principal, 1e-30))
    bounding = float(np.linalg.norm(shell, axis=1).max() * 1.02)
    return SceneParticle(builder=builder, home_root=home, input_root=root, node_ints=ni,
                         node_reals=nr, shell=shell, mass=float(props["mass"]),
                         volume=float(props["volume"]), com=com, principal=principal,
                         quat=np.asarray(props["quat"]), inv_inertia_unit=inv_unit,
                         bounding_radius=bounding)
