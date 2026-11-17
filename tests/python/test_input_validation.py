"""Contract: the public setters refuse a NaN/Inf or physically inadmissible value with ValueError.

Validation lives in the binding TU (src/dem_bindings.cpp, helpers at its top); the message names
the setter, the parameter, the admissible range with its dimension and the offending value. Each
validated argument is covered twice: every bad value raises ValueError naming the parameter, and
one valid value is accepted (where a call needs a fixture this file cannot build cheaply -- an
analytic wall tree, an MPI decomposition -- the valid value is shown to pass validation by the
call reaching the C++ check behind it, which raises RuntimeError instead).
"""
import math
import re

import numpy as np
import pytest

from peclet import dem

NAN, INF = math.nan, math.inf


def _sim():
    s = dem.Simulation(8)
    s.initialize_shape("sphere", 0.5)
    s.set_domain((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
    s.set_positions(np.array([[1.0, 1.0, 1.0], [2.5, 1.0, 1.0]], dtype=np.float32))
    return s


def _grid():
    """A small world-space SDF wall grid: a floor at z = 0.5 (positive above it)."""
    z = np.arange(5, dtype=np.float32)
    return np.broadcast_to(z[None, None, :] - 0.5, (5, 5, 5)).astype(np.float32)


def _with_wall(s):
    return s.add_sdf_wall(_grid(), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def _sdf_shape(s, fn, origin=(-1.0, -1.0, -1.0), spacing=(0.5, 0.5, 0.5),
               inv_inertia=(2.5, 2.5, 2.5), bounding_radius=1.0):
    ax = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    grid = (np.sqrt(x * x + y * y + z * z) - 0.8).astype(np.float32)
    shell = np.array([[0.8, 0, 0], [-0.8, 0, 0], [0, 0.8, 0], [0, -0.8, 0], [0, 0, 0.8],
                      [0, 0, -0.8]], dtype=np.float32)
    return getattr(s, fn)(grid, origin, spacing, shell, inv_inertia, bounding_radius)


def _ints(*v):
    return np.array(v, dtype=np.int32)


_EMPTY_I = np.zeros(0, dtype=np.int32)
_EMPTY_F = np.zeros(0, dtype=np.float32)
_NO_TREE = (RuntimeError, "counts disagree")  # empty node arrays: past validation, refused by C++

# (id, call(sim, v), bad values, good value, parameter as the message names it, good outcome)
# `good outcome` None = the call returns; (exc, regex) = it gets past validation into a C++ check.
CASES = [
    ("Simulation.capacity", lambda s, v: dem.Simulation(v), [-1], 0, "capacity", None),
    ("initialize_shape.radius", lambda s, v: s.initialize_shape("sphere", v), [NAN, INF, 0.0, -1.0],
     0.5, "radius", None),
    ("initialize_shape.height", lambda s, v: s.initialize_shape("hollow_cylinder", 0.5, v, 0.1),
     [NAN, -0.1], 1.0, "height", None),
    ("initialize_shape.thickness", lambda s, v: s.initialize_shape("hollow_cylinder", 0.5, 1.0, v),
     [NAN, -0.1], 0.1, "thickness", None),
    ("add_shape.radius", lambda s, v: s.add_shape("sphere", v), [NAN, 0.0], 0.3, "radius", None),
    ("add_shape.height", lambda s, v: s.add_shape("hollow_cylinder", 0.5, v, 0.1), [INF, -1.0], 1.0,
     "height", None),
    ("add_shape.thickness", lambda s, v: s.add_shape("hollow_cylinder", 0.5, 1.0, v), [NAN, -1.0],
     0.1, "thickness", None),
    ("set_sdf_shape.origin", lambda s, v: _sdf_shape(s, "set_sdf_shape", origin=(-1.0, v, -1.0)),
     [NAN, INF], -1.0, "origin[1]", None),
    ("set_sdf_shape.spacing", lambda s, v: _sdf_shape(s, "set_sdf_shape", spacing=(0.5, 0.5, v)),
     [NAN, 0.0, -0.5], 0.5, "spacing[2]", None),
    ("set_sdf_shape.inv_inertia",
     lambda s, v: _sdf_shape(s, "set_sdf_shape", inv_inertia=(v, 2.5, 2.5)), [NAN, -1.0], 0.0,
     "inv_inertia[0]", None),
    ("set_sdf_shape.bounding_radius",
     lambda s, v: _sdf_shape(s, "set_sdf_shape", bounding_radius=v), [NAN, 0.0], 1.0,
     "bounding_radius", None),
    ("add_sdf_shape.spacing", lambda s, v: _sdf_shape(s, "add_sdf_shape", spacing=(v, 0.5, 0.5)),
     [INF, 0.0], 0.5, "spacing[0]", None),
    ("add_sdf_shape.origin", lambda s, v: _sdf_shape(s, "add_sdf_shape", origin=(v, -1.0, -1.0)),
     [NAN], -1.0, "origin[0]", None),
    ("add_sdf_shape.inv_inertia",
     lambda s, v: _sdf_shape(s, "add_sdf_shape", inv_inertia=(2.5, v, 2.5)), [NAN, -2.5], 2.5,
     "inv_inertia[1]", None),
    ("add_sdf_shape.bounding_radius",
     lambda s, v: _sdf_shape(s, "add_sdf_shape", bounding_radius=v), [-1.0], 1.0,
     "bounding_radius", None),
    ("add_scene_shape.inv_inertia",
     lambda s, v: s.add_scene_shape(_EMPTY_I, _EMPTY_F, 0, _EMPTY_F, (2.5, 2.5, v), 1.0),
     [NAN, -1.0], 2.5, "inv_inertia[2]", _NO_TREE),
    ("add_scene_shape.bounding_radius",
     lambda s, v: s.add_scene_shape(_EMPTY_I, _EMPTY_F, 0, _EMPTY_F, (2.5, 2.5, 2.5), v),
     [NAN, 0.0], 1.0, "bounding_radius", _NO_TREE),
    ("set_domain.min", lambda s, v: s.set_domain((v, 0.0, 0.0), (4.0, 4.0, 4.0)), [NAN, INF], 0.0,
     "min[0]", None),
    ("set_domain.max", lambda s, v: s.set_domain((0.0, 0.0, 0.0), (4.0, v, 4.0)),
     [NAN, 0.0, -1.0], 4.0, "max[1]", None),
    ("set_domain.extent", lambda s, v: s.set_domain(extent=(4.0, 4.0, v)), [NAN, 0.0, -4.0], 4.0,
     "extent[2]", None),
    ("set_domain.origin", lambda s, v: s.set_domain(extent=(4.0, 4.0, 4.0), origin=(0.0, v, 0.0)),
     [INF], -2.0, "origin[1]", None),
    ("set_gravity", lambda s, v: s.set_gravity((0.0, 0.0, v)), [NAN, -INF], -9.81, "gravity[2]",
     None),
    ("set_thermostat.temperature", lambda s, v: s.set_thermostat(v, 0.1), [NAN, -1.0], 0.0,
     "temperature", None),
    ("set_thermostat.tau", lambda s, v: s.set_thermostat(1.0, v), [NAN, -0.1], 0.0, "tau", None),
    ("set_thermostat.kB", lambda s, v: s.set_thermostat(1.0, 0.1, v), [NAN, 0.0], 1.0, "kB", None),
    ("set_solver_iterations.pos", lambda s, v: s.set_solver_iterations(v, 0), [-1], 0, "pos", None),
    ("set_solver_iterations.vel", lambda s, v: s.set_solver_iterations(8, v), [-1], 4, "vel", None),
    ("set_hertz_material.youngs", lambda s, v: s.set_hertz_material(0, v, 0.3), [NAN, 0.0, -1e7],
     1e7, "youngs", None),
    ("set_hertz_material.poisson", lambda s, v: s.set_hertz_material(0, 1e7, v),
     [NAN, -1.0, 0.6], 0.5, "poisson", None),
    ("set_global_scale", lambda s, v: s.set_global_scale(v), [NAN, 0.0, -1.0], 1.0, "scale", None),
    ("set_dt", lambda s, v: s.set_dt(v), [NAN, INF, 0.0, -1e-3], 1e-3, "dt", None),
    ("set_material_params.restitution_normal", lambda s, v: s.set_material_params(v),
     [NAN, -0.1, 1.1], 1.0, "restitution_normal", None),
    ("set_material_params.restitution_tangent", lambda s, v: s.set_material_params(0.5, v),
     [NAN, INF], 1.0, "restitution_tangent", None),
    ("set_material_params.friction", lambda s, v: s.set_material_params(0.5, 0.0, v),
     [NAN, -0.1], 0.4, "friction", None),
    ("set_material_ids", lambda s, v: s.set_material_ids([0, v]), [-1, 8], 7, "ids[1]", None),
    ("set_pair_material.restitution", lambda s, v: s.set_pair_material(0, 1, v, 0.2),
     [NAN, 1.5], 0.7, "restitution", None),
    ("set_pair_material.friction", lambda s, v: s.set_pair_material(0, 1, 0.7, v), [INF, -1.0],
     0.0, "friction", None),
    ("set_wall_material_id", lambda s, v: s.set_wall_material_id(_with_wall(s), v), [8], -1, "mat",
     None),
    ("add_analytic_wall.restitution",
     lambda s, v: s.add_analytic_wall(_EMPTY_I, _EMPTY_F, 0, True, v, 0.0), [NAN, -0.5, 2.0], 0.5,
     "restitution", _NO_TREE),
    ("add_analytic_wall.friction",
     lambda s, v: s.add_analytic_wall(_EMPTY_I, _EMPTY_F, 0, True, 0.5, v), [NAN, -0.5], 0.3,
     "friction", _NO_TREE),
    ("add_plane.point", lambda s, v: s.add_plane((0.0, v, 0.0), (0.0, 0.0, 1.0)), [NAN], 0.0,
     "point[1]", None),
    ("add_plane.normal", lambda s, v: s.add_plane((0.0, 0.0, 0.0), (0.0, 0.0, v)), [NAN, 0.0], 1.0,
     "normal", None),
    ("add_sdf_wall.origin", lambda s, v: s.add_sdf_wall(_grid(), (0.0, 0.0, v), (1.0, 1.0, 1.0)),
     [NAN], 0.0, "origin[2]", None),
    ("add_sdf_wall.spacing", lambda s, v: s.add_sdf_wall(_grid(), (0.0, 0.0, 0.0), (1.0, v, 1.0)),
     [NAN, 0.0, -1.0], 1.0, "spacing[1]", None),
    ("add_sdf_wall.restitution",
     lambda s, v: s.add_sdf_wall(_grid(), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), v), [NAN, 1.01], 0.0,
     "restitution", None),
    ("add_sdf_wall.friction",
     lambda s, v: s.add_sdf_wall(_grid(), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0, v), [-0.2], 0.5,
     "friction", None),
    ("set_wall_velocity.lin_vel", lambda s, v: s.set_wall_velocity(_with_wall(s), (v, 0.0, 0.0)),
     [NAN, INF], 1.0, "lin_vel[0]", None),
    ("set_wall_velocity.ang_vel",
     lambda s, v: s.set_wall_velocity(_with_wall(s), ang_vel=(0.0, 0.0, v)), [NAN], 2.0,
     "ang_vel[2]", None),
    ("set_wall_velocity.center",
     lambda s, v: s.set_wall_velocity(_with_wall(s), center=(0.0, v, 0.0)), [INF], 2.0,
     "center[1]", None),
    # A grid-SDF wall has no tree to move: a valid placement passes validation and is refused there.
    ("set_wall_transform.translation",
     lambda s, v: s.set_wall_transform(_with_wall(s), (v, 0.0, 0.0)), [NAN], 1.0, "translation[0]",
     (RuntimeError, "GRID SDF")),
    ("set_wall_transform.quat",
     lambda s, v: s.set_wall_transform(_with_wall(s), quat=(0.0, 0.0, 0.0, v)), [NAN, INF], 1.0,
     "quat[3]", (RuntimeError, "GRID SDF")),
    ("set_scales_uniform", lambda s, v: s.set_scales_uniform(v), [NAN, 0.0, -1.0], 1.0, "scale",
     None),
    ("set_growth_params.rate", lambda s, v: s.set_growth_params(v), [NAN, INF], 1.0, "rate", None),
    ("set_growth_params.new_factor", lambda s, v: s.set_growth_params(1.0, v), [NAN], 0.15,
     "new_factor", None),
    ("set_sleeping.threshold_scale", lambda s, v: s.set_sleeping(True, threshold_scale=v), [NAN],
     2.0, "threshold_scale", None),
    ("set_sleeping.wake_scale", lambda s, v: s.set_sleeping(True, wake_scale=v), [INF], 40.0,
     "wake_scale", None),
    ("set_sleeping.immovable_frac", lambda s, v: s.set_sleeping(True, immovable_frac=v), [NAN],
     0.0, "immovable_frac", None),
    ("set_verlet_skin", lambda s, v: s.set_verlet_skin(v), [NAN, -0.1], 0.0, "skin_frac", None),
    ("get_sdf_grid", lambda s, v: s.get_sdf_grid((4, v, 4)), [-1, 0], 1, "resolution[1]", None),
]
if hasattr(dem.Simulation, "init_mpi"):
    # Only the refusals: they are checked before MPI_Init, while a valid call would initialize MPI
    # (and enable_mpi_step needs init_mpi first). The valid paths are the MPI tests' business.
    _BOX = dict(origin=(0.0, 0.0, 0.0), extent=(4.0, 4.0, 4.0), cells=(4, 4, 4),
                periodic=(False, False, False))
    CASES += [
        ("init_mpi.origin", lambda s, v: s.init_mpi(**{**_BOX, "origin": (v, 0.0, 0.0)}), [NAN],
         None, "origin[0]", None),
        ("init_mpi.extent", lambda s, v: s.init_mpi(**{**_BOX, "extent": (4.0, v, 4.0)}),
         [NAN, 0.0], None, "extent[1]", None),
        ("init_mpi.cells", lambda s, v: s.init_mpi(**{**_BOX, "cells": (4, 4, v)}), [0, -2], None,
         "cells[2]", None),
        ("enable_mpi_step.rcut", lambda s, v: s.enable_mpi_step(rcut=v), [NAN, -1.0], None, "rcut",
         None),
        ("enable_mpi_step.verlet_skin", lambda s, v: s.enable_mpi_step(verlet_skin=v), [INF, -0.5],
         None, "verlet_skin", None),
    ]

_BAD = [(c[0], c[1], b, c[4]) for c in CASES for b in c[2]]
_GOOD = [(c[0], c[1], c[3], c[5]) for c in CASES if c[3] is not None]


@pytest.mark.parametrize("case,call,value,param", _BAD,
                         ids=[f"{c[0]}={c[2]}" for c in _BAD])
def test_bad_value_raises_value_error_naming_the_parameter(case, call, value, param):
    s = _sim()
    with pytest.raises(ValueError) as err:
        call(s, value)
    msg = str(err.value)
    assert re.search(rf"{re.escape(param)}(\[\d\])? must be ", msg), msg
    assert "got" in msg, msg


@pytest.mark.parametrize("case,call,value,outcome", _GOOD, ids=[c[0] for c in _GOOD])
def test_valid_value_is_accepted(case, call, value, outcome):
    s = _sim()
    if outcome is None:
        call(s, value)
    else:
        exc, match = outcome
        with pytest.raises(exc, match=match):
            call(s, value)


def test_message_states_range_unit_and_value():
    s = _sim()
    with pytest.raises(ValueError, match=r"^set_hertz_material: youngs must be finite and > 0 "
                                         r"\(force/area\), got nan$"):
        s.set_hertz_material(0, NAN, 0.3)
    with pytest.raises(ValueError, match=r"^set_gravity: gravity\[1\] must be finite "
                                         r"\(length/time\^2\), got inf$"):
        s.set_gravity((0.0, INF, 0.0))


def test_refused_value_leaves_the_setting_unchanged():
    s = _sim()
    s.set_dt(1e-3)
    s.set_gravity((0.0, 0.0, -9.81))
    with pytest.raises(ValueError):
        s.set_dt(NAN)
    with pytest.raises(ValueError):
        s.set_gravity((NAN, 0.0, 0.0))
    assert s.dt == pytest.approx(1e-3)
    assert s.gravity == pytest.approx((0.0, 0.0, -9.81))
