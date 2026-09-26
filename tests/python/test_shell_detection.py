"""Two-way shell detection (Simulation.set_shell_detection, docs/solver_details.md).

The one-way narrow phase tests only the LOWER-index body's probes against the other body's SDF
(docs/contact_physics_followups.md F2). A thin tube rim, or a small sphere, pressed into the flat
face of a box between the box's shell points is then invisible when the box has the lower index.
Each test below pins both halves: the miss with 'one_way' (the negative control -- if it stops
missing, the scene no longer exercises the defect) and the detection with 'two_way'.
"""
import numpy as np
import pytest

from peclet import dem

DEPTH = 0.02  # imposed penetration of the probe body into the box's +y face (the face is y = 1)


def _box_and(probe, box_first=True, lift=0.0):
    """A fixed unit box (half-extent 1, shell spacing 0.5: face points at x, z in {-1, -0.5, 0,
    0.5, 1}) and a probe body whose lowest point sits DEPTH - lift below the box's top face,
    centred at (0.25, ., 0.25) so no box face point lies inside the probe's footprint."""
    sim = dem.Simulation(2)
    sim.set_domain([-4.0, -4.0, -4.0], [4.0, 4.0, 4.0])
    sim.initialize_shape('box', radius=1.0)
    if probe == 'tube':  # outer radius 0.3, wall 0.06, height 0.6, axis y (canonical)
        sid = sim.add_shape('hollow_cylinder', radius=0.3, height=0.6, thickness=0.06)
        y = 1.0 + 0.3 - DEPTH
    else:  # sphere of radius 0.1
        sid = sim.add_shape('sphere', radius=0.1)
        y = 1.0 + 0.1 - DEPTH
    box, other = [0.0, 0.0, 0.0], [0.25, y + lift, 0.25]
    rows = [box, other] if box_first else [other, box]
    sim.set_positions(np.array(rows, dtype=np.float32))
    sim.set_shape_ids(np.array([0, sid] if box_first else [sid, 0], dtype=np.int32))
    fixed = np.array([0.0, 1.0] if box_first else [1.0, 0.0], dtype=np.float32)  # the box is fixed
    sim.set_inv_mass(fixed)
    sim.set_inv_inertia(sim.get_inv_inertia() * fixed[:, None])
    return sim


@pytest.mark.parametrize('probe', ['tube', 'sphere'])
def test_face_penetration_visible_only_two_way(probe):
    sim = _box_and(probe)
    assert sim.shell_detection == 'one_way'  # the default
    one_way = sim.compute_overlaps()
    sim.set_shell_detection('two_way')
    assert sim.shell_detection == 'two_way'
    two_way = sim.compute_overlaps()
    print(f"{probe}: one_way {one_way:.3e}  two_way {two_way:.3e}  (imposed {DEPTH})")
    assert one_way < 1e-6, "negative control: the one-way probe was expected to miss this"
    assert abs(two_way - DEPTH) < 1e-4, "two-way detection must see the imposed penetration"


@pytest.mark.parametrize('probe', ['tube', 'sphere'])
def test_probe_body_first_is_seen_either_way(probe):
    """With the probe body at the lower index the one-way pass already tests its points, and
    two-way reports the same depth: the reverse probes add nothing that was already seen."""
    sim = _box_and(probe, box_first=False)
    one_way = sim.compute_overlaps()
    sim.set_shell_detection('two_way')
    two_way = sim.compute_overlaps()
    assert abs(one_way - DEPTH) < 1e-4
    assert abs(two_way - DEPTH) < 1e-4


def test_tube_rests_on_box_only_two_way():
    """Dynamics: a tube dropped onto the fixed box's face falls THROUGH it one-way (the rim is
    never seen) and comes to rest on it two-way."""
    rest_y = {}
    for mode in ('one_way', 'two_way'):
        sim = _box_and('tube', lift=DEPTH + 0.01)  # start 0.01 clear of the face
        sim.set_shell_detection(mode)
        sim.set_gravity((0.0, -9.81, 0.0))
        sim.set_solver_iterations(20, 8)
        sim.set_dt(1e-3)
        sim.step(400)
        rest_y[mode] = float(sim.get_positions()[1, 1])
        print(f"{mode}: tube centre y after 0.4 s = {rest_y[mode]:.4f} (resting = 1.3)")
    assert rest_y['one_way'] < 0.9, "negative control: the one-way tube was expected to sink"
    assert abs(rest_y['two_way'] - 1.3) < 5e-3


def test_rejects_unknown_mode():
    sim = _box_and('sphere')
    with pytest.raises(ValueError):
        sim.set_shell_detection('both')
