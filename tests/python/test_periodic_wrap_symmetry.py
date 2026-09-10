"""Single-rank periodic wrap contacts are resolved symmetrically, wherever the pair sits.

Two equal spheres overlapping THROUGH a periodic face must each move by half the overlap, exactly
as the same pair would inside the box. The single-rank step images a particle only within
`ghostBand` of a periodic face and writes position corrections to the raw slot (a ghost's
correction is discarded at commit), so the band must cover the FARTHER partner of every pair the
narrow phase can report: 2 R_max + margin. With the old band of one R_max a pair with
max(a, b) > R_max (a, b the two distances to the face) was detected from one side only and the
near partner absorbed the whole de-penetration -- this test is RED on that band.
"""
import numpy as np
import pytest
from peclet import dem

L = 10.0
R = 1.0
OVERLAP = 0.2


def _run(xa, xb, use_step, axis=0):
    s = dem.Simulation(2)
    s.initialize_shape('sphere', radius=R)
    s.set_domain((0.0, 0.0, 0.0), (L, L, L))
    s.set_periodic(axis == 0, axis == 1, axis == 2)
    s.set_gravity((0.0, 0.0, 0.0))
    s.set_material_params(0.0, 0.0, 0.0)
    s.set_solver_iterations(4, 0)
    pos = np.full((2, 3), 5.0, np.float32)
    pos[0, axis], pos[1, axis] = xa, xb
    s.set_positions(pos)
    if use_step:
        s.set_dt(0.01)
        s.step()
    else:
        s.relax()
    p = s.get_positions().reshape(2, 3)
    return float(p[0, axis] - xa), float(p[1, axis] - xb)


@pytest.mark.parametrize("use_step", [False, True], ids=["relax", "step"])
@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_asymmetric_wrap_pair_moves_both(use_step, axis):
    # A at 1.5 from the low face, B at 0.3 from the high face: 1.8 apart through the wrap
    # (overlap 0.2), and A sits beyond one radius of its face.
    da, db = _run(1.5, L - 0.3, use_step, axis)
    # the reference: the same overlap, symmetric about the face (both within one radius)
    ra, rb = _run(0.9, L - 0.9, use_step, axis)
    assert ra > 0.0 and rb < 0.0 and abs(ra + rb) < 1e-5, (ra, rb)
    assert da > 0.0 and db < 0.0, f"one-sided: A moved {da:.4f}, B moved {db:.4f}"
    assert abs(da + db) < 1e-5, f"not equal and opposite: {da:.5f} vs {db:.5f}"
    assert abs(da - ra) < 1e-5 and abs(db - rb) < 1e-5, "result depends on where the pair sits"


def test_wrap_pair_matches_in_box_pair():
    # The same pair inside the box (no face between them) is the ground truth: A, on the low side
    # of the face, is pushed +x like the in-box body on the high side (ib), and B like ia.
    da, db = _run(1.5, L - 0.3, False)
    ia, ib = _run(4.0, 4.0 + 2 * R - OVERLAP, False)
    assert abs(da - ib) < 1e-5 and abs(db - ia) < 1e-5, (da, db, ia, ib)
