"""Two overlapping hollow cylinders: the point-shell narrow-phase must see the contact and the
position solve must reduce the interpenetration."""
import numpy as np
from peclet import dem


def test_static_overlap():
    sim = dem.Simulation(2)
    sim.set_domain([-0.45, -1.0, -1.0], [0.45, 1.0, 1.0])
    sim.initialize_shape('hollow_cylinder', radius=0.5, height=1.0, thickness=0.5)
    pos = np.array([[-0.1, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 1.0]], dtype=np.float32)
    sim.set_positions(pos)
    sim.set_velocities(np.zeros((2, 3), dtype=np.float32))
    sim.set_quaternions(np.array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.float32))
    sim.set_growth_params(1.0, 0.2)
    dt = 0.01
    sim.set_dt(dt)
    sim.step()
    n0, m0, ov0 = sim.num_contacts, sim.num_manifolds, sim.max_overlap
    print(f"step 0: contacts {n0}  manifolds {m0}  max overlap {ov0:.4f}")
    # Detection, not residual overlap (docs/contact_solve_framework.md §12 S8): the pair's contact
    # points are one position unit swept in order, so the position loop removes the overlap inside
    # the substep and its last-iteration residual (max_overlap) reads 0. The scene has two bodies,
    # no walls and no periodic axis, so every narrow-phase contact and manifold is the cylinder pair.
    assert n0 > 0, "narrow-phase missed the contact"
    assert m0 >= 1, "narrow-phase found no contact manifold between the two cylinders"
    sim.set_dt(dt)
    for i in range(1, 100):
        sim.step()
    ov = sim.max_overlap
    print(f"step 99: contacts {sim.num_contacts}  manifolds {sim.num_manifolds}  max overlap {ov:.4f}")
    p = sim.get_positions()
    assert np.isfinite(p).all()
    assert abs(p[1][0] - p[0][0]) > 0.2, "bodies did not separate"
