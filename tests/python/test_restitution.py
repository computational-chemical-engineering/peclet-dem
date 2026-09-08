"""Boundary restitution + sliding friction on an explicit floor plane at y = 0."""
import numpy as np
from peclet import dem


def _sim():
    sim = dem.Simulation(1)
    sim.set_domain(np.array([-10, 0, -10], dtype=np.float32), np.array([10, 20, 10], dtype=np.float32))
    sim.add_plane([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])   # floor at y = 0
    sim.set_gravity(0.0, -9.8, 0.0)                    # default gravity is ZERO
    sim.set_solver_iterations(8, 4)                    # the velocity solve is OFF by default
    return sim


def test_normal_restitution():
    """Drop from h=5 on the floor with e_n = 0.8: the bounce velocity ratio must be ~0.8."""
    sim = _sim()
    sim.set_material_params(0.8, 0.0, 0.0)
    sim.initialize_shape(0, radius=0.5)
    sim.set_positions(np.array([[0, 5.0, 0, 1.0]], dtype=np.float32))
    sim.set_velocities(np.zeros((1, 4), dtype=np.float32))
    sim.set_scales(np.ones(1, dtype=np.float32))
    dt = 0.01
    bounce = None
    for i in range(200):
        v_prev = sim.get_velocities()[0][1]
        sim.step(dt)
        v = sim.get_velocities()[0][1]
        if v_prev < 0 and v > 0:
            bounce = (i, v_prev, v)
            break
    assert bounce is not None, "did not bounce"
    i, v_in, v_out = bounce
    ratio = -v_out / v_in
    print(f"bounce at step {i}: v_in={v_in:.4f} v_out={v_out:.4f} ratio={ratio:.4f} (want 0.8)")
    assert abs(ratio - 0.8) < 0.1


def test_sliding_friction():
    """A sphere sliding on the floor with mu = 0.5 loses horizontal velocity."""
    sim = _sim()
    sim.set_material_params(0.5, 0.0, 0.5)
    sim.initialize_shape(0, radius=0.5)
    sim.set_positions(np.array([[0, 2.0, 0, 1.0]], dtype=np.float32))
    sim.set_velocities(np.array([[5.0, -1.0, 0.0, 0.0]], dtype=np.float32))
    sim.set_scales(np.ones(1, dtype=np.float32))
    for _ in range(150):
        sim.step(0.01)
    vx = float(sim.get_velocities()[0][0])
    print(f"vx: 5.0 -> {vx:.4f}")
    assert vx < 5.0 - 0.1, "no friction observed"
