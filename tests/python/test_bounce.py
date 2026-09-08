"""Two touching elastic spheres approaching head-on must bounce (the velocity solver)."""
import numpy as np
from peclet import dem


def test_bounce():
    sim = dem.Simulation(2)
    sim.initialize_shape('sphere', 1.0, 0, 0)          # sphere, radius 1
    sim.set_material_params(1.0, 0.0, 0.0)      # elastic
    sim.set_solver_iterations(8, 4)             # the velocity solve is OFF by default (0 iterations)
    sim.set_gravity((0, 0, 0))
    # 0.01 overlap so the narrow-phase sees a contact on the first step
    sim.set_positions(np.array([[-0.995, 0, 0], [0.995, 0, 0]], dtype=np.float32))
    sim.set_scales(np.array([1.0, 1.0], dtype=np.float32))
    sim.set_velocities(np.array([[1.0, 0, 0], [-1.0, 0, 0]], dtype=np.float32))
    sim.set_dt(0.01)
    sim.step()
    v = sim.get_velocities()
    print(f"after one step: v0={v[0]}  v1={v[1]}")
    assert v[0][0] < 0 and v[1][0] > 0, "particles did not bounce"
    assert abs(v[0][0] + v[1][0]) < 1e-4, "momentum not conserved"
