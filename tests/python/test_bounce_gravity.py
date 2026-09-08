"""A sphere dropped on a plane with e=0.5 must come to rest on it within 2 s."""
import numpy as np
from peclet import dem


def test_bounce_gravity():
    sim = dem.Simulation(1)
    sim.initialize_shape(0, radius=0.5)
    sim.set_material_params(0.5, 0.0, 0.0)
    sim.set_solver_iterations(8, 4)             # the velocity solve is OFF by default (0 iterations)
    sim.set_gravity(0, -9.8, 0)
    sim.add_plane([0, -1.0, 0], [0, 1.0, 0])    # floor at y=-1
    sim.set_positions(np.array([[0, 1.0, 0]], dtype=np.float32))
    sim.set_scales(np.array([1.0], dtype=np.float32))   # radius 0.5 -> rests at y = -0.5
    sim.set_velocities(np.zeros((1, 3), dtype=np.float32))
    dt = 0.01
    for i in range(200):
        sim.step(dt)
        if i % 50 == 0:
            p = sim.get_positions()[0]
            v = sim.get_velocities()[0]
            print(f"step {i}: y={p[1]:.4f} vy={v[1]:.4f}")
    p = sim.get_positions()[0]
    v = sim.get_velocities()[0]
    print(f"final: y={p[1]:.4f} vy={v[1]:.4f}")
    assert abs(v[1]) < 0.2, "particle did not settle"
    assert abs(p[1] - (-0.5)) < 0.05, "particle (radius 0.5) is not resting on the floor at y=-1"
