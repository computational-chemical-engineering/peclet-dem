from peclet import dem
import numpy as np
import time

def test_bounce_gravity():
    print("--- 1-Particle Gravity Bounce Test ---")
    
    # Init
    sim = dem.Simulation(1)
    sim.initialize(0) # Sphere
    
    # Material: Restitution=0.5
    sim.set_material_params(0.5, 0.0, 0.0) 
    sim.set_gravity(0, -9.8, 0)
    
    # Add Floor Plane
    sim.add_plane([0, -1.0, 0], [0, 1.0, 0])
    
    # Initial: Height 1.0 above floor (y=0, Floor=-1, R=1 -> Touching)
    # Let's drop from y=1.0 (Gap 1.0).
    pos = np.array([[0, 1.0, 0]], dtype=np.float32)
    scales = np.array([1.0], dtype=np.float32) # Radius 1.0
    
    sim.set_positions(pos)
    sim.set_scales(scales)
    sim.set_velocities(np.zeros((1, 3), dtype=np.float32))
    
    dt = 0.01 
    steps = 200 # 2 seconds
    
    print(f"Dropping particle... (dt={dt})")
    
    for i in range(steps):
        sim.step(dt)
        p = sim.get_positions()[0]
        v = sim.get_velocities()[0]
        
        # Monitor y and vy
        if i % 10 == 0:
            print(f"Step {i}: Y={p[1]:.4f}, Vy={v[1]:.4f}")
            
    # Check final state
    p = sim.get_positions()[0]
    v = sim.get_velocities()[0]
    print(f"Final: Y={p[1]:.4f}, Vy={v[1]:.4f}")
    
    if abs(v[1]) < 0.2:
        print("SUCCESS: Particle settled (approx).")
    else:
        print("FAILURE: Particle did not settle.")

if __name__ == "__main__":
    test_bounce_gravity()
