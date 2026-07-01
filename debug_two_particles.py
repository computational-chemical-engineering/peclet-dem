from peclet import dem
import numpy as np

def test_two_particle_overlap():
    print("--- 2-Particle Overlap Test ---")
    
    # Init
    sim = dem.Simulation(2)
    sim.initialize(0) # Sphere
    
    # Set Params
    sim.set_material_params(0.0, 0.0, 0.5) # Zero restitution
    sim.set_gravity(0, 0, 0) # No gravity
    
    # Setup Positions: Overlap by 0.2 (Radius 1.0, Dist 1.8)
    # Radii will be 1.0 (Scale 1.0)
    # Pos 0: (-0.9, 0, 0)
    # Pos 1: ( 0.9, 0, 0)
    # Dist = 1.8. 
    
    pos = np.array([
        [-0.9, 0, 0],
        [ 0.9, 0, 0]
    ], dtype=np.float32)
    
    sim.set_positions(pos)
    
    scales = np.array([1.0, 1.0], dtype=np.float32)
    sim.set_scales(scales)
    
    vel = np.zeros((2, 3), dtype=np.float32)
    sim.set_velocities(vel)
    
    print("Initial State:")
    print(f"Pos 0: {pos[0]}")
    print(f"Pos 1: {pos[1]}")
    
    dt = 0.001
    
    # Run Steps
    for i in range(5):
        sim.step(dt)
        
        p = sim.get_positions()
        v = sim.get_velocities()
        
        print(f"--- Step {i+1} ---")
        print(f"Pos 0: {p[0]}")
        print(f"Pos 1: {p[1]}")
        print(f"Vel 0: {v[0]}")
        print(f"Vel 1: {v[1]}")
        
        # Calc separation speed
        rel_v = v[0] - v[1]
        speed = np.linalg.norm(rel_v)
        dist = np.linalg.norm(p[0][:3] - p[1][:3])
        print(f"Dist: {dist:.4f}, Rel Speed: {speed:.4f}")

if __name__ == "__main__":
    test_two_particle_overlap()
