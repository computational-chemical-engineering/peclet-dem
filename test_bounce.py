import dem
import numpy as np

def test_bounce():
    print("--- 2-Particle Bounce Test (Velocity Solver) ---")
    
    # Init
    sim = dem.Simulation(2)
    sim.initialize(0, 1.0, 0, 0) # Sphere, Radius 1.0
    
    # Set Params: Elastic Bounce (e=1.0)
    sim.set_material_params(1.0, 0.0, 0.0) 
    sim.set_gravity(0, 0, 0) # No gravity
    
    # Setup Positions: Touching
    # P0 at -1.0, P1 at +1.0. Radius 1.0. 
    # Dist = 2.0. Overlap = 0? Or maybe small overlap to trigger contact?
    # Narrowphase usually needs margin or overlap.
    # Let's put slightly overlapping: 0.01 overlap.
    # Dist 1.99.
    # P0 = -0.995, P1 = 0.995
    pos = np.array([
        [-0.995, 0, 0],
        [ 0.995, 0, 0]
    ], dtype=np.float32)
    
    sim.set_positions(pos)
    
    scales = np.array([1.0, 1.0], dtype=np.float32)
    sim.set_scales(scales)
    
    # Velocities: Approaching at speed 1.0 each
    vel = np.array([
        [ 1.0, 0, 0],
        [-1.0, 0, 0]
    ], dtype=np.float32)
    sim.set_velocities(vel)
    
    print("Initial State:")
    print(f"Vel 0: {vel[0]}")
    print(f"Vel 1: {vel[1]}")
    
    dt = 0.01
    
    # Run 1 Step
    print("\nRunning Step 1...")
    sim.step(dt)
    
    p = sim.get_positions()
    v = sim.get_velocities()
    
    print(f"Pos 0: {p[0]}")
    print(f"Pos 1: {p[1]}")
    print(f"Vel 0: {v[0]}")
    print(f"Vel 1: {v[1]}")
    
    # Expected:
    # Elastic bounce => Vel reflected.
    # V0 should be approx -1.0.
    # V1 should be approx +1.0.
    
    if v[0][0] < 0 and v[1][0] > 0:
        print("SUCCESS: Particles bounced!")
    else:
        print("FAILURE: Particles did not bounce correctly.")

if __name__ == "__main__":
    test_bounce()
