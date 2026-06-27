
import sys
import os

# Adjust path to find the build module
sys.path.append(os.path.join(os.getcwd(), 'build'))

import dem
import numpy as np

def verify_periodic_vti():
    # 1. Initialize Simulation
    sim = dem.Simulation(num_particles=1)
    
    # Domain 10x10x10, centered at 0
    sim.set_domain((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))
    
    # Enable Periodicity in X only for this test
    # But wait, default is all periodic if we use constructor? No, default is false unless set_domain used.
    # set_domain enabled periodicity by default.
    sim.enable_periodicity(True, True, True) # X periodic
    
    # 2. Place particle at X boundary
    # Domain is -5 to 5. width 10.
    # Place at x = -4.5. Radius 1.0. 
    # Initialize radius first to allocate memory
    #sim.initialize(shape_type=2, radius=1.0) # Sphere
    sim.initialize(shape_type=2, radius=1.0, height=1.0, thickness=0.1)
    pos = np.array([[-4.9, -4.9, 4.9, 1.0]], dtype=np.float32) # Added w=1.0 for mass
    sim.set_positions(pos)
    sim.export_sdf("output/test_periodic.vti", (256, 256, 256))

    # Numerical Verification
    res = (20, 10, 10)
    sdf_flat = sim.get_sdf_grid(res)
    sdf = sdf_flat.reshape((10, 10, 20)) #(Z, Y, X)
    
    # Check left side (near particle) - should be similar to right
    # Particle at x=-4.9. Boundary at x=-5.0.
    # wrapped x=-0.1 relative to -4.9 is 0.8? No.
    # Dist at -5.0 is dist to center (-4.9) = 0.1.
    # Dist at +5.0 should be same.
    
    # Left edge index 0 -> x = -4.75. Dist to -4.9 = 0.15.
    # Right edge index 19 -> x = 4.75.
    # 4.75 is 0.25 from 5.0. 5.0 wraps    # Particle at X=-4.9, Y=-4.9, Z=4.9.
    # Resolution (20, 10, 10).
    # X (20): -4.9 -> Index 0. Wrapped Right -> Index 19.
    # Y (10): -4.9 -> Index 0.
    # Z (10): 4.9 -> Index 9.
    
    # Check left side (near particle)
    # Z=9, Y=0, X=0
    dist_left = sdf[9, 0, 0]
    print(f"Dist at x=0 (Left): {dist_left}")
    
    # Check right side (should be wrapped)
    # Z=9, Y=0, X=19
    dist_right = sdf[9, 0, 19]
    print(f"Dist at x=19 (Right): {dist_right}")
    
    if dist_right > 2.0:
        print(f"FAILURE: Right side dist {dist_right} too high!")
    else:
        print("SUCCESS: Right side detects particle across boundary.")

if __name__ == "__main__":
    verify_periodic_vti()
