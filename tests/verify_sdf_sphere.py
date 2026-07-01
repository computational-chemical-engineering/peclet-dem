
import os
import sys
import numpy as np
import time

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
from peclet import dem

def run_sphere_sdf_test():
    # Setup
    num_particles = 10
    sim = dem.Simulation(num_particles)
    sim.initialize(shape_type=1) # Sphere (ID=1)
    
    # Domain
    domain_size = 10.0
    sim.set_domain((-domain_size, -domain_size, -domain_size), 
                   (domain_size, domain_size, domain_size))
    
    # Set positions (Grid)
    pos = np.zeros((num_particles, 4), dtype=np.float32)
    # Place one sphere at origin, r=1.0
    pos[0] = [0, 0, 0, 1.0] 
    # Place another at (3,0,0)
    pos[1] = [3, 0, 0, 1.0]

    sim.set_positions(pos)
    
    # Export SDF
    os.makedirs("output/sdf", exist_ok=True)
    filename = "output/sdf/sphere_test.vti"
    print(f"Exporting SDF to {filename}...")
    sim.export_sdf(filename, (64, 64, 64))
    
    if os.path.exists(filename):
        print("Success: File created.")
    else:
        print("Error: File not created.")

if __name__ == "__main__":
    run_sphere_sdf_test()
