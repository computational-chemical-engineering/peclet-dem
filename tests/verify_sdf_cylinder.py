
import os
import sys
import numpy as np
import time

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
import dem

def run_cylinder_sdf_test():
    # Setup
    num_particles = 10
    sim = dem.Simulation(num_particles)
    sim.initialize(shape_type=2) # Cylinder (ID=2)
    
    # Domain
    domain_size = 10.0
    sim.set_domain((-domain_size, -domain_size, -domain_size), 
                   (domain_size, domain_size, domain_size))
    
    # Set positions (Grid)
    pos = np.zeros((num_particles, 4), dtype=np.float32)
    quat = np.zeros((num_particles, 4), dtype=np.float32)
    quat[:, 3] = 1.0 # Identity
    
    # Place one cylinder at origin
    pos[0] = [0, 0, 0, 1.0] 
    
    # Place another rotated 90 degrees around X
    pos[1] = [3, 0, 0, 1.0]
    # Euler to Quat (90 deg over X)
    # q = [sin(45) 0 0 cos(45)]
    s = np.sin(np.pi/4)
    c = np.cos(np.pi/4)
    quat[1] = [s, 0, 0, c]

    sim.set_positions(pos)
    # Need to expose set_quaternions if not already?
    # Based on main_binding.cpp, set_quaternions is NOT exposed.
    # Wait, getting quats is exposed, setting is not?
    # Checking main_binding.cpp... set_quaternions is missing!
    # I should add it.
    
    # But wait, looking at my memory of main_binding.cpp...
    # I see .def("get_quaternions", ...)
    # I do NOT see .def("set_quaternions", ...) in the view_file output.
    # I must add it to be able to test rotation.
    
    # Export SDF
    os.makedirs("output/sdf", exist_ok=True)
    filename = "output/sdf/cylinder_test.vti"
    print(f"Exporting SDF to {filename}...")
    sim.export_sdf(filename, (64, 64, 64))
    
    if os.path.exists(filename):
        print("Success: File created.")
    else:
        print("Error: File not created.")

if __name__ == "__main__":
    run_cylinder_sdf_test()
