import os
import math
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

import demgpu

def run_cylinder_test():
    # 1. Setup
    num_particles = 100
    sim = demgpu.Simulation(num_particles)
    
    # Initialize with ShapeType 1 (Hollow Cylinder)
    sim.initialize(shape_type=1)
    
    sim.set_domain((-5, -5, -5), (5, 5, 5))
    sim.set_gravity(0, -9.81, 0)
    sim.set_global_scale(1.0)
    # Materials (restitution, tangent_restitution, friction)
    sim.set_material_params(0.5, 0.5, 0.5)
    
    # 2. Arrange Particles (Grid to avoid overlap)
    positions = np.zeros((num_particles, 4), dtype=np.float32)
    quaternions = np.zeros((num_particles, 4), dtype=np.float32)
    
    # Grid parameters
    grid_dim = int(math.ceil(num_particles**(1/3.0)))
    spacing = 1.5 # Cylinder is height 2.0, radius 0.5. Max dim 2.0. So 2.5 spacing safe?
    # R=0.5, H=2.0.
    
    idx = 0
    for x in range(grid_dim):
        for y in range(grid_dim):
            for z in range(grid_dim):
                if idx >= num_particles:
                    break
                # Center around 0, start high up
                px = (x - grid_dim/2) * spacing
                py = (y - grid_dim/2) * spacing + 10.0 # Drop height
                pz = (z - grid_dim/2) * spacing
                
                positions[idx] = [px, py, pz, 1.0] # w=1 (inv_mass)
                
                # Randomize orientation slightly?
                # Axis: (0,1,0). Small random rotation.
                # Identity for now to check stacking stability
                quaternions[idx] = [0, 0, 0, 1] 
                
                idx += 1
                
    sim.set_positions(positions)
    # sim.set_quaternions(quaternions) # If binding exists? No, only set_positions/velocities exposed?
    # Let's check bindings. get_quaternions exists. set_quaternions MISSING in main_binding.cpp!
    # Simulation::initialize sets internal quats to identity.
    # If I want random orientations, I need set_quaternions binding!
    # For now, start with Identity (Vertical alignment).
    
    # 3. Time Loop
    dt = 0.005
    steps = 1000
    
    os.makedirs("output/cylinders", exist_ok=True)
    
    print(f"Starting Cylinder Verification with {num_particles} particles...")
    for i in range(steps):
        sim.step(dt)
        
        if i % 10 == 0:
            vel = sim.get_velocities()
            v_max = np.max(np.linalg.norm(vel[:, :3], axis=1))
            print(f"Step {i}: Max Vel = {v_max:.4f}")
            
            sim.write_vtp(f"output/cylinders/step_{i:04d}.vtp")
            
            if v_max > 100.0:
                print("Explosion detected!")
                break

if __name__ == "__main__":
    run_cylinder_test()
