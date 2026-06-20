
import os
import sys
import numpy as np
import math

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
import dem

def run_packing_sdf():
    # Parameters
    domain_size = 10.0 # [-5, 5]
    radius = 0.5
    phi = 0.45
    
    # Calculate N
    vol_domain = domain_size ** 3
    vol_particle = (4.0/3.0) * math.pi * (radius ** 3)
    num_particles = int((phi * vol_domain) / vol_particle)
    
    print(f"Generating packing for Phi={phi}")
    print(f"Domain Payload: {vol_domain}")
    print(f"Particle Radius: {radius} (Vol {vol_particle:.4f})")
    print(f"Num Particles: {num_particles}")
    
    # Simulation Setup
    sim = dem.Simulation(num_particles) # Exact count
    sim.initialize(shape_type=1) # Sphere
    
    half_d = domain_size / 2.0
    sim.set_domain((-half_d, -half_d, -half_d), 
                   (half_d, half_d, half_d))
    
    # Initialize Positions (Grid to avoid overlap)
    # 10x10x10 domain. 
    # Try to fit roughly cube root(N) per dim
    k = int(math.ceil(num_particles**(1.0/3.0))) # e.g. 10
    spacing = domain_size / k
    
    pos = np.zeros((num_particles, 4), dtype=np.float32)
    scales = np.ones(num_particles, dtype=np.float32) * radius
    
    idx = 0
    for z in range(k):
        for y in range(k):
            for x in range(k):
                if idx >= num_particles: break
                
                # Center coords
                px = -half_d + (x + 0.5) * spacing
                py = -half_d + (y + 0.5) * spacing
                pz = -half_d + (z + 0.5) * spacing
                
                # Jitter slightly
                px += np.random.uniform(-spacing*0.1, spacing*0.1)
                py += np.random.uniform(-spacing*0.1, spacing*0.1)
                pz += np.random.uniform(-spacing*0.1, spacing*0.1)
                
                pos[idx] = [px, py, pz, 1.0] # w=inv_mass/type
                idx += 1
    
    sim.set_positions(pos)
    sim.set_scales(scales) # Set radius specifically
    
    # Run a few steps to let them jiggle (optional, but good for validity)
    sim.set_gravity(0, -9.8, 0)
    print("Simulating 500 steps to settle...")
    for i in range(500):
        sim.step(0.005)
        
    # Export SDF
    os.makedirs("output/sdf", exist_ok=True)
    filename = "output/sdf/packing_phi045.vti"
    print(f"Exporting SDF to {filename}...")
    # Higher resolution for packing
    sim.export_sdf(filename, (128, 128, 128))
    
    if os.path.exists(filename):
        print("Success: File created.")

if __name__ == "__main__":
    run_packing_sdf()
