
import os
import sys
import math
import numpy as np

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
import demgpu

def run_cylinder_packing():
    # Parameters
    # User Request: 200 cylinders, H/D=1, Phi=0.3
    num_particles = 200
    phi = 0.3
    radius = 0.5
    diameter = 1.0
    height = 1.0 # H/D = 1
    thickness = 0.1 # Arbitrary hollow thickness
    
    # Calculate Domain Volume
    # V_cyl = H * pi * (R^2 - r_in^2) = H * pi * (2Rt - t^2)
    vol_particle = height * math.pi * (2 * radius * thickness - thickness**2)
    # vol_particle = 1.0 * pi * (0.1 - 0.01) = 0.09pi approx 0.283
    
    vol_total_particles = num_particles * vol_particle
    vol_domain = vol_total_particles / phi
    domain_side = vol_domain ** (1.0/3.0)
    
    print(f"Generating Packing for User Request:")
    print(f"Count: {num_particles}")
    print(f"Params: R={radius}, H={height}, T={thickness}")
    print(f"Particle Vol: {vol_particle:.4f}")
    print(f"Total Vol: {vol_total_particles:.4f}")
    print(f"Req Domain Vol: {vol_domain:.4f} (Cube Side: {domain_side:.2f})")
    
    # Setup Sim
    sim = demgpu.Simulation(num_particles)
    # New initialize signature
    sim.initialize(shape_type=2, radius=radius, height=height, thickness=thickness)
    
    half_d = domain_side / 2.0
    sim.set_domain((-half_d, -half_d, -half_d), (half_d, half_d, half_d))
    
    # Place particles roughly
    pos = np.zeros((num_particles, 4), dtype=np.float32)
    k = int(math.ceil(num_particles**(1.0/3.0)))
    spacing = domain_side / k
    
    idx = 0
    for z in range(k):
        for y in range(k):
            for x in range(k):
                if idx >= num_particles: break
                px = -half_d + (x + 0.5) * spacing
                py = -half_d + (y + 0.5) * spacing
                pz = -half_d + (z + 0.5) * spacing
                pos[idx] = [px, py, pz, 1.0]
                idx += 1
                
    sim.set_positions(pos)
    
    # Random Orientations
    # Cylinders need valid quaternions
    import random
    quat = np.zeros((num_particles, 4), dtype=np.float32)
    for i in range(num_particles):
        # Random axis
        u = np.random.normal(0, 1, 3)
        u /= np.linalg.norm(u)
        angle = random.uniform(0, math.pi*2)
        q = np.array([math.sin(angle/2)*u[0], math.sin(angle/2)*u[1], math.sin(angle/2)*u[2], math.cos(angle/2)])
        quat[i] = q
    sim.set_quaternions(quat)
    
    # Settle
    sim.set_gravity(0, -9.8, 0)
    print("Simulating 1000 steps to settle...")
    for i in range(1000):
        sim.step(0.005)
        
    # Export
    os.makedirs("output/sdf", exist_ok=True)
    filename = "output/sdf/packing_cylinders_200.vti"
    print(f"Exporting to {filename}...")
    sim.export_sdf(filename, (128, 128, 128))
    print("Success.")

if __name__ == "__main__":
    run_cylinder_packing()
