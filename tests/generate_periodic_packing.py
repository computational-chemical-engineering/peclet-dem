
import os
import sys
import math
import numpy as np

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
import dem

def run_periodic_growth():
    # Parameters
    num_particles = 200
    phi = 0.1
    
    # Cylinder Params (Base Scale 1.0)
    radius_base = 0.5
    diameter_base = 1.0
    height_base = 1.0 # H/D = 1
    thickness_base = 0.2
    
    # Calculate Domain for Phi=0.1 at Scale=1.0
    vol_particle = height_base * math.pi * (2 * radius_base * thickness_base - thickness_base**2)
    # Vol ~ 0.28
    
    vol_total = num_particles * vol_particle
    vol_domain = vol_total / phi
    domain_side = vol_domain ** (1.0/3.0)
    
    print(f"Periodic Packing (Phi={phi}, N={num_particles})")
    print(f"Domain Side: {domain_side:.3f}")
    
    sim = dem.Simulation(num_particles)
    sim.initialize(shape_type=2, radius=radius_base, height=height_base, thickness=thickness_base)
    
    # Set Domain (Automatic Periodicity)
    half_d = domain_side / 2.0
    sim.set_domain((-half_d, -half_d, -half_d), (half_d, half_d, half_d))
    
    # Zero Gravity (Floating/Jamming)
    sim.set_gravity(0, 0, 0)
    sim.enable_periodicity(True, True, True)
    
    # Initialize Random Positions & Orientations
    # Since we grow from 0, overlaps initially don't matter much, but good to spread.
    np.random.seed(42)
    pos = np.random.uniform(-half_d, half_d, (num_particles, 4)).astype(np.float32)
    pos[:, 3] = 1.0 # Mass
    sim.set_positions(pos)
    
    # Random Quats
    quat = np.zeros((num_particles, 4), dtype=np.float32)
    for i in range(num_particles):
        u = np.random.normal(0, 1, 3)
        u /= np.linalg.norm(u)
        angle = np.random.uniform(0, math.pi*2)
        q = np.array([math.sin(angle/2)*u[0], math.sin(angle/2)*u[1], math.sin(angle/2)*u[2], math.cos(angle/2)])
        quat[i] = q
    sim.set_quaternions(quat)
    
    # Growth Loop
    num_steps = 2000
    # Growth phase: 0 to 80% steps? Or 100%?
    # Let's grow for 1000 steps, then settle for 1000.
    grow_steps = 1000
    settle_steps = 1000
    
    print("Starting Growth Phase...")
    scales = np.zeros(num_particles, dtype=np.float32)
    
    for i in range(grow_steps):
        s = (i + 1) / grow_steps
        scales[:] = s
        sim.set_scales(scales)
        sim.step(0.005)
        if i % 100 == 0:
            print(f"Step {i}/{grow_steps}: Scale {s:.3f}")
            
    print("Starting Settle Phase (Scale 1.0)...")
    scales[:] = 1.0
    sim.set_scales(scales)
    for i in range(settle_steps):
        sim.step(0.005)
        if i % 200 == 0:
            print(f"Settle {i}/{settle_steps}")
            
    # Export
    os.makedirs("output/sdf", exist_ok=True)
    filename = "output/sdf/periodic_cylinders.vti"
    print(f"Exporting to {filename}...")
    sim.export_sdf(filename, (256, 256, 256))
    print("Success.")

if __name__ == "__main__":
    run_periodic_growth()
