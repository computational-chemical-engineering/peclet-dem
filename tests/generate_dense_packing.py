
import os
import sys
import math
import numpy as np

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
import demgpu

def run_dense_packing():
    # Parameters
    num_particles = 300 # Dense pack
    # Target high phi. Let's aim for 0.60?
    phi_target = 0.60
    
    radius_base = 0.5
    height_base = 1.0 # H/D=1
    thickness_base = 0.2
    
    # Volume at Scale 1.0
    vol_particle = height_base * math.pi * (2 * radius_base * thickness_base - thickness_base**2)
    vol_total = num_particles * vol_particle
    vol_domain = vol_total / phi_target
    domain_side = vol_domain ** (1.0/3.0)
    
    print(f"Goal: Dense Packing (Phi={phi_target}, N={num_particles})")
    print(f"Domain Side: {domain_side:.3f}")
    
    sim = demgpu.Simulation(num_particles)
    sim.initialize(shape_type=2, radius=radius_base, height=height_base, thickness=thickness_base)
    
    half_d = domain_side / 2.0
    sim.set_domain((-half_d, -half_d, -half_d), (half_d, half_d, half_d))
    sim.set_gravity(0, 0, 0)
    
    # Init Random
    np.random.seed(42)
    pos = np.random.uniform(-half_d, half_d, (num_particles, 4)).astype(np.float32)
    pos[:, 3] = 1.0
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
    scale = 0.1
    max_scale = 1.2 # Overshoot slightly?
    growth_rate = 0.001
    overlap_tolerance = 0.02 # 2% of unit size
    
    print("Starting Compression...")
    scales = np.zeros(num_particles, dtype=np.float32)
    
    steps_per_growth = 10
    total_steps = 0
    
    while scale <= max_scale:
        scales[:] = scale
        sim.set_scales(scales)
        
        # Relax
        for _ in range(steps_per_growth):
            sim.step(0.005)
            
        total_steps += steps_per_growth
            
        # Check Overlap
        max_ov = sim.get_max_overlap()
        
        if total_steps % 100 == 0:
            phi_current = (vol_total * scale**3) / (domain_side**3)
            print(f"Scale: {scale:.3f}, Phi: {phi_current:.3f}, Max Overlap: {max_ov:.4f}")
            
        if max_ov > overlap_tolerance:
            print(f"Overlap Exceeded Tolerance ({max_ov:.4f} > {overlap_tolerance}). Attempting to settle...")
            # Try to settle for 500 steps
            settled = False
            for k in range(5):
                for _ in range(100): sim.step(0.005)
                max_ov = sim.get_max_overlap()
                if max_ov < overlap_tolerance:
                    settled = True
                    break
                print(f"  Settle {k*100}: Ov={max_ov:.4f}")
            
            if not settled:
                print(f"CRITICAL: Could not resolve overlap at Scale {scale:.3f}. Stopping growth.")
                break
        
        scale += growth_rate
        
    print(f"Final State: Scale {scale:.3f}")
    
    # Final Settle
    print("Final Settle (1000 steps)...")
    for _ in range(1000):
        sim.step(0.005)
    
    max_ov = sim.get_max_overlap()
    phi_final = (vol_total * scale**3) / (domain_side**3)
    print(f"Final Stats: Phi={phi_final:.3f}, Max Overlap={max_ov:.4f}")
    
    os.makedirs("output/sdf", exist_ok=True)
    sim.export_sdf("output/sdf/dense_packing.vti", (128, 128, 128))
    print("Exported SDF.")

if __name__ == "__main__":
    run_dense_packing()
