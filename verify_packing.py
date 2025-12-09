import demgpu
import numpy as np
import time
import sys

def verify_packing(target_density, scale):
    print(f"--- Verifying Packing Density {target_density} (Scale {scale}) ---")
    
    # Initialize
    # N=1000, Box=1000
    sim = demgpu.Simulation(1000)
    sim.initialize(0) # 0 = Sphere
    
    # zero restitution and friction for stability testing
    sim.set_material_params(0.0, 0.0, 0.0)
    
    # Set Domain (verify defaults)
    # Default is -5 to 5 (size 10), Volume 1000.
    
    # Initial Setup with small particles to avoid overlap
    initial_scale = 0.1
    scales = np.full(1000, initial_scale, dtype=np.float32)
    sim.set_scales(scales)
    
    # Settle?
    print("Settling initial state...")
    for i in range(50):
        sim.step(0.001)
        
    # Expand to target
    print(f"Expanding to scale {scale}...")
    scales[:] = scale
    sim.set_scales(scales)
    
    particle_radii = scales * 1.0 
    
    def compute_overlap_stats(pos, radii):
        n = len(pos)
        max_ov = 0.0
        tot_ov = 0.0
        count = 0
        # Simple N^2 check is fine for N=1000 (0.5M pairs)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(pos[i] - pos[j])
                rad_sum = radii[i] + radii[j]
                if dist < rad_sum:
                    ov = rad_sum - dist
                    max_ov = max(max_ov, ov)
                    tot_ov += ov
                    count += 1
        mean_ov = tot_ov / count if count > 0 else 0.0
        return count, max_ov, mean_ov

    # Run Relaxation
    dt = 0.001
    max_steps = 5000
    print(f"Running relaxation for {max_steps} steps...")
    
    for i in range(max_steps):
        sim.step(dt)
        
        if i % 100 == 0:
            vels = sim.get_velocities()
            v_sq = np.sum(vels**2, axis=1)
            mean_ke = 0.5 * np.mean(v_sq)
            max_v = np.max(np.sqrt(v_sq))
            
            # Expensive overlap check
            pos = sim.get_positions()
            ov_count, ov_max, ov_mean = compute_overlap_stats(pos, particle_radii)
            
            print(f"Step {i}: Max Vel={max_v:.4f}, Mean KE={mean_ke:.6f}, Overlaps={ov_count}, Max Ov={ov_max:.6f}, Mean Ov={ov_mean:.6f}")

    # Final Check
    vels = sim.get_velocities()
    v_mag = np.linalg.norm(vels, axis=1)
    final_max_v = np.max(v_mag)
    
    print(f"Final Max Velocity: {final_max_v:.6f}")
    
    # Calculate Kinetic Energy
    vels = sim.get_velocities()
    v_sq = np.sum(vels**2, axis=1)
    mean_ke = 0.5 * np.mean(v_sq) # Assuming mass=1 for simplicity
    print(f"Final Mean Kinetic Energy: {mean_ke:.6f}")

    # Check Overlaps (Brute Force)
    pos = sim.get_positions()
    radii = scales * 0.5 # Assuming unit sphere r=0.5 * scale? verify broadphase
    # Broadphase used radius = 1.0 * s; (where s = scale * global_scale). global_scale=1.0.
    # So radius = scale.
    # Verify shape type 0 radius in `initialize`.
    # Simulation::initialize(0) calls shapes/point_sampler.cpp usually.
    # Let's assume radius = scale * 1.0 based on my fix in Step 546.
    
    particle_radii = scales * 1.0 
    
    print("Checking overlaps...")
    max_overlap = 0.0
    total_overlap = 0.0
    overlap_count = 0
    
    n = len(pos)
    # Vectorized distance check
    # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # dist = np.linalg.norm(diff, axis=2)
    # rad_sum = particle_radii[:, np.newaxis] + particle_radii[np.newaxis, :]
    # overlaps = dist - rad_sum
    # # Ignore self-collision (diagonal)
    # np.fill_diagonal(overlaps, 100.0) 
    
    # Memory efficient loop
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pos[i] - pos[j])
            rad_sum = particle_radii[i] + particle_radii[j]
            if dist < rad_sum:
                overlap = rad_sum - dist
                max_overlap = max(max_overlap, overlap)
                total_overlap += overlap
                overlap_count += 1
                
    print(f"Overlap Analysis:")
    print(f"  Count: {overlap_count}")
    print(f"  Max Overlap: {max_overlap:.6f}")
    if overlap_count > 0:
        print(f"  Mean Overlap: {total_overlap / overlap_count:.6f}")
    
    output_filename = f"packing_density_{target_density:.2f}.vtp"
    sim.write_vtp(output_filename)
    print(f"Exported to {output_filename}")
    
    if final_max_v > 1.0: # Arbitrary high threshold for "Explosion"
        print("FAILURE: High velocity detected. Possible explosion.")
        return False
    
    return True

if __name__ == "__main__":
    # Density 0.45 -> Scale ~0.475
    # Density 0.64 -> Scale ~0.534
    
    success_045 = verify_packing(0.45, 0.475)
    
    if not success_045:
        print("Density 0.45 FAILED")
