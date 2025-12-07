import demgpu
import numpy as np
import time

def verify_spheres():
    print("Initializing Simulation with SPHERES (Type 0)...")
    num_particles = 1000
    sim = demgpu.Simulation(num_particles)
    sim.initialize(shape_type=0) # 0 for Sphere

    # Set scales to 1.0
    scales = np.ones(num_particles, dtype=np.float32)
    sim.set_scales(scales)
    
    print("Running settling phase...")
    t0 = time.time()
    for i in range(200):
        sim.step(0.002)
    print(f"Simulation done in {time.time()-t0:.3f}s")
    
    # Analysis
    pos = sim.get_positions().reshape(-1, 4) # x,y,z,w
    scales = sim.get_scales()
    
    # Check bounds (roughly)
    print(f"Positions range: X[{pos[:,0].min():.2f}, {pos[:,0].max():.2f}] Y[{pos[:,1].min():.2f}, {pos[:,1].max():.2f}]")
    
    # Check Overlaps
    # Brute force O(N^2) is fine for 1000 particles for verification
    print("Checking overlaps...")
    
    positions = pos[:, :3]
    radii = 0.5 * scales # Base radius 0.5
    
    max_overlap = 0.0
    overlap_count = 0
    
    # Simple spatial hash or just brute force
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            dist = np.linalg.norm(positions[i] - positions[j])
            sum_radii = radii[i] + radii[j]
            if dist < sum_radii:
                overlap = sum_radii - dist
                if overlap > 0.01: # Tolerance
                    overlap_count += 1
                    max_overlap = max(max_overlap, overlap)
                    
    print(f"Overlap Analysis:")
    print(f"  Total pairs checked: {num_particles*(num_particles-1)//2}")
    print(f"  Significant Overlaps (>0.01): {overlap_count}")
    print(f"  Max Overlap: {max_overlap:.4f}")
    
    if overlap_count < 100: # Some overlaps expected during settling or if compressed
        print("PASS: Sphere packing looks physically reasonable (low overlap).")
    else:
        print("FAIL: Too many overlaps!")

if __name__ == "__main__":
    verify_spheres()
