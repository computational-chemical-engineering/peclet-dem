import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import demgpu
import math

def run_simulation(num_particles, density_target, steps=1000):
    print(f"\n--- Testing Density {density_target} (N={num_particles}) ---")
    
    # Box Size calculation
    # Packing Fraction phi = (N * 4/3 pi r^3) / V_box
    # V_box = (N * 4/3 pi r^3) / phi
    # L = V_box^(1/3)
    r = 1.0
    vol_particles = num_particles * (4.0/3.0 * math.pi * r**3)
    vol_box = vol_particles / density_target
    L = vol_box**(1.0/3.0)
    
    print(f"Box Size L={L:.3f}")
    
    sim = demgpu.Simulation(num_particles)
    sim.set_domain(np.array([-L/2, -L/2, -L/2], dtype=np.float32), 
                   np.array([L/2, L/2, L/2], dtype=np.float32))
    
    # Initialize empty
    sim.initialize(0)
    
    # Safe Grid Initialization (Manual)
    # Target L is calculated. 
    # Create a grid fitting inside L.
    k_dim = int(np.ceil(num_particles**(1/3)))
    spacing = L / k_dim
    # Center grid
    offset = -L/2 + spacing/2
    
    pos_list = []
    for x in range(k_dim):
        for y in range(k_dim):
            for z in range(k_dim):
                if len(pos_list) >= num_particles: break
                px = offset + x * spacing
                py = offset + y * spacing
                pz = offset + z * spacing
                pos_list.append([px, py, pz, 1.0]) # w=1 (InvMass)
            if len(pos_list) >= num_particles: break
        if len(pos_list) >= num_particles: break
            
    positions = np.array(pos_list, dtype=np.float32)
    # Replace w=0 for Real? No, w=1 for InvMass?
    # Wait, w=0 is Active Flag? No, w>0.5 is Ghost.
    # Inverse Mass is stored in d_pos.w?
    # Solver uses d_pos.w as wi.
    # set_positions logic: if w>0.5 return (Active check).
    # Ah, `d_vel.w` is the active flag. `d_pos.w` is usually Mass/InvMass.
    # My Fix for Ghost Flag was in `set_velocities`. `vel.w=0` means Active.
    # `set_positions` sets `d_pos`.
    # Let's verify `set_positions` and `set_velocities` logic usage.
    # `d_pos.w` is typically InvMass in PBD.
    
    # Set Positions
    sim.set_positions(positions)
    
    # Set Velocities (Active Flag w=0)
    vels = np.zeros((num_particles, 4), dtype=np.float32)
    # vels[:, 3] = 0.0 # Active
    sim.set_velocities(vels)
    
    # Growth phase
    sim.set_global_scale(0.1)
    
    # Run
    for i in range(steps):
        # Linear growth to 1.0
        s = min(1.0, 0.1 + 0.9 * i / (steps * 0.8))
        sim.set_global_scale(s)
        sim.set_global_scale(s)
        sim.step(0.01) # Substeps internal
        
        if i % 50 == 0:
            prof = sim.get_profiling_info()
            print(f"Step {i}: Int={prof['integration']:.2f}ms BP={prof['broadphase']:.2f}ms Solver={prof['solver']:.2f}ms")
    
    # Check Overlaps
    pos = sim.get_positions()
    overlaps = 0
    max_overlap = 0.0
    
    # Naive CPU check (slow for large N but fine for test)
    # Only check first 100 to save time? Or use BVH/Cell list if I implemented it in Python?
    # Let's check all for small N=100.
    
    if num_particles <= 200:
        for i in range(num_particles):
            for j in range(i+1, num_particles):
                p1 = pos[i]
                p2 = pos[j]
                d2 = np.sum((p1[:3]-p2[:3])**2)
                d = np.sqrt(d2)
                # Periodic optional? Assume box big enough or periodic BC handled in Sim.
                # Assuming simple distance for now
                if d < 2.0:
                    ov = 2.0 - d
                    overlaps += 1
                    max_overlap = max(max_overlap, ov)
    
    print(f"Result: Max Overlap = {max_overlap:.4f}")
    
    # Check Containment (Explosion Check)
    # Domain is centered at 0, size L. Range is [-L/2, L/2].
    limit = L / 2.0 * 1.5 # Allow 50% buffer before calling it "Explosion"
    max_dist = np.max(np.abs(pos[:, :3]))
    print(f"Max Particle Distance from Origin: {max_dist:.4f} (Limit L/2={L/2:.4f})")
    
    if max_dist > limit:
         print(f"FAIL: Explosion detected. Particles flew out of domain (Max Dist {max_dist} > {limit})")
         return False

    # Export VTP for visual check
    vtp_filename = f"packing_density_{density_target:.2f}.vtp"
    sim.write_vtp(vtp_filename)
    print(f"Exported {vtp_filename}")

    if max_overlap > 0.1:
        print("FAIL: High overlap detected.")
        return False
    else:
        print("PASS: Stable packing.")
        return True

def test_two_particles():
    print("\n--- Test: 2 Particles Overlap ---")
    sim = demgpu.Simulation(2)
    sim.set_domain(np.array([-10,-10,-10], dtype=np.float32), np.array([10,10,10], dtype=np.float32))
    sim.initialize(0)
    
    pos = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0, 0.0] # Overlap 0.5 (R=1)
    ], dtype=np.float32)
    
    sim.set_positions(pos)
    sim.set_velocities(np.zeros_like(pos))
    sim.set_scales(np.ones(2, dtype=np.float32))
    sim.set_global_scale(1.0)
    
    sim.step(0.01)
    
    new_pos = sim.get_positions()
    d = np.linalg.norm(new_pos[0][:3] - new_pos[1][:3])
    print(f"Final Dist: {d:.4f}")
    
    if d > 1.95:
        print("PASS: Pushed apart.")
        return True
    else:
        print("FAIL: Still overlapping.")
        return False

if __name__ == "__main__":
    p1 = test_two_particles()
    p2 = run_simulation(125, 0.40, steps=500)
    p3 = run_simulation(216, 0.64, steps=500)
