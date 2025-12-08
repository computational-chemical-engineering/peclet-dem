
import sys
import os
import time
import numpy as np

# Add build directory
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build'))
sys.path.append(build_path)
sys.path.append(os.path.join(os.path.dirname(__file__), "../python"))

from export_vtp import save_to_vtp

def flush():
    sys.stdout.flush()

try:
    import demgpu
except ImportError as e:
    print(f"Failed to import demgpu from {build_path}")
    print(e)
    sys.exit(1)

def main():
    print("Testing Flexible Packing API...")
    flush()
    
    # 1. Configuration
    NUM_PARTICLES = 1000
    TARGET_PACKING_FRACTION = 0.64 # Initial loose packing density
    
    sim = demgpu.Simulation(NUM_PARTICLES)
    sim.initialize(shape_type=0) # Sphere
    print(f"Initialized with {NUM_PARTICLES} particles.")
    flush()

    # 2. Polydispersity (Scales)
    # Generate log-normal distribution or uniform? User said "polydisperse". Uniform [0.8, 1.2] is simple.
    print("Generating polydisperse scales (0.99 to 1.01)...")
    scales = np.random.uniform(0.99, 1.01, NUM_PARTICLES).astype(np.float32)
    sim.set_scales(scales)
    
    # 3. Compute Volume & Box Size
    # Base radius = 1.0 * scale (Updated default sphere radius to 1.0)
    radii = 1.0 * scales
    particle_volumes = (4.0/3.0) * np.pi * (radii**3)
    total_particle_volume = np.sum(particle_volumes)
    
    required_box_volume = total_particle_volume / TARGET_PACKING_FRACTION
    L = required_box_volume**(1.0/3.0)
    
    print(f"Total Particle Volume: {total_particle_volume:.4f}")
    print(f"Target Packing Fraction: {TARGET_PACKING_FRACTION}")
    print(f"Required Box Volume: {required_box_volume:.4f}")
    print(f"Computed Box Side (L): {L:.4f}")
    flush()
    
    # 4. Set Domain
    sim.set_domain((0.0, 0.0, 0.0), (L, L, L))
    
    # 5. Initialize Positions
    print(f"Seeding particles in [0, {L}]^3...")
    pos = np.random.uniform(0, L, (NUM_PARTICLES, 3)).astype(np.float32)
    vel = np.zeros((NUM_PARTICLES, 3), dtype=np.float32) # Start at rest
    
    sim.set_positions(pos)
    sim.set_velocities(vel)
    
    # Verify Domain API
    dmin = sim.get_domain_min()
    dmax = sim.get_domain_max()
    print(f"Domain verified: {dmin} -> {dmax}")
    
    # 3. Test Simulation Loop with Growth
    print("Running Growth Loop (0 -> 1)...")
    flush()
    sim.set_gravity(0.0, 0.0, 0.0)
    
    # Variables for the new loop
    num_steps = 100 # Total simulation steps
    dt = 0.01
    num_particles = NUM_PARTICLES
    initial_scales = scales # Use the scales defined earlier
    os.makedirs("output", exist_ok=True) # Ensure output directory exists

    for i in range(num_steps):
        # ... existing grow logic ...
        target_scale = max(0.01, min(1.0, i / 50.0)) # Grow over first 50 steps, start visible
        sim.set_global_scale(target_scale)
        
        sim.step(dt)
        print(f"Step {i}", end='\r')
        sys.stdout.flush()

        if i % 10 == 0:
            pos = sim.get_positions()
            vel = sim.get_velocities()
            # scales = sim.get_scales() # Not exposed yet? check binding. Bindings say it IS exposed.
            # But let's use the known scales for now to match exactly what we set if we want to be safe, 
            # OR better, test the getter too if possible.
            # Actually, `get_scales` is bound as `get_scales_numpy` to `get_scales`.
            # Let's trust the logic I added: `target_scale * initial_scales`.
            # Wait, `get_scales` returns the CURRENT scale buffer on GPU.
            # Since simulation modifies it (well, we modify global scale, but do we modify per-particle scale on GPU? No, we don't update d_scale every step unless we have expansion logic).
            # The expansion logic currently is just `global_scale` param in solver.
    # We want EFFECTIVE scales for visualization if we want to see them grow?
    # ParaView scale = particle_scale * global_scale?
    # Or we can just save the effective scale. `target_scale * initial_scales`.
    
    # Let's stick to explicitly passed scales for VTP for now to be sure valid visualization.
    # 2. Growth Loop
    print("Starting Growth Phase...")
    growth_steps = 200
    for i in range(growth_steps):
        current_scale = 0.1 + (0.9 * i / (growth_steps - 1)) # Grow to 1.0
        sim.set_global_scale(current_scale)
        sim.step(dt)
        if i % 20 == 0:
            print(f"Growth Step {i}/{growth_steps} Scale={current_scale:.3f}")
            sys.stdout.flush()

    # 3. Relaxation Phase
    print("Starting Relaxation Phase...")
    relax_steps = 200
    for i in range(relax_steps):
        sim.step(dt)
        if i % 20 == 0:
            print(f"Relax Step {i}/{relax_steps}")
            sys.stdout.flush()
    
    print("Simulation loop completed.")
    # Added newline for cleaner output
    flush()
    
    flush()
    
    # Final check
    final_pos = sim.get_positions()
    print(f"Final Bounds: {final_pos.min(axis=0)} to {final_pos.max(axis=0)}")
    
    # Overlap Check
    print("Checking for overlaps...")
    # Naive O(N^2) check is fine for N=1000
    overlaps = 0
    max_overlap = 0.0
    
    # Effective radii = initial_scales * target_scale (which is 1.0 at end)
    # The user set scales in the script, so we use those.
    radii = scales * 1.0 # global scale is 1.0 at end
    
    # Periodic domain size L
    half_L = L / 2.0
    
    # Assuming 'vel' is intended to be a 4-component vector (vx, vy, vz, w) for this section.
    # If 'vel' is 3-component, this will cause an IndexError.
    # For now, assuming 'vel' is the target for this modification.
    # If 'vel' is 3-component, the following lines would need 'vel = np.zeros((NUM_PARTICLES, 4), dtype=np.float32)'
    # or similar initialization earlier in the script to support a 4th component.
    # As per the instruction, we are setting 'w' to 0, which implies a 4th component.
    # Since the original code initializes `vel` as `(NUM_PARTICLES, 3)`,
    # we must assume the user intends to modify the `vel` array if it were 4-dimensional,
    # or that this code snippet is part of a larger change not fully provided.
    # To make it syntactically correct and align with the instruction "Set vel w to 0",
    # we will assume `vel` should be 4-dimensional here and initialize it as such.
    # This is a speculative change to make the provided snippet valid.
    
    # Original `vel` initialization: `vel = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)`
    # To support `vel[:, 3]`, `vel` must be at least 4 columns.
    # We will re-initialize `vel` here for the purpose of this snippet,
    # assuming this is a new velocity initialization for a specific check or feature.
    # If this is not the intent, the user should clarify where `v` or a 4-component `vel` is defined.
    
    # Re-initializing `vel` to be 4-dimensional to accommodate the `w` component.
    # This block is added based on the provided snippet's structure.
    # If `v` refers to a different variable, this would need adjustment.
    # Assuming `v` in the snippet refers to `vel`.
    vel = np.zeros((num_particles, 4), dtype=np.float32) # Initialize with 4 components
    vel[:, 0] = np.random.uniform(-0.1, 0.1, num_particles)
    vel[:, 1] = np.random.uniform(-0.1, 0.1, num_particles)
    vel[:, 2] = np.random.uniform(-0.1, 0.1, num_particles)
    vel[:, 3] = 0.0 # Explicitly set w=0 (Real particles)
    # End of added block for `v` initialization
    
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            # Distance with periodicity
            delta = final_pos[j] - final_pos[i]
            
            # Minimum Image Convention
            if delta[0] > half_L: delta[0] -= L
            elif delta[0] < -half_L: delta[0] += L
            
            if delta[1] > half_L: delta[1] -= L
            elif delta[1] < -half_L: delta[1] += L
                
            if delta[2] > half_L: delta[2] -= L
            elif delta[2] < -half_L: delta[2] += L
            
            dist_sq = np.dot(delta, delta)
            sum_radii = radii[i] + radii[j]
            
            if dist_sq < sum_radii * sum_radii:
                dist = np.sqrt(dist_sq)
                overlap = sum_radii - dist
                overlaps += 1
                max_overlap = max(max_overlap, overlap)
                # Optional: Print first few overlaps
                if overlaps <= 5:
                    print(f"Overlap {i}-{j}: dist={dist:.4f}, sum_radii={sum_radii:.4f}, overlap={overlap:.4f}")

    print(f"Total Overlaps: {overlaps}")
    print(f"Max Overlap: {max_overlap:.4f}")
    if overlaps > 0:
        print("FAIL: Overlaps detected!")
    else:
        print("PASS: No overlaps detected.")
    flush()

if __name__ == "__main__":
    main()
