
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

def check_overlaps(positions, radii, L):
    overlaps = 0
    max_overlap = 0.0
    num_particles = len(positions)
    
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            # Distance with periodicity
            delta = positions[j] - positions[i]
            
            # Minimum Image Convention
            delta -= L * np.floor(delta / L + 0.5)
            
            dist_sq = np.dot(delta, delta)
            sum_radii = radii[i] + radii[j]
            
            if dist_sq < sum_radii * sum_radii:
                dist = np.sqrt(dist_sq)
                overlap = sum_radii - dist
                overlaps += 1
                max_overlap = max(max_overlap, overlap)
                
    return overlaps, max_overlap

def main():
    print("Testing Flexible Packing API...")
    flush()
    
    # 1. Configuration
    NUM_PARTICLES = 1000
    TARGET_PACKING_FRACTION = 0.64 # Initial loose packing density
    GRANULAR_TEMPERATURE = 0.1 # One-third of the mean square of these fluctuation velocities
    SEED = 42

    rng = np.random.default_rng(seed=SEED)

    sim = demgpu.Simulation(NUM_PARTICLES)
    sim.initialize(shape_type=0) # Sphere
    print(f"Initialized with {NUM_PARTICLES} particles.")
    flush()

    # 2. Polydispersity (Scales)
    # Generate log-normal distribution or uniform? User said "polydisperse". Uniform [0.8, 1.2] is simple.
    print("Generating polydisperse scales (0.99 to 1.01)...")
    scales = rng.uniform(0.99, 1.01, NUM_PARTICLES).astype(np.float32)
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
    pos = rng.uniform(0, L, (NUM_PARTICLES, 3)).astype(np.float32)
    # Initialize velocities with Normal distribution
    # T_g = <v^2>/3 = sigma^2  => sigma = sqrt(T_g)
    sigma = np.sqrt(GRANULAR_TEMPERATURE)
    vel = rng.normal(0.0, sigma, (NUM_PARTICLES, 3)).astype(np.float32)

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

    print("Starting Growth Phase...")
    growth_steps = 200
    for i in range(growth_steps):
        current_scale = i / (growth_steps - 1) # Grow to 1.0
        sim.set_global_scale(current_scale)
        sim.step(dt)
        if i % 20 == 0:
            print(f"Growth Step {i}/{growth_steps} Scale={current_scale:.3f}")
            flush()

    # 3. Relaxation Phase
    print("Starting Relaxation Phase...")
    relax_steps = 200
    for i in range(relax_steps):
        sim.step(dt)
        if i % 20 == 0:
            print(f"Relax Step {i}/{relax_steps}")
            flush()
    
    print("Simulation loop completed.")
    flush()
    
    # Final check
    final_pos = sim.get_positions()
    print(f"Final Bounds: {final_pos.min(axis=0)} to {final_pos.max(axis=0)}")
    
    # Overlap Check
    print("Checking for overlaps...")
    
    # Effective radii = scales * 1.0 (global scale is 1.0 at end)
    radii = scales * 1.0

    overlaps, max_overlap = check_overlaps(final_pos, radii, L)

    print(f"Total Overlaps: {overlaps}")
    print(f"Max Overlap: {max_overlap:.4f}")
    if overlaps > 0:
        print("FAIL: Overlaps detected!")
    else:
        print("PASS: No overlaps detected.")
    flush()

if __name__ == "__main__":
    main()
