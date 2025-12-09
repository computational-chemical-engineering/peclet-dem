
import sys
import os
import time
import numpy as np

# Add build directory
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build'))
sys.path.append(build_path)
sys.path.append(os.path.join(os.path.dirname(__file__), "../python"))

from export_vtp import save_to_vtp

try:
    import demgpu
except ImportError as e:
    print(f"Failed to import demgpu from {build_path}")
    print(e)
    sys.exit(1)

# Numba overlap check
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Using slow Numpy fallback.")

def flush():
    sys.stdout.flush()

if HAS_NUMBA:
    @njit(parallel=True)
    def check_overlaps_numba(positions, radii, L):
        overlaps = 0
        max_overlap = 0.0
        num_particles = positions.shape[0]
        
        # Naive N^2 check (acceptably fast for N=1000 with Numba)
        for i in prange(num_particles):
            for j in range(i + 1, num_particles):
                # Delta
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                
                # MIC
                dx -= L * np.floor(dx / L + 0.5)
                dy -= L * np.floor(dy / L + 0.5)
                dz -= L * np.floor(dz / L + 0.5)
                
                dist_sq = dx*dx + dy*dy + dz*dz
                sum_radii = radii[i] + radii[j]
                
                if dist_sq < sum_radii * sum_radii:
                    dist = np.sqrt(dist_sq)
                    overlap = sum_radii - dist
                    # Race condition on scalar add? Numba handles reduction or use atomic?
                    # For simple counter/max, manually reduction is safer or just use array
                    # But for simple boolean check:
                    if overlap > 1e-4: # Tolerance
                         # overlaps += 1 # Numba parallel reduction limitation
                         max_overlap = max(max_overlap, overlap)
                         overlaps += 1
                         
        return overlaps, max_overlap
else:
    def check_overlaps_numba(positions, radii, L):
        # Fallback to Numpy
        overlaps = 0
        max_overlap = 0.0
        num_particles = len(positions)
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                delta = positions[j] - positions[i]
                delta -= L * np.floor(delta / L + 0.5)
                dist_sq = np.dot(delta, delta)
                sum_radii = radii[i] + radii[j]
                if dist_sq < sum_radii * sum_radii:
                    dist = np.sqrt(dist_sq)
                    overlap = sum_radii - dist
                    if overlap > 1e-4:
                        overlaps += 1
                        max_overlap = max(max_overlap, overlap)
        return overlaps, max_overlap

def run_simulation(num_particles, packing_density, granular_temperature):
    """
    Runs a packing simulation and returns (overlaps, max_overlap, duration_sec).
    """
    SEED = 42
    rng = np.random.default_rng(seed=SEED)

    sim = demgpu.Simulation(num_particles)
    sim.initialize(shape_type=0) # Sphere

    # Scales (Polydisperse)
    scales = rng.uniform(0.99, 1.01, num_particles).astype(np.float32)
    sim.set_scales(scales)
    
    # Domain Calculation
    radii = 1.0 * scales
    particle_volumes = (4.0/3.0) * np.pi * (radii**3)
    total_particle_volume = np.sum(particle_volumes)
    required_box_volume = total_particle_volume / packing_density
    L = required_box_volume**(1.0/3.0)
    
    sim.set_domain((0.0, 0.0, 0.0), (L, L, L))
    
    # Seeding
    pos = rng.uniform(0, L, (num_particles, 3)).astype(np.float32)
    
    # Velocities
    sigma = np.sqrt(granular_temperature)
    vel = rng.normal(0.0, sigma, (num_particles, 3)).astype(np.float32)

    sim.set_positions(pos)
    sim.set_velocities(vel)
    
    # Simulation Loop
    # Simulation Loop
    sim.set_gravity(0.0, 0.0, 0.0)
    dt = 0.001 # Reduced from 0.01 for stability
    
    start_time = time.perf_counter()
    
    # Growth Phase
    growth_steps = 2000 # Increased from 200
    for i in range(growth_steps):
        current_scale = i / (growth_steps - 1)
        sim.set_global_scale(current_scale)
        sim.step(dt)
        
    # Relaxation Phase
    relax_steps = 5000 # Increased from 500
    for i in range(relax_steps):
        sim.step(dt)
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # Analysis
    final_pos = sim.get_positions()
    radii_final = scales * 1.0 # Global scale is 1.0
    
    overlaps, max_overlap = check_overlaps_numba(final_pos, radii_final, L)
    
    # Export VTP
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    vtp_filename = os.path.join(output_dir, f"packing_T{granular_temperature:.1f}_rho{packing_density:.2f}.vtp")
    sim.write_vtp(vtp_filename)
    
    return overlaps, max_overlap, duration

def main():
    print("===================================================")
    print("  DEM-GPU: High Density Packing Parameter Study")
    print("===================================================")
    
    num_particles = 1000
    densities = [0.40, 0.50, 0.55, 0.60, 0.62, 0.63, 0.64, 0.65]
    temperatures = [0.1, 1.0] 
    
    results = []
    
    for T in temperatures:
        print(f"\n--- Temperature: {T} ---")
        best_density = 0.0
        
        for rho in densities:
            print(f"Testing Rho={rho:.2f}...", end="")
            flush()
            
            try:
                overlaps, max_ov, duration = run_simulation(num_particles, rho, T)
                print(f" Done. Time={duration:.2f}s, Overlaps={overlaps}, MaxOv={max_ov:.4f}")
                
                results.append({
                    "T": T,
                    "rho": rho,
                    "overlaps": overlaps,
                    "max_ov": max_ov,
                    "time": duration
                })
                
                if overlaps == 0:
                    best_density = max(best_density, rho)
                
            except Exception as e:
                print(f" Failed: {e}")
                
        print(f"-> Max Reachable Density at T={T}: {best_density:.2f}")

    print("\n===================================================")
    print("  Summary Results")
    print("===================================================")
    print(f"{'Temp':<6} {'Density':<8} {'Overlaps':<10} {'MaxOv':<10} {'Time(s)':<8}")
    for res in results:
        print(f"{res['T']:<6.1f} {res['rho']:<8.2f} {res['overlaps']:<10} {res['max_ov']:<10.4f} {res['time']:<8.2f}")
    print("===================================================")

if __name__ == "__main__":
    main()
