
import sys
import os
import time
import numpy as np

# Add build directory
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build'))
sys.path.append(build_path)
sys.path.append(os.path.join(os.path.dirname(__file__), "../python"))

import demgpu

# Numba overlap check (copied from verify_flexible.py)
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Using slow Numpy fallback.")

if HAS_NUMBA:
    @njit(parallel=True)
    def check_overlaps_numba(positions, radii, L):
        overlaps = 0
        max_overlap = 0.0
        num_particles = positions.shape[0]
        
        for i in prange(num_particles):
            for j in range(i + 1, num_particles):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                
                dx -= L * np.floor(dx / L + 0.5)
                dy -= L * np.floor(dy / L + 0.5)
                dz -= L * np.floor(dz / L + 0.5)
                
                dist_sq = dx*dx + dy*dy + dz*dz
                sum_radii = radii[i] + radii[j]
                
                if dist_sq < sum_radii * sum_radii:
                    dist = np.sqrt(dist_sq)
                    overlap = sum_radii - dist
                    if overlap > 1e-4:
                        overlaps += 1
                        max_overlap = max(max_overlap, overlap)
                        
        return overlaps, max_overlap
else:
    def check_overlaps_numba(positions, radii, L):
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

def run_convergence_test():
    num_particles = 1000
    density = 0.45
    print(f"--- Convergence Test: Rho={density}, N={num_particles} ---")
    
    SEED = 42
    rng = np.random.default_rng(seed=SEED)

    sim = demgpu.Simulation(num_particles)
    sim.initialize(shape_type=0)

    scales = rng.uniform(0.99, 1.01, num_particles).astype(np.float32)
    sim.set_scales(scales)
    
    radii = 1.0 * scales
    particle_volumes = (4.0/3.0) * np.pi * (radii**3)
    total_particle_volume = np.sum(particle_volumes)
    required_box_volume = total_particle_volume / density
    L = required_box_volume**(1.0/3.0)
    print(f"Box Size L: {L:.4f}")
    
    sim.set_domain((0.0, 0.0, 0.0), (L, L, L))
    
    pos = rng.uniform(0, L, (num_particles, 3)).astype(np.float32)
    vel = np.zeros((num_particles, 3), dtype=np.float32)

    sim.set_positions(pos)
    sim.set_velocities(vel)
    sim.set_gravity(0.0, 0.0, 0.0)
    
    # Material params: standard
    sim.set_material_params(0.5, 0.5, 0.5)
    
    dt = 0.01
    
    # Run for many steps
    total_steps = 2000
    check_interval = 100
    
    print(f"{'Step':<8} {'Time':<8} {'Overlaps':<10} {'MaxOv':<10}")
    
    # Growth phase? No, user said "iterating long enough overlaps go to zero". 
    # Usually this implies relaxation from a random state.
    # The random state at 0.45 density will have HUGE overlaps initially.
    # We want to see the solver push them apart.
    
    sim.set_global_scale(1.0) # Full size immediately
    
    for i in range(total_steps + 1):
        sim.step(dt)
        
        if i % check_interval == 0:
            pos_curr = sim.get_positions()
            # Radii are fixed at 1.0 * scales
            overlaps, max_ov = check_overlaps_numba(pos_curr, radii, L)
            print(f"{i:<8} {i*dt:<8.2f} {overlaps:<10} {max_ov:<10.4f}")
            sys.stdout.flush()
            
            if overlaps == 0:
                print("CONVERGED: Overlaps reached 0.")
                break

if __name__ == "__main__":
    run_convergence_test()
