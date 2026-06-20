
import os
import sys
import math
import numpy as np

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
import dem

def run_study():
    # Fixed Params
    num_particles = 300
    radius = 0.5
    vol_particle = (4.0/3.0) * math.pi * radius**3
    
    # Study Grid
    # stiffness (iterations) vs timestep (dt)
    # Higher iterations = stiffer solver, better overlap resolution
    # Smaller dt = better stability, but slower growth
    
    iterations_list = [10, 50, 100]
    dt_list = [0.005, 0.002] 
    
    print(f"Sphere Packing Optimization Study (N={num_particles})")
    print(f"{'Iter':<5} {'dt':<8} {'MaxPhi':<8} {'FinalOv':<10} {'Notes'}")
    print("-" * 50)
    
    best_phi = 0.0
    best_config = None
    
    for iters in iterations_list:
        for dt in dt_list:
            # Run one experiment
            
            # Target a high Phi to define domain
            phi_ref = 0.65
            vol_total_ref = num_particles * vol_particle
            vol_domain_ref = vol_total_ref / phi_ref
            domain_side = vol_domain_ref ** (1.0/3.0)
            
            sim = dem.Simulation(num_particles)
            sim.initialize(shape_type=1, radius=radius) # Sphere
            
            half_d = domain_side / 2.0
            sim.set_domain((-half_d, -half_d, -half_d), (half_d, half_d, half_d))
            sim.set_gravity(0, 0, 0)
            sim.enable_periodicity(True, True, True) # User requested periodic packing
            sim.set_solver_iterations(iters, 0) # Only Position Solver Used for Overlap
            
            # Init Random
            np.random.seed(123)
            pos = np.random.uniform(-half_d, half_d, (num_particles, 4)).astype(np.float32)
            pos[:, 3] = 1.0
            sim.set_positions(pos)
            
            # Growth
            scale = 0.1
            max_phi_reached = 0.0
            overlap_tolerance = 0.02 # 2%
            
            # Step limit? Or Growth limit?
            steps = 0
            limit_steps = 5000
            growth_rate = 0.0005 # Slow growth
            
            scales = np.zeros(num_particles, dtype=np.float32)
            
            final_ov = 0.0
            reason = "Limit"
            
            while steps < limit_steps:
                scales[:] = scale
                sim.set_scales(scales)
                
                sim.step(dt)
                steps += 1
                
                current_ov = sim.get_max_overlap()
                
                # Check Overlap
                if current_ov > overlap_tolerance:
                    # Try simple settle
                    settled = False
                    for _ in range(50): # Quick settle check
                         sim.step(dt)
                         steps += 1
                         if sim.get_max_overlap() < overlap_tolerance:
                             settled = True
                             break
                    
                    if not settled:
                        final_ov = sim.get_max_overlap()
                        reason = "Jam"
                        break
                
                # Update Max Phi (valid state)
                phi_current = (num_particles * vol_particle * scale**3) / (domain_side**3)
                max_phi_reached = phi_current
                
                # Move to next scale
                scale += growth_rate
                
            print(f"{iters:<5} {dt:<8.4f} {max_phi_reached:<8.3f} {final_ov:<10.3f} {reason}")
            
            if max_phi_reached > best_phi:
                best_phi = max_phi_reached
                best_config = (iters, dt)

    print("-" * 50)
    print(f"Best Result: Phi={best_phi:.3f} with Iter={best_config[0]}, dt={best_config[1]}")

if __name__ == "__main__":
    run_study()
