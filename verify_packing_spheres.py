# Fixed Params
import sys
sys.path.append('./build')
import demgpu
import numpy as np
import time
import math
import os

def verify_packing():
    num_particles = 300
    radius = 0.5
    vol_particle = (4.0/3.0) * math.pi * radius**3
    growth_rate = 1.0 # Slow growth

    # Study Grid
    # stiffness (iterations) vs timestep (dt)
    # Higher iterations = stiffer solver, better overlap resolution
    # Smaller dt = better stability, but slower growth

    output_dir = "./output/sphere_packing"
    os.makedirs(output_dir, exist_ok=True)
    iterations_list = [100]
    dt_list = [0.01]
    T_gran = [0.0]

    best_phi = 0.0
    best_config = None

    phi_ref = 0.6
    vol_total_ref = num_particles * vol_particle
    vol_domain_ref = vol_total_ref / phi_ref
    domain_side = vol_domain_ref ** (1.0/3.0)

    sim = demgpu.Simulation(num_particles)
    sim.initialize(shape_type=1, radius=radius) # Sphere

    half_d = domain_side / 2.0
    sim.set_domain((-half_d, -half_d, -half_d), (half_d, half_d, half_d))
    sim.set_gravity(0, 0, 0)
    rng = np.random.default_rng(42)

    print(f"Sphere Packing Optimization Study (N={num_particles})")
    print(f"{'Iter':<5} {'dt':<8} {'TGran':<8} {'T':<8} {'MaxPhi':<8} {'FinalOv':<10} {'Notes'}")
    print("-" * 50)

    for iters in iterations_list:
        for dt in dt_list:
            for T in T_gran:    
                # Run one experiment
                
                # Target a high Phi to define domain

                sim.set_material_params(0.0, 0.0, 0.0) # Inelastic for Packing
                sim.set_solver_iterations(iters, iters) # Hybrid Solver (Pos+Vel)
            
                # Init Random
                pos = rng.uniform(-half_d, half_d, (num_particles, 4)).astype(np.float32)
                pos[:, 3] = 1.0
                sim.set_positions(pos)
                sigma = np.sqrt(T)
                vel = rng.normal(0, sigma, (num_particles, 4)).astype(np.float32)
                vel[:, 3] = 0.0
                sim.set_velocities(vel) 
                
                # Growth
                max_phi_reached = 0.0
                overlap_tolerance = 1e-4
                
                # Step limit? Or Growth limit?
                steps = 0
                limit_steps = np.ceil(1.5 / (dt * growth_rate)).astype(int)
                
                sim.set_scales(np.full(num_particles, 1.0, dtype=np.float32))
                sim.set_growth_params(growth_rate, 0.05)
                        
                for i in range(limit_steps):
                    sim.step(dt)
                    if i % 1 == 0:
                        p = sim.get_positions()
                        v = sim.get_velocities()
                        # Assuming get_quaternions returns numpy array.
                        q = sim.get_quaternions()
                        s = sim.get_scales()
                        
                        # export_lammps(filename, step, pos, vel, quats, radii, ...)
                        # export_lammps(filename, step, pos, vel, quats, radii, ...)
                        sim.export_lammps(f"{output_dir}/dump.stacking.{i}.lammps", i)
                    
                # Update Max Phi (valid state)
                final_ov = sim.get_max_overlap()
                vel = sim.get_velocities()
                T_current = sum(vel[:, 0:3].ravel()**2) / (3*num_particles)
                phi_current = (num_particles * vol_particle) / (domain_side**3)
                max_phi_reached = phi_current
                
                
                print(f"{iters:<5} {dt:<8.4f} {T:<8.4f} {T_current:<8.4f} {max_phi_reached:<8.3f} {final_ov:<10.3f}") 

if __name__ == "__main__":
    verify_packing() 