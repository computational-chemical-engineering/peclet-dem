
import sys
sys.path.append('./build')
import dem
import numpy as np
import math
import os

def run_sphere_test():
    # --- Configuration ---
    num_particles = 2
    radius = 0.5
    
    # Simulation Params
    dt = 0.01
    duration = 5.0 
    limit_steps = int(duration / dt)
    
    # Solver
    sim_iterations_pos = 20
    sim_iterations_vel = 20
    
    restitution = 1.0 
    restitution_t = 0.0 # frictionless
    friction = 0.0
    
    impact_velocity = 5.0
    initial_dist = 4.0*radius 
    
    # Test Cases: 
    # 0: Head-On
    # 1: Off-Center (Impact Parameter b = 0.5*Radius)
    test_cases = [0.0, 0.5*radius]
    
    output_dir = "./output/collision_test_spheres"
    os.makedirs(output_dir, exist_ok=True)
    
    for offset in test_cases:
        print(f"\nRunning Sphere Test with Offset={offset}")
        
        sim = dem.Simulation(num_particles)
        sim.initialize(shape_type=1, radius=radius, height=0, thickness=0) # Sphere
        
        domain_size = 6.0*radius
        sim.set_domain((-domain_size, -domain_size, -domain_size), 
                       (domain_size, domain_size, domain_size))
        sim.enable_periodicity(False, False, False)
        
        sim.set_gravity(0, 0, 0) 
        sim.set_material_params(restitution, restitution_t, friction) 
        sim.set_solver_iterations(sim_iterations_pos, sim_iterations_vel)
        
        # Init
        pos = np.zeros((num_particles, 4), dtype=np.float32)
        vel = np.zeros((num_particles, 3), dtype=np.float32) 
        quat = np.zeros((num_particles, 4), dtype=np.float32)
        
        # P0 Left
        pos[0] = [-initial_dist/2.0, -offset/2.0, 0.0, 1.0] 
        vel[0] = [impact_velocity, 0.0, 0.0]
        quat[0] = [0,0,0,1]
        
        # P1 Right
        pos[1] = [initial_dist/2.0, offset/2.0, 0.0, 1.0] 
        vel[1] = [-impact_velocity, 0.0, 0.0]
        quat[1] = [0,0,0,1]
        
        sim.set_positions(pos)
        sim.set_velocities(vel)
        sim.set_quaternions(quat)
        sim.set_scales(np.ones(num_particles, dtype=np.float32))
        sim.set_angular_velocities(np.zeros((num_particles, 3), dtype=np.float32))
        
        # Metrics
        def fast_ke(sim):
            v = sim.get_velocities()
            m = sim.get_masses()
            ke = 0.0
            for i in range(2):
                ke += 0.5 * m[i] * np.dot(v[i], v[i])
            return ke

        ke_init = fast_ke(sim)
        print(f"Initial KE: {ke_init:.6f}")
        
        for i in range(limit_steps):
            sim.step(dt)
            if i % 100 == 0:
                p = sim.get_positions()
                d = np.linalg.norm(p[0, :3] - p[1, :3])
                ke = fast_ke(sim)
                #print(f"Step {i}: Dist={d:.4f}, KE={ke:.6f}")
        
        ke_final = fast_ke(sim)
        print(f"Final KE: {ke_final:.6f}")
        
        if abs(ke_final - ke_init) < 1e-4:
            print("Status: PASS (Conserved)")
        else:
            print(f"Status: FAIL (Diff={ke_final - ke_init})")

if __name__ == "__main__":
    run_sphere_test()
