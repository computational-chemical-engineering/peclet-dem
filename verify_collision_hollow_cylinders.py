
import sys
sys.path.append('./build')
import demgpu
import numpy as np
import time
import math
import os
import struct

def generate_unit_sdf_stl(radius, height, thickness, filename):
    """
    Generates a unit STL using SDF + Marching Cubes (skimage).
    Adapted from verify_packing_hollow_cylinders.py
    """
    print(f"Generating Unit STL via SDF: {filename}")
    
    # 1. Create a 1-particle simulation
    sim_unit = demgpu.Simulation(1)
    sim_unit.initialize(shape_type=2, radius=radius, height=height, thickness=thickness)
    
    # Domain large enough to contain the unit shape
    bound = max(height, 2.0*radius) * 1.5 
    sim_unit.set_domain((-bound, -bound, -bound), (bound, bound, bound))
    
    # Place particle at origin
    sim_unit.set_positions(np.array([[0,0,0,1]], dtype=np.float32))
    sim_unit.set_scales(np.array([1.0], dtype=np.float32))
    
    # Get SDF
    resolution = (64, 64, 64)
    sdf_grid = sim_unit.get_sdf_grid(resolution)
    
    # Mesh
    import skimage.measure
    from stl import mesh
    
    verts, faces, normals, values = skimage.measure.marching_cubes(sdf_grid, level=0.0)
    
    # Transform to World
    min_b = np.array([-bound, -bound, -bound])
    max_b = np.array([bound, bound, bound])
    domain_size = max_b - min_b
    voxel_size = domain_size / np.array(resolution)
    
    verts_world = min_b + verts * voxel_size
    
    # Save STL
    data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
    unit_mesh = mesh.Mesh(data, remove_empty_areas=False)
    unit_mesh.vectors = verts_world[faces]
    unit_mesh.save(filename)
    print(f"Unit STL saved to {filename}")

def run_collision_test():
    # --- Configuration ---
    num_particles = 2
    
    # Geometry (Hollow Cylinder)
    radius = 0.5
    height = 1.0  # Aspect Ratio 1
    thickness = 0.3
    
    
    # Simulation Params
    dt = 0.01
    duration = 0.8 # Seconds
    limit_steps = int(duration / dt)
    
    # Solver
    sim_iterations_pos = 20
    sim_iterations_vel = 20
    
    restitution = 1.0 # Normal Restitution
    restitution_t = 1.0 # Tangential Restitution
    friction = 0.0 # Enable friction/tangential interaction
    
    # Collision Setup
    impact_velocity = -5.0
    spin_0_y = 0.0 # rad/s
    spin_1_y = spin_0_y
    initial_dist = 4.0*radius # Sufficient to not overlap
    offset_y = 0.0    # Impact parameter (Head-On)
    
    
    output_dir = "./output/collision_test_hollow_cylinder"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Initialize Simulation ---
    sim = demgpu.Simulation(num_particles)
    sim.initialize(shape_type=2, radius=radius, height=height, thickness=thickness)
    
    # Large domain, periodic
    domain_size = 6.0*radius
    sim.set_domain((-domain_size, -domain_size, -domain_size), 
                   (domain_size, domain_size, domain_size))
    # sim.enable_periodicity(False, False, False) # Default is False for Large Domain? 
    # Actually set_domain enables it by default in code. Disable explicitly for test.
    sim.enable_periodicity(True, True, True)
    
    sim.set_gravity(0, 0, 0) # Disable Gravity
    sim.set_material_params(restitution, restitution_t, friction) # e_n, e_t, mu
    sim.set_solver_iterations(sim_iterations_pos, sim_iterations_vel)
    
    # --- Helper: Metrics ---
    def calculate_metrics(sim):
        pos = sim.get_positions()
        vel = sim.get_velocities()
        ang_vel = sim.get_angular_velocities()
        inv_I = sim.get_inv_inertia()
        masses = sim.get_masses()
        quat = sim.get_quaternions()
        
        total_P = np.zeros(3)
        total_L = np.zeros(3)
        total_KE = 0.0
        
        n = pos.shape[0]
        for i in range(n):
            mass = masses[i]
            if mass <= 0.0: continue
            
            v = vel[i]
            omega_world = ang_vel[i]
            q = quat[i] # (x,y,z,w) from binding 
            
            # --- Linear ---
            P_i = mass * v
            total_P += P_i
            KE_trans = 0.5 * mass * np.dot(v, v)
            
            # --- Angular ---
            # 1. Orbital L
            p_i = pos[i, :3]
            L_orb = np.cross(p_i, P_i)
            
            # 2. Spin L (Requires World Inertia)
            # Construct Rotation Matrix from Quaternion (x,y,z,w)
            x, y, z, w = q
            R = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
            ])
            
            # World Omega -> Local Omega
            omega_local = R.T @ omega_world
            
            # Local Inertia (Diagonal)
            I_local = np.zeros(3)
            # Avoid div zero
            if inv_I[i, 0] > 0: I_local[0] = 1.0/inv_I[i, 0]
            if inv_I[i, 1] > 0: I_local[1] = 1.0/inv_I[i, 1]
            if inv_I[i, 2] > 0: I_local[2] = 1.0/inv_I[i, 2]
            
            # L_local = I_local * omega_local
            L_spin_local = I_local * omega_local
            
            # L_world = R * L_spin_local
            L_spin = R @ L_spin_local
            
            total_L += (L_orb + L_spin)
            
            # Rotational KE = 0.5 * omega_local . (I_local * omega_local)
            KE_rot = 0.5 * np.dot(omega_local, L_spin_local)
            
            total_KE += (KE_trans + KE_rot)
            
        return total_P, total_L, total_KE

    # Init Data
    pos = np.zeros((num_particles, 4), dtype=np.float32)
    vel = np.zeros((num_particles, 3), dtype=np.float32) 
    quat = np.zeros((num_particles, 4), dtype=np.float32)
    ang_vel = np.zeros((num_particles, 3), dtype=np.float32)
    scales = np.ones(num_particles, dtype=np.float32) # Scale 1.0 (Fixed)
    
    # Particle 0 (Left, moving Right)
    pos[0] = [-initial_dist/2.0, -offset_y/2.0, 0.0, 1.0] # Mass=1
    vel[0] = [impact_velocity, 0.0, 0.0]
    quat[0] = [0,0,0,1] # Identity
    ang_vel[0] = [0, spin_0_y, 0]
    
    # Particle 1 (Right, moving Left)
    pos[1] = [initial_dist/2.0, offset_y/2.0, 0.0, 1.0] # Mass=1
    vel[1] = [-impact_velocity, 0.0, 0.0]
    quat[1] = [0,0,0,1] # Identity
    ang_vel[1] = [0, spin_1_y, 0]
    
    # Upload
    sim.set_positions(pos)
    sim.set_velocities(vel)
    sim.set_quaternions(quat)
    sim.set_scales(scales)
    sim.set_angular_velocities(ang_vel)
    
    # Generate Unit STL for visualization
    r_unit = 1.0
    h_unit = height / radius
    t_unit = thickness / radius
    stl_path = f"{output_dir}/hollow_cylinder_unit.stl"
    if not os.path.exists(stl_path):
        generate_unit_sdf_stl(r_unit, h_unit, t_unit, stl_path)
    
    print(f"Starting Hollow Cylinder Collision Test: dt={dt}, Offset={offset_y}, Vel={impact_velocity}")
    print(f"Output: {output_dir}")
    
    # Initial Metrics
    P_init, L_init, KE_init = calculate_metrics(sim)
    print("--- Initial State ---")
    print(f"Total Momentum: {P_init}")
    print(f"Total Ang Mom : {L_init}")
    print(f"Total KE      : {KE_init:.6f}")
    
    # Loop
    dump_interval = 1
    if dump_interval < 1: dump_interval = 1
    
    for i in range(limit_steps):
        # Step
        sim.step(dt)
        
        # Logging
        if i % dump_interval == 0:
            sim.export_lammps(f"{output_dir}/dump.collision.{i}.lammps", i)
            
            # Simple distance check and contact count
            p = sim.get_positions()
            d = np.linalg.norm(p[0, :3] - p[1, :3])
            num_contacts = sim.get_num_contacts()
            _, _, KE = calculate_metrics(sim)
            print(f"Step {i}: Dist={d:.4f}, Contacts={num_contacts}, KE={KE:.6f}")
            
    # Final Metrics
    P_final, L_final, KE_final = calculate_metrics(sim)
    print("--- Final State ---")
    print(f"Total Momentum: {P_final}")
    print(f"Total Ang Mom : {L_final}")
    print(f"Total KE      : {KE_final:.6f}")
    
    print("--- Change ---")
    print(f"Delta Momentum: {P_final - P_init}")
    print(f"Delta Ang Mom : {L_final - L_init}")
    print(f"Delta KE      : {KE_final - KE_init:.6f}")

    print("Collision Test Completed.")

if __name__ == "__main__":
    run_collision_test()
