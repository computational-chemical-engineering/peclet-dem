
import sys
sys.path.append('./build')
import demgpu
import numpy as np
import time
import math
import os
import struct

def generate_unit_sdf_stl(radius, filename):
    """
    Generates a unit STL for Sphere using SDF + Marching Cubes (skimage).
    """
    print(f"Generating Unit STL via SDF: {filename}")
    
    # 1. Create a 1-particle simulation
    sim_unit = demgpu.Simulation(1)
    # Shape Type 0 = Sphere. Height/Thickness ignored.
    sim_unit.initialize(shape_type=0, radius=radius, height=0.0, thickness=0.0)
    
    # Domain large enough
    bound = 2.0 * radius * 1.5 
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
    
    # Geometry (Sphere)
    radius = 0.5
    
    # Simulation Params
    dt = 0.01

    
    # Solver
    sim_iterations_pos = 20
    sim_iterations_vel = 20
    
    restitution = 1.0 # Normal Restitution
    restitution_t = 0.0 # Tangential Restitution
    friction = 1.0 # Enable friction/tangential interaction
    
    # Collision Setup
    impact_velocity = 5.0
    spin_0_y = 5.0 # rad/s
    spin_1_y = -spin_0_y
    initial_dist = 2.5*radius 
    offset_y = 0.0    # Impact parameter to induce spin/shear?
    
    duration = 2.0*(initial_dist-np.sqrt(4*radius**2 - offset_y**2))/(2.0*impact_velocity) # Seconds
    limit_steps = int(duration / dt) # Give some time after collision

    output_dir = "./output/collision_test_sphere"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Initialize Simulation ---
    # Shape Type 0 = Sphere (Analytic)
    sim = demgpu.Simulation(num_particles)
    sim.initialize(shape_type=0, radius=radius) # Radius is set here
    
    domain_size = 10.0
    sim.set_domain((-domain_size, -domain_size, -domain_size), 
                   (domain_size, domain_size, domain_size))
    
    sim.set_gravity(0.0, 0.0, 0.0) # Disable Gravity
    sim.set_material_params(restitution, restitution_t, friction) # e_n, e_t, mu
    sim.set_solver_iterations(sim_iterations_pos, sim_iterations_vel)
    
    # --- Helper: Random Quaternions ---
    def random_quaternion():
        u1 = np.random.random() # Changed from random.random() to np.random.random() for consistency
        u2 = np.random.random()
        u3 = np.random.random()
        q = [
            math.sqrt(1-u1) * math.sin(2*math.pi*u2),
            math.sqrt(1-u1) * math.cos(2*math.pi*u2),
            math.sqrt(u1) * math.sin(2*math.pi*u3),
            math.sqrt(u1) * math.cos(2*math.pi*u3)
        ]
        return q

    # Set Positions & Velocities
    pos = np.zeros((num_particles, 4), dtype=np.float32)
    quat = np.zeros((num_particles, 4), dtype=np.float32)
    vel = np.zeros((num_particles, 3), dtype=np.float32)
    ang_vel = np.zeros((num_particles, 3), dtype=np.float32)
    scale = np.ones(num_particles, dtype=np.float32)
    
    # Particle 0 (Left -> Right)
    pos[0] = [-initial_dist/2.0, -offset_y/2.0, 0.0, 1.0] # Mass=1
    vel[0] = [impact_velocity, 0.0, 0.0]
    quat[0] = [0, 0, 0, 1] 
    ang_vel[0] = [0, spin_0_y, 0] # Initial Spin
    
    # Particle 1 (Right -> Left)
    pos[1] = [initial_dist/2.0, offset_y/2.0, 0.0, 1.0] # Mass=1
    vel[1] = [-impact_velocity, 0.0, 0.0]
    quat[1] = [0, 0, 0, 1]
    ang_vel[1] = [0, spin_1_y, 0] # Initial Spin
    
    sim.set_positions(pos)
    sim.set_velocities(vel)
    sim.set_quaternions(quat)
    sim.set_scales(scale)
    sim.set_angular_velocities(ang_vel) # Set Initial Angular Velocity
    
    print("Simulation initialized. Seeding particles...")
    # Generate Unit STL for visualization
    r_unit = 1.0
    stl_path = f"{output_dir}/sphere_unit.stl"
    if not os.path.exists(stl_path):
        generate_unit_sdf_stl(r_unit, stl_path)
    
    
    # --- Helper: Metrics ---
    def calculate_metrics(sim):
        pos = sim.get_positions()
        vel = sim.get_velocities()
        ang_vel = sim.get_angular_velocities()
        inv_I = sim.get_inv_inertia()
        masses = sim.get_masses()
        
        total_P = np.zeros(3)
        total_L = np.zeros(3)
        total_KE = 0.0
        
        n = pos.shape[0]
        for i in range(n):
            mass = masses[i]
            if mass <= 0.0: continue
            
            v = vel[i]
            omega = ang_vel[i]
            
            # Linear Momentum
            P_i = mass * v
            total_P += P_i
            
            # Angular Momentum
            # L = r x p + I * omega
            # I is diagonal in local frame. For Sphere, World I = Local I.
            # I = 1.0 / inv_I
            I_diag = np.zeros(3)
            # Handle infinite inertia (fixed rotation) if needed, but here dynamics are active
            if inv_I[i,0] > 0: I_diag[0] = 1.0/inv_I[i,0]
            if inv_I[i,1] > 0: I_diag[1] = 1.0/inv_I[i,1]
            if inv_I[i,2] > 0: I_diag[2] = 1.0/inv_I[i,2]
            
            L_spin = I_diag * omega # Element-wise for diagonal matrix
            L_orbit = np.cross(pos[i, :3], P_i)
            total_L += (L_orbit + L_spin)
            
            # Kinetic Energy
            # KE = 0.5*m*v^2 + 0.5*omega*I*omega
            KE_trans = 0.5 * mass * np.dot(v, v)
            KE_rot = 0.5 * np.dot(omega, I_diag * omega)
            total_KE += (KE_trans + KE_rot)
            
        return total_P, total_L, total_KE

    print(f"Starting Sphere Collision Test: dt={dt}, Offset={offset_y}, Vel={impact_velocity}")
    print(f"Output: {output_dir}")
    
    # Initial Metrics
    P_init, L_init, KE_init = calculate_metrics(sim)
    print("--- Initial State ---")
    print(f"Total Momentum: {P_init}")
    print(f"Total Ang Mom : {L_init}")
    print(f"Total KE      : {KE_init:.6f}")
    
    # Loop
    dump_interval = int(duration/dt/1e2 + 1.0)
    
    for i in range(limit_steps):
        # Step
        sim.step(dt)
        
        # Logging
        if i % dump_interval == 0:
            sim.export_lammps(f"{output_dir}/dump.collision.{i}.lammps", i)
            
            # Simple distance check
            p = sim.get_positions()
            d = np.linalg.norm(p[0, :3] - p[1, :3])
            
            print(f"Step {i}: Dist={d:.4f}")
            
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
