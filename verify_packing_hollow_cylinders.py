# Fixed Params
import sys
sys.path.append('./build')
import dem
import numpy as np
import time
import math
import os
import struct

def generate_unit_sdf_stl(radius, height, thickness, filename):
    """
    Generates a unit STL using SDF + Marching Cubes (skimage).
    """
    print(f"Generating Unit STL via SDF: {filename}")
    
    # 1. Create a 1-particle simulation
    sim_unit = dem.Simulation(1)
    # Use exact same shape params
    sim_unit.initialize(shape_type=2, radius=radius, height=height, thickness=thickness)
    
    # Domain large enough to contain the unit shape
    # Max dimension is likely Height or Diameter. 
    # Diameter = 2*Radius. Height = H.
    bound = max(height, 2.0*radius) * 1.5 
    sim_unit.set_domain((-bound, -bound, -bound), (bound, bound, bound))
    
    # Place particle at origin, identity rotation, scale 1.0
    sim_unit.set_positions(np.array([[0,0,0,1]], dtype=np.float32))
    sim_unit.set_scales(np.array([1.0], dtype=np.float32))
    sim_unit.set_quaternions(np.array([[0,0,0,1]], dtype=np.float32)) # Identity? (0,0,0,1) is w=1?
    # Eigen/CUDA quat: x,y,z,w. Identity is 0,0,0,1 usually.
    
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

def verify_packing():
    num_particles = 150
    radius = 0.5
    height = 0.3# H/D = 1
    thickness = 0.15 # Arbitrary hollow thickness

    restitution = 1.0 # Normal Restitution
    restitution_t = 1.0 # Tangential Restitution
    friction = 0.0 # Enable friction/tangential interaction

    T = 1.0 # Granular temperature
    dt = 0.001
    limit_steps = int(3.0/dt)
    dump_interval = int(1.0/(100*dt))
    iters = 20

    scale_init = 5e-2
    growth_rate_init = -math.log(scale_init)
    growth_rate = growth_rate_init
    i_switch_cooling = 2000 # Switch to cooling after this step


    # Calculate Domain Volume
    # V_cyl = H * pi * (R^2 - r_in^2) = H * pi * (2Rt - t^2)
    vol_particle = math.pi * height * (2 * radius * thickness - thickness**2)

    # Study Grid
    # stiffness (iterations) vs timestep (dt)
    # Higher iterations = stiffer solver, better overlap resolution
    # Smaller dt = better stability, but slower growth

    output_dir = f"./output/ring_packing_N={num_particles}"
    os.makedirs(output_dir, exist_ok=True)

    # Target: High density jam
    phi_ref = 0.6
    criterion_ov = 1e-3
    vol_total_ref = num_particles * vol_particle
    vol_domain_ref = vol_total_ref / phi_ref
    domain_side = vol_domain_ref ** (1.0/3.0)
    
    # --- Generate Unit STL (Skimage) ---
    # Normalize dimensions so that outer radius is 1.0
    # Ovito scales mesh by 'Radius' property, so mesh must be Unit Size.
    r_unit = 1.0
    h_unit = height / radius
    t_unit = thickness / radius
    
    generate_unit_sdf_stl(r_unit, h_unit, t_unit, f"{output_dir}/ring_unit.stl")

    sim = dem.Simulation(num_particles)
    sim.initialize(shape_type=2, radius=radius, height=height, thickness=thickness) #hollow cylinder


    half_d = domain_side / 2.0
    sim.set_domain((-half_d, -half_d, -half_d), (half_d, half_d, half_d))
    
    sim.set_gravity(0, 0, 0)
    rng = np.random.default_rng(42)

    print(f"Hollow Cylinder Packing Optimization Study (N={num_particles})")
    print(f"{'Iter':<5} {'dt':<8} {'TGran':<8} {'T':<8} {'MaxPhi':<8} {'FinalOv':<10} {'Notes'}")
    print("-" * 50)

    
    # Run one experiment
    
    vol_domain_ref = vol_total_ref / phi_ref
    domain_side = vol_domain_ref ** (1.0/3.0)
    half_d = domain_side / 2.0
    
    # Update Simulation Domain to new size
    sim.set_domain((-half_d, -half_d, -half_d), (half_d, half_d, half_d))

    sim.set_material_params(restitution, restitution_t, friction)
    sim.set_solver_iterations(iters, iters)

    # Init Random
    pos = rng.uniform(-half_d, half_d, (num_particles, 4)).astype(np.float32)
    pos[:, 3] = 1.0 # InvMass
    sim.set_positions(pos)

    # Init Velocities
    vel = np.zeros((num_particles, 3), dtype=np.float32)
    vel[:, :3] = rng.normal(0.0, math.sqrt(T), (num_particles, 3)).astype(np.float32)
    sim.set_velocities(vel)

    # Init Quaternions (Uniform Random)
    quat = rng.normal(0.0, 1.0, (num_particles, 4)).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    sim.set_quaternions(quat)

    # Init Angular Velocities (Fix: Was uninitialized garbage)
    # C++ backend has also been patched to default to 0, but being explicit is safer.
    ang_vel = np.zeros((num_particles, 3), dtype=np.float32)
    sim.set_angular_velocities(ang_vel)                
    sim.set_growth_params(growth_rate, scale_init) 
    sim.set_thermostat(T, dt)

    print(f"Jamming Study: Target Phi={phi_ref}, Growth Rate={growth_rate}, Steps={limit_steps}")
      
    for i in range(limit_steps):
        if i==i_switch_cooling:
            restitution = 0.5
            sim.set_material_params(restitution, restitution_t, friction)
            sim.set_thermostat(0, 1e4*dt)
        sim.step(dt)
        max_ov = sim.get_max_overlap()
        is_jammed = max_ov > criterion_ov
        if is_jammed:
            do_iter = True
            num_iter = 0
            while do_iter:
                #sim.set_solver_iterations(0, iters)
                sim.step(0.0)
                num_iter += 1
                max_ov_new = sim.get_max_overlap()
                if max_ov_new >= 0.95*max_ov and num_iter > 6:
                    do_iter = False
                max_ov = max_ov_new
            is_jammed = max_ov > criterion_ov
            if is_jammed:
                growth_factor = sim.get_growth_factor()
                growth_factor *= math.exp(-growth_rate*dt)
                growth_rate *= 0.95
                sim.set_growth_params(growth_rate, growth_factor)
        else:
            growth_rate = min(growth_rate*1.02, growth_rate_init)
            growth_factor = sim.get_growth_factor()
            sim.set_growth_params(growth_rate, growth_factor)
        if (i % dump_interval == 0):
            s = sim.get_scales()
            # Calculate current Phi based on scales
            # Phi = Sum(Vol_i) / Vol_Domain
            mean_scale3 = np.mean(s**3)
            phi_current = phi_ref * mean_scale3
            vel = sim.get_velocities()
            T_current = np.sum(vel[:, 0:3]**2) / (3*num_particles)
            
            num_contacts = sim.get_num_contacts()
            num_manifolds = sim.get_num_manifolds()

            print(f"Step {i}: Scale={np.mean(s):.4f}, Growth Rate={growth_rate:.4f}, T={T_current:.4f}, Phi={phi_current:.4f}, Overlap={max_ov}, Contacts={num_contacts}, Manifolds={num_manifolds}")
            sim.export_lammps(f"{output_dir}/dump.jamming.{i}.lammps", i)
    
    # Export Final VTI
    print(f"Generating SDF VTI for Phi={phi_current:.3f}...")
    sim.export_sdf(f"{output_dir}/packing_ring.vti", (128, 128, 128))

if __name__ == "__main__":
    verify_packing() 