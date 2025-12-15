# Fixed Params
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
    """
    print(f"Generating Unit STL via SDF: {filename}")
    
    # 1. Create a 1-particle simulation
    sim_unit = demgpu.Simulation(1)
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
    print(f"Unit STL saved to {filename}")

def verify_packing():
    num_particles = 300
    radius = 0.5
    diameter = 1.0
    height = 1.0 # H/D = 1
    thickness = 0.3 # Arbitrary hollow thickness
    # Calculate Domain Volume
    # V_cyl = H * pi * (R^2 - r_in^2) = H * pi * (2Rt - t^2)
    vol_particle = math.pi * (2 * radius * thickness - thickness**2)
    growth_rate = 1.0 # Slow growth

    # Study Grid
    # stiffness (iterations) vs timestep (dt)
    # Higher iterations = stiffer solver, better overlap resolution
    # Smaller dt = better stability, but slower growth

    output_dir = "./output/hollow_cylinder_packing"
    os.makedirs(output_dir, exist_ok=True)
    iterations_list = [100]
    dt_list = [0.0001]
    T_gran = [0.0]

    best_phi = 0.0
    best_config = None

    # Target: High density jam
    phi_ref = 0.70
    vol_total_ref = num_particles * vol_particle
    vol_domain_ref = vol_total_ref / phi_ref
    domain_side = vol_domain_ref ** (1.0/3.0)
    
    # --- Generate Unit STL (Skimage) ---
    # Normalize dimensions so that outer radius is 1.0
    # Ovito scales mesh by 'Radius' property, so mesh must be Unit Size.
    r_unit = 1.0
    h_unit = height / radius
    t_unit = thickness / radius
    
    generate_unit_sdf_stl(r_unit, h_unit, t_unit, f"{output_dir}/hollow_cylinder_unit.stl")

    sim = demgpu.Simulation(num_particles)
    sim.initialize(shape_type=2, radius=radius, height=height, thickness=thickness) #hollow cylinder


    half_d = domain_side / 2.0
    sim.set_domain((-half_d, -half_d, -half_d), (half_d, half_d, half_d))
    
    sim.set_gravity(0, 0, 0)
    rng = np.random.default_rng(42)

    print(f"Hollow Cylinder Packing Optimization Study (N={num_particles})")
    print(f"{'Iter':<5} {'dt':<8} {'TGran':<8} {'T':<8} {'MaxPhi':<8} {'FinalOv':<10} {'Notes'}")
    print("-" * 50)

    for iters in iterations_list:
        for dt in dt_list:
            for T in T_gran:    
                # Run one experiment
                
                # Target Loose Packing to verify rotation
                phi_target = 0.40 
                # Ref: 0.25 was loose. 0.7 jammed. 0.4 should flow.
                
                # Update Density
                phi_ref = phi_target
                vol_domain_ref = vol_total_ref / phi_ref
                domain_side = vol_domain_ref ** (1.0/3.0)
                half_d = domain_side / 2.0
                
                # Update Simulation Domain to new size
                sim.set_domain((-half_d, -half_d, -half_d), (half_d, half_d, half_d))
                
                sim.set_material_params(0.0, 0.5, 0.0) # Friction 0.5 for rotation
                sim.set_solver_iterations(20, 20)
                
                # Growth
                growth_rate = 0.1 # Slow and stable
                sim.set_growth_params(growth_rate, 0.05)

                # Init Random
                pos = rng.uniform(-half_d, half_d, (num_particles, 4)).astype(np.float32)
                pos[:, 3] = 1.0 # InvMass
                sim.set_positions(pos)
                
                # Run
                limit_steps = 50000 
                
                print(f"Jamming Study: Target Phi={phi_target}, Growth Rate={growth_rate}, Steps={limit_steps}")
                        
                for i in range(limit_steps):
                    sim.step(dt)
                    if i % 1000 == 0:
                        s = sim.get_scales()
                        # Calculate current Phi based on scales
                        # Phi = Sum(Vol_i) / Vol_Domain
                        mean_scale3 = np.mean(s**3)
                        phi_current = phi_target * mean_scale3
                        max_ov = sim.get_max_overlap()
                        num_contacts = sim.get_num_contacts()
                        
                        print(f"Step {i}: Scale={np.mean(s):.4f}, Phi={phi_current:.4f}, Overlap={max_ov:.4f}, Contacts={num_contacts}")
                        
                        # Stop if jammed (scale stops increasing or overlap high)
                        if max_ov > 1e-3:
                            print("Jamming detected (High Overlap).")
                            break
                        
                        # Export rarely
                        if i % 2000 == 0:
                            sim.export_lammps(f"{output_dir}/dump.jamming.{i}.lammps", i)
                    
                # Final Stats
                final_ov = sim.get_max_overlap()
                s = sim.get_scales()
                phi_final = phi_target * np.mean(s**3)
                
                # Calculate T_current
                vel = sim.get_velocities()
                T_current = np.sum(vel[:, 0:3]**2) / (3*num_particles)
                
                print(f"FINAL: Max Random Packing Density = {phi_final:.4f}")
                print(f"{iters:<5} {dt:<8.4f} {T:<8.4f} {T_current:<8.4f} {phi_final:<8.3f} {final_ov:<10.3f}") 
            
            # Export Final VTI
            print(f"Generating SDF VTI for Phi={phi_final:.3f}...")
            sim.export_sdf(f"{output_dir}/packing_hollow_cylinder_jammed.vti", (256, 256, 256))

if __name__ == "__main__":
    verify_packing() 