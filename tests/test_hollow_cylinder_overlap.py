import sys
import os
import math
import numpy as np

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
from peclet import dem

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

def test_static_overlap():
    print("Initializing Static Overlap Test...")
    
    # 1. Setup Simulation (N=2)
    sim = dem.Simulation(2)
    # Box domain
    sim.set_domain(
        [-0.45, -1.0, -1.0],
        [0.45, 1.0, 1.0]
    )
    
    # 2. Initialize Hollow Cylinder
    # Type 2 = Hollow Cylinder
    # Radius=0.1, Height=0.4, Thickness=0.05
    sim.initialize(shape_type=2, radius=0.5, height=1.0, thickness=0.5)
    
    # 3. Add 2 Particles with Known Overlap
    # P1 at (0,0,0)
    # P2 at (0.15, 0, 0) -> Distance 0.15. 
    # Radius is 0.1. So Gap should be 0.15 - 0.1 - 0.1 = -0.05 (Overlap)
    # They should overlap by 0.05.
    
    pos = np.array([
        [-0.1, 0.0, 0.0, 1.0],      # q=(0,0,0,1)
        [0.1, 0.0, 0.0, 1.0]      # q=(0,0,0,1)
    ], dtype=np.float32)
    
    vel = np.zeros_like(pos)
    quat = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # Set data
    sim.set_positions(pos)
    sim.set_velocities(vel)
    sim.set_quaternions(quat)
    
    
    # Create Output Directory
    output_dir = "output/debug_overlap"
    os.makedirs(output_dir, exist_ok=True)

    # Convert user params to unit params for STL
    # Params used: sim.initialize(shape_type=2, radius=0.5, height=1.0, thickness=0.5)
    # Unit STL expects dimensions relative to R=1.0
    r_sim = 0.5
    h_sim = 1.0
    t_sim = 0.5
    
    r_unit = 1.0
    h_unit = h_sim / r_sim # 2.0
    t_unit = t_sim / r_sim # 1.0
    
    generate_unit_sdf_stl(r_unit, h_unit, t_unit, f"{output_dir}/hollow_cylinder_unit.stl")

    # 4. Run 1 Step (Detection)
    sim.set_growth_params(1.0, 0.2)
    dt = 0.01
    for i in range(100):
        sim.step(dt)
        
        # Expert LAMMPS
        sim.export_lammps(f"{output_dir}/step.{i}.lammps", i)
    
        # 5. Check Results
        num_contacts = sim.get_num_contacts()
        num_manifolds = sim.get_num_manifolds()
        max_overlap = sim.get_max_overlap()
        
        print(f"{i}, Contacts: {num_contacts}, Manifolds: {num_manifolds}, Max Overlap: {max_overlap}")
        
    if num_contacts > 0:
        print("SUCCESS: Narrowphase detected contact.")
    else:
        print("FAILURE: Narrowphase missed contact.")
        
    if max_overlap > 0:
        print(f"SUCCESS: Solver detected overlap C < 0. (Penetration={max_overlap})")
    else:
        print("FAILURE: Solver reports 0 overlap (C >= 0).")

if __name__ == "__main__":
    test_static_overlap()
