
import os
import sys
import math
import numpy as np

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
from peclet import dem

def run_cylinder_highres():
    # Setup
    num_particles = 1
    sim = dem.Simulation(num_particles)
    sim.initialize(shape_type=2) # Cylinder
    
    # Domain: Focus on the single cylinder
    # Cylinder default: Radius 0.5, Height 2.0. Scale 1.0 (Wait, Simulation::initialize generates scale 1.0?)
    # Simulation::initialize uses default H_inv_inertia 1.0. Does not set scales!
    # "ParticleSystem.cuh" d_scale might be uninitialized if not set?
    # Wait, "Simulation::initialize" calls "shape_manager_.createAnalyticShape".
    # But d_scale?
    # In "verify_sdf_cylinder.py", we didn't set scales.
    # We should set scale to 1.0 explicit.
    
    sim.set_scales(np.array([1.0], dtype=np.float32))

    # Domain size: Cylinder is H=2.0 (Y-axis presumably). Radius=0.5.
    # So bounds Y=[-1, 1]. X,Z=[-0.5, 0.5].
    # Let's verify orientation.
    # Default quaternion is identity? (0,0,0,1).
    # Cylinder axis defaults to Y? 
    # Logic in `sdf_hollow_cylinder`: r = sqrt(x*x + z*z). This implies Y-axis alignment. Correct.
    
    domain_min = (-1.5, -1.5, -1.5)
    domain_max = (1.5, 1.5, 1.5)
    sim.set_domain(domain_min, domain_max)
    
    # Position at origin
    pos = np.zeros((1, 4), dtype=np.float32)
    pos[0] = [0, 0, 0, 1.0]
    sim.set_positions(pos)
    
    # Export High Res
    os.makedirs("output/sdf", exist_ok=True)
    filename = "output/sdf/cylinder_highres.vti"
    print(f"Exporting High Res Cylinder to {filename}...")
    # 128x128x128 on a 3x3x3 domain gives voxel size ~ 0.023.
    # Thickness is 0.2. So ~8-9 voxels across thickness. Good.
    sim.export_sdf(filename, (128, 128, 128))
    
    print("Success.")

if __name__ == "__main__":
    run_cylinder_highres()
