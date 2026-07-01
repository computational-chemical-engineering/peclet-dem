import sys
sys.path.append('./build')
from peclet import dem
import numpy as np
import time
import sys
print(f"DEBUG: dem loaded from: {dem.__file__}")

def verify_stacking_inelastic():
    print("--- Stacking Stability Test (Inelastic e=0) ---")
    
    shape = (4,1,1)
    n = shape[0]*shape[1]*shape[2]
    sim = dem.Simulation(n)
    radius = 0.6
    sim.initialize(0, radius=radius)  # Base Radius 1.0 so Scale acts as Radius 
    
    # Material: Restitution=0.0, Friction=0.0
    sim.set_material_params(0.0, 0.0, 0.0) # Inelastic
    sim.set_gravity(0, -9.8, 0)
    sim.set_solver_iterations(10, 100) # Pos=10, Vel=20
    sim.set_global_scale(1.0)
    
    sim.add_plane([0, -5.0, 0], [0, 1.0, 0])
    
    initial_scale = 0.5
    scales = np.full(n, initial_scale, dtype=np.float32)
    sim.set_scales(scales)
    
    pos = []
    spacing = 2.1
    for y in range(shape[0]):
        for x in range(shape[1]):
            for z in range(shape[2]):
                px = (x - 2) * spacing
                pz = (z - 2) * spacing
                py = -4.0 + y * spacing
                pos.append([px, py, pz])
    
    sim.set_positions(np.array(pos, dtype=np.float32))
    sim.set_velocities(np.zeros((n, 3), dtype=np.float32))
    # The following lines are added based on the provided Code Edit block.
    # Note: 'n' is not defined in the original code, assuming it should be 200.
    # 'quat' is also not defined, assuming it should be initialized or derived.
    # For now, I will use the original values for velocities and scales,
    # and add a placeholder for quaternions if it's a new line.
    # Re-evaluating the instruction: "Add diagnosis prints".
    # The provided "Code Edit" block shows the *result* of the change,
    # which includes more than just diagnosis prints.
    # I will apply the changes as literally as possible from the "Code Edit" block,
    # assuming 'n' is 200 and 'quat' needs to be defined.
    # Given the context, 'quat' is likely an array of identity quaternions.
    quat = np.array([[1.0, 0.0, 0.0, 0.0]] * n, dtype=np.float32) # Identity quaternions
    
    sim.set_positions(np.array(pos, dtype=np.float32))
    sim.set_velocities(np.zeros((n, 3), dtype=np.float32))
    sim.set_quaternions(np.array(quat, dtype=np.float32))
    sim.set_scales(scales)

    # Export Initial State
    dt = 0.05
    steps = 40
    
    import os
    output_dir = "output/stacking_inelastic"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running {steps} steps (dt={dt})...")
    
    for i in range(steps):
        sim.step(dt)
        
        if i % 1 == 0:
            p = sim.get_positions()
            v = sim.get_velocities()
            # Assuming get_quaternions returns numpy array.
            q = sim.get_quaternions()
            s = sim.get_scales()
            
            # export_lammps(filename, step, pos, vel, quats, radii, ...)
            dem.export_lammps(f"{output_dir}/dump.stacking.{i}.lammps", i, p, v, q, radius*s)
            
            # Trace Particle 1
            print(f"Step {i}: P1_y={p[1][1]:.6f}, P1_vy={v[1][1]:.6f}")
    
    vels = sim.get_velocities()
    v_mag = np.linalg.norm(vels, axis=1)
    max_v = np.max(v_mag)
    mean_ke = 0.5 * np.mean(v_mag**2)
    
    max_overlap = sim.compute_overlaps()
    print(f"Max V={max_v:.4f}, Mean KE={mean_ke:.6f}, Max Overlap={max_overlap:.6f}")

            
    vels = sim.get_velocities()
    v_mag = np.linalg.norm(vels, axis=1)
    final_max_v = np.max(v_mag)
    

    print(f"Final Max Velocity: {final_max_v:.4f}")
    
    if final_max_v < 0.2: 
        print("SUCCESS: Stack is stable (Inelastic).")
    else:
        print("FAILURE: Inelastic stack did not settle.")

    # --- Final Analysis ---
    final_pos = sim.get_positions()
    final_scales = sim.get_scales()
    
    print("\n--- Final State Analysis ---")
    
    min_gap = float('inf')
    min_gap_indices = (-1, -1)
    
    # Particle-Particle Gaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(final_pos[i] - final_pos[j])
            gap = dist - radius*(final_scales[i] + final_scales[j])
            
            if gap < min_gap:
                min_gap = gap
                min_gap_indices = (i, j)

    print(f"Minimum Particle-Particle Gap: {min_gap:.6f} between {min_gap_indices}")
    if min_gap > 1e-4:
        print("RESULT: SEPARATION (Particles are not touching)")
    elif min_gap < -1e-4:
        print("RESULT: OVERLAP (Particles are penetrating)")
    else:
        print("RESULT: TOUCHING (Perfect contact)")
        
    # Wall Gap (Plane at y = -5.0)
    # Bottom plane normal is (0,1,0), point is (0,-5,0).
    # Distance to plane = dot(p - plane_p, n)
    # In this case = p.y - (-5.0) = p.y + 5.0
    
    min_wall_gap = float('inf')
    min_wall_idx = -1
    
    for i in range(n):
        dist_to_plane = (final_pos[i][1] - (-5.0))
        gap = dist_to_plane - radius*final_scales[i]
        
        if gap < min_wall_gap:
            min_wall_gap = gap
            min_wall_idx = i
            
    print(f"Minimum Wall Gap: {min_wall_gap:.6f} (Particle {min_wall_idx})")
    
    if min_wall_gap > 1e-4:
        print("RESULT: WALL SEPARATION (Floating above floor)")
    elif min_wall_gap < -1e-4:
        print("RESULT: WALL OVERLAP (Sinking into floor)")
    else:
        print("RESULT: WALL CONTACT")

    p = p.reshape((*shape, -1))
    v = v.reshape((*shape, -1))

if __name__ == "__main__":
    verify_stacking_inelastic()
