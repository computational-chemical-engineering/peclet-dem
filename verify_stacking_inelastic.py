import demgpu
import numpy as np
import time
import sys
print(f"DEBUG: demgpu loaded from: {demgpu.__file__}")

def verify_stacking_inelastic():
    print("--- Stacking Stability Test (Inelastic e=0) ---")
    
    sim = demgpu.Simulation(200)
    sim.initialize(0) 
    
    # Material: Restitution=0.0, Friction=0.0
    sim.set_material_params(0.0, 0.0, 0.) 
    sim.set_gravity(0, -9.8, 0)
    
    sim.add_plane([0, -5.0, 0], [0, 1.0, 0])
    
    initial_scale = 0.5
    scales = np.full(200, initial_scale, dtype=np.float32)
    sim.set_scales(scales)
    
    pos = []
    spacing = 1.2
    for y in range(8):
        for x in range(5):
            for z in range(5):
                px = (x - 2) * spacing
                pz = (z - 2) * spacing
                py = -4.0 + y * spacing
                pos.append([px, py, pz])
    
    sim.set_positions(np.array(pos, dtype=np.float32))
    sim.set_velocities(np.zeros((200, 3), dtype=np.float32))
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
    
    # Let's define 'n' and 'quat' based on common DEMGPU usage and the context.
    n = 200 # Number of particles
    quat = np.array([[1.0, 0.0, 0.0, 0.0]] * n, dtype=np.float32) # Identity quaternions
    
    sim.set_positions(np.array(pos, dtype=np.float32))
    sim.set_velocities(np.zeros((n, 3), dtype=np.float32))
    sim.set_quaternions(np.array(quat, dtype=np.float32))
    sim.set_scales(scales)

    # Export Initial State
    dt = 0.005
    steps = 1000
    
    import os
    output_dir = "output/stacking_inelastic"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running {steps} steps (dt={dt})...")
    
    for i in range(steps):
        sim.step(dt)
        
        # Export for Ovito every 10 steps
        if i % 10 == 0:
            p = sim.get_positions()
            v = sim.get_velocities()
            # If get_quaternions is not standard, we might need to handle it.
            # Simulation usually has it. 
            # Note: The binding we made accepts numpy arrays.
            # Assuming get_quaternions returns numpy array.
            q = sim.get_quaternions()
            s = sim.get_scales()
            
            # export_lammps(filename, step, pos, vel, quats, radii, ...)
            demgpu.export_lammps(f"{output_dir}/dump.stacking.{i}.lammps", i, p, v, q, s)
            
        if i % 100 == 0:
            vels = sim.get_velocities()
            v_mag = np.linalg.norm(vels, axis=1)
            max_v = np.max(v_mag)
            mean_ke = 0.5 * np.mean(v_mag**2)
            
            max_overlap = sim.compute_overlaps()
            print(f"Step {i}: Max V={max_v:.4f}, Mean KE={mean_ke:.6f}, Max Overlap={max_overlap:.6f}")
            
    vels = sim.get_velocities()
    v_mag = np.linalg.norm(vels, axis=1)
    final_max_v = np.max(v_mag)
    
    print(f"Final Max Velocity: {final_max_v:.4f}")
    
    if final_max_v < 0.2: 
        print("SUCCESS: Stack is stable (Inelastic).")
    else:
        print("FAILURE: Inelastic stack did not settle.")

if __name__ == "__main__":
    verify_stacking_inelastic()
