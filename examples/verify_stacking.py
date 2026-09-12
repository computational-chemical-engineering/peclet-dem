from peclet import dem
import numpy as np
import time

def verify_stacking():
    print("--- Stacking Stability Test ---")
    
    # Init
    sim = dem.Simulation(200) # Smaller number for quick test
    sim.initialize_shape('sphere', radius=0.5) # Sphere
    
    # Material: Restitution=0.5, Friction=0.3
    sim.set_material_params(0.5, 0.5, 0.)
    sim.set_gravity((0, -9.8, 0))
    # velocityIterations defaults to 0 (no velocity solve, so no restitution/contact impulse and
    # velocities integrate under gravity unbounded even though positions are constraint-clamped) --
    # dem/CLAUDE.md's documented legacy-script trap. Without this the friction and no-friction
    # scripts both "fail" identically on unbounded velocity, which verifies nothing.
    sim.set_solver_iterations(12, 8)

    # Add Floor Plane at y = -5.0
    sim.add_plane([0, -5.0, 0], [0, 1.0, 0])
    
    # Initial Setup
    initial_scale = 0.5
    scales = np.full(200, initial_scale, dtype=np.float32)

    # Seed positions above floor
    # Grid: 5x5x8
    pos = []
    spacing = 1.2
    for y in range(8):
        for x in range(5):
            for z in range(5):
                px = (x - 2) * spacing
                pz = (z - 2) * spacing
                py = -4.0 + y * spacing
                pos.append([px, py, pz])
    
    # If not enough points, fill rest randomly?
    # 5*5*8 = 200. Perfect.
    sim.set_positions(np.array(pos, dtype=np.float32))
    # set_scales (and every other per-particle setter) must follow set_positions, which is what
    # sizes the particle set and resets each particle's scale to 1.
    sim.set_scales(scales)
    sim.set_velocities(np.zeros((200, 3), dtype=np.float32))

    dt = 0.005 # 5ms
    steps = 1000
    
    print(f"Running {steps} steps (dt={dt})...")
    
    sim.set_dt(dt)
    for i in range(steps):
        sim.step()
        if i % 100 == 0:
            vels = sim.get_velocities()
            v_mag = np.linalg.norm(vels, axis=1)
            max_v = np.max(v_mag)
            mean_ke = 0.5 * np.mean(v_mag**2)
            print(f"Step {i}: Max V={max_v:.4f}, Mean KE={mean_ke:.4f}")
            
    # Final Check
    vels = sim.get_velocities()
    v_mag = np.linalg.norm(vels, axis=1)
    final_max_v = np.max(v_mag)
    
    print(f"Final Max Velocity: {final_max_v:.4f}")
    
    sim.write_vtp("stacking_test.vtp")
    
    if final_max_v < 1.0: # Should be near zero (settled)
        print("SUCCESS: Stack is stable.")
    else:
        print("FAILURE: Particles did not settle (Jitter/Explosion).")

if __name__ == "__main__":
    verify_stacking()
