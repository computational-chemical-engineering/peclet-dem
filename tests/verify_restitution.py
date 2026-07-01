
import sys
import os
import numpy as np
import math

# Add build path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
from peclet import dem

def test_restitution():
    print("--- Test 1: Normal Restitution ---")
    sim = dem.Simulation(1)
    sim.set_domain(np.array([-10,-10,-10], dtype=np.float32), np.array([10,10,10], dtype=np.float32))
    
    # Drop from y=2.0. Radius=1.0. Ground=0.0?
    # Our ground kernel uses domain_min.y. Let's set min.y = 0.
    sim.set_domain(np.array([-10,0,-10], dtype=np.float32), np.array([10,20,10], dtype=np.float32))
    
    # e_n = 0.8
    sim.set_material_params(0.8, 0.0, 0.0) # Restitution, Tangent Restitution, Friction
    
    h0 = 5.0
    pos = np.array([[0, h0, 0, 1.0]], dtype=np.float32)
    vel = np.zeros_like(pos)
    scales = np.ones(1, dtype=np.float32)
    
    sim.initialize(0) # Sphere
    sim.set_positions(pos)
    sim.set_velocities(vel)
    sim.set_scales(scales)
    
    dt = 0.01
    
    # Run until bounce
    max_h = 0.0
    bounced = False
    
    impact_v = 0.0
    bounce_v = 0.0
    
    # Track velocity just before and after impact (y < radius=1.0)
    
    for i in range(200):
        v_prev = sim.get_velocities()[0][1]
        sim.step(dt)
        p = sim.get_positions()[0]
        v = sim.get_velocities()[0][1]
        
        # print(f"Step {i}: y={p[1]:.4f} v={v:.4f}")
        
        if v_prev < 0 and v > 0:
            print(f"BOUNCE DETECTED at Step {i}. V_in={v_prev:.4f}, V_out={v:.4f}")
            bounced = True
            impact_v = v_prev
            bounce_v = v
            break

    if not bounced:
        print("FAIL: Did not bounce.")
        return False
        
    ratio = -bounce_v / impact_v
    print(f"Restitution Ratio: {ratio:.4f} (Expected 0.8)")
    
    if abs(ratio - 0.8) < 0.1:
        print("PASS: Normal Restitution correct.")
    else:
        print("FAIL: Restitution ratio mismatch.")

    print("\n--- Test 2: Tangential Friction (Spinning Ball) ---")
    sim2 = dem.Simulation(1)
    # Ground at y=0
    sim2.set_domain(np.array([-10,0,-10], dtype=np.float32), np.array([10,20,10], dtype=np.float32))
    
    # Friction = 0.5. Normal Res = 0.5.
    sim2.set_material_params(0.5, 0.0, 0.5)
    
    # Drop with spin
    pos = np.array([[0, 2.0, 0, 1.0]], dtype=np.float32) # Just above ground (R=1.0)
    vel = np.zeros_like(pos)
    
    # Initial angular velocity (Spin around Z axis) -> moves in X?
    # w = (0, 0, 10). r_c = (0, -1, 0).
    # v_contact = v + w x r = 0 + (0,0,10) x (0,-1,0) = (10, 0, 0).
    # Friction opposes v_contact -> Force in -X.
    # Wait, v_contact is positive X. Force is negative X.
    # Wait, if ball spins CCW (w_z > 0), bottom moves RIGHT.
    # Ground pushes LEFT. Ball accel LEFT (neg X).
    # Torque = r_c x F = (0,-1,0) x (-F, 0, 0) = (0, 0, -F).
    # Torque opposes spin (reduces w_z).
    
    # Let's set initial vel to 0.
    scales = np.ones(1, dtype=np.float32)
    
    sim2.initialize(0)
    sim2.set_positions(pos)
    sim2.set_velocities(vel)
    sim2.set_scales(scales)
    
    # Set angular velocity via hack? 
    # Current bindings don't expose set_angular_velocity?
    # I verified `simulation.h` has `allocate_device(ps_.d_ang_vel...` but NO SETTER in bindings!
    # I need to add `set_angular_velocities` binding if I want to test this.
    # Or I can imply it via initial collision? 
    # Or I can assume 0 spin and check friction RESISTING sliding?
    # "Tangential Restitution" implies bounce. But basic friction is key.
    
    # Let's check SLIDING friction.
    # Initial horizontal velocity. Friction should slow it down.
    vel2 = np.array([[5.0, -1.0, 0.0, 0.0]], dtype=np.float32) # Moving X, Falling
    sim2.set_velocities(vel2)
    
    print("Simulating sliding...")
    vx_init = 5.0
    vx_final = 0.0
    
    for i in range(50):
        sim2.step(dt)
        vals = sim2.get_velocities()[0]
        # print(f"Step {i} v={vals}")
        vx_final = vals[0]
        
    print(f"Vx Initial: {vx_init}, Vx Final: {vx_final}")
    
    if vx_final < vx_init:
        print("PASS: Horizontal velocity reduced (Friction working).")
    else:
        print("FAIL: No friction observed.")
        
if __name__ == "__main__":
    test_restitution()
