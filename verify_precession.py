import numpy as np
import os
import sys
import math

# Adjust path to find the shared library
sys.path.append(os.path.join(os.path.dirname(__file__), "build"))
from peclet import dem

def verify_precession():
    # Setup
    radius = 0.5
    height = 1.0
    thickness = 0.2
    
    dt = 1e-4
    num_steps = 5000 # 0.5 seconds
    
    # Init Simulation
    sim = dem.Simulation(1)
    sim.initialize(shape_type=2, radius=radius, height=height, thickness=thickness)
    
    # Domain (Arbitrary, no boundaries needed for free flight)
    sim.set_domain((-10, -10, -10), (10, 10, 10))
    sim.set_gravity(0, 0, 0)
    
    # Particle Setup
    pos = np.zeros((1, 4), dtype=np.float32)
    pos[0] = [0, 0, 0, 1.0] # Mass=1
    
    vel = np.zeros((1, 3), dtype=np.float32)
    
    quat = np.zeros((1, 4), dtype=np.float32)
    quat[0] = [0, 0, 0, 1] # Identity
    
    # Initial Spin (Non-Principal Axis)
    # Cylinder I_xx = I_zz != I_yy.
    # Spin (5, 5, 5) triggers precession.
    ang_vel = np.zeros((1, 3), dtype=np.float32)
    ang_vel[0] = [5.0, 5.0, 5.0]
    
    scales = np.ones(1, dtype=np.float32)
    
    sim.set_positions(pos)
    sim.set_velocities(vel)
    sim.set_quaternions(quat)
    sim.set_angular_velocities(ang_vel)
    sim.set_scales(scales)
    
    print("Starting Precession Verification...")
    print(f"Initial W: {ang_vel[0]}")
    
    # Helper for Metrics
    def get_metrics(sim):
        # Need current state
        w_world = sim.get_angular_velocities()[0]
        q = sim.get_quaternions()[0]
        inv_I_data = sim.get_inv_inertia()[0] # (inv_xx, inv_yy, inv_zz, 0)
        
        # Construct R
        x, y, z, w = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        
        # Local Omega
        w_local = R.T @ w_world
        
        # Local Inertia
        I_local = np.zeros(3)
        if inv_I_data[0] > 0: I_local[0] = 1.0/inv_I_data[0]
        if inv_I_data[1] > 0: I_local[1] = 1.0/inv_I_data[1]
        if inv_I_data[2] > 0: I_local[2] = 1.0/inv_I_data[2]
        
        # L_local
        L_local = I_local * w_local
        
        # L_world
        L_world = R @ L_local
        
        # Energy
        E = 0.5 * np.dot(w_local, L_local)
        
        return w_world, L_world, E
        
    # Run
    w0, L0, E0 = get_metrics(sim)
    print(f"Initial Metrics -- L: {L0}, E: {E0:.6f}")
    
    history_w = []
    
    for i in range(num_steps):
        sim.step(dt)
        if i % 100 == 0:
            w, L, E = get_metrics(sim)
            # print(f"Step {i}: W={w}, L={L}, E={E:.6f}")
            # Check Deviation
            dL = np.linalg.norm(L - L0)
            dE = abs(E - E0)
            history_w.append(w)
            
            if dL > 1e-2 or dE > 1e-3:
                 print(f"WARNING: Drift at step {i}! dL={dL:.6f}, dE={dE:.6f}")
                 
    w_final, L_final, E_final = get_metrics(sim)
    print(f"Final Metrics -- L: {L_final}, E: {E_final:.6f}")
    
    # Check if W changed (It should!)
    dw = np.linalg.norm(w_final - w0)
    print(f"Change in W vector (Precession): {dw:.6f}")
    if dw < 1e-3:
        print("ERROR: Omega did not change! Precession failed.")
    else:
        print("SUCCESS: Omega evolved (Precession verified).")
        
    # Check L conservation
    dL = np.linalg.norm(L_final - L0)
    print(f"L Conservation Error: {dL:.6f}")
    if dL < 0.1: # Allow some numerical drift for Euler integration
        print("SUCCESS: L approximately conserved.")
    else:
        print("ERROR: L drift too high.")

if __name__ == "__main__":
    verify_precession()
