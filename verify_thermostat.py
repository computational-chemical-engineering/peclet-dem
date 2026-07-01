
import sys
sys.path.append('./build')
from peclet import dem
import numpy as np
import matplotlib.pyplot as plt

def verify_thermostat():
    # 1. Initialize
    n_particles = 1000
    sim = dem.Simulation(n_particles)
    sim.initialize(shape_type=1, radius=0.5) # Spheres
    
    # 2. Domain (Periodic)
    L = 20.0
    half = L/2
    sim.set_domain((-half, -half, -half), (half, half, half))
    sim.enable_periodicity(True, True, True)
    
    # 3. Random Positions
    pos = (np.random.rand(n_particles, 3) * L) - half
    pos4 = np.hstack((pos, np.ones((n_particles, 1)))) # w=inv_mass=1.0
    sim.set_positions(pos4)
    
    # 4. Constant Velocity
    vel = np.zeros((n_particles, 3))
    vel[:, 0] = 1.0 # CONSTANT 1.0 X Velocity
    sim.set_velocities(vel)
    
    # Angular Vel
    ang_vel = (np.random.rand(n_particles, 3) - 0.5) * 10.0
    sim.set_angular_velocities(ang_vel)
    
    # Mass/Inertia
    # Sphere I = 2/5 m r^2. m=1, r=0.5 => I = 0.4 * 1 * 0.25 = 0.1
    # inv_I = 10.0
    # Actually initialize sets this based on radius. W=inv_mass.
    # Let's read them.
    
    # 5. Configure Thermostat
    target_T = 1.0
    tau = 0.1
    kB = 1.0
    sim.set_thermostat(target_T, tau, kB)
    
    dt = 0.01
    steps = 500
    
    T_trans_history = []
    T_rot_history = []
    
    print(f"Starting Thermostat Verification. Target T={target_T}, Tau={tau}")
    
    # Check initial state
    v_init = sim.get_velocities()
    print(f"Initial Max V: {np.max(np.abs(v_init))}")
    print(f"Initial Mean V^2: {np.mean(np.sum(v_init**2, axis=1))}")
    inv_I_init = sim.get_inv_inertia()
    print(f"Initial Inv I sample: {inv_I_init[0]}")

    for i in range(steps):
        sim.step(dt)
        
        # Compute T manually
        v = sim.get_velocities() # (N, 3)
        w = sim.get_angular_velocities() # (N, 3)
        inv_I = sim.get_inv_inertia() # (N, 4)
        quats = sim.get_quaternions() # (N, 4)
        
        # KE Trans (Assuming m=1 for all, inv_mass=1)
        # Actually should read mass from pos[:,3] but we set it to 1.
        ke_trans = 0.5 * np.sum(v**2) 
        N = n_particles
        # T = 2*KE / (3Nk)
        T_trans = 2.0 * ke_trans / (3.0 * N * kB)
        
        # KE Rot
        # Need to rotate w to body frame? Or simply w . I . w if I is isotropic?
        # Sphere I is isotropic diagonal. So I_world = I_body.
        # I = 1/inv_I. inv_I is (N,4).
        # We can assume isotropic for spheres.
        
        I_val = 0.0
        if inv_I[0,0] > 0:
            I_val = 1.0 / inv_I[0,0]
            
        ke_rot = 0.5 * I_val * np.sum(w**2)
        T_rot = 2.0 * ke_rot / (3.0 * N * kB)
        
        T_trans_history.append(T_trans)
        T_rot_history.append(T_rot)
        
        if i % 50 == 0:
            print(f"Step {i}: T_trans={T_trans:.4f} T_rot={T_rot:.4f}")
            
    # Check convergence
    final_T_trans = T_trans_history[-1]
    final_T_rot = T_rot_history[-1]
    
    print(f"Final T_trans: {final_T_trans:.4f}")
    print(f"Final T_rot: {final_T_rot:.4f}")
    
    if abs(final_T_trans - target_T) < 0.05 and abs(final_T_rot - target_T) < 0.05:
        print("SUCCESS: Temperature converged to target.")
    else:
        print("FAILURE: Temperature did not converge.")

if __name__ == "__main__":
    verify_thermostat()
