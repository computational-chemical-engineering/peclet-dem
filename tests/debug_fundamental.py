import dem
import numpy as np
import sys
import math

def print_state(sim, step, label):
    pos = sim.get_positions()
    vel = sim.get_velocities()
    print(f"--- {label} (Step {step}) ---")
    for i in range(len(pos)):
        print(f"P{i}: Pos=[{pos[i][0]:.4f}, {pos[i][1]:.4f}, {pos[i][2]:.4f}], Vel=[{vel[i][0]:.4f}, {vel[i][1]:.4f}, {vel[i][2]:.4f}]")

def test_center_overlap():
    print("\n=== TEST 1: Two Particles Center Overlap ===")
    sim = dem.Simulation(2)
    sim.initialize(0)
    sim.set_domain((-10, -10, -10), (10, 10, 10))
    sim.set_gravity(0, 0, 0)
    sim.set_global_scale(1.0)
    sim.set_material_params(0.5, 0.5, 0.5)

    pos = np.array([[-0.5, 0, 0], [0.5, 0, 0]], dtype=np.float32)
    vel = np.zeros((2, 3), dtype=np.float32)
    sim.set_positions(pos)
    sim.set_velocities(vel)
    
    # Force scales to 1.0
    scales = np.ones(2, dtype=np.float32)
    sim.set_scales(scales)

    print_state(sim, 0, "Initial")
    
    sim.step(0.01)
    
    print_state(sim, 1, "After Step 1")

    pos_new = sim.get_positions()
    dist = np.linalg.norm(pos_new[1] - pos_new[0])
    print(f"Distance: {dist:.4f} (Ideal: >= 2.0?)") 
    # Radius=1.0 per particle => SumRadii=2.0
    
    if dist > 1.001:
        print("SUCCESS: Particles pushed apart.")
    else:
        print("FAILURE: Particles not pushed apart.")
        
    # Check Step 2
    sim.step(0.01)
    print_state(sim, 2, "After Step 2")


def test_3_particles():
    print("\n=== TEST 2: Three Particles Linear Overlap ===")
    sim = dem.Simulation(3)
    sim.initialize(0)
    sim.set_domain((-10, -10, -10), (10, 10, 10))
    sim.set_gravity(0, 0, 0)
    sim.set_global_scale(1.0)
    
    # 3 particles in line at x=-0.5, 0.0, 0.5. All radius 1.0. All overlap.
    pos = np.array([[-0.5, 0, 0], [0.0, 0, 0], [0.5, 0, 0]], dtype=np.float32)
    vel = np.zeros((3, 3), dtype=np.float32)
    sim.set_positions(pos)
    sim.set_velocities(vel)
    sim.set_scales(np.ones(3, dtype=np.float32))
    
    print_state(sim, 0, "Initial")
    sim.step(0.01)
    print_state(sim, 1, "After Step 1")
    
    pos_new = sim.get_positions()
    # Check distances
    print(f"Dist P0-P1: {np.linalg.norm(pos_new[1]-pos_new[0]):.4f}")
    print(f"Dist P1-P2: {np.linalg.norm(pos_new[2]-pos_new[1]):.4f}")


def test_periodic_overlap():
    print("\n=== TEST 3: Periodic Overlap ===")
    sim = dem.Simulation(2)
    sim.initialize(0)
    # Domain -5 to 5 (Size 10)
    sim.set_domain((-5, -5, -5), (5, 5, 5)) 
    sim.set_gravity(0, 0, 0)
    sim.set_global_scale(1.0)
    sim.set_scales(np.ones(2, dtype=np.float32))
    
    # P0 at -4.5. P1 at +4.5.
    # Distance in regular space = 9.0.
    # Periodic distance = 1.0.
    # Sum Radii = 2.0.
    # Overlap = 1.0.
    
    pos = np.array([[-4.5, 0, 0], [4.5, 0, 0]], dtype=np.float32)
    vel = np.zeros((2, 3), dtype=np.float32)
    sim.set_positions(pos)
    sim.set_velocities(vel)
    
    print_state(sim, 0, "Initial")
    sim.step(0.01)
    print_state(sim, 1, "After Step 1")
    
    vel_new = sim.get_velocities()
    
    # P0 (-4.5) should be pushed Right (+vel) by P1's ghost at (-5.5) 
    # OR pushed Left (-vel) by P1's ghost ??
    # P1 is at 4.5. Ghost at 4.5 - 10 = -5.5.
    # P0 is at -4.5.
    # P0 > GhostP1. P0 pushed +X (Right).
    
    # P1 (4.5). Ghost of P0 (at -4.5 + 10 = 5.5).
    # P1 < GhostP0. P1 pushed -X (Left).
    
    p0_vel_x = vel_new[0][0]
    p1_vel_x = vel_new[1][0]
    
    print(f"P0 Vel X: {p0_vel_x}")
    print(f"P1 Vel X: {p1_vel_x}")
    
    if p0_vel_x > 0 and p1_vel_x < 0:
         print("SUCCESS: Periodic repulsion detected (Correct Direction).")
    elif p0_vel_x < 0 and p1_vel_x > 0:
         print("FAILURE: Particles attracted or wrong side?")
    else:
         print("FAILURE: No interaction or asymmetric.")

if __name__ == "__main__":
    test_center_overlap()
    test_3_particles()
    test_periodic_overlap()
