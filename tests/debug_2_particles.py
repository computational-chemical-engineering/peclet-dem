
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import demgpu

def test_2_particles():
    sim = demgpu.Simulation(2)
    sim.set_domain(make_float3(0,0,0), make_float3(10,10,10))
    
    # 2 Particles overlapping
    # R=1.0. Sum R=2.0
    # Dist = 1.0 (Overlap 1.0)
    pos = np.array([
        [5.0, 5.0, 5.0, 1.0],
        [6.0, 5.0, 5.0, 1.0]
    ], dtype=np.float32)
    
    vel = np.zeros_like(pos)
    vel[:, 3] = 0.0 # Important: w>0 marks ghost!
    scales = np.ones(2, dtype=np.float32)

    sim.initialize(0) # Sphere
    sim.set_positions(pos)
    sim.set_velocities(vel)
    sim.set_scales(scales)
    sim.set_global_scale(1.0)
    
    print("Running 1 step...")
    sim.step(0.01)
    
    # Check positions
    new_pos = sim.get_positions()
    p0 = new_pos[0]
    p1 = new_pos[1]
    
    dist = np.linalg.norm(p0[:3] - p1[:3])
    print(f"Final Dist: {dist}")
    
    if dist > 1.05:
        print("PASS: Particles moved apart.")
    else:
        print("FAIL: Particles did not move.")

def make_float3(x,y,z):
    return np.array([x,y,z], dtype=np.float32)

if __name__ == "__main__":
    test_2_particles()
