from peclet import dem
import numpy as np
import time

def main():
    print("Initializing DEM-GPU...")
    sim = dem.Simulation(1000)
    sim.initialize()

    t0 = time.time() # Define t0 for the new code block
    print("Running expansion test...")
    # Initial scale check
    scales = sim.get_scales()
    print(f"Initial Scale (first 5): {scales[:5]}")

    # Run some steps
    for i in range(10):
        sim.step(0.001)
        if i % 10 == 0:
            print(f"Step {i}")

    # Expand
    print("Expanding particles...")
    scales[:] = 1.5  # Grow by 50%
    sim.set_scales(scales)
    scales_read = sim.get_scales()
    print(f"New Scale (first 5): {scales_read[:5]}")

    # Run equilibrium
    for i in range(50):
        sim.step(0.001)
        if i % 10 == 0:
            print(f"Step {i}")

    pos = sim.get_positions()
    # Reshape to (N, 4)
    pos = pos.reshape((-1, 4))
    print(f"Final Positions (first 5):\n{pos[:5]}")

    print(f"Done in {time.time() - t0:.4f}s")

if __name__ == "__main__":
    main()
