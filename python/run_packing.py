import demgpu
import numpy as np
import time

def main():
    print("Initializing DEM-GPU...")
    sim = demgpu.Simulation(1000)
    sim.initialize()

    dt = 0.01
    steps = 100

    print(f"Running {steps} steps...")
    start = time.time()
    for i in range(steps):
        sim.step(dt)
        if i % 10 == 0:
            print(f"Step {i}")

            # Optional: Visualize
            # pos = sim.get_positions()
            # print(pos[0])

    end = time.time()
    print(f"Done in {end - start:.4f}s")

if __name__ == "__main__":
    main()
