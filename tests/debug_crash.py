
import sys
import os
import time

# Add build directory
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build'))
sys.path.append(build_path)

print(f"Loading from {build_path}")
sys.stdout.flush()

import demgpu
print("Imported demgpu.")
sys.stdout.flush()

try:
    sim = demgpu.Simulation(100)
    print("Constructor ok.")
    sys.stdout.flush()
except Exception as e:
    print(f"Constructor failed: {e}")
    sys.exit(1)

try:
    sim.initialize(0)
    print("Initialize ok.")
    sys.stdout.flush()
except Exception as e:
    print(f"Initialize failed: {e}")
    sys.exit(1)

try:
    print("Running step...")
    sys.stdout.flush()
    sim.step(0.01)
    print("Step ok.")
    sys.stdout.flush()
except Exception as e:
    print(f"Step failed: {e}")
    sys.exit(1)

print("Exiting normally.")
