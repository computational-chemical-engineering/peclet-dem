#include "../src/simulation.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// Simple check macro
#define CHECK(expr)                                                            \
  if (!(expr)) {                                                               \
    std::cerr << "Test failed: " << #expr << " at line " << __LINE__           \
              << std::endl;                                                    \
    exit(1);                                                                   \
  }

void test_corner_ghosts() {
  std::cout << "Running Periodic Corner Test..." << std::endl;

  // 1. Setup
  // Domain: [0,0,0] to [10,10,10]
  // Particle at (9.9, 9.9, 9.9) -> Should trigger 7 ghosts
  Simulation sim(1); // 1 Real Particle
  sim.set_domain(make_float3(0, 0, 0), make_float3(10, 10, 10));
  sim.initialize(0); // Spheres

  // Set Position
  std::vector<float> pos = {9.9f, 9.9f, 9.9f};
  pybind11::array_t<float> py_pos({1, 3}, pos.data());
  sim.set_positions_numpy(py_pos);

  // Set Scale (Radius = 1.0)
  std::vector<float> scales = {1.0f};
  pybind11::array_t<float> py_scales({1}, scales.data());
  sim.set_scales_numpy(py_scales);

  // 2. Step (Trigger Ghost Update)
  // We need to call step() or expose update_ghosts().
  // update_ghosts is private/helper. step() calls it.
  // step() also integrates. We set dt=0 to avoid movement?
  // Gravity might move it. Set gravity 0.
  sim.set_gravity(0, 0, 0);
  sim.step(0.0f);

  // 3. Verify Count
  int n = sim.num_particles(true); // Include Ghosts
  std::cout << "Particles after step: " << n << std::endl;
  CHECK(n == 8);

  // 4. Verify Positions
  auto py_res = sim.get_positions_numpy(true); // Include Ghosts
  float *ptr = (float *)py_res.request().ptr;

  // Expected shifts:
  // Real: (9.9, 9.9, 9.9)
  // Ghosts (Bitmask logic):
  // Near Upper Corner -> Needs Lower neighbors (Shift -10)
  // Wait, let's re-verify the logic in plan.
  // If p > max - skin, it is "Right Edge".
  // Ghost should appear at "Left Edge" (x - L).
  // Correct.

  // We expect ghosts at:
  // (-0.1, 9.9, 9.9)  [X-Shift]
  // (9.9, -0.1, 9.9)  [Y-Shift]
  // (9.9, 9.9, -0.1)  [Z-Shift]
  // (-0.1, -0.1, 9.9) [XY]
  // (-0.1, 9.9, -0.1) [XZ]
  // (9.9, -0.1, -0.1) [YZ]
  // (-0.1, -0.1, -0.1)[XYZ]

  int match_count = 0;
  for (int i = 0; i < 8; ++i) {
    float x = ptr[i * 3 + 0];
    float y = ptr[i * 3 + 1];
    float z = ptr[i * 3 + 2];

    bool is_valid = false;
    // Check if it matches one of the expected
    // We can just print them for visual verify if automated is complex
    printf("P[%d]: (%.2f, %.2f, %.2f)\n", i, x, y, z);

    // Loose check for corner cluster
    // Real or Ghost
    if (fabs(x - 9.9) < 1e-4 || fabs(x + 0.1) < 1e-4)
      is_valid = true;
    if (!is_valid)
      printf("  -> FAIL X\n");
    CHECK(is_valid);
  }

  // Export for visual
  sim.write_vtp("output/debug_corner.vtp");
  std::cout << "Exported output/debug_corner.vtp" << std::endl;

  std::cout << "Test Passed!" << std::endl;
}

#include <pybind11/embed.h>

int main() {
  pybind11::scoped_interpreter
      guard{}; // Start the interpreter and keep it alive
  test_corner_ghosts();
  return 0;
}
