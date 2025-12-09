#pragma once
#include "simulation.h"
#include <string>
#include <vector>

namespace dem {

class OutputGenerator {
public:
  OutputGenerator(Simulation *sim);
  ~OutputGenerator();

  void generateAndSaveVTI(const std::string &filename, int3 resolution,
                          float3 bounds_min, float3 bounds_max);

private:
  Simulation *sim_;
  float *d_grid_ping_;
  float *d_grid_pong_;
  int *d_state_;
  size_t allocated_voxels_;

  void allocate(int3 dims);
  void free();
};

} // namespace dem
