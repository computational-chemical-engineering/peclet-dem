#include "Exporter.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>

void compute_bounds(const std::vector<float3> &pos,
                    const std::vector<float> &radii, float3 &min_out,
                    float3 &max_out) {
  if (pos.empty()) {
    min_out = {0.0f, 0.0f, 0.0f};
    max_out = {0.0f, 0.0f, 0.0f};
    return;
  }

  min_out = {std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max()};
  max_out = {-std::numeric_limits<float>::max(),
             -std::numeric_limits<float>::max(),
             -std::numeric_limits<float>::max()};

  for (size_t i = 0; i < pos.size(); ++i) {
    float r = (i < radii.size()) ? radii[i] : 0.0f;

    min_out.x = std::min(min_out.x, pos[i].x - r);
    min_out.y = std::min(min_out.y, pos[i].y - r);
    min_out.z = std::min(min_out.z, pos[i].z - r);

    max_out.x = std::max(max_out.x, pos[i].x + r);
    max_out.y = std::max(max_out.y, pos[i].y + r);
    max_out.z = std::max(max_out.z, pos[i].z + r);
  }

  // Add small margin (10%)
  float dx = max_out.x - min_out.x;
  float dy = max_out.y - min_out.y;
  float dz = max_out.z - min_out.z;
  float margin = 0.05f; // 5% on each side

  min_out.x -= dx * margin;
  min_out.y -= dy * margin;
  min_out.z -= dz * margin;
  max_out.x += dx * margin;
  max_out.y += dy * margin;
  max_out.z += dz * margin;
}

void export_lammps_dump(const std::string &filename, int step,
                        const std::vector<float3> &pos,
                        const std::vector<float3> &vel,
                        const std::vector<float4> &quats,
                        const std::vector<float> &radii,
                        const float3 *box_min_ptr, const float3 *box_max_ptr,
                        bool pbc_enabled) {
  // 1. Determine Bounds
  float3 bmin, bmax;
  if (box_min_ptr && box_max_ptr) {
    bmin = *box_min_ptr;
    bmax = *box_max_ptr;
  } else {
    compute_bounds(pos, radii, bmin, bmax);
  }

  // 2. Write Header
  std::ofstream file(filename);
  file << "ITEM: TIMESTEP\n" << step << "\n";
  file << "ITEM: NUMBER OF ATOMS\n" << pos.size() << "\n";

  std::string boundary = pbc_enabled ? "pp pp pp" : "ff ff ff";
  file << "ITEM: BOX BOUNDS " << boundary << "\n";
  file << bmin.x << " " << bmax.x << "\n";
  file << bmin.y << " " << bmax.y << "\n";
  file << bmin.z << " " << bmax.z << "\n";

  // 3. Write Data
  // Ovito Format: id type x y z radius vx vy vz qw qx qy qz
  file << "ITEM: ATOMS id type x y z radius vx vy vz qw qx qy qz\n";

  file << std::fixed << std::setprecision(6);

  for (size_t i = 0; i < pos.size(); ++i) {
    float r = (i < radii.size()) ? radii[i] : 1.0f;
    float3 v = (i < vel.size()) ? vel[i] : float3{0, 0, 0};
    float4 q = (i < quats.size()) ? quats[i] : float4{1, 0, 0, 0}; // w, x, y, z

    file << (i + 1) << " 1 " // ID, Type (fixed at 1)
         << pos[i].x << " " << pos[i].y << " " << pos[i].z << " " << r << " "
         << v.x << " " << v.y << " " << v.z << " " << q.w << " " << q.x << " "
         << q.y << " " << q.z << "\n";
  }
}
