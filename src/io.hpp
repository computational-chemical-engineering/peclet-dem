/// @file
/// @brief dem — portable (CUDA-free) I/O helpers for the Kokkos dem module: a LAMMPS dump writer and
/// a scalar-SDF VTI writer. Operate on flat std::vector<float> host arrays (no CUDA vector_types), so they
/// compile on every Kokkos backend. Faithful to the CUDA io/Exporter.cpp LAMMPS format and Simulation::
/// export_sdf VTI convention.
#ifndef DEM_IO_HPP
#define DEM_IO_HPP

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

namespace peclet::dem {

// LAMMPS "dump custom" (Ovito): id type x y z radius vx vy vz qw qx qy qz.
// pos/vel are flat [n*3]; quat is flat [n*4] in (x,y,z,w) order; radii is [n]. Bounds: explicit box if
// provided, else computed from the particle AABBs + a 5% margin (matches io/Exporter.cpp::compute_bounds).
inline void writeLammpsDump(const std::string& filename, int step, const std::vector<float>& pos,
                            const std::vector<float>& vel, const std::vector<float>& quat,
                            const std::vector<float>& radii, const float* boxMin, const float* boxMax,
                            bool pbcEnabled) {
  const std::size_t n = pos.size() / 3;
  float bmin[3], bmax[3];
  if (boxMin && boxMax) {
    for (int d = 0; d < 3; ++d) { bmin[d] = boxMin[d]; bmax[d] = boxMax[d]; }
  } else if (n == 0) {
    for (int d = 0; d < 3; ++d) bmin[d] = bmax[d] = 0.0f;
  } else {
    for (int d = 0; d < 3; ++d) { bmin[d] = std::numeric_limits<float>::max(); bmax[d] = -std::numeric_limits<float>::max(); }
    for (std::size_t i = 0; i < n; ++i) {
      const float r = (i < radii.size()) ? radii[i] : 0.0f;
      for (int d = 0; d < 3; ++d) {
        bmin[d] = std::min(bmin[d], pos[3 * i + d] - r);
        bmax[d] = std::max(bmax[d], pos[3 * i + d] + r);
      }
    }
    for (int d = 0; d < 3; ++d) { const float m = 0.05f * (bmax[d] - bmin[d]); bmin[d] -= m; bmax[d] += m; }
  }

  std::ofstream file(filename);
  if (!file) throw std::runtime_error("Could not open file for writing: " + filename);
  file << "ITEM: TIMESTEP\n" << step << "\n";
  file << "ITEM: NUMBER OF ATOMS\n" << n << "\n";
  file << "ITEM: BOX BOUNDS " << (pbcEnabled ? "pp pp pp" : "ff ff ff") << "\n";
  for (int d = 0; d < 3; ++d) file << bmin[d] << " " << bmax[d] << "\n";
  file << "ITEM: ATOMS id type x y z radius vx vy vz qw qx qy qz\n";
  file << std::fixed << std::setprecision(6);
  for (std::size_t i = 0; i < n; ++i) {
    const float r = (i < radii.size()) ? radii[i] : 1.0f;
    const float vx = (3 * i + 2 < vel.size()) ? vel[3 * i] : 0.0f;
    const float vy = (3 * i + 2 < vel.size()) ? vel[3 * i + 1] : 0.0f;
    const float vz = (3 * i + 2 < vel.size()) ? vel[3 * i + 2] : 0.0f;
    float qx = 0, qy = 0, qz = 0, qw = 1;  // (x,y,z,w) input
    if (4 * i + 3 < quat.size()) { qx = quat[4 * i]; qy = quat[4 * i + 1]; qz = quat[4 * i + 2]; qw = quat[4 * i + 3]; }
    file << (i + 1) << " 1 " << pos[3 * i] << " " << pos[3 * i + 1] << " " << pos[3 * i + 2] << " " << r
         << " " << vx << " " << vy << " " << vz << " " << qw << " " << qx << " " << qy << " " << qz << "\n";
  }
}

// Scalar-SDF ImageData VTI (ParaView), x-fastest grid [rx*ry*rz], origin = domain min, spacing = extent/res.
inline void writeSdfVti(const std::string& filename, const std::vector<float>& grid, int rx, int ry, int rz,
                        const float* minB, const float* maxB) {
  const double sx = rx > 1 ? (double(maxB[0]) - minB[0]) / (rx - 1) : 1.0;
  const double sy = ry > 1 ? (double(maxB[1]) - minB[1]) / (ry - 1) : 1.0;
  const double sz = rz > 1 ? (double(maxB[2]) - minB[2]) / (rz - 1) : 1.0;
  std::ofstream out(filename);
  if (!out) throw std::runtime_error("Could not open file for writing: " + filename);
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  out << "  <ImageData WholeExtent=\"0 " << (rx - 1) << " 0 " << (ry - 1) << " 0 " << (rz - 1)
      << "\" Origin=\"" << minB[0] << " " << minB[1] << " " << minB[2]
      << "\" Spacing=\"" << sx << " " << sy << " " << sz << "\">\n";
  out << "    <Piece Extent=\"0 " << (rx - 1) << " 0 " << (ry - 1) << " 0 " << (rz - 1) << "\">\n";
  out << "      <PointData Scalars=\"sdf\">\n";
  out << "        <DataArray type=\"Float32\" Name=\"sdf\" format=\"ascii\">\n";
  for (float v : grid) out << v << " ";
  out << "\n        </DataArray>\n";
  out << "      </PointData>\n";
  out << "    </Piece>\n";
  out << "  </ImageData>\n";
  out << "</VTKFile>\n";
  std::printf("Exported SDF VTI: %s\n", filename.c_str());
}

}  // namespace peclet::dem

#endif  // DEM_IO_HPP
