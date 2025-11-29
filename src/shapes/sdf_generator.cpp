// sdf_generator.cpp - CPU code to generate SDF grids (placeholder)
#include <vector>
#include <cmath>

namespace demgpu {

// Generates a sphere SDF grid in row-major order
std::vector<float> generateSphereSDF(int nx, int ny, int nz, float radius) {
    std::vector<float> grid(nx * ny * nz, 0.f);
    float cx = (nx - 1) * 0.5f;
    float cy = (ny - 1) * 0.5f;
    float cz = (nz - 1) * 0.5f;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                float dx = (i - cx);
                float dy = (j - cy);
                float dz = (k - cz);
                float d = std::sqrt(dx*dx + dy*dy + dz*dz);
                grid[k*ny*nx + j*nx + i] = d - radius;
            }
        }
    }
    return grid;
}

} // namespace demgpu
