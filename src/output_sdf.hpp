/// @file
/// @brief dem — portable (Kokkos) SDF-grid reconstruction: the get_sdf_grid pipeline.
///
/// Kokkos port of output_sdf.cu (init_grid + splat_particles + Jacobi-Eikonal solve). For each grid
/// voxel: (1) init to +inf (100), (2) SPLAT — every particle writes the EXACT transformed analytic
/// SDF over its AABB band via atomic-min (and marks those cells fixed), (3) EIKONAL — a Jacobi
/// fast-iterative-method propagates the (positive) distance into the far field, periodic-wrapped. A
/// faithful copy of the CUDA kernels (same band, same bounding radius, same quadratic solve, same
/// max(res)*4 iteration cap), over the Kokkos particle SoA + the ported analytic SDFs.
/// Deterministic (atomic-min + Jacobi) -> matches CUDA for the same particle configuration. Runs on
/// any Kokkos backend.
#ifndef DEM_OUTPUT_SDF_HPP
#define DEM_OUTPUT_SDF_HPP

#include <Kokkos_Core.hpp>
#include <vector>

#include "dem_portable.hpp"  // F3/F4, invRotateVector, sdfSphere/sdfHollowCylinder, ShapeKind
#include "narrowphase.hpp"   // ShapeDesc, CpExec/CpMem, view aliases

namespace peclet::dem {

// Generate the SDF grid (flat, x-fastest: i = x + y*rx + z*rx*ry; negative inside solid) over
// [min,max].
inline std::vector<float> generateSdfKokkos(int rx, int ry, int rz, F3 dmin, F3 dmax, int numReal,
                                            PosView pos, QuatView quat, ScalarF scale,
                                            ScalarI shapeId,
                                            Kokkos::View<const ShapeDesc*, CpMem> shapes, bool px,
                                            bool py, bool pz) {
  CpExec space;
  const long total = (long)rx * ry * rz;
  const F3 origin = dmin;
  const F3 vox{(dmax.x - dmin.x) / rx, (dmax.y - dmin.y) / ry, (dmax.z - dmin.z) / rz};

  Kokkos::View<float*, CpMem> grid("sdf_grid", total);
  Kokkos::View<int*, CpMem> state("sdf_state", total);
  Kokkos::deep_copy(grid, 100.0f);
  Kokkos::deep_copy(state, 0);

  // SPLAT: each particle writes the exact analytic signed distance over its AABB band (atomic-min).
  Kokkos::parallel_for(
      "peclet::dem::sdf::splat", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        const F3 pw = loadF3(pos, i);
        const F4 q = loadF4(quat, i);
        const float sc = scale(i);
        const ShapeDesc shp = shapes(shapeId(i));
        float rbound = 1.0f;
        if (shp.type == HOLLOW_CYLINDER) {
          const float r = shp.params.x, h = shp.params.y;
          rbound = Kokkos::sqrt(r * r + (h * 0.5f) * (h * 0.5f));
        } else if (shp.type == SPHERE)
          rbound = shp.params.x;
        rbound *= sc;
        rbound *= 1.2f;

        int minx = (int)Kokkos::floor((pw.x - rbound - origin.x) / vox.x),
            maxx = (int)Kokkos::ceil((pw.x + rbound - origin.x) / vox.x);
        int miny = (int)Kokkos::floor((pw.y - rbound - origin.y) / vox.y),
            maxy = (int)Kokkos::ceil((pw.y + rbound - origin.y) / vox.y);
        int minz = (int)Kokkos::floor((pw.z - rbound - origin.z) / vox.z),
            maxz = (int)Kokkos::ceil((pw.z + rbound - origin.z) / vox.z);
        if (!px) {
          minx = minx < 0 ? 0 : minx;
          maxx = maxx > rx - 1 ? rx - 1 : maxx;
        }
        if (!py) {
          miny = miny < 0 ? 0 : miny;
          maxy = maxy > ry - 1 ? ry - 1 : maxy;
        }
        if (!pz) {
          minz = minz < 0 ? 0 : minz;
          maxz = maxz > rz - 1 ? rz - 1 : maxz;
        }

        for (int z = minz; z <= maxz; ++z)
          for (int y = miny; y <= maxy; ++y)
            for (int x = minx; x <= maxx; ++x) {
              const F3 vp{origin.x + (x + 0.5f) * vox.x, origin.y + (y + 0.5f) * vox.y,
                          origin.z + (z + 0.5f) * vox.z};
              const F3 prel = sub3(vp, pw);
              const F3 plocal = scale3(invRotateVector(q, prel), 1.0f / sc);
              float distc = 100.0f;
              if (shp.type == HOLLOW_CYLINDER)
                distc = sdfHollowCylinder(plocal, shp.params);
              else if (shp.type == SPHERE)
                distc = sdfSphere(plocal, shp.params);
              const float dist = distc * sc;
              const int wx = (x % rx + rx) % rx, wy = (y % ry + ry) % ry, wz = (z % rz + rz) % rz;
              if (!px && (x < 0 || x >= rx))
                continue;
              if (!py && (y < 0 || y >= ry))
                continue;
              if (!pz && (z < 0 || z >= rz))
                continue;
              const long vi = (long)wz * rx * ry + (long)wy * rx + wx;
              Kokkos::atomic_min(&grid(vi), dist);
              if (dist < rbound)
                state(vi) = 1;
            }
      });
  space.fence();

  // EIKONAL: Jacobi fast-iterative-method (ping-pong), max(res)*4 iterations, periodic wrap.
  Kokkos::View<float*, CpMem> grid2("sdf_grid2", total);
  const int iters = Kokkos::max(Kokkos::max(rx, ry), rz) * 4;
  const float h = vox.x;
  Kokkos::View<float*, CpMem> in = grid, out = grid2;
  for (int it = 0; it < iters; ++it) {
    Kokkos::View<float*, CpMem> din = in, dout = out;
    Kokkos::View<int*, CpMem> st = state;
    Kokkos::parallel_for(
        "peclet::dem::sdf::eikonal", Kokkos::RangePolicy<CpExec>(space, 0, total),
        KOKKOS_LAMBDA(long idx) {
          if (st(idx) == 1) {
            dout(idx) = din(idx);
            return;
          }
          const long sy = rx, sz = (long)rx * ry;
          const int z = (int)(idx / sz);
          const long rem = idx % sz;
          const int y = (int)(rem / sy), x = (int)(rem % sy);
          float xm = 100.0f, xp = 100.0f, ym = 100.0f, yp = 100.0f, zm = 100.0f, zp = 100.0f;
          if (x > 0)
            xm = din(idx - 1);
          else if (px)
            xm = din(idx - 1 + rx);
          if (x < rx - 1)
            xp = din(idx + 1);
          else if (px)
            xp = din(idx + 1 - rx);
          if (y > 0)
            ym = din(idx - sy);
          else if (py)
            ym = din((long)z * sz + (long)(ry - 1) * sy + x);
          if (y < ry - 1)
            yp = din(idx + sy);
          else if (py)
            yp = din((long)z * sz + x);
          if (z > 0)
            zm = din(idx - sz);
          else if (pz)
            zm = din((long)(rz - 1) * sz + (long)y * sy + x);
          if (z < rz - 1)
            zp = din(idx + sz);
          else if (pz)
            zp = din((long)y * sy + x);
          float a = Kokkos::fmin(xm, xp), b = Kokkos::fmin(ym, yp), c = Kokkos::fmin(zm, zp);
          if (a > b) {
            float t = a;
            a = b;
            b = t;
          }
          if (b > c) {
            float t = b;
            b = c;
            c = t;
          }
          if (a > b) {
            float t = a;
            a = b;
            b = t;
          }
          float u = 100.0f, u1 = a + h;
          if (u1 <= b)
            u = u1;
          else {
            const float d2 = 2.0f * h * h - (a - b) * (a - b);
            float u2 = (d2 < 0.0f) ? 100.0f : (a + b + Kokkos::sqrt(d2)) * 0.5f;
            if (u2 <= c)
              u = u2;
            else {
              const float s = a + b + c, qd = s * s - 3.0f * (a * a + b * b + c * c - h * h);
              u = (qd < 0.0f) ? 100.0f : (s + Kokkos::sqrt(qd)) / 3.0f;
            }
          }
          dout(idx) = u;
        });
    space.fence();
    auto tmp = in;
    in = out;
    out = tmp;
  }

  std::vector<float> h_grid(total);
  auto hv = Kokkos::create_mirror_view(in);
  Kokkos::deep_copy(hv, in);
  for (long i = 0; i < total; ++i)
    h_grid[i] = hv(i);
  return h_grid;
}

}  // namespace peclet::dem

#endif  // DEM_OUTPUT_SDF_HPP
