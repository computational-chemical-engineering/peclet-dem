/// @file
/// @brief dem — portable (Kokkos) periodic ghost generation (periodicity.cu / integration.cu).
///
/// Per real particle near a periodic face, emit shifted ghost copies into atomically-allocated slots
/// past num_real, copying the full per-particle state and mapping each ghost back to its real owner
/// (real_indices) for momentum conservation. Faithful port of generate_ghosts_bitmask_kernel; ghost
/// positions use the predicted state to align with the solver. Decoupled from any BVH (the CUDA
/// candidate pre-pass was just an optimization; the boundary test is inlined here).
#ifndef DEM_PERIODICITY_HPP
#define DEM_PERIODICITY_HPP

#include <Kokkos_Core.hpp>

#include "dem_portable.hpp"
#include "integration.hpp"  // Domain, V3/V4/Vf/Vi, detail::st3/st4

namespace dem {

/// Generate periodic ghosts for particles [0,numReal). topGhost must be pre-seeded to numReal (the
/// first free slot); on return it holds the total particle count. Slots beyond `capacity` are
/// dropped. `skin` is the ghost band width at each periodic face.
inline void generateGhostsKokkos(int numReal, int capacity, Domain dom, float skin, V3 pos, Vf invMass,
                                 V3 posPred, V3 vel, V3 velPred, V4 quat, V4 quatPred, V3 angVel,
                                 V3 angVelPred, Vf scale, Vi shapeId, Vi realIndices,
                                 Kokkos::View<int, CpMem> topGhost) {
  using detail::st3;
  using detail::st4;
  CpExec space;
  Kokkos::parallel_for(
      "dem::generate_ghosts", Kokkos::RangePolicy<CpExec>(space, 0, numReal), KOKKOS_LAMBDA(int i) {
        const F3 p = ldF3(pos, i);
        const int sx = dom.periodic_x ? ((p.x < dom.min.x + skin) ? 1 : ((p.x > dom.max.x - skin) ? -1 : 0)) : 0;
        const int sy = dom.periodic_y ? ((p.y < dom.min.y + skin) ? 1 : ((p.y > dom.max.y - skin) ? -1 : 0)) : 0;
        const int sz = dom.periodic_z ? ((p.z < dom.min.z + skin) ? 1 : ((p.z > dom.max.z - skin) ? -1 : 0)) : 0;
        const int ax[2] = {0, sx}, ay[2] = {0, sy}, az[2] = {0, sz};
        const int nx = (sx == 0) ? 1 : 2, ny = (sy == 0) ? 1 : 2, nz = (sz == 0) ? 1 : 2;

        const F3 pPred = ldF3(posPred, i);
        for (int a = 0; a < nx; ++a)
          for (int b = 0; b < ny; ++b)
            for (int c = 0; c < nz; ++c) {
              const int ix = ax[a], iy = ay[b], iz = az[c];
              if (ix == 0 && iy == 0 && iz == 0) continue;
              const int slot = Kokkos::atomic_fetch_add(&topGhost(), 1);
              if (slot >= capacity) {
                Kokkos::atomic_add(&topGhost(), -1);
                continue;
              }
              const F3 shift{ix * dom.size.x, iy * dom.size.y, iz * dom.size.z};
              st3(pos, slot, add3(p, shift));
              st3(posPred, slot, add3(pPred, shift));
              invMass(slot) = invMass(i);
              st3(vel, slot, ldF3(vel, i));
              st3(velPred, slot, ldF3(velPred, i));
              st4(quat, slot, ldF4(quat, i));
              st4(quatPred, slot, ldF4(quatPred, i));
              st3(angVel, slot, ldF3(angVel, i));
              st3(angVelPred, slot, ldF3(angVelPred, i));
              scale(slot) = scale(i);
              shapeId(slot) = shapeId(i);
              realIndices(slot) = i;
            }
      });
  space.fence();
}

}  // namespace dem

#endif  // DEM_PERIODICITY_HPP
