// packing-gpu — portable (ArborX) broad-phase, the Kokkos-native replacement for the CUDA-only
// cuBQL broad-phase in broadphase.cu.
//
// Same semantics as find_collisions(): build axis-aligned bounding boxes (half-width radius+margin)
// over ALL particles, then for each REAL particle query the BVH for overlapping boxes and emit the
// candidate pairs (i,j) with i<j into a preallocated buffer guarded by an atomic counter. An
// AABB-overlap broad-phase, identical in criterion to cuBQL's fixedBoxQuery, so it yields the same
// candidate set. Decoupled from ParticleSystemData (plain Kokkos Views) so it builds and is tested
// standalone before being wired into the solver; runs on whatever backend Kokkos was built for.
#ifndef DEM_BROADPHASE_ARBORX_HPP
#define DEM_BROADPHASE_ARBORX_HPP

#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

#include <utility>

namespace dem {

using BpExec = Kokkos::DefaultExecutionSpace;
using BpMem = BpExec::memory_space;

/// Emit candidate collision pairs (i<j) for real particles into outPairs/outCount.
///   pos[numParticles][3]   positions of all particles (real first, then ghosts)
///   rad[numParticles]      effective radius per particle (scale * global_scale)
///   numReal                only particles [0,numReal) issue queries (own real particles)
///   margin                 safety margin added to every box half-width (cuBQL uses 0.1*global_scale)
///   outPairs[maxPairs][2]  preallocated output; outCount is the (clamped) number found
/// Returns the number of pairs found (may exceed maxPairs if the buffer is too small).
// View params are templated so callers can pass managed Views (tests) OR unmanaged Views wrapping
// raw device buffers (the CUDA demgpu module's float4*/int2* arrays). Element types/layout must be
// the usual (positions [N][3], radii [N], pairs [N][2], count scalar).
template <class PosV, class RadV, class PairsV, class CountV>
inline int findCollisionsArborX(PosV pos, RadV rad, int numParticles, int numReal, float margin,
                                PairsV outPairs, CountV outCount) {
  BpExec space;
  using Box = ArborX::Box<3>;

  // AABBs over all particles (these are the BVH primitives).
  Kokkos::View<Box*, BpMem> boxes(Kokkos::view_alloc(space, "dem::bp::boxes", Kokkos::WithoutInitializing),
                                  numParticles);
  Kokkos::parallel_for(
      "dem::bp::aabb", Kokkos::RangePolicy<BpExec>(space, 0, numParticles),
      KOKKOS_LAMBDA(int i) {
        const float b = rad(i) + margin;
        boxes(i) = Box{{pos(i, 0) - b, pos(i, 1) - b, pos(i, 2) - b},
                       {pos(i, 0) + b, pos(i, 1) + b, pos(i, 2) + b}};
      });

  ArborX::BoundingVolumeHierarchy const tree(space, ArborX::Experimental::attach_indices(boxes));

  // One intersection query per real particle (box of the same half-width).
  using Predicate = decltype(ArborX::intersects(std::declval<Box>()));
  Kokkos::View<Predicate*, BpMem> preds(
      Kokkos::view_alloc(space, "dem::bp::preds", Kokkos::WithoutInitializing), numReal);
  Kokkos::parallel_for(
      "dem::bp::preds", Kokkos::RangePolicy<BpExec>(space, 0, numReal), KOKKOS_LAMBDA(int i) {
        const float b = rad(i) + margin;
        preds(i) = ArborX::intersects(Box{{pos(i, 0) - b, pos(i, 1) - b, pos(i, 2) - b},
                                          {pos(i, 0) + b, pos(i, 1) + b, pos(i, 2) + b}});
      });

  Kokkos::View<typename decltype(tree)::value_type*, BpMem> values("dem::bp::values", 0);
  Kokkos::View<int*, BpMem> offsets("dem::bp::offsets", 0);
  tree.query(space, preds, values, offsets);

  // Compact the (i, j) results with i<j into the output buffer (matches cuBQL's i<j filter).
  Kokkos::deep_copy(space, outCount, 0);
  const int maxPairs = static_cast<int>(outPairs.extent(0));
  Kokkos::View<int* [2], BpMem> pairs = outPairs;
  Kokkos::View<int, BpMem> cnt = outCount;
  Kokkos::parallel_for(
      "dem::bp::emit", Kokkos::RangePolicy<BpExec>(space, 0, numReal), KOKKOS_LAMBDA(int i) {
        for (int k = offsets(i); k < offsets(i + 1); ++k) {
          const int j = values(k).index;
          if (i < j) {
            const int slot = Kokkos::atomic_fetch_add(&cnt(), 1);
            if (slot < maxPairs) {
              pairs(slot, 0) = i;
              pairs(slot, 1) = j;
            }
          }
        }
      });
  space.fence();

  int h_count = 0;
  Kokkos::deep_copy(h_count, outCount);
  return h_count;
}

}  // namespace dem

#endif  // DEM_BROADPHASE_ARBORX_HPP
