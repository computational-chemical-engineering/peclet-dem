#pragma once
#include "ParticleSystem.cuh" // For ParticleSystemData
#include <cuda_runtime.h>

namespace dem {

struct PeriodicConfig {
  float3 min;       // Domain Min
  float3 max;       // Domain Max
  float3 size;      // max - min
  float skin_width; // max_particle_radius + safety_margin
};

// Functor for cuBQL Traversal
struct BoundarySkinQuery {
  float3 inner_min; // domain.min + skin
  float3 inner_max; // domain.max - skin

  // cuBQL Query Interface:
  // Returns true if we should prune (STOP), false if we should visit (CONTINUE)
  // Logic: If Node AABB is fully inside the "Safe Zone", it cannot touch the
  // boundary.
  __device__ __forceinline__ bool prune(const float3 &node_min,
                                        const float3 &node_max) const {
    return (node_min.x > inner_min.x && node_max.x < inner_max.x &&
            node_min.y > inner_min.y && node_max.y < inner_max.y &&
            node_min.z > inner_min.z && node_max.z < inner_max.z);
  }

  // Leaf Check
  __device__ __forceinline__ bool is_candidate(float3 p, float r) const {
    // Check if sphere touches any boundary "skin" region
    // It is a candidate if it is OUTSIDE the "Inner Safe Box"
    return (p.x - r < inner_min.x || p.x + r > inner_max.x ||
            p.y - r < inner_min.y || p.y + r > inner_max.y ||
            p.z - r < inner_min.z || p.z + r > inner_max.z);
  }
};

// Host Launcher Declarations
void launch_find_ghost_candidates(ParticleSystemData ps, PeriodicConfig config,
                                  int *d_candidates, int *d_candidate_count);

void launch_generate_ghosts_bitmask(int *d_candidates, int *d_candidate_count,
                                    ParticleSystemData ps,
                                    PeriodicConfig config);

} // namespace dem
