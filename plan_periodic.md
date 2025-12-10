Here is the detailed implementation plan to fast-track **Periodic Ghosts** into **Phase 1**. This allows you to verify the boundary logic using simple Spheres/AABBs before dealing with complex SDFs.

You can feed this directly to your coding agent.

-----

# Technical Implementation Plan: Periodic Ghosts (Phase 1)

**Goal:** Implement efficient Periodic Boundary Conditions (PBC) using `cuBQL` to detect and generate ghost particles.
**Context:** This module runs **before** the main collision detection. It populates the `ParticleSystem` with ghost copies so the solver sees "neighbors" across the boundary.

-----

## 1\. Data Structures (`src/cuda/periodicity.cuh`)

We need a configuration struct and a temporary buffer for candidates.

```cpp
struct PeriodicConfig {
    float3 min;        // Domain Min
    float3 max;        // Domain Max
    float3 size;       // max - min
    float skin_width;  // max_particle_radius + safety_margin
};

// Functor for cuBQL Traversal
struct BoundarySkinQuery {
    float3 inner_min; // domain.min + skin
    float3 inner_max; // domain.max - skin

    // cuBQL Query Interface: 
    // Returns true if we should prune (STOP), false if we should visit (CONTINUE)
    // Note: Verify cuBQL API (some versions use 'overlap' vs 'prune' semantics).
    // Logic: If Node AABB is fully inside the "Safe Zone", it cannot touch the boundary.
    __device__ __forceinline__ bool prune(const float3& node_min, const float3& node_max) const {
        return (node_min.x > inner_min.x && node_max.x < inner_max.x &&
                node_min.y > inner_min.y && node_max.y < inner_max.y &&
                node_min.z > inner_min.z && node_max.z < inner_max.z);
    }
    
    // Leaf Check
    __device__ __forceinline__ bool is_candidate(float3 p, float r) const {
        // Check if sphere touches any boundary face
        if (p.x - r < inner_min.x - r) return true; // Near Left (Wait, inner_min includes skin?)
        // Correct Logic: Check against actual Domain Bounds + Skin
        // Actually, simpliest check: Is it OUTSIDE the "Inner Safe Box"?
        return (p.x - r < inner_min.x || p.x + r > inner_max.x ||
                p.y - r < inner_min.y || p.y + r > inner_max.y ||
                p.z - r < inner_min.z || p.z + r > inner_max.z);
    }
};
```

-----

## 2\. Kernels (`src/cuda/periodicity.cu`)

### Task 2.1: Candidate Finder Kernel

Traverse the *local* cuBQL tree to find particles near the edge.

  * **Kernel:** `find_ghost_candidates`
  * **Input:** `cuBQL::Model` (Local), `PeriodicConfig`.
  * **Output:** `int* d_candidates`, `int* d_candidate_count`.
  * **Logic:**
      * Use `cuBQL::traverse<BoundarySkinQuery>`.
      * In the leaf callback:
          * Load particle `pos` and `radius`.
          * If `query.is_candidate(pos, radius)`:
              * `idx = atomicAdd(d_candidate_count, 1)`.
              * `d_candidates[idx] = particle_id`.

### Task 2.2: Ghost Generator Kernel (The Bitmask Logic)

Generate up to 7 ghosts per candidate in a single pass.

  * **Kernel:** `generate_ghosts_bitmask`
  * **Input:** `d_candidates`, `d_candidate_count`, `ParticleSystem`.
  * **Logic:**
    1.  Thread $i$ processes `candidate[i]`.
    2.  Load `pos`.
    3.  **Compute Mask:**
          * `mask_x = (pos.x < min.x + skin) ? -1 : (pos.x > max.x - skin ? 1 : 0)`
          * `mask_y = ...`
          * `mask_z = ...`
    4.  **Iterate Permutations:**
          * Nested loops: `ix` from `(mask_x == -1 ? -1 : 0)` to `(mask_x == 1 ? 1 : 0)`.
          * Same for `iy`, `iz`.
          * **Skip Center:** If `ix==0 && iy==0 && iz==0` continue.
    5.  **Create Ghost:**
          * `ghost_idx = atomicAdd(p.d_num_particles, 1)`.
          * **Safety Check:** Ensure `ghost_idx < max_capacity`.
          * `shift = make_float3(ix, iy, iz) * domain_size`.
          * `p.d_pos[ghost_idx] = p.d_pos[real_id] + shift`.
          * Copy `vel`, `quat`, `ang_vel`, `shape_id`.
          * **Set Flag:** `p.d_vel[ghost_idx].w = FLAG_GHOST`.

Add a Guard Clause inside the kernel to prevent memory corruption if the heuristic fails (e.g., extreme explosion).

```cpp
// Inside generate_ghosts_bitmask kernel
// ...
int ghost_idx = atomicAdd(p.d_num_particles, 1);

// SAFETY CHECK
if (ghost_idx >= p.max_capacity) {
    // Decrement back to avoid corrupting the count for future frames (optional)
    atomicSub(p.d_num_particles, 1);
    // Mark Error Flag (p.d_error_code = 1)
    return; 
}
// ... proceed to write ...
```
-----

## 3\. Host Integration (`src/cpp/Simulation.cpp`)

Implement the `updateGhosts()` method in the main loop.

### Task 3.1: The Pipeline

1.  **Reset:** `num_particles = num_real_particles`. (Reset count to discard old ghosts).
2.  **Build Local BVH:** Call `cubql::build` on range `[0, num_real]`.
3.  **Query:**
      * Reset `d_candidate_count` to 0.
      * Launch `find_ghost_candidates`.
4.  **Generate:**
      * Launch `generate_ghosts_bitmask`.
      * Retrieve new `num_particles` (Real + Ghosts).
5.  **Build Global BVH:** Call `cubql::build` on range `[0, num_total]`.
      * *Note:* This global BVH is what gets passed to the Collision Detection module.

-----

## 4\. Verification Plan (Corner Test)

Create a specific unit test to validate the "1 to 7" logic.

### Task 4.1: `tests/test_periodicity_corner.cpp`

  * **Setup:**
      * Domain: `[0,0,0]` to `[10,10,10]`.
      * Particle Radius: `1.0`.
      * Skin: `1.1`.
      * **Particle 0:** Place at `(9.9, 9.9, 9.9)` (Top-Right-Back Corner).
  * **Action:** Run `updateGhosts()`.
  * **Assertions:**
    1.  `num_particles` should be **8**.
    2.  Check positions of ghosts. Expect:
          * X-shift: `(-0.1, 9.9, 9.9)`
          * Y-shift: `(9.9, -0.1, 9.9)`
          * Z-shift: `(9.9, 9.9, -0.1)`
          * XY-shift: `(-0.1, -0.1, 9.9)`
          * ... and so on for XZ, YZ, XYZ.
    3.  Check `d_vel[i].w` flag is set to GHOST for indices 1–7.

### Task 4.2: Visualization

  * Export the 8 particles to `.vtp`.
  * Verify visually that they form a $2 \times 2 \times 2$ cube cluster (the "Periodic Corner Cluster").

-----

## Implementation Notes for Agent

  * **Capacity:** 

Dynamic Memory Allocation (src/cpp/Simulation.cpp)
Implement a helper to calculate optimal buffer size.

```cpp
// Helper: Calculate robust capacity
size_t calculate_capacity(int n_real, float3 box_size, float skin_width) {
    // 1. Calculate Safe Inner Dimensions
    float3 inner = make_float3(
        std::max(0.0f, box_size.x - 2.0f * skin_width),
        std::max(0.0f, box_size.y - 2.0f * skin_width),
        std::max(0.0f, box_size.z - 2.0f * skin_width)
    );

    // 2. Volume Ratios
    double vol_total = box_size.x * box_size.y * box_size.z;
    double vol_inner = inner.x * inner.y * inner.z;
    double vol_shell = vol_total - vol_inner;

    // 3. Fraction of particles expected to be ghosts
    double ghost_fraction = vol_shell / vol_total;

    // 4. Heuristic:
    // If box is tiny (ghost_fraction > 0.5), assume worst case (8x).
    // Else, use volume fraction with safety margin (e.g. 2.0x for clustering).
    double estimated_ghosts = 0;
    
    if (ghost_fraction > 0.5) {
        // Small box regime: Worst case corner overlap
        estimated_ghosts = n_real * 7.0; 
    } else {
        // Large box regime: Surface area scaling
        // Safety Factor 2.0 handles density fluctuations near borders
        estimated_ghosts = n_real * ghost_fraction * 2.0;
    }

    // 5. Final Capacity
    // Add base N_real + Estimated Ghosts + minimal fixed buffer (e.g. 1024)
    return n_real + (size_t)estimated_ghosts + 1024;
}
```

  * **cuBQL API:** Use the `cuBQL::gpuBuilder` for dynamic building every frame. It is fast enough ($< 1ms$ for 100k particles).
  * **Atomic Safety:** In `generate_ghosts`, always check `if (ghost_idx >= max_capacity)` before writing, to prevent buffer overflows during explosion instabilities.