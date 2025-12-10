# Verification Failure Report

## Status
**Verification Failed**: The simulation crashed with a `CUDA Error: an illegal memory access` during the initial settling phase.

## failure Analysis
The crash occurs in `solve_ground_kernel` or `apply_deltas_kernel`. The most probable cause is **Ghost Buffer Overflow**.

1.  **Configuration**:
    *   **Domain**: $10 \times 10 \times 10$ (-5 to 5).
    *   **Boundary Margin**: Hardcoded to `2.0 * global_scale` in `Simulation::step`.
    *   **Ghost Capacity**: Allocated as `2 * num_particles` (Total 2000 for 1000 particles).

2.  **The Issue**:
    *   A margin of 2.0 units on a 10-unit box means any particle within 2.0 units of the boundary creates a ghost.
    *   The "safe" inner volume is $6 \times 6 \times 6 = 216$.
    *   The "ghost-generating" shell volume is $1000 - 216 = 784$.
    *   Approximately **78%** of uniformly distributed particles are in the ghost region.
    *   Many particles (corners/edges) generate **multiple ghosts** (up to 7).
    *   **Result**: The number of ghosts exceeds the 1000 slots available in the buffer. While the kernel has an overflow check, `total_particles` might be reporting the clamped maximum (2000), and any subsequent access logic or race conditions in `generate_ghosts` could be fragile, or more likely, the logic in `solve_ground` might be accessing `d_pos_star` or other arrays that were only allocated for 2000, but if `top_ghost` incremented beyond (before decrementing?), or if the sheer number of ghosts implies we need more robust handling.
    *   *Correction*: The `atomicSub` prevents writing, so memory corruption *during* generation is unlikely. However, if valid particles are dropped, the simulation state is inconsistent. The "Illegal Access" might be due to `total_particles` being used in a kernel launch that maps threads to indices, and if any buffer (like `d_contacts` or others depending on `num_particles`) wasn't sized for the *actual* potential ghosts, or if `step` uses `total_particles` for something else.
    *   Regardless, **2x capacity is insufficient** for this simulation configuration.

## Proposed Remediation Plan
1.  **Increase Ghost Capacity**: Increase allocation to `8 * num_particles` to handle worst-case 3D periodic boundaries (all particles could theoretically be ghosts if box is small relative to margin).
2.  **Dynamic Margin**: Calculate margin based on `max_velocity * dt` rather than a fixed `2.0`. This will significantly reduce the number of ghosts for slow-moving/settling particles.
3.  **Fix Destructor**: Permanently apply the fix to `src/simulation.cpp` (re-closing the destructor scope) which was required to build the project.

## Next Steps
Once these stability fixes are applied, we can resume the verification for densities 0.45 and 0.64.
