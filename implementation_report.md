# Hybrid Physics Implementation Report

## Status Summary
**Functional for Spheres, but Incomplete for General Shapes.**

The Hybrid XPBD pipeline (Position Solve $\to$ Pre-Velocity Update $\to$ Velocity Solve) is implemented and matches the architectural flow of `plan_physics.md`. However, the specific solver algorithms and shape support deviate from the plan.

## Detailed Findings

### 1. Hybrid Pipeline Structure (Fully Implemented)
The execution sequence in `Simulation::step` correctly follows the Two-Pass Hybrid model:
1.  **Integration**: `predict_position_kernel` (Phase 1).
2.  **Position Solve**: `launch_solver` (Phase 2).
3.  **Pre-Velocity Update**: `launch_pre_velocity` (Phase 3).
    *   Correctly calculates $v = \Delta x / \Delta t$.
    *   Correctly calculates $\omega$ using the Quaternion Derivative method required by the plan (`solver.cu`: `pre_velocity_update_kernel`).
4.  **Velocity Solve**: `launch_velocity_solve` (Phase 4).

### 2. Velocity & Angular Velocity Updates (Partially Implemented)
The `solve_velocity_kernel` in `solver.cu` implements the core physics logic:
*   **Normal Restitution**: Implemented using $v_{target} = -e_n v_{old}$. Correctly accesses `d_vel_old`.
*   **Tangential Restitution & Friction**: Implemented with Coulomb clamping based on Normal Impulse.
*   **Angular Updates**: Torque is calculated and applied to `d_ang_vel`.

**Issues / Deviations:**
*   **Solver Algorithm**: The plan specified **Projected Gauss-Seidel (PGS) with Graph Coloring**. The current implementation uses **Jacobi with Atomics**.
    *   *Impact*: Convergence might be slower, but it is parallel-safe. Graph/Coloring data structures (`d_sorted_constraints`) are missing.
*   **Shape Support (Critical)**: The Velocity Solver logic is **hardcoded for Spheres**.
    *   Inertia calculation assumes a solid sphere: $I^{-1} = 2.5 \cdot m^{-1} / r^2$.
    *   Lever arms $r$ are calculated assuming spherical contact points ($r = -n \cdot radius$).
    *   The "Generalized Inverse Mass" (Sandwich Product) for OBB/Cylinders (Plan Section 4B) is **NOT implemented**.
    *   *Impact*: Cylinders or other non-spherical shapes will have incorrect rotational physics in the velocity solve.

### 3. Missing Components
*   **Graph Coloring**: No `d_batch_offsets` or coloring logic in `Simulation.cpp` or kernels.
*   **Generalized Mass Calculation**: The `compute_generalized_mass` device function described in the plan is missing.

## Recommendations for Refined Plan
1.  **Implement Generalized Inverse Mass**: Replace the hardcoded sphere inertia in `solve_velocity_kernel` with the generic helper from the plan to support Cylinders.
2.  **Switch to Graph Coloring (Optional but Recommended)**: The current atomic-add Jacobi works for spheres but PGS is generally more stable for stacking. If you want to stick with Jacobi, increase iteration counts or add damping. if you want PGS, the coloring infrastructure needs to be built.
