# Current Velocity Solver Algorithm

This document outlines the exact algorithm implemented in `src/cuda/solver_velocity.cu` and `src/cuda/integration.cu` as of the current build.

## Overview
The solver implements a **Iterative "One-Shot" Jacobi Velocity Solver**. 
- **Type:** Stateless Jacobi (Impulses are recalculated from zero every iteration; no warm starting).
- **Update Method:** **Unweighted Summation** (Impulses from all contacts are summed directly).

## Algorithm Pipeline

The solver runs within `Simulation::step()` in **Step 4: Phase A**.

### 1. Iteration Loop
The solver repeats the following two kernels for `velocity_iterations_` (currently configured to 20).

```cpp
for (k = 0; k < velocity_iterations; ++k) {
    launch_velocity_solve(ps);         // Kernel A
    launch_apply_velocity_deltas(ps);  // Kernel B
}
```

---

### Step A: Compute Impulses (`solve_velocity_jacobi_kernel`)
This kernel runs in parallel over **contacts**.

1.  **Inputs**:
    *   Contact Normal $n$, Distances $r_A, r_B$.
    *   Current Predicted Velocities $v_A, \omega_A, v_B, \omega_B$. (From `d_vel_pred`).
    *   Inverse Masses $w_A = 1/m_A, I_A^{-1}$.

2.  **Relative Velocity**:
    $$v_{rel} = (v_A + \omega_A \times r_A) - (v_B + \omega_B \times r_B)$$
    $$v_n = v_{rel} \cdot n$$

3.  **Effective Mass**:
    $$w_{eff} = w_A + (r_A \times n)^T I_A^{-1} (r_A \times n) + w_B + \dots$$
    $$m_{eff} = 1 / w_{eff}$$

4.  **Normal Impulse (Restitution)**:
    *   Target velocity $v_{target}$ is calculated based on restitution coefficient $e$ and *original* velocities.
    *   Delta Velocity: $\Delta v = v_{target} - v_n$.
    *   Constraint: If $v_n \ge 0$ (separating) and $v_{target} \approx 0$, no impulse.
    *   Lambda: $\lambda_n = \Delta v / w_{eff}$.
    *   **Clamping**: If $\lambda_n < 0$ (pulling), $\lambda_n = 0$.

5.  **Friction Impulse**:
    *   Tangential direction $t = (v_{rel} - v_n n)$.
    *   $\lambda_t$ computed to oppose sliding.
    *   **Coulomb Clamp**: $|\lambda_t| \le \mu \lambda_n$.

6.  **Output (Accumulation)**:
    The computed impulse $J = \lambda_n n + \lambda_t t$ is applied to the bodies via atomic addition:
    *   `atomicAdd(d_delta_vel[idA], J * w_A)`
    *   `atomicAdd(d_delta_vel[idB], -J * w_B)`
    *   `atomicAdd(d_constraint_counts[idA], 1)`

---

### Step B: Apply Updates (`apply_velocity_deltas_kernel`)
This kernel runs in parallel over **particles**.

1.  **Inputs**:
    *   Accumulated Delta Velocity buffer `d_delta_vel`.
    *   Current Predicted Velocity `d_vel_pred`.

2.  **Update Rule (Unweighted Summation)**:
    The solver adds the sum of all impulses directly to the current velocity **without averaging**.
    $$v_{pred}^{k+1} = v_{pred}^{k} + \sum_{j} \Delta v_{j}$$

3.  **Cleanup**:
    *   Buffers `d_delta_vel` and `d_constraint_counts` are cleared to 0 for the next iteration.

## Key Characteristic: Unweighted Summation
Unlike a "Weighted Jacobi" or "Projected Gauss-Seidel" solver, this implementation does **not** divide the accumulated impulse by the number of constraints ($N$).

*   **Behavior**: If a particle is squeezed between two walls, it receives full repulsion from *both* walls simultaneously in the same iteration.
*   **Result**: $v_{update} = \Delta v_{left} + \Delta v_{right} \approx 2 \times \text{Required Correction}$.
*   **Stability**: This over-correction causes oscillation ("micro-jitter") as the particle bounces back and forth between constraints within the timeframe, instead of converging effectively.
