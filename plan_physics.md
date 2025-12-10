Here is the comprehensive **Technical Implementation Plan** for Phase 1 and 2. It incorporates the **Post-Stabilization** (Velocity-based) architecture, **Projected Jacobi** solvers, and **Speculative Contact** logic we agreed upon.

You can copy and paste this entire block directly to your coding agent.

-----

# Technical Implementation Plan: HollowPack-HPC

**Project:** High-Performance GPU Granular Physics Engine
**Architecture:** C++17 / CUDA / Custom MPI / Python
**Physics Model:** Velocity-Based Post-Stabilization (Hybrid XPBD) with Projected Jacobi Solvers.

-----

## Global Architectural Constraints (For the Agent)

1.  **Memory Layout:** Use Structure of Arrays (SoA). All vector types must be `float4` (16-byte aligned) for coalesced access. Avoid `float3`.
2.  **Solver Type:** **Projected Jacobi**. Never update `d_pos` or `d_vel` directly inside a solver loop. Use `atomicAdd` to accumulation buffers (`d_delta_pos`, `d_delta_vel`), then apply in a separate pass.
3.  **Collision Logic:** Use **Speculative Contacts**. Detect contacts with a safety margin. Solvers must check `dist` and `relative_velocity` to determine if a constraint is active.
4.  **No Virtual Functions:** Use data-driven design (Texture Objects and ID lookups) for shape handling.

-----

## Phase 1: Core Physics Engine (Spheres)

**Goal:** Implement the "Time Stepper" and Solvers using simple Spheres to validate the physics stability before introducing complex SDFs.

### 1.1 Data Structures (`src/cuda/ParticleSystem.cuh`)

Define the Kernel View struct. It must be trivially copyable.

```cpp
struct ContactConstraint {
    int bodyA;
    int bodyB;
    float4 normal;   // .xyz = normal, .w = padding
    float4 rA;       // .xyz = vector from COM_A to contact point
    float4 rB;       // .xyz = vector from COM_B to contact point
    float dist;      // Signed distance (Negative = Overlap, Positive = Gap)
    float friction_lambda_n; // Stored from prev frame or pos solve (for clamping)
};

struct ParticleSystem {
    // --- State Arrays (Read Only during Sub-step) ---
    float4* d_pos;        // .xyz = pos, .w = inv_mass
    float4* d_quat;       // .xyzw = quaternion
    float4* d_vel;        // .xyz = lin_vel, .w = padding
    float4* d_ang_vel;    // .xyz = ang_vel, .w = padding
    float4* d_inv_inertia;// .xyz = diagonal inertia

    // --- Predicted State (The "Working" Copy) ---
    float4* d_pos_pred;
    float4* d_quat_pred;
    float4* d_vel_pred;   // Intermediate velocity state
    float4* d_ang_vel_pred;

    // --- Jacobi Accumulators (Write Only) ---
    float4* d_delta_pos;
    float4* d_delta_quat;
    float4* d_delta_vel;
    float4* d_delta_ang_vel;
    int* d_constraint_counts;

    // --- Constraint Buffer ---
    ContactConstraint* d_contacts;
    int* d_contact_count; // Atomic counter
    
    // --- Configuration ---
    float dt;
    float4 gravity;
};
```

### 1.2 Math Utilities (`src/cuda/math_utils.cuh`)

Implement `atomicAddVector`.

  * **Logic:** Since CUDA `atomicAdd` supports `float` but not `float4`, verify the implementation casts `float4*` to `float*` and calls atomic add on components 0, 1, 2, 3 individually.

### 1.3 Integration Kernels (`src/cuda/integration.cu`)

**Kernel A: `predict_and_clear`**

  * **Goal:** Apply gravity and predict position. Reset accumulators.
  * **Logic:**
    1.  `v_pred = v_curr + gravity * dt`
    2.  `x_pred = x_curr + v_pred * dt`
    3.  `q_pred = IntegrateRotation(q_curr, omega_curr, dt)`
    4.  **Memset:** Set `d_delta_pos`, `d_delta_vel`, `d_constraint_counts` (etc.) to 0.
    5.  **Reset:** Set `*d_contact_count = 0`.

**Kernel B: `apply_updates`**

  * **Goal:** Apply averaged Jacobi corrections.
  * **Logic:**
    1.  Read `count = d_constraint_counts[id]`.
    2.  If `count > 0`:
          * `x_pred += d_delta_pos[id] / count`
          * `q_pred += d_delta_quat[id] / count` (Normalize q\!)
          * `v_pred += d_delta_vel[id] / count`
          * `omega_pred += d_delta_ang_vel[id] / count`

**Kernel C: `final_commit`**

  * **Goal:** Update state for next frame.
  * **Logic:**
      * `pos = pos_pred`
      * `vel = v_pred` (Note: This `v_pred` has been corrected by the Velocity Solver).
      * `quat = q_pred`
      * `ang_vel = omega_pred`

### 1.4 Narrowphase: Spheres (`src/cuda/narrowphase.cu`)

**Kernel: `detect_contacts_sphere`**

  * **Input:** `pos_pred`.
  * **Logic:**
    1.  Grid-stride loop over pairs (Brute force for Phase 1 is fine, or simple grid).
    2.  Calculate distance $d = |p_a - p_b| - (r_a + r_b)$.
    3.  **Speculative Check:** If $d < \text{SAFETY\_MARGIN}$:
          * Calculate normal, $r_A$, $r_B$.
          * `idx = atomicAdd(d_contact_count, 1)`.
          * Write to `d_contacts[idx]`.

### 1.5 Solver: Velocity (`src/cuda/solver_velocity.cu`)

**Kernel: `solve_velocity_jacobi`**

  * **Input:** `d_contacts`, `v_pred`, `omega_pred`.
  * **Logic (Per Thread):**
    1.  Read constraint.
    2.  Compute $v_{rel} = (v_A + \omega_A \times r_A) - (v_B + \dots)$.
    3.  **Activity Check:**
          * If `dist > 0` (Speculative) AND $v_{rel} \cdot n > 0$ (Separating): **Return**.
    4.  **Normal Impulse:**
          * Target $v_n = -e \cdot (v_{initial} \cdot n)$. (Use simple 0 target for first pass).
          * $\Delta J_n = M_{eff} \cdot (v_{target} - v_{current})$.
          * Clamp total accumulated impulse $\ge 0$.
    5.  **Tangent Impulse (Friction):**
          * Target $v_t = 0$.
          * Clamp against $\mu \cdot J_n$.
    6.  **Atomic Accumulate:** Add $J$ to `d_delta_vel` / `d_delta_ang_vel`.
    7.  **Count:** `atomicAdd(d_constraint_counts, 1)`.

### 1.6 Solver: Position (`src/cuda/solver_position.cu`)

**Kernel: `solve_position_jacobi`**

  * **Input:** `d_contacts`, `pos_pred`.
  * **Logic:**
    1.  Read constraint.
    2.  **Re-evaluate Overlap:** $C(x) = \text{dist} - \text{current\_separation}$.
    3.  **Activity Check:**
          * If $C(x) \ge 0$: **Return** (Separated).
    4.  **Projection:**
          * $\Delta \lambda = -C / w_{total}$.
          * $\Delta x = w \cdot \Delta \lambda \cdot n$.
    5.  **Atomic Accumulate:** Add to `d_delta_pos` / `d_delta_quat`.

-----

## Phase 2: General Shapes (SDF Architecture)

**Goal:** Replace hardcoded sphere logic with Data-Driven SDFs.

### 2.1 Host Shape Management (`src/shape_manager.h`)

Create a class `ShapeRegistry`.

  * **Method:** `uploadShape(float* sdf_grid, int3 dim, float* points, int num_points)`
  * **Action:**
    1.  Allocate `cudaArray`.
    2.  Copy 3D grid data.
    3.  Create `cudaTextureObject_t` with Linear Filtering and Clamp Border.
    4.  Upload points to a global `d_all_shell_points` buffer.
    5.  Store offsets in `d_shape_descriptors`.

### 2.2 Device Shape Structures

Update `ParticleSystem.cuh`:

```cpp
struct ShapeDescriptor {
    cudaTextureObject_t sdf_texture;
    float3 aabb_min; // For UVW mapping
    float3 aabb_max;
    int shell_start_idx;
    int shell_count;
};
// Add ShapeDescriptor* d_shapes to ParticleSystem
```

### 2.3 General Narrowphase Kernel (`src/cuda/narrowphase.cu`)

**Kernel: `detect_contacts_sdf`**

  * **Input:** Pairs from Broadphase (cuBQL).
  * **Logic:**
    1.  Get `shapeID_A`, `shapeID_B`.
    2.  **Loop** over point shell of A (using `d_shapes[id].shell_start`).
    3.  Transform Point $P_{local}$ to World $P_{world}$ using `pos_pred`, `quat_pred`.
    4.  Transform $P_{world}$ to B's texture space $UVW$.
    5.  **Sample:** `float dist = tex3D<float>(d_shapes[B].sdf_texture, uvw)`.
    6.  **Check:** If `dist < SAFETY_MARGIN`:
          * **Gradient:** Sample 6 neighbors ($\pm \epsilon$) to get normal.
          * Rotate normal to world space.
          * Atomic Write to `d_contacts`.

### 2.4 Python Shape Generator (`python/generate_shapes.py`)

Create a utility to generate the "Hollow Cylinder" test case.

  * Use `numpy` to generate a 3D grid ($64^3$ or $128^3$).
  * Calculate signed distance to a hollow tube.
  * Generate Poisson-disk sampled points on the surface.
  * Export to binary/numpy format for C++ loading.

-----

## Execution Checklist for the Agent

1.  **Start with Phase 1.** Do not implement textures yet. Hardcode `detect_contacts_sphere` to verify the `atomicAdd` Jacobi logic and `predict -> solve -> commit` loop.
2.  **Verify Solvers:**
      * Create a test: 2 spheres colliding.
      * Verify `delta_v` effectively reverses velocity (restitution).
      * Verify `delta_x` removes overlap.
3.  **Implement Phase 2:**
      * Implement `ShapeRegistry`.
      * Replace `detect_contacts_sphere` with `detect_contacts_sdf`.
      * Load the Hollow Cylinder data.
      * **Optimization:** Ensure the inner loop of `detect_contacts_sdf` minimizes register usage to keep occupancy high.