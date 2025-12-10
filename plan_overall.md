Here is the comprehensive **Technical Implementation Plan (PRD)**. It consolidates all our discussions: **Post-Stabilization (Velocity-based)** physics, **Projected Jacobi** solvers, **Analytic SDFs** (Hollow Cylinders), **Analytic Walls**, and **Periodic Ghosts**.

You can feed this document directly to your coding agent.

-----

# Technical Implementation Plan: HollowPack-HPC

**Project:** High-Performance GPU Granular Physics Engine
**Architecture:** C++17 / CUDA / Custom MPI / Python
**Physics Model:** Velocity-Based Post-Stabilization (Hybrid XPBD) with Projected Jacobi Solvers.

-----

## 1\. Global Architectural Constraints

1.  **Memory Layout:** Use Structure of Arrays (SoA). All vector types must be `float4` (16-byte aligned) for coalesced access. Avoid `float3`.
2.  **Solver Type:** **Projected Jacobi**. Never update `d_pos` or `d_vel` directly inside a solver kernel. Use `atomicAdd` to accumulation buffers (`d_delta_pos`, `d_delta_vel`), then apply in a separate "Apply" kernel.
3.  **Collision Logic:** Use **Speculative Contacts**. Detect contacts with a safety margin (`dist < margin`). Solvers must check `dist` and `relative_velocity` to determine if a constraint is active.
4.  **No Virtual Functions:** Use data-driven design (Enums, Texture Objects, and Parameter lookups) for shape handling.
5.  **Wall Handling:** Do **not** decompose walls into voxels. Use Analytic SDFs for boundaries (infinite mass).

-----

## 2\. Core Data Structures (`src/cuda/ParticleSystem.cuh`)

```cpp
enum ShapeType {
    SHAPE_GRID_SDF = 0,
    SHAPE_ANALYTIC_SPHERE = 1,
    SHAPE_ANALYTIC_HOLLOW_CYLINDER = 2,
    SHAPE_ANALYTIC_WALL_PLANE = 3,
    SHAPE_ANALYTIC_WALL_CYLINDER = 4
};

struct ShapeDescriptor {
    ShapeType type;
    // Type A: Grid
    cudaTextureObject_t sdf_texture;
    float3 aabb_min, aabb_max;
    // Type B: Analytic Params (R, h, thickness, etc.)
    float4 params; 
    // Collision Shell
    int shell_start_idx;
    int shell_count;
};

struct ContactConstraint {
    int bodyA;
    int bodyB;       // If -1, bodyB is a static wall
    float4 normal;   // .xyz = normal, .w = padding
    float4 rA;       // Vector from COM_A to contact point
    float4 rB;       // Vector from COM_B to contact point
    float dist;      // Signed distance (Negative=Overlap, Positive=Gap)
    float friction_lambda_n; // Accumulated Normal Impulse (for Friction clamping)
};

struct ParticleSystem {
    // --- State Arrays (Read Only during Sub-step) ---
    float4* d_pos;        // .xyz = pos, .w = inv_mass
    float4* d_quat;       // .xyzw = quaternion
    float4* d_vel;        // .xyz = lin_vel, .w = padding
    float4* d_ang_vel;    // .xyz = ang_vel, .w = padding
    float4* d_inv_inertia;// .xyz = diagonal inertia
    int* d_shape_ids;

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
    int* d_contact_count; 
    
    // --- Shapes ---
    ShapeDescriptor* d_shapes;
    float4* d_shell_points; // Global buffer of all surface points
};
```

-----

## Phase 1: Core Physics Engine (Spheres)

**Goal:** Implement the "Time Stepper" and Solvers using simple Spheres to validate the physics stability.

### Task 1.1: Integration Kernels (`integration.cu`)

**Kernel A: `predict_and_clear`**

  * **Logic:**
    1.  `v_pred = v_curr + gravity * dt`
    2.  `x_pred = x_curr + v_pred * dt`
    3.  `q_pred = IntegrateRotation(q_curr, omega_curr, dt)`
    4.  **Memset:** Set all `d_delta_*` and `d_constraint_counts` to 0.
    5.  **Reset:** Set `*d_contact_count = 0`.

**Kernel B: `apply_updates`**

  * **Logic:**
    1.  Read `count`. If `count > 0`:
          * `x_pred += d_delta_pos / count`
          * `q_pred += d_delta_quat / count` (Normalize\!)
          * `v_pred += d_delta_vel / count`
          * `omega_pred += d_delta_ang_vel / count`

**Kernel C: `final_commit`**

  * **Logic:**
      * `pos = pos_pred`
      * `vel = v_pred` (**Crucial:** Do not derive from displacement. Use solved velocity).
      * `quat = q_pred`
      * `ang_vel = omega_pred`

### Task 1.2: Narrowphase (Contact Generator) (`narrowphase.cu`)

**Kernel:** `detect_contacts_sphere`

  * **Input:** `pos_pred`.
  * **Logic:**
    1.  Compute distance $d$.
    2.  **Speculative Check:** If $d < \text{SAFETY\_MARGIN}$:
          * Calculate normal, $r_A$, $r_B$.
          * `idx = atomicAdd(d_contact_count, 1)`.
          * Write to `d_contacts[idx]`.

### Task 1.3: Velocity Solver (Jacobi) (`solver_velocity.cu`)

**Kernel:** `solve_velocity_jacobi`

  * **Input:** `d_contacts`, `v_pred`, `omega_pred`.
  * **Logic (Per Thread):**
    1.  Compute $v_{rel} = (v_A + \omega_A \times r_A) - (v_B + \dots)$.
    2.  **Activity Check:** If `dist > 0` (Speculative) AND $v_{rel} \cdot n > 0$ (Separating): **Return**.
    3.  **Restitution:** Target $v_n = -e \cdot (v_{initial} \cdot n)$.
    4.  **Friction:** Target $v_t = 0$. Clamp against $\mu \cdot J_n$.
    5.  **Accumulate:** Add $J$ to `d_delta_vel` using `atomicAddVector`.

### Task 1.4: Position Solver (Jacobi) (`solver_position.cu`)

**Kernel:** `solve_position_jacobi`

  * **Input:** `d_contacts`, `pos_pred`.
  * **Logic:**
    1.  **Re-evaluate Overlap:** $C(x) = \text{dist} - \text{current\_separation}$.
    2.  **Activity Check:** If $C(x) \ge 0$: **Return** (Separated).
    3.  **Projection:** Compute $\Delta x = w \cdot \Delta \lambda \cdot n$.
    4.  **Accumulate:** Add to `d_delta_pos` using `atomicAddVector`.

-----

## Phase 2: Advanced Shapes & Boundaries

**Goal:** Implement Analytic Hollow Cylinders, Analytic Walls, and Periodic Ghosts.

### Task 2.1: Analytic SDFs (`shapes/sdf_analytic.cuh`)

Implement the Hollow Cylinder math.

```cpp
__device__ inline float sdf_hollow_cylinder(float3 p, float4 params) {
    float h = params.x, r_outer = params.y, thick = params.z;
    // Map to 2D (r, y)
    float r = sqrtf(p.x*p.x + p.z*p.z);
    // Center profile
    float r_mid = r_outer - (thick * 0.5f);
    float2 d = make_float2(fabsf(r - r_mid) - thick*0.5f, fabsf(p.y) - h*0.5f);
    // Box distance logic
    return length(max(d, 0.0f)) + min(max(d.x, d.y), 0.0f);
}
```

### Task 2.2: General Narrowphase (`narrowphase.cu`)

Replace the sphere kernel with a `switch(type)` kernel.

**Kernel:** `detect_contacts_general`

  * **Input:** Broadphase Pairs.
  * **Logic:**
    1.  Loop over Point Shell of A.
    2.  Transform point to Local Space of B.
    3.  **Switch (B.type):**
          * `GRID`: `tex3D(B.texture)`.
          * `HOLLOW_CYL`: `sdf_hollow_cylinder(p_local, B.params)`.
          * `WALL_PLANE`: `dot(p_world, wall.normal) - wall.d`.
    4.  **Gradient:** If `dist < MARGIN`, compute normal via **Central Difference** (sample +/- eps).
    5.  **Write:** Store constraint.

### Task 2.3: Periodic Ghosts (Simultaneous Bitmask)

**Kernel:** `generate_ghosts`

  * **Logic:**
    1.  Thread $i$ checks Particle $i$.
    2.  `mask = 0`.
    3.  If `p.x < min.x + R`: `mask |= 1` (Left).
    4.  If `p.x > max.x - R`: `mask |= 2` (Right).
    5.  ... same for Y and Z ...
    6.  **Loop over mask:** If `mask` has bits for X and Y, generate ghosts for `(X)`, `(Y)`, AND `(XY)` shifts.
    7.  **Append:** Add ghost copies to `ParticleSystem` (resize if needed).

### Task 2.4: Broadphase Integration

  * Update `cuBQL` builder to include the Ghost Particles.
  * Ensure Narrowphase filters pairs: Ignore `Ghost vs Ghost`.

-----

## Phase 3: Domain Decomposition (MPI)

**Goal:** Scale to multiple GPUs using Z-Curve partitioning.

### Task 3.1: Z-Curve Sort

  * Implement Morton Code calculation kernel.
  * Use `cub::DeviceRadixSort` to sort `ParticleSystem`.

### Task 3.2: Halo Exchange

  * **Query:** Use `cuBQL` to find particles in the "Halo Skin".
  * **Pack:** Copy particle data to `d_send_buffer`.
  * **MPI:** Use CUDA-Aware `MPI_Isend` / `MPI_Irecv`.
  * **Unpack:** Append received particles as **Ghosts** (Reuse the Ghost flag from Phase 2).

-----

## Execution Checklist for the Agent

1.  **Phase 1 First:** Build `integration`, `solver_vel`, `solver_pos` for Spheres. Verify restitution ($e=1$ bounces forever) and friction ($v_t \to 0$).
2.  **Verify Solvers:** Ensure `detect_contacts` runs **before** solvers. Do not re-detect inside solvers.
3.  **Phase 1 Ghosts:** Implement the Bitmask Ghost kernel. Verify corner cases (7 ghosts).
4.  **Phase 2 Shapes:** Implement `sdf_analytic.cuh`. Create the "Hollow Cylinder" Python generator (points + params).
5.  **Phase 2 Walls:** Add `SHAPE_ANALYTIC_WALL` logic to Narrowphase. Test a cylinder hitting a floor.
