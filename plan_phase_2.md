Here is the updated **Technical Implementation Plan for Phase 2**. I have swapped the focus from Hollow Spheres to **Hollow Cylinders** (Raschig Rings) as the primary analytic primitive, while keeping the architecture for Grid SDFs intact.

You can feed this directly to your coding agent.

-----

# Technical Implementation Plan: Phase 2 (General Shapes & SDFs)

**Goal:** Transition from hardcoded logic to a Data-Driven Shape Engine. Support both high-performance Analytic Primitives (specifically **Hollow Cylinders**) and general Voxelized Shapes.

## 1\. Shape Architecture (`src/cuda/shapes/`)

We define a unified interface where the solver treats all objects as "Shapes", but the Narrowphase kernel branches for performance.

### Task 2.1: Shape Data Structures (`ParticleSystem.cuh`)

**Enum for Shape Types:**

```cpp
enum ShapeType {
    SHAPE_GRID_SDF = 0,
    SHAPE_ANALYTIC_SPHERE = 1,
    SHAPE_ANALYTIC_HOLLOW_CYLINDER = 2, // The primary target
    SHAPE_ANALYTIC_BOX = 3
};
```

**Shape Descriptor Struct:**
This struct acts as a "Union" to hold data for any shape type.

```cpp
struct ShapeDescriptor {
    ShapeType type;
    
    // --- Type A: Grid SDF (Texture) ---
    cudaTextureObject_t sdf_texture;
    float3 aabb_min;      // For UVW mapping (World -> Texture Space)
    float3 aabb_max;
    
    // --- Type B: Analytic Parameters ---
    // Pack parameters into float4 registers to save memory bandwidth
    // For Hollow Cylinder: 
    // .x = height (h)
    // .y = outer_radius (R)
    // .z = thickness (t)
    // .w = unused (or inner_radius precalc)
    float4 params; 
    
    // --- Common: Point Shell (for Collision Source) ---
    // Index into the global 'd_all_shell_points' buffer
    int shell_start_idx;
    int shell_count;
};
```

-----

## 2\. Analytic SDF Implementations (`src/cuda/shapes/`)

### Task 2.2: Hollow Cylinder SDF (Math)

**Concept:** A hollow cylinder is the solid of revolution of a 2D rectangular profile. We map 3D space $(x,y,z)$ to 2D radial space $(r, y)$ and check distance to a 2D box.

  * **Parameters:** Height $h$, Outer Radius $R$, Thickness $t$.
  * **Derived:** $R_{mid} = R - t/2$, Radial Half-width $w_r = t/2$, Vertical Half-width $w_y = h/2$.

**File:** `src/cuda/shapes/sdf_analytic.cuh`

```cpp
__device__ inline float sdf_hollow_cylinder(float3 p, float4 params) {
    float h = params.x;
    float r_outer = params.y;
    float thick = params.z;
    
    // 1. Map to 2D Radial Plane
    // r = distance from central axis (Y-axis assumption)
    float r = sqrtf(p.x * p.x + p.z * p.z);
    float y = p.y;
    
    // 2. Center the "Wall Profile"
    float r_mid = r_outer - (thick * 0.5f);
    
    // 3. 2D Box SDF Logic
    // Vector d = distance from the center of the wall segment
    // We treat the wall cross-section as a box of size (thick, h)
    float2 d;
    d.x = fabsf(r - r_mid) - (thick * 0.5f);
    d.y = fabsf(y) - (h * 0.5f);
    
    // 4. Combine Exterior and Interior distances
    // length(max(d, 0)) -> Distance to outside of box
    // min(max(d.x, d.y), 0) -> Distance to inside of box (negative)
    float outside_dist = length(make_float2(fmaxf(d.x, 0.0f), fmaxf(d.y, 0.0f)));
    float inside_dist = fminf(fmaxf(d.x, d.y), 0.0f);
    
    return outside_dist + inside_dist;
}
```

### Task 2.3: Gradient Calculation

For analytic shapes, we can implement an exact gradient, but **Central Difference** sampling is often faster on GPUs than evaluating complex derivative branches, and it handles sharp edges (like the rim of the cylinder) more robustly for collision normals.

-----

## 3\. General Narrowphase Kernel (`src/cuda/narrowphase.cu`)

We implement a single monolithic kernel that switches logic based on `ShapeType`.

### Task 2.4: The `detect_contacts` Kernel

**Input:** Broadphase Pairs, `x_pred`, `q_pred`, `d_shapes`.
**Logic:**

```cpp
// 1. Setup
int shape_id_A = p.d_shape_ids[body_A];
int shape_id_B = p.d_shape_ids[body_B];
ShapeDescriptor shape_B = d_shapes[shape_id_B];

// 2. Loop over Point Shell of Body A
for (int i = 0; i < d_shapes[shape_id_A].shell_count; ++i) {
    // ... Load point, Transform to World, Transform to Local B ...
    float3 p_local = ...; 

    float dist = 0.0f;
    float3 normal_local;

    // 3. Switch based on Target Shape Type
    switch (shape_B.type) {
        case SHAPE_GRID_SDF: {
            // Map p_local to UVW (0..1)
            float3 uvw = (p_local - shape_B.aabb_min) / (shape_B.aabb_max - shape_B.aabb_min);
            dist = tex3D<float>(shape_B.sdf_texture, uvw.x, uvw.y, uvw.z);
            break;
        }
        case SHAPE_ANALYTIC_HOLLOW_CYLINDER: {
            dist = sdf_hollow_cylinder(p_local, shape_B.params);
            break;
        }
        // ... other types ...
    }

    // 4. Generate Constraint
    if (dist < SAFETY_MARGIN) {
        // Calculate Normal (Gradient) via Central Difference
        // This works for both Texture and Analytic types identically!
        float eps = 1e-4f;
        float3 d_dx, d_dy, d_dz; // +/- samples
        // ... sample sdf at p_local +/- eps ...
        normal_local = normalize(make_float3(dx, dy, dz));
        
        // ... Transform Normal to World ...
        // ... Atomic Add to ConstraintBuffer ...
    }
}
```

-----

## 4\. Host-Side Management (`src/cpp/`)

### Task 2.5: Shape Manager Class

We need C++ code to manage the GPU memory for these shapes.

  * **File:** `ShapeManager.h` / `.cpp`
  * **Method:** `createAnalyticShape(ShapeType type, float4 params, vector<float3> shell_points)`
      * Uploads points to the global buffer.
      * Stores `params` in the descriptor.
      * Returns `int shape_id`.
  * **Method:** `createGridShape(float* grid_data, int3 dim, float3 min, float3 max)`
      * Creates `cudaArray`, `cudaMemcpy3D`.
      * Creates `cudaTextureObject`.
      * Returns `int shape_id`.

### Task 2.6: Python Generation Scripts (`python/shapes/`)

We need scripts to generate the **Point Shells** (the "feelers" for collision) for the hollow cylinder.

  * **File:** `gen_hollow_cylinder.py`
      * **Logic:** Generate points on the surface of the cylinder.
      * **Crucial:** You must generate points on **ALL** surfaces:
        1.  **Outer Wall:** Cylinder at radius $R$.
        2.  **Inner Wall:** Cylinder at radius $R - t$.
        3.  **Top Rim:** Annulus (Ring) at $y = +h/2$.
        4.  **Bottom Rim:** Annulus (Ring) at $y = -h/2$.
      * **Output:** NumPy array `(N, 4)` (x, y, z, padding).

-----

## 5\. Execution Order for the Agent

1.  **Implement `sdf_analytic.cuh`**: Add the Hollow Cylinder device function.
2.  **Update Structs**: Modify `ShapeDescriptor` in `ParticleSystem.cuh` to include `float4 params`.
3.  **Update Narrowphase**: Rewrite the kernel to include the `switch(type)` statement and the `SHAPE_ANALYTIC_HOLLOW_CYLINDER` case.
4.  **Test**:
      * Create a test scene with 2 interlocking Hollow Cylinders (like links in a chain).
      * Verify they collide correctly and do not pass through each other's walls.
      * Verify particles can sit *inside* the hollow cylinder without triggering collision (since $dist > 0$ in the void).