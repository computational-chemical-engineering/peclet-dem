Here is the detailed Technical Implementation Plan (PRD) formatted as a Markdown file. You can pass this directly to an AI coding assistant or use it as the master document for your development team.

# ---

**Technical Implementation Plan: Multi-GPU XPBD Engine for Granular Dynamics**

Project Name: HollowPack-HPC  
Target Application: Dense packing of non-spherical particles (Hollow Cylinders) in a periodic domain.  
Architecture: C++17 Core / CUDA / Custom MPI / Python Interface.

## ---

**1\. System Architecture Overview**

This engine avoids event-driven logic in favor of a stateless, time-stepping approach optimized for massive parallelism on GPUs.

### **Core Pipelines**

1. **Integration (XPBD Prediction):** Predict positions $x^\* \= x \+ v\\Delta t$. Apply growth (expansion) if in packing mode.  
2. **Domain Decomposition (MPI):** Z-Curve partitioning with Halo Exchange via GPUDirect.  
3. **Broadphase (cuBQL):** Build/Refit QBVH on GPU. Handle periodic ghosts.  
4. **Narrowphase (SDF \+ Shells):** Two-Level Hierarchical masking to generate contact constraints.  
5. **Solver (XPBD):** Projected Jacobi iteration to resolve Non-Penetration, Friction, and Rolling Resistance.  
6. **Commit:** Update velocities and positions.

## ---

**2\. Technology Stack**

* **Language:** C++17 (Host), CUDA C++ (Device), Python 3.10+ (Steering).  
* **Build System:** CMake 3.20+.  
* **MPI:** OpenMPI or MVAPICH2 (Must be built with CUDA awareness/GPUDirect support).  
* **GPU Broadphase:** **cuBQL** (NVIDIA Research) \- for high-performance QBVH construction.  
* **Bindings:** **pybind11** or **nanobind**.  
* **Math:** Native CUDA float4 intrinsics (no GLM/Eigen on device to ensure alignment).

## ---

**3\. Directory Structure**

Plaintext

/HollowPack-HPC  
├── CMakeLists.txt  
├── src  
│   ├── main\_binding.cpp         \# Pybind11 entry point  
│   ├── simulation.h/cpp         \# Main C++ Class (Memory Owner)  
│   ├── cuda  
│   │   ├── ParticleSystem.cuh   \# SoA Data Structures (The "View")  
│   │   ├── memory\_utils.cuh     \# Helpers for aligned allocs  
│   │   ├── integration.cu       \# Predict, Expansion, Update kernels  
│   │   ├── broadphase.cu        \# cuBQL wrapper & Motion AABB generation  
│   │   ├── narrowphase.cu       \# SDF sampling & Constraint generation  
│   │   ├── solver.cu            \# XPBD Projected Jacobi Kernels  
│   │   └── shapes  
│   │       ├── sdf\_cylinder.cuh \# Signed Distance Field math  
│   │       └── point\_shell.cuh  \# Shell generation logic  
│   └── mpi  
│       ├── domain.cpp           \# Z-Curve sorting & load balancing  
│       └── communicator.cpp     \# CUDA-Aware MPI buffers & Halo exchange  
├── python  
│   ├── run\_packing.py           \# Simulation script  
│   └── visualize.py             \# Paraview/VTP exporter  
└── tests  
    └── test\_periodicity.cpp     \# Unit tests for boundary wrapping

## ---

**4\. Key Data Structures (src/cuda/ParticleSystem.cuh)**

We use **Structure of Arrays (SoA)** with **128-bit alignment**.

C++

struct ParticleSystem {  
    // \--- Dynamic State \---  
    float4\* d\_pos;        // .xyz \= position, .w \= inv\_mass  
    float4\* d\_quat;       // .xyzw \= quaternion  
    float4\* d\_vel;        // .xyz \= lin\_vel, .w \= particle\_phase (active/ghost)  
    float4\* d\_ang\_vel;    // .xyz \= ang\_vel, .w \= particle\_scale (for expansion)

    // \--- XPBD Solver State \---  
    float4\* d\_pos\_star;   // Predicted Position  
    float4\* d\_quat\_star;  // Predicted Orientation  
    float4\* d\_delta\_pos;  // Accumulator for Jacobi  
    int\* d\_constraint\_counts;

    // \--- Static/Shape Data \---  
    float4\* d\_inv\_inertia; // .xyz \= diagonal inertia tensor  
    int\* d\_shape\_ids;   // Index into ShapeData  
      
    // \--- System Bounds \---  
    float3  domain\_min;  
    float3  domain\_size;   // For Periodic Mapping  
    int     num\_particles;  
};

struct ShapeData {  
    cudaTextureObject\_t sdf\_texture; // 3D Texture for general shapes  
    float4\* d\_proxies;               // Coarse Level 1 Mask  
    float4\* d\_fine\_points;           // Level 2 Point Shell  
    int2\* d\_tile\_descriptors;      // Lookup for tiles  
};

## ---

**5\. Implementation Strategy: Key Modules**

### **A. Periodicity & Broadphase (The "Ghost" Strategy)**

cuBQL does not natively support toroidal topology. We simulate it via **Ghost Replication** before the BVH build.

1. **Periodic Boundary Condition (PBC) Kernel:**  
   * Iterate all local particles.  
   * Check if pos is within cutoff\_radius of domain\_min or domain\_max.  
   * If yes, copy the particle to a temporary "Ghost List", applying the domain shift (e.g., pos.x \+= domain\_size.x).  
   * *Note:* A particle in a corner might generate up to 7 ghosts.  
2. **Build BVH:**  
   * Input to cuBQL \= \[Local\_Particles\] \+ \[Ghost\_Particles\].  
   * This ensures cuBQL detects collisions across the boundary naturally.  
3. **Narrowphase Filter:**  
   * Ignore collisions between two Ghost particles.  
   * Only process collisions where at least one particle is "Real".

### **B. The Hollow Cylinder Application**

We do not use a voxel grid for simple primitives to save memory; we use analytic SDFs injected into the kernel via templates.

* **SDF Definition:**  
  * Aligned along Y-axis, Outer Radius $R$, Thickness $t$.  
  * d\_xz \= length(p.xz) \- R  
  * d\_y \= abs(p.y) \- height/2  
  * dist\_tube \= length(max(vec2(d\_xz, d\_y), 0.0)) \+ min(max(d\_xz, d\_y), 0.0)  
  * **Hollow Modification:** We need the *shell*. The SDF is the distance to the *solid wall*.  
  * dist\_signed \= abs(dist\_tube\_center) \- thickness/2  
* **Expansion Logic:**  
  * In integration.cu, update particle\_scale based on a growth rate.  
  * The SDF function uses pos / particle\_scale to evaluate distance, effectively shrinking the world (or growing the particle).

### **C. MPI Domain Decomposition**

1. **Sort:** Every $N$ steps, sort ParticleSystem by Morton Code (using cub::DeviceRadixSort).  
2. **Partition:** Rank $i$ owns indices $\[i \\cdot N/P, (i+1) \\cdot N/P\]$.  
3. **Coarse Map:** Construct a coarse CPU octree. Map Morton ranges to octree leaves. Determine neighbor ranks.  
4. **Halo Exchange:**  
   * Query QBVH with Halo AABBs.  
   * Pack buffer using thrust::copy\_if (GPUDirect).  
   * Send/Recv via MPI\_Isend / MPI\_Irecv.

## ---

**6\. Step-by-Step Task List**

### **Phase 1: The Core (Single GPU / Spheres)**

* \[ \] **Scaffold:** Setup CMake, C++ project, and Pybind11 entry point.  
* \[ \] **Memory:** Implement ParticleSystem allocation/free and View generation.  
* \[ \] **Integration:** Write integrate kernel (Euler step) and reset\_deltas.  
* \[ \] **Broadphase:** Integrate cuBQL. Write wrapper to convert particle pos to cuBQL AABBs (including Motion Expansion for CCD).  
* \[ \] **Solver:** Implement simple Sphere-Sphere XPBD constraint solver (Projected Jacobi).  
* \[ \] **Python:** Write test script to drop 1000 spheres in a box.

### **Phase 2: Shapes & Periodicity**

* \[ \] **Analytic SDF:** Implement the SDF\_HollowCylinder device function.  
* \[ \] **Point Shell:** Write CPU code to generate points on a cylinder surface. Upload to GPU ShapeData.  
* \[ \] **Narrowphase:** Implement the "Two-Level Masking" kernel (Coarse Proxies \-\> Fine Points).  
* \[ \] **Periodicity:** Implement the "Ghost Particle Generation" kernel. Update BVH build to include ghosts.  
* \[ \] **Expansion:** Add scale parameter to ParticleSystem. Update SDF to respect scale.

### **Phase 3: Friction & Stability**

* \[ \] **Friction:** Implement the "Stateless Friction" logic in the solver (using relative motion in the current time step).  
* \[ \] **Rolling Resistance:** Add angular velocity damping constraint based on normal force $\\lambda\_n$.  
* \[ \] **Sub-stepping:** Update Python loop to trigger multiple solver substeps per frame.

### **Phase 4: MPI & Scale**

* \[ \] **Z-Curve:** Implement Morton encoding kernel and cub::RadixSort.  
* \[ \] **Partitioning:** Implement logic to split the sorted array among ranks.  
* \[ \] **Comm:** Implement PackHalo and UnpackHalo kernels.  
* \[ \] **Wiring:** Connect MPI Init/Finalize and the Exchange loop in the main step function.

## ---

**7\. Configuration for "Google Antigravity" Input**

When providing this to the coding assistant, emphasize the following constraints:

1. **No STL on GPU:** Do not use std::vector or std::map inside kernels.  
2. **Explicit Alignment:** Always use \_\_align\_\_(16) for custom structs if not using float4.  
3. **No Virtual Functions:** Use C++ Templates ("Curiously Recurring Template Pattern" or simple Instantiation) for handling different shapes (Cylinders vs Boxes) to avoid warp divergence overhead.  
4. **Raw Pointers:** The MPI buffer packing must operate on raw device pointers to enable Direct Memory Access (DMA) from the network card.