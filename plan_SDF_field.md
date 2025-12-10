# Technical Instruction: SDF Field Output Module

**Task:** Implement a module to generate a global Signed Distance Field (SDF) of the simulation domain and export it to Paraview (`.vti`) format.
**Algorithm:** Hybrid approach.

1.  **Splatting:** Particles write their exact local SDF values into the grid (Atomic Min).
2.  **Eikonal Solver:** A parallel Fast Iterative Method (FIM) fills the "holes" between particles to ensure a continuous Euclidean distance field.

-----

## 1\. File Structure

  * `src/cuda/output_sdf.cu`: Kernels for splatting and solving.
  * `src/cpp/OutputGenerator.cpp`: Host class to manage buffers and write VTI files.
  * `src/cuda/math_utils.cuh`: (Re-verify `atomicMinFloat` exists).

## 2\. Kernel Logic (`src/cuda/output_sdf.cu`)

### Kernel 1: `init_grid`

**Goal:** Initialize the grid with the distance to the Container Walls (the "Background").

  * **Input:** `d_grid`, `grid_dimensions`, `voxel_size`, `ContainerShape`.
  * **Logic:**
      * Map Global Thread ID to Voxel Coordinate $(x,y,z)$.
      * Calculate World Position $P_{world}$.
      * Evaluate **Analytic SDF** of the container (e.g., Cylinder: $R - \sqrt{x^2+z^2}$).
      * Write to `d_grid[idx]`.
      * Set corresponding `d_state[idx] = 0` (Active/Unknown) or `1` (Fixed/Known) depending on if you treat walls as fixed boundary conditions. *Recommendation: Treat walls as Fixed.*

### Kernel 2: `splat_particles`

**Goal:** Rasterize particle SDFs into the grid.
**Input:** `ParticleSystem`, `d_grid`, `d_state`.
**Logic:**

1.  Thread per **Particle**.
2.  Calculate Particle AABB in Grid Space. Clamp to Grid Dimensions.
3.  **Loop** over voxels in AABB ($x, y, z$).
4.  **Transform:**
      * $P_{world} = \text{GridToWorld}(x, y, z)$.
      * $P_{local} = \text{TransformInv}(P_{world}, \text{ParticlePos}, \text{ParticleQuat})$.
5.  **Canonical Bounds Check (Crucial):**
      * Check if $P_{local}$ is inside the Shape's Canonical OBB (e.g., $[-R, R]$).
      * **If Outside:** Continue (Skip). Do not write.
6.  **Evaluate:**
      * `switch(shape.type)`:
          * `ANALYTIC`: Call `sdf_hollow_cylinder(P_local, params)`.
          * `GRID`: Call `tex3D(shape.texture, P_local_normalized)`.
7.  **Write:**
      * `atomicMinFloat(&d_grid[voxel_idx], dist)`.
      * `d_state[voxel_idx] = 1` (Mark as FIXED).

### Kernel 3: `solve_eikonal_jacobi`

**Goal:** Propagate distance into the voids.
**Input:** `d_grid_in`, `d_grid_out`, `d_state`.
**Logic (Per Voxel):**

1.  If `d_state[idx] == 1` (Fixed):
      * `d_grid_out[idx] = d_grid_in[idx]`.
      * Return.
2.  **Godunov Update (Eikonal Update):**
      * Read neighbors ($x\pm1, y\pm1, z\pm1$). Handle boundaries (use Infinity).
      * Find min neighbor in each axis:
          * $a = \min(val_{x-1}, val_{x+1})$
          * $b = \min(val_{y-1}, val_{y+1})$
          * $c = \min(val_{z-1}, val_{z+1})$
      * Sort so $a \le b \le c$.
      * **Solve Quadratic:** $|\nabla \phi| = 1$.
          * Try 1D ($u = a + h$). If $u \le b$, keep.
          * Else try 2D ($u = \text{quad_solve}(a, b, h)$). If $u \le c$, keep.
          * Else try 3D ($u = \text{quad_solve}(a, b, c, h)$).
3.  **Write:** `d_grid_out[idx] = u`.

-----

## 3\. Host Class (`OutputGenerator.cpp`)

Implement a function `generateAndSaveVTI(filename, resolution)`:

1.  **Allocation:** Allocate `d_grid_Ping`, `d_grid_Pong`, `d_state`.
2.  **Execution Sequence:**
      * Launch `init_grid` (Write to Ping).
      * Launch `splat_particles` (Write to Ping, Update State).
      * **Loop (e.g., 20-50 iterations):**
          * Launch `solve_eikonal_jacobi` (Ping $\to$ Pong).
          * Swap pointers.
3.  **Download:** Copy final grid to Host Memory `std::vector<float>`.
4.  **Write VTI (XML):**
      * Use raw binary appending for speed.
      * Format structure:
    <!-- end list -->
    ```xml
    <VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">
      <ImageData WholeExtent="0 N 0 N 0 N" Origin="0 0 0" Spacing="dx dy dz">
        <Piece Extent="0 N 0 N 0 N">
          <PointData Scalars="SDF">
            <DataArray type="Float32" Name="SDF" format="appended" offset="0"/>
          </PointData>
        </Piece>
      </ImageData>
      <AppendedData encoding="raw">
        _ [BINARY_PAYLOAD_HERE]
      </AppendedData>
    </VTKFile>
    ```

-----

## 4\. Specific Implementation Constraints

1.  **Double Buffering:** The Eikonal solver MUST use ping-pong buffers. Reading and writing to the same buffer during propagation causes race conditions and artifacts.
2.  **Atomic Min:** Ensure `math_utils.cuh` implements the `atomicCAS` loop for float min, as standard `atomicMin` is integer-only.
3.  **Oversize Grid:** If the particle is partially outside the domain (periodic), the splat kernel must handle wrapping or clamping indices to avoid segfaults. *Simplest approach:* Clamp indices to `[0, dim-1]`.
4.  **Visualization:** In Paraview, the user will apply a "Contour" filter on value `0.0` to see the isosurface.