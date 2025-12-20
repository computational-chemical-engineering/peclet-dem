#include "ParticleSystem.cuh"
#include "math_utils.cuh"
#include "shapes/sdf_analytic.cuh"

namespace dem {

// -----------------------------------------------------------------------------
// Kernel: Initialize Grid
// -----------------------------------------------------------------------------
__global__ void init_grid_kernel(float *d_grid, int *d_state, int3 dims,
                                 float3 origin, float3 voxel_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_voxels = dims.x * dims.y * dims.z;
  if (idx >= total_voxels)
    return;

  // Convert idx to x,y,z
  int z = idx / (dims.x * dims.y);
  int rem = idx % (dims.x * dims.y);
  int y = rem / dims.x;
  int x = rem % dims.x;

  // World Position
  float3 p = make_float3(origin.x + (x + 0.5f) * voxel_size.x,
                         origin.y + (y + 0.5f) * voxel_size.y,
                         origin.z + (z + 0.5f) * voxel_size.z);

  // Initial Background: Container Walls (Cylinder radius 50.0 assumed for now,
  // or infinity) For now, let's initialize to a large positive value (Infinity)
  // to represent "void" The Eikonal solver will propagate from the particles
  // (Source) d_grid[idx] = 1e6f;

  // Better: Initialize to distance from origin for testing, or just MAX_FLOAT
  d_grid[idx] =
      100.0f;       // Sufficiently large for typical domain, acts as infinity
  d_state[idx] = 0; // 0 = Unknown/Active, 1 = Fixed
}

// -----------------------------------------------------------------------------
// Kernel: Splat Particles
// -----------------------------------------------------------------------------
__global__ void splat_particles_kernel(ParticleSystemData ps, float *d_grid,
                                       int *d_state, int3 dims, float3 origin,
                                       float3 voxel_size, float3 domain_min,
                                       float3 domain_max) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_real)
    return;

  float4 pos = ps.d_pos[idx];
  float4 quat = ps.d_quat[idx];
  float scale = ps.d_scale[idx]; // Read Scale
  int shape_id = ps.d_shape_ids[idx];
  ShapeDescriptor shape = ps.d_shapes[shape_id];

  // Determine Bounds of the particle in Grid Coords
  // Ideally we use AABB of the shape. For Sphere/Cylinder, we can approximate
  // with a bounding sphere Or for Cylinder: diagonal of AABB

  // Note: shape.params.x is 'radius' or similar. We must scale it.
  float radius_bound = 1.0f; // Default unit size if not specified
  if (shape.type == SHAPE_ANALYTIC_HOLLOW_CYLINDER) {
    // params: x=outer_radius, y=height, z=thickness
    float r = shape.params.x;
    float h = shape.params.y;
    // Bounds: Max of (OuterRadius, Height/2 in y)
    // Actually bounding sphere radius
    radius_bound = sqrtf(r * r + (h / 2.0f) * (h / 2.0f));
  } else if (shape.type == SHAPE_ANALYTIC_SPHERE) {
    radius_bound = shape.params.x; // Radius
  }

  // Apply Uniform Scale
  radius_bound *= scale;

  // Expand slightly for conservative rasterization
  radius_bound *= 1.2f;

  // Grid Bounds calculation
  float3 p_world = make_float3(pos.x, pos.y, pos.z);

  // Min/Max indices (Unclamped initially)
  int min_x = (int)floorf((p_world.x - radius_bound - origin.x) / voxel_size.x);
  int max_x = (int)ceilf((p_world.x + radius_bound - origin.x) / voxel_size.x);
  int min_y = (int)floorf((p_world.y - radius_bound - origin.y) / voxel_size.y);
  int max_y = (int)ceilf((p_world.y + radius_bound - origin.y) / voxel_size.y);
  int min_z = (int)floorf((p_world.z - radius_bound - origin.z) / voxel_size.z);
  int max_z = (int)ceilf((p_world.z + radius_bound - origin.z) / voxel_size.z);

  // Clamp if NOT periodic
  if (!ps.periodic_x) {
    min_x = max(0, min_x);
    max_x = min(dims.x - 1, max_x);
  }
  if (!ps.periodic_y) {
    min_y = max(0, min_y);
    max_y = min(dims.y - 1, max_y);
  }
  if (!ps.periodic_z) {
    min_z = max(0, min_z);
    max_z = min(dims.z - 1, max_z);
  }

  for (int z = min_z; z <= max_z; ++z) {
    for (int y = min_y; y <= max_y; ++y) {
      for (int x = min_x; x <= max_x; ++x) {

        // Voxel Center in World Space (Unwrapped loop variable usage)
        float3 voxel_pos = make_float3(origin.x + (x + 0.5f) * voxel_size.x,
                                       origin.y + (y + 0.5f) * voxel_size.y,
                                       origin.z + (z + 0.5f) * voxel_size.z);

        // Inverse Transform: World -> Local
        float3 p_rel = voxel_pos - p_world;
        float3 p_local_unscaled = inv_rotate_vector(quat, p_rel);

        // Apply Inverse Scale to get to Canonical Space (where radius is
        // params.x)
        float3 p_local = p_local_unscaled / scale;

        // Eval Canonical SDF
        float dist_canonical = 100.0f;
        if (shape.type == SHAPE_ANALYTIC_HOLLOW_CYLINDER) {
          dist_canonical = sdf_hollow_cylinder(p_local, shape.params);
        } else if (shape.type == SHAPE_ANALYTIC_SPHERE) {
          dist_canonical = sdf_sphere(p_local, shape.params);
        }

        // Scale distance back to World (approximate for general shapes, exact
        // for uniform scaled convex)
        float dist = dist_canonical * scale;

        // Optimization: Only write if meaningful (e.g. within some band)
        // But for exact SDF, we want to write correct values close to surface.

        // Wrap indices for periodic boundaries
        int wrapped_x = (x % dims.x + dims.x) % dims.x;
        int wrapped_y = (y % dims.y + dims.y) % dims.y;
        int wrapped_z = (z % dims.z + dims.z) % dims.z;

        // If NOT periodic, and out of bounds, skip (should be handled by loop
        // bounds but safe to double check)
        if (!ps.periodic_x && (x < 0 || x >= dims.x))
          continue;
        if (!ps.periodic_y && (y < 0 || y >= dims.y))
          continue;
        if (!ps.periodic_z && (z < 0 || z >= dims.z))
          continue;

        int voxel_idx =
            wrapped_z * (dims.x * dims.y) + wrapped_y * dims.x + wrapped_x;

        // Atomic Update
        atomicMinFloat(&d_grid[voxel_idx], dist);

        // Mark as Fixed Source if close to surface or inside (<= 0)
        // Actually, for FIM, we treat ANY place we write exact SDF as "Fixed"
        // source But typically we only fix the negative region and a small band
        // around it? For now, let's fix everything we touch that is "close
        // enough" to be accurate? Or simplistic: Fix everything inside the AABB
        if (dist < radius_bound) {
          d_state[voxel_idx] = 1;
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Kernel: Eikonal Update (FIM / Jacobi)
// -----------------------------------------------------------------------------
__device__ float solve_quadratic(float a, float b, float h) {
  // Solves (u-a)^2 + (u-b)^2 = h^2 for u > a, u > b
  // 2u^2 - 2(a+b)u + a^2 + b^2 - h^2 = 0
  // u = (a+b + sqrt(2h^2 - (a-b)^2)) / 2
  float d = 2.0f * h * h - (a - b) * (a - b);
  if (d < 0.0f)
    return 100.0f; // Should not happen if condition |a-b| < h checked
  return (a + b + sqrtf(d)) * 0.5f;
}

__device__ float solve_quadratic(float a, float b, float c, float h) {
  // Solves (u-a)^2 + (u-b)^2 + (u-c)^2 = h^2
  // 3u^2 - 2(a+b+c)u + a^2+b^2+c^2 - h^2 = 0
  float s = a + b + c;
  float q = a * a + b * b + c * c - h * h;
  // 3u^2 - 2su + q = 0
  // u = (2s + sqrt(4s^2 - 12q)) / 6 = (s + sqrt(s^2 - 3q))/3
  float delta = s * s - 3.0f * q;
  if (delta < 0.0f)
    return 100.0f;
  return (s + sqrtf(delta)) / 3.0f;
}

// Helper to wrap coordinate
__device__ inline int wrap_coord(int val, int dim) {
  return (val % dim + dim) % dim;
}

__global__ void solve_eikonal_jacobi_kernel(float *d_in, float *d_out,
                                            int *d_state, int3 dims,
                                            float3 voxel_size, bool px, bool py,
                                            bool pz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_voxels = dims.x * dims.y * dims.z;
  if (idx >= total_voxels)
    return;

  // If active fixed source, don't update
  if (d_state[idx] == 1) {
    d_out[idx] = d_in[idx];
    return;
  }

  // Convert idx to x,y,z
  int stride_y = dims.x;
  int stride_z = dims.x * dims.y;
  int z = idx / stride_z;
  int rem = idx % stride_z;
  int y = rem / stride_y;
  int x = rem % stride_y;

  // Sample Neighbors with Wrapping
  float val_x_m = 100.0f;
  if (x > 0)
    val_x_m = d_in[idx - 1];
  else if (px)
    val_x_m = d_in[idx - 1 + dims.x];

  float val_x_p = 100.0f;
  if (x < dims.x - 1)
    val_x_p = d_in[idx + 1];
  else if (px)
    val_x_p = d_in[idx + 1 - dims.x];

  float val_y_m = 100.0f;
  if (y > 0)
    val_y_m = d_in[idx - stride_y];
  else if (py)
    val_y_m = d_in[idx - stride_y +
                   stride_z]; // stride_z is size of one Z slice? No, stride_z =
                              // x*y = total size of one Z slice. Wait.
  // stride_y = dims.x.  Offset for Y-1 is -stride_y.
  // If y=0, we want y=dims.y-1 => index + (dims.y-1)*stride_y = index -
  // stride_y + stride_y*dims.y ... Simplest is to recompute index. idx(x, y-1)
  // when y=0 -> idx(x, dims.y-1) idx_wrapped = z*stride_z + (dims.y-1)*stride_y
  // + x
  else if (py)
    val_y_m = d_in[z * stride_z + (dims.y - 1) * stride_y + x];

  float val_y_p = 100.0f;
  if (y < dims.y - 1)
    val_y_p = d_in[idx + stride_y];
  else if (py)
    val_y_p = d_in[z * stride_z + 0 * stride_y + x];

  float val_z_m = 100.0f;
  if (z > 0)
    val_z_m = d_in[idx - stride_z];
  else if (pz)
    val_z_m = d_in[(dims.z - 1) * stride_z + y * stride_y + x];

  float val_z_p = 100.0f;
  if (z < dims.z - 1)
    val_z_p = d_in[idx + stride_z];
  else if (pz)
    val_z_p = d_in[0 * stride_z + y * stride_y + x];

  // Min in each dimension
  float a = fminf(val_x_m, val_x_p);
  float b = fminf(val_y_m, val_y_p);
  float c = fminf(val_z_m, val_z_p);

  // Sort a, b, c
  // simple bubble sort
  if (a > b) {
    float t = a;
    a = b;
    b = t;
  }
  if (b > c) {
    float t = b;
    b = c;
    c = t;
  }
  if (a > b) {
    float t = a;
    a = b;
    b = t;
  }

  // Assume isotropic voxel size for now (h = voxel_size.x)
  float h = voxel_size.x;
  float u = 100.0f;

  // Try 1D
  float u1 = a + h;
  if (u1 <= b) {
    u = u1;
  } else {
    // Try 2D
    float u2 = solve_quadratic(a, b, h);
    if (u2 <= c) {
      u = u2;
    } else {
      // Try 3D
      u = solve_quadratic(a, b, c, h);
    }
  }

  d_out[idx] = u;
}

// Wrappers
void launch_init_grid(float *d_grid, int *d_state, int3 dims, float3 origin,
                      float3 voxel_size) {
  int total = dims.x * dims.y * dims.z;
  int block = 256;
  int grid = (total + block - 1) / block;
  init_grid_kernel<<<grid, block>>>(d_grid, d_state, dims, origin, voxel_size);
}

void launch_splat_particles(ParticleSystemData ps, float *d_grid, int *d_state,
                            int3 dims, float3 origin, float3 voxel_size,
                            float3 d_min, float3 d_max) {
  int block = 128;
  int grid = (ps.num_real + block - 1) / block; // Use num_real
  splat_particles_kernel<<<grid, block>>>(ps, d_grid, d_state, dims, origin,
                                          voxel_size, d_min, d_max);
}

void launch_eikonal_update(float *d_in, float *d_out, int *d_state, int3 dims,
                           float3 voxel_size, bool px, bool py, bool pz) {
  int total = dims.x * dims.y * dims.z;
  int block = 256;
  int grid = (total + block - 1) / block;
  solve_eikonal_jacobi_kernel<<<grid, block>>>(d_in, d_out, d_state, dims,
                                               voxel_size, px, py, pz);
}

} // namespace dem
