/// @file
/// @brief dem — the shape registry: the host-side authoritative list of particle shapes (analytic
/// sphere / hollow cylinder / box, grid SDF, composed analytic scene) with their surface point
/// shells, unit-mass inverse inertias and bounding radii, and the upload that rebuilds the device
/// Views (`P_.shapes` / `P_.shell` / `P_.sdfGrid` / `P_.shapeNodes`) from it. `Simulation`
/// (sim.hpp) derives from it; the registry owns the particle SoA `P_` because every shape change
/// re-sizes the contact buffers around it.
#ifndef DEM_SHAPE_REGISTRY_HPP
#define DEM_SHAPE_REGISTRY_HPP

#include <algorithm>
#include <cstddef>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "narrowphase.hpp"  // ShapeDesc, SHAPE_GRID_SDF / SHAPE_SCENE
#include "particles.hpp"
#include "peclet/core/geom/scene_builder.hpp"
#include "shapes_portable.hpp"  // genCylinderShell / genBoxShell
#include "step_solve.hpp"       // growContactBuffers

namespace peclet::dem {

/// Host-side shape registry (Layer 1). The device Views (P_.shapes / P_.shell / P_.sdfGrid) are
/// rebuilt from the host vectors by uploadShapes(), which is what assigns each shape its
/// shellOffset and grid offset into the shared pools. Keeping the authoritative copy on the host is
/// what makes a MIXTURE of shapes possible at all: the old code wrote descriptor slot 0 directly
/// and had nowhere to put a second shell.
class ShapeRegistry {
 public:
  void setSphereShape(float radius) { initializeShape(SPHERE, radius, 0.0f, 0.0f); }

  // Mirror of CUDA Simulation::initialize(shape_type, radius, height, thickness): builds shape 0's
  // descriptor + surface point shell (cylinder/box) and records the per-shape base radius and
  // (uniform-mass=1) inverse inertia applied to every particle by setPositions. shape_type uses the
  // peclet::dem::ShapeKind values (SPHERE=1, HOLLOW_CYLINDER=2, BOX=3).
  void initializeShape(int shape_type, float radius, float height, float thickness) {
    baseRadius_ = radius;
    P_.baseRadius =
        radius;  // effective radius = scale*globalScale*baseRadius (broadphase + ghost band)
    F4 params{radius, 0, 0, 0};
    std::vector<F3> shell;

    if (shape_type == HOLLOW_CYLINDER) {
      params = F4{radius, height, thickness, 0};
      // Dynamic spacing (faithful to CUDA): >=4 pts across thickness, >=20 around circumference.
      float min_dim = std::min(radius, thickness);
      if (min_dim < 1e-4f)
        min_dim = radius;  // safety if thickness 0
      float spacing = std::min(radius * 0.3f, min_dim * 0.5f);
      if (spacing < 1e-3f)
        spacing = 1e-3f;
      shell = genCylinderShell(radius, height, thickness, spacing);
    } else if (shape_type == BOX) {
      // Cube with half-extent = radius (side = 2*radius).
      params = F4{radius, radius, radius, 0};
      float spacing = std::max(radius * 0.5f, 1e-3f);
      shell = genBoxShell(radius, radius, radius, spacing);
    } else {
      shape_type = SPHERE;
      params = F4{radius, 0, 0, 0};  // sphere: analytic single-probe, no shell
    }

    const int nPts = static_cast<int>(shell.size());

    // Per-shape inverse inertia (mass=1), faithful to CUDA Simulation::initialize.
    float ix = 1.0f, iy = 1.0f, iz = 1.0f;
    if (shape_type == SPHERE) {
      if (baseRadius_ > 0.0f) {
        float v = 2.5f / (baseRadius_ * baseRadius_);
        ix = iy = iz = v;
      }
    } else if (shape_type == HOLLOW_CYLINDER) {
      float r_out = baseRadius_, r_in = baseRadius_ - thickness;
      if (r_in < 0)
        r_in = 0;
      float term_r = r_out * r_out + r_in * r_in;
      float I_zz = 0.5f * term_r;
      float I_xx = (1.0f / 12.0f) * (3.0f * term_r + height * height);
      if (I_xx > 1e-6f) {
        ix = 1.0f / I_xx;
        iy = 1.0f / I_xx;
      }
      if (I_zz > 1e-6f)
        iz = 1.0f / I_zz;
    } else if (shape_type == BOX) {
      float L = 2.0f * baseRadius_;
      float I = (1.0f / 6.0f) * L * L;
      if (I > 1e-6f) {
        ix = iy = iz = 1.0f / I;
      }
    }
    // RESET the registry to this one shape -- initializeShape's documented behaviour is "shape 0
    // becomes this". Use addShape() to build a mixture instead.
    clearShapes();
    ShapeDesc sd{shape_type, params, 0, nPts};
    appendShape(sd, shell, F3{ix, iy, iz}, radius);
    uploadShapes();
    ensureContactCapacity();
  }

  /// Append an analytic shape and return its index, for simulations with a MIXTURE of shapes.
  /// Same arguments as initializeShape, which stays the "single shape" entry point (it resets the
  /// registry). Assign the returned index to particles with setShapeIds().
  int addShape(int shape_type, float radius, float height, float thickness) {
    const std::size_t before = shapesHost_.size();
    if (before == 0)
      throw std::runtime_error("addShape: call initialize_shape/set_sphere_shape first");
    // Reuse initializeShape's descriptor + shell construction by running it into a scratch
    // registry, then splice the result back on top of the existing shapes.
    std::vector<ShapeDesc> keepS = shapesHost_;
    std::vector<F3> keepShell = shellHost_;
    std::vector<F3> keepInvI = invIHost_;
    std::vector<float> keepBase = baseRadiusHost_;
    std::vector<float> keepSamples = sdfSamplesHost_;        // initializeShape clears these too
    initializeShape(shape_type, radius, height, thickness);  // leaves exactly one shape
    ShapeDesc added = shapesHost_[0];
    const std::vector<F3> addedShell = shellHost_;
    const F3 addedInvI = invIHost_[0];
    const float addedBase = baseRadiusHost_[0];
    shapesHost_ = keepS;
    shellHost_ = keepShell;
    invIHost_ = keepInvI;
    baseRadiusHost_ = keepBase;
    sdfSamplesHost_ = keepSamples;
    appendShape(added, addedShell, addedInvI, addedBase);
    uploadShapes();
    ensureContactCapacity();
    return static_cast<int>(shapesHost_.size()) - 1;
  }

  /// Per-particle shape assignment. Also refreshes each particle's inverse inertia from its new
  /// shape, so the order of setPositions/setShapeIds does not matter.
  void setShapeIds(const std::vector<int>& ids) {
    if (static_cast<int>(ids.size()) != P_.numReal)
      throw std::runtime_error("set_shape_ids: expected one id per particle");
    const int nShapes = static_cast<int>(shapesHost_.size());
    for (int v : ids)
      if (v < 0 || v >= nShapes)
        throw std::runtime_error("set_shape_ids: shape id out of range");
    auto sid = Kokkos::create_mirror_view(P_.shapeId);
    auto ii = Kokkos::create_mirror_view(P_.invInertia);
    Kokkos::deep_copy(sid, P_.shapeId);
    Kokkos::deep_copy(ii, P_.invInertia);
    for (int i = 0; i < P_.numReal; ++i) {
      sid(i) = ids[i];
      ii(i, 0) = invIHost_[ids[i]].x;
      ii(i, 1) = invIHost_[ids[i]].y;
      ii(i, 2) = invIHost_[ids[i]].z;
    }
    Kokkos::deep_copy(P_.shapeId, sid);
    Kokkos::deep_copy(P_.invInertia, ii);
  }

  /// Append a grid-SDF shape (the general non-spherical particle) to the mixture and return its
  /// index. Same arguments as set_sdf_shape, which stays the single-shape entry point. The samples
  /// are appended to the shared pool and the descriptor's offset set accordingly.
  int addSdfShape(const std::vector<float>& grid, int nx, int ny, int nz, F3 origin, F3 spacing,
                  const std::vector<float>& shellFlat, F3 invInertia, float boundingRadius) {
    if (shapesHost_.empty())
      throw std::runtime_error("add_sdf_shape: call initialize_shape/set_sdf_shape first");
    if (nx < 2 || ny < 2 || nz < 2)
      throw std::runtime_error("add_sdf_shape: grid dims must be >= 2 on each axis");
    if (static_cast<long>(nx) * ny * nz != static_cast<long>(grid.size()))
      throw std::runtime_error("add_sdf_shape: grid.size() must equal nx*ny*nz");
    const std::vector<float> shellUse =
        shellFlat.empty() ? autoShell(grid, nx, ny, nz, origin, spacing, boundingRadius)
                          : shellFlat;
    const int nPts = static_cast<int>(shellUse.size() / 3);
    if (nPts <= 0)
      throw std::runtime_error(
          "add_sdf_shape: surface point shell is empty and could not be "
          "generated from the field");
    ShapeDesc sd{};
    sd.type = SHAPE_GRID_SDF;
    sd.params = F4{boundingRadius, 0, 0, 0};
    sd.grid.nx = nx;
    sd.grid.ny = ny;
    sd.grid.nz = nz;
    sd.grid.offset = static_cast<int>(sdfSamplesHost_.size());  // append to the shared pool
    sd.grid.origin = toCoreVec(origin);
    sd.grid.invSpacing = peclet::core::Vec3<float>{spacing.x > 0 ? 1.0f / spacing.x : 0.0f,
                                                   spacing.y > 0 ? 1.0f / spacing.y : 0.0f,
                                                   spacing.z > 0 ? 1.0f / spacing.z : 0.0f};
    sd.grid.extension = peclet::core::geom::GridExtension::kObject;
    sdfSamplesHost_.insert(sdfSamplesHost_.end(), grid.begin(), grid.end());
    std::vector<F3> shellPts(nPts);
    for (int i = 0; i < nPts; ++i)
      shellPts[i] = F3{shellUse[3 * i], shellUse[3 * i + 1], shellUse[3 * i + 2]};
    appendShape(sd, shellPts, invInertia, boundingRadius);
    uploadShapes();
    ensureContactCapacity();
    return static_cast<int>(shapesHost_.size()) - 1;
  }

  int numShapes() const { return static_cast<int>(shapesHost_.size()); }

  // Import a general particle as a grid SDF (canonical, unit-scale space) + a surface point shell +
  // its unit-mass principal-frame diagonal inverse inertia. Replaces shape 0, so every particle
  // becomes an instance of this shape (per-particle position/quaternion/scale still apply). `grid`
  // holds nx*ny*nz signed-distance samples, x-fastest (idx = x + y*nx + z*nx*ny), at lattice nodes
  // q = origin + (x,y,z)*spacing (negative inside). `shellFlat` is the flat [nPts*3] surface point
  // set the collision probes body A against. boundingRadius is the canonical radius enclosing the
  // surface (broad-phase + VTI splat bound). invInertia is the per-unit-mass diagonal inverse
  // inertia in the body (principal) frame; like the analytic shapes it becomes the default applied
  // to every particle by setPositions (override afterwards with set_inv_inertia / set_inv_mass for
  // a real density). See peclet.dem.particle_builder for the Python side that produces these arrays
  // from an implicit-solid SDF (marching-cubes shell + voxel-integrated mass properties).
  void setSdfShape(const std::vector<float>& grid, int nx, int ny, int nz, F3 origin, F3 spacing,
                   const std::vector<float>& shellFlat, F3 invInertia, float boundingRadius) {
    if (nx < 2 || ny < 2 || nz < 2)
      throw std::runtime_error("setSdfShape: grid dims must be >= 2 on each axis");
    if (static_cast<long>(nx) * ny * nz != static_cast<long>(grid.size()))
      throw std::runtime_error("setSdfShape: grid.size() must equal nx*ny*nz");
    // An EMPTY shell means "generate one from the field itself" (Layer 1).
    const std::vector<float> shellUse =
        shellFlat.empty() ? autoShell(grid, nx, ny, nz, origin, spacing, boundingRadius)
                          : shellFlat;
    const int nPts = static_cast<int>(shellUse.size() / 3);
    if (nPts <= 0)
      throw std::runtime_error(
          "setSdfShape: surface point shell is empty and could not be "
          "generated from the field");

    // Upload the signed-distance samples.
    P_.sdfGrid = Kokkos::View<float*, CpMem>("sdfGrid", grid.size());
    auto hg = Kokkos::create_mirror_view(P_.sdfGrid);
    for (size_t i = 0; i < grid.size(); ++i)
      hg(i) = grid[i];
    Kokkos::deep_copy(P_.sdfGrid, hg);

    // Upload the surface point shell.
    P_.shell = Kokkos::View<float* [3], CpMem>("shell", nPts);
    auto hs = Kokkos::create_mirror_view(P_.shell);
    for (int i = 0; i < nPts; ++i) {
      hs(i, 0) = shellFlat[3 * i];
      hs(i, 1) = shellFlat[3 * i + 1];
      hs(i, 2) = shellFlat[3 * i + 2];
    }
    Kokkos::deep_copy(P_.shell, hs);

    // Shape descriptor: grid SDF for the field, shell points for the probes.
    ShapeDesc sd{};
    sd.type = SHAPE_GRID_SDF;
    sd.params = F4{boundingRadius, 0, 0, 0};
    sd.shellOffset = 0;
    sd.numPoints = nPts;
    sd.grid.offset = 0;
    sd.grid.nx = nx;
    sd.grid.ny = ny;
    sd.grid.nz = nz;
    sd.grid.origin = toCoreVec(origin);
    sd.grid.invSpacing = peclet::core::Vec3<float>{spacing.x > 0 ? 1.0f / spacing.x : 0.0f,
                                                   spacing.y > 0 ? 1.0f / spacing.y : 0.0f,
                                                   spacing.z > 0 ? 1.0f / spacing.z : 0.0f};
    sd.grid.extension = peclet::core::geom::GridExtension::kObject;  // a body

    std::vector<F3> shellPts(nPts);
    for (int i = 0; i < nPts; ++i)
      shellPts[i] = F3{shellUse[3 * i], shellUse[3 * i + 1], shellUse[3 * i + 2]};
    clearShapes();
    sdfSamplesHost_ = grid;  // this shape owns the whole pool when it is the only shape
    sd.grid.offset = 0;
    appendShape(sd, shellPts, invInertia, boundingRadius);
    uploadShapes();
    ensureContactCapacity();
  }

  // Composed-analytic particle shape (SHAPE_SCENE): decode core's flat node encoding into the
  // shared per-Simulation pool (child indices rebased to ABSOLUTE pool indices, exactly like
  // addAnalyticWall), register a ShapeDesc whose tree pointer is patched at uploadShapes. The
  // canonical frame is trusted to be the PRINCIPAL frame -- SceneBuilder.principal_frame emits
  // that; passing a non-principal tree runs the diagonal-inertia rotational update on the wrong
  // frame, silently, which is why the docstring shouts about it.
  int addSceneShape(const std::vector<int>& nodeInts, const std::vector<float>& nodeReals, int root,
                    const std::vector<float>& shellFlat, F3 invInertia, float boundingRadius) {
    namespace g = peclet::core::geom;
    if (nodeInts.size() % g::kNodeIntStride || nodeReals.size() % g::kNodeRealStride)
      throw std::runtime_error("add_scene_shape: node arrays are not a whole number of records");
    const int n = static_cast<int>(nodeInts.size() / g::kNodeIntStride);
    if (n == 0 || static_cast<int>(nodeReals.size() / g::kNodeRealStride) != n)
      throw std::runtime_error("add_scene_shape: node int/real counts disagree");
    if (root < 0 || root >= n)
      throw std::runtime_error("add_scene_shape: root out of range");
    const int base = static_cast<int>(shapeNodesHost_.size());
    for (int i = 0; i < n; ++i) {
      g::ShapeNode<float> nd = g::decodeNode<float>(&nodeInts[i * g::kNodeIntStride],
                                                    &nodeReals[i * g::kNodeRealStride]);
      if (nd.kind == g::kGrid)
        throw std::runtime_error(
            "add_scene_shape: grid leaves are not supported in particle trees (use set_sdf_shape "
            "for sampled shapes)");
      if (nd.kind >= g::kCsgBase) {
        nd.aux0 += base;
        nd.aux1 += base;
      }
      shapeNodesHost_.push_back(nd);
    }
    std::vector<F3> shellPts;
    if (shellFlat.size() % 3)
      throw std::runtime_error("add_scene_shape: shell must be (M,3)");
    shellPts.reserve(shellFlat.size() / 3);
    for (std::size_t i = 0; i + 2 < shellFlat.size(); i += 3)
      shellPts.push_back(F3{shellFlat[i], shellFlat[i + 1], shellFlat[i + 2]});
    if (shellPts.empty())
      throw std::runtime_error(
          "add_scene_shape: an empty shell would make the body invisible to contacts -- bake the "
          "tree and generate one (SceneBuilder.bake + the marching-cubes shell path)");
    ShapeDesc sd{};
    sd.type = SHAPE_SCENE;
    sd.params = F4{boundingRadius, 0, 0, 0};
    sd.nodeCount = 0;  // patched at uploadShapes (whole-pool count; root is absolute)
    sd.shapeRoot = base + root;
    appendShape(sd, shellPts, invInertia, boundingRadius);
    uploadShapes();
    ensureContactCapacity();  // every other shape adder does this; its absence dropped contacts
    return static_cast<int>(shapesHost_.size()) - 1;
  }

  /// Generate a collision shell for a grid SDF by sampling its own zero level set, so a caller
  /// does not have to supply one. Layer 1: previously every shape needed a hand-written generator
  /// (there were only two, for the cylinder and the box) or a marching-cubes shell computed in
  /// Python; core's surfacePoints() is driven by the SDF itself and works for any geometry.
  std::vector<float> autoShell(const std::vector<float>& grid, int nx, int ny, int nz, F3 origin,
                               F3 spacing, float boundingRadius) const {
    namespace g = peclet::core::geom;
    g::SceneBuilder<float> b;
    const int node = b.addGrid(
        grid, nx, ny, nz, peclet::core::Vec3<float>{origin.x, origin.y, origin.z},
        peclet::core::Vec3<float>{spacing.x, spacing.y, spacing.z}, g::GridExtension::kObject);
    // Pitch: fine enough that the shell resolves the body, coarse enough not to explode the
    // contact buffers -- ~1/12 of the bounding radius matches the density of the hand-written
    // cylinder/box shells.
    const float pitch = std::max(boundingRadius / 12.0f, 1e-4f);
    const float r = boundingRadius * 1.05f;
    const std::vector<peclet::core::Vec3<float>> pts =
        g::surfacePoints<float>(b.view(), node, pitch, peclet::core::Vec3<float>{-r, -r, -r},
                                peclet::core::Vec3<float>{r, r, r});
    std::vector<float> flat;
    flat.reserve(pts.size() * 3);
    for (const auto& q : pts) {
      flat.push_back(q.x);
      flat.push_back(q.y);
      flat.push_back(q.z);
    }
    return flat;
  }

 protected:
  // Size the contact/manifold buffers so no contact is dropped: a shell point sits inside at most
  // one neighbour (body-body ~ capacity*shellPoints) plus one per wall it touches
  // (capacity*shellPoints per wall). Boundary/wall contacts are appended AFTER body-body ones, so
  // an undersized buffer silently drops them and grains tunnel through walls. Floored at the
  // analytic default; grows only.
  void ensureContactCapacity() {
    P_.shellPoints = shellPoints_;
    growContactBuffers(P_, P_.capacity);
  }

  Particles P_;
  float baseRadius_ = 1.0f;
  // --- host-side shape registry (Layer 1) -------------------------------------------------
  // The device Views (P_.shapes / P_.shell / P_.sdfGrid) are rebuilt from these by uploadShapes(),
  // which is what assigns each shape its shellOffset and grid offset into the shared pools. Keeping
  // the authoritative copy on the host is what makes a MIXTURE of shapes possible at all: the old
  // code wrote descriptor slot 0 directly and had nowhere to put a second shell.
  std::vector<ShapeDesc> shapesHost_;
  std::vector<F3> shellHost_;          // concatenated shells; ShapeDesc::shellOffset indexes it
  std::vector<F3> invIHost_;           // per-shape unit-mass inverse inertia
  std::vector<float> baseRadiusHost_;  // per-shape canonical bounding radius
  std::vector<float> sdfSamplesHost_;  // concatenated grid-SDF samples

  void clearShapes() {
    shapesHost_.clear();
    shellHost_.clear();
    invIHost_.clear();
    baseRadiusHost_.clear();
    sdfSamplesHost_.clear();
  }

  /// Append one shape, taking ownership of its shell. Fills in shellOffset from the running
  /// concatenation; the caller has already set the grid fields (if any).
  void appendShape(ShapeDesc sd, const std::vector<F3>& shell, F3 invI, float baseRadius) {
    sd.shellOffset = static_cast<int>(shellHost_.size());
    sd.numPoints = static_cast<int>(shell.size());
    shellHost_.insert(shellHost_.end(), shell.begin(), shell.end());
    shapesHost_.push_back(sd);
    invIHost_.push_back(invI);
    baseRadiusHost_.push_back(baseRadius);
  }

  /// Rebuild the device Views from the host registry.
  void uploadShapes() {
    const int nShapes = static_cast<int>(shapesHost_.size());
    if (nShapes == 0)
      return;
    // Composed-analytic node pool first, so the ShapeDesc pointers below can refer to it. Rebuilt
    // whole (KBs); every SHAPE_SCENE descriptor points at the pool BASE with an absolute root, so
    // one View serves every tree (the wallNodes pattern).
    if (!shapeNodesHost_.empty()) {
      P_.shapeNodes = Kokkos::View<peclet::core::geom::ShapeNode<float>*, CpMem>(
          "shapeNodes", shapeNodesHost_.size());
      auto hn = Kokkos::create_mirror_view(P_.shapeNodes);
      for (std::size_t i = 0; i < shapeNodesHost_.size(); ++i)
        hn(i) = shapeNodesHost_[i];
      Kokkos::deep_copy(P_.shapeNodes, hn);
    }
    if (static_cast<int>(P_.shapes.extent(0)) < nShapes)
      P_.shapes = Kokkos::View<ShapeDesc*, CpMem>("shapes", nShapes);
    auto hs = Kokkos::create_mirror_view(P_.shapes);
    for (int i = 0; i < nShapes; ++i) {
      hs(i) = shapesHost_[i];
      if (hs(i).type == SHAPE_SCENE) {
        hs(i).nodes = P_.shapeNodes.data();
        hs(i).nodeCount = static_cast<int>(P_.shapeNodes.extent(0));
      }
    }
    Kokkos::deep_copy(P_.shapes, hs);

    const int nPts = static_cast<int>(shellHost_.size());
    if (nPts > 0) {
      P_.shell = Kokkos::View<float* [3], CpMem>("shell", nPts);
      auto hsh = Kokkos::create_mirror_view(P_.shell);
      for (int i = 0; i < nPts; ++i) {
        hsh(i, 0) = shellHost_[i].x;
        hsh(i, 1) = shellHost_[i].y;
        hsh(i, 2) = shellHost_[i].z;
      }
      Kokkos::deep_copy(P_.shell, hsh);
    }

    if (!sdfSamplesHost_.empty()) {
      P_.sdfGrid = Kokkos::View<float*, CpMem>("sdfGrid", sdfSamplesHost_.size());
      auto hg = Kokkos::create_mirror_view(P_.sdfGrid);
      for (std::size_t i = 0; i < sdfSamplesHost_.size(); ++i)
        hg(i) = sdfSamplesHost_[i];
      Kokkos::deep_copy(P_.sdfGrid, hg);
    }

    // Broad-phase band and contact-buffer sizing must cover the LARGEST shape present, not the
    // most recently added one.
    float rmax = 0.0f;
    int shellMax = 0;
    for (int i = 0; i < nShapes; ++i) {
      rmax = std::max(rmax, baseRadiusHost_[i]);
      shellMax = std::max(shellMax, shapesHost_[i].numPoints);
    }
    baseRadius_ = rmax;
    P_.baseRadius = rmax;
    defaultInvI_ = invIHost_[0];
    shellPoints_ = shellMax;
  }

  int shellPoints_ = 0;  // LARGEST surface-shell size across shapes (contact-buffer sizing)
  std::vector<peclet::core::geom::ShapeNode<float>> shapeNodesHost_;  // SHAPE_SCENE pool
  F3 defaultInvI_{2.5f, 2.5f, 2.5f};
};

}  // namespace peclet::dem

#endif  // DEM_SHAPE_REGISTRY_HPP
