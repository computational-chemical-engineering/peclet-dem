// dem — momentum conservation of the distributed contact solve (diagnostic; the future gate).
//
// A closed, force-free granular cluster (no walls within reach, a non-periodic box much larger
// than the cluster) is advanced by the distributed step and the global invariants are summed over
// the owned bodies of every rank (host, DOUBLE, MPI_Allreduce) after each step:
//
//   P = sum m v                         linear momentum
//   X = sum m x / M                     centre of mass
//   L = sum (m x x v + I w)             angular momentum (spheres: I isotropic, I w = w / invI)
//
// Every contact impulse of an exact scheme is equal and opposite and acts along a common line, so
// P is conserved to round-off and X moves at the constant P/M. A rank-local processor-block
// Gauss-Seidel sweeps a cross-rank contact twice, once on each owner from that owner's state, and
// keeps only its own body's half: the two halves no longer cancel.
//
// The XPBD step writes the stored velocity in its VELOCITY phase only (applyVelocityAndPredict-
// Position), then projects overlaps in its POSITION phase, which moves positions without touching
// v. So the per-step record separates the two phases without instrumenting the solver:
//
//   dP     max_t |P(t) - P(0) - M g t| / sum m|v(0)|                      velocity phase
//   dX     max_t |X(t) - X(0) - V(0) t - g t^2/2| / R                    both phases
//   dXpos  max_t |sum_steps [X(n+1) - X(n) - dt (P(n) + P(n+1)) / 2M]| / R   position phase
//          (X(n+1) = X(n) + dt (P(n) + P(n+1)) / 2M exactly when the projection's position
//          corrections are mass-weighted equal and opposite)
//   dL     max_t |L(t) - L(0)| / Lscale, L about the origin              both phases
//   dLcm   the same about the moving CoM: sum m (x - X) x (v - V) + I w  both phases
//   dLvel  max_t |sum_steps sum_i [m (x_pred - X_pred) x (dv - g dt) + I dw]| / Lscale
//          velocity phase, lever arms at the predicted positions x_pred = x + (v + g dt) dt the
//          phase solves at, taken about their centre of mass X_pred
//
// R is the base radius and Lscale = sum (m |x - X| |v - V| + I |w|) at t = 0 (the angular
// momentum "content" about the centre of mass). The box is centred on the origin and so is the
// cluster (within 0.4), so L about the origin carries only a short lever arm on a linear-momentum
// defect and the float positions keep ~6 bits more than at |x| ~ 16. With gravity (cluster_pgs)
// the whole cluster falls freely -- the relative dynamics are those of g = 0, but gravity switches
// on the production PGS path (warm start, persistent contacts, friction cone) -- and dL is then
// taken about the CoM too (uniform gravity exerts no torque there).
//
// NOTE the XPBD scheme itself does not conserve L exactly even on one rank: the position phase
// moves x without touching v (sum m dx x v != 0), and friction impulses act at each body's own
// surface point, which differ by the overlap. dL at np = 1 is that intrinsic level; the velocity-
// phase dLvel is exact on one rank for frictionless spheres (central impulses).
//
// Modes (argv[1]):
//   cluster           XPBD, g = 0, frictionless: one-shot coloured GS restitution + overlap GS
//   cluster_friction  XPBD, g = 0, mu = 0.4, random initial spins (friction pass, L exercised)
//   cluster_pgs       XPBD, free fall (g != 0), mu = 0.4, spins, stabilization off: the warm-
//                     started PGS velocity solve with the friction cone (the production path)
//   cluster_posonly   XPBD, g = 0, velocity solve OFF (the dem default): the position phase alone
//   cluster_jacobi    cluster with velocityUseGS off: the legacy count-averaged Jacobi velocity
//                     and position solves, which divide each body's summed correction by its
//                     GLOBAL per-body contact count (syncContactCounts). Per-body averaging is not
//                     momentum-conserving in serial either (docs/mpi_momentum_conservation.md
//                     §4.2), so the gate is the np = 1 level, not round-off.
//   hertz             the force-based Hertz-Mindlin engine (step_hertz_mpi), mu = 0.4, spins
//   cluster_sync3     cluster_friction with sync_every = 3 (owner/ghost reconciliation every 3rd
//                     sweep)
//   cluster_norot     cluster_friction with forward_rotation = false
//   cluster_periodic  cluster in a FULLY PERIODIC box [-16, 16]^3, centred on the box corner, so
//                     wrap pairs exist on every axis (periodic self-ghost twins at np = 1, cross-
//                     rank wraps at np >= 2). Only dP is meaningful there (X and L are not
//                     conserved quantities of a periodic box): dX, dXpos and dLvel print n/a.
//   perf_gas / perf_pgs  timing only (not a ctest): N = 20000, fully periodic, ms/step
//
// REPORT-ONLY modes of docs/contact_evidence/FOLLOWUPS.md (never gated; kFollowupGate below):
//   hub               defect 1 (colour overflow): one grain of scale --hub=S (default 10, radius
//                     S R) at the cluster centre, touching a shell of ~2.5 (S+1)^2 unit grains
//                     (scale 1 +- 0.1) that move inward; g = 0, frictionless, velocity solve on.
//                     The hub's contact degree is far above 63, so the colourings give several
//                     of its contacts colour 62 (same-colour pairs at one body). Prints a HUB
//                     line: the largest per-body degree of the rank-local colouring graphs and the
//                     same-colour pairs per body (velConf / posConf, max over steps, summed over
//                     ranks; a valid colouring has 0).
//   hub_posonly       the same with the velocity solve off (the dem default).
//   friction_pair     defect 2 (legacy-friction couple): two unit spheres sliding past each other
//                     at an overlap --delta=d (in R, at the predicted positions the velocity phase
//                     solves at) with a normal approach, g = 0 (the legacy friction pass). One
//                     step; prints the measured velocity-phase angular impulse about the origin
//                     and the prediction (p_A - p_B) x J_t = -delta n x J_t. np = 1 only.
//   friction_pair_pgs the same under free fall (g != 0: the PGS friction cone, manifold midpoint).
//   cluster_friction also prints a FRIC line: the mean |dist| / R of the friction-active body-body
//                     contacts and their mean count per step.
//
// REPORT-ONLY modes of docs/contact_solve_framework.md WO-0 (never gated: tolOf returns -1; the
// framework's gates G1-G7 read their lines):
//   tri               the review scene (docs/contact_evidence/review): unit spheres A1 (-0.42,
//                     0.45), B (0.45, 0), A2 (-0.42, -0.45) in the (axis, axis+1) plane, A1 and A2
//                     moving at 1 along --axis toward B, g = 0, frictionless, 3 steps; the rank
//                     faces at 0 split A1/A2 from B
//   tri_pgs           tri under free fall (g != 0: the warm-started PGS path), --vel-iters
//   cluster_e09, cluster_e10   cluster with restitution 0.9 / 1.0
//   cluster_poisson   cluster_pgs with the Poisson restitution model
//   cluster_multilevel, cluster_escalate, cluster_ordered, cluster_onesided   cluster_pgs with that
//                     stabilization mode instead of 'off'
//   hub_pgs           hub under free fall (g != 0)
//   ring_mini         27 hollow cylinders (outer diameter 1, height 1.5, wall 0.18, unit mass
//                     through the shape registry) on a jittered 3 x 3 x 3 lattice at spacing 0.9
//                     about the cluster centre, orientations uniformly random (mt19937(11)),
//                     cluster-recipe velocities, g = 0, friction 0.02, pos/vel iterations 20/8,
//                     dt 1e-2, 10 steps: per-point position-graph degrees far above 64
// Options (after the mode): --dump=<path>, --dt=<dt>, --posit=<position iterations>,
// --hub=<scale>, --delta=<overlap / R>, --solo (single-rank demStep via Simulation::step instead
// of step_mpi; np = 1 only), --e=<normal restitution> (default 0.5; cluster_e09/_e10 set theirs),
// --steps=<n> (overrides the mode's step count), --vel-iters=<n> (velocity iterations),
// --axis=<0|1|2> (tri: the approach axis), --relabel=<seed> (seed 0 = identity; otherwise the body
// list is std::shuffle'd with std::mt19937(seed) before the gids are assigned, which samples
// another serial Gauss-Seidel order of the same physical scene).
//
// Output: parseable lines per run,
//   MOMENTUM mode=.. np=.. thr=.. N=.. steps=.. dP=.. dX=.. dXpos=.. dL=.. dLcm=.. dLvel=.. ovl=..
//   KE mode=.. np=.. thr=.. s1=.. s2=..      the centre-of-mass-frame kinetic energy (translational
//            + rotational, host double, Allreduced) after every (recorded) step
//   CONFLICTS mode=.. np=.. thr=.. vel=.. pos=.. degVel=.. degPos=.. colVel=.. colPos=..
//            (not hertz) the same-colour pairs of the rank-local velocity (manifold) and position
//            (contact) colourings of each step -- sum over (body slot, colour) of (count - 1), the
//            quantity Simulation::debugColoringConflicts counts, tallied on the host over the items
//            the step coloured -- the largest per-body-slot degree and the colour count, each the
//            maximum over steps and ranks; leftVel / leftPos count the items the step left
//            uncoloured (-1, the count-averaged fallback's set), maximum over steps and ranks
// ovl is the maximum over the steps of the Allreduce-MAX of max_overlap (the position loop's last
// residual; it under-reports the committed overlap but is measured the same way before and after).
// Angular momentum and rotational energy use the world-frame inertia R diag(1/invI) R^T of the
// orientation (x, y, z, w); for isotropic bodies (spheres) that is exactly the scalar I w.
//
// Optional second argument --dump=<path>: after the last step rank 0 writes every owned body of
// every rank, sorted by its global body index (the index into the test's body list, which is the
// same at every np), as raw records {int32 gid; float32 pos[3], vel[3], angVel[3], quat[4]} --
// the committed state (for hertz: the committed state of the force engine). Used for the bitwise
// comparisons of the conservation work (docs/mpi_momentum_conservation.md, G3/G4).
// The conservation GATE (docs/mpi_momentum_conservation.md §7 G1): every contact is owned by
// exactly one rank and ghost increments are reverse-accumulated, so the drifts are round-off at
// every np and thread count; the per-mode thresholds are in tolOf() below (a non-finite state
// always fails).
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/particle_migrator.hpp"
#include "sim.hpp"

using peclet::core::IVec;
using peclet::dem::Simulation;

// Simulation with its Particles exposed, for the report-only colouring / contact diagnostics.
struct ProbeSim : Simulation {
  using Simulation::Simulation;
  const peclet::dem::Particles& parts() const { return P_; }
  peclet::dem::Particles& parts() { return P_; }
};

// ---- the conservation gate (G1) ----
static constexpr bool kGate = true;
// ---- the FOLLOWUPS report-only modes (hub*): never fail today. When the colour overflow is
// fixed, the gate is velConf == posConf == 0 for hub / hub_posonly at every np and thread count.
// ----
static constexpr bool kFollowupGate = false;
// ---- friction_pair (FOLLOWUPS defect 2, fixed by the single application point, WO-1): the couple
// ratio |dL| / |dist n x J_t| <= 1e-2 (docs/contact_solve_framework.md §12 S1: the float floor;
// before the fix 1.000, after <= 1.3e-3). friction_pair_pgs stays a report-only reference. ----
static constexpr bool kFrictionPairGate = true;
// Per-mode thresholds (max over runs; < 0 = not gated). dLvel for the XPBD modes, dL for hertz.
struct Tol {
  double dP, dX, dXpos, dL, dLvel;
};
static Tol tolOf(const std::string& mode) {
  if (mode == "cluster")  // np 1: 2.8e-9 / 3e-7 / 3e-9; broken >= 3.3e-3
    return {1e-6, 1e-5, 1e-5, -1, 1e-6};
  if (mode == "cluster_friction")  // midpoint arms (WO-1): dLvel <= 6.4e-9 at np 1..8 (was 1.9e-5)
    return {1e-6, 1e-5, 1e-5, -1, 1e-7};
  if (mode == "cluster_sync3" || mode == "cluster_norot")
    return {1e-6, 1e-5, 1e-5, -1, 1e-4};  // the serial legacy-friction floor dLvel was 1.9e-5
  if (mode == "cluster_pgs")              // free-fall float accumulation floor dP 7.9e-7 at np 1
    return {5e-6, 1e-5, 1e-5, -1, 1e-6};
  if (mode == "cluster_posonly")  // the velocity increments are exact zeros
    return {1e-12, 1e-5, 1e-5, -1, 1e-12};
  // cluster_jacobi: mass-split Jacobi (WO-2) is conservative at every iterate: dP <= 7.6e-9,
  // dXpos <= 2.6e-7 at np 1..8. The count-averaged apply it replaced drifted (np 1: dP 9.0e-3,
  // dXpos 8.2e-4); rank-local counts gave dXpos 2.4e-2+.
  if (mode == "cluster_jacobi")
    return {1e-6, 1e-5, 1e-5, -1, 1e-6};
  if (mode == "cluster_periodic")  // dP only
    return {1e-6, -1, -1, -1, -1};
  if (mode == "hertz")  // regression guard; today 2.8e-8 / 8.8e-7 / 4.4e-7
    return {1e-6, 1e-5, -1, 1e-5, -1};
  return {-1, -1, -1, -1, -1};
}
static bool within(double v, double tol) {
  return tol < 0 || v <= tol;  // NaN fails
}

static constexpr double L = 32.0;  // closed box [-16, 16]^3; the ORB splits every axis at 0
static constexpr double LO = -16.0;
static constexpr int GX = 32;
static constexpr float RAD = 0.5f;  // base radius (scale 1)

using D3 = std::array<double, 3>;
static D3 cross(const D3& a, const D3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
static double norm(const D3& a) {
  return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

struct Body {
  float x[3], v[3], w[3], scale;
  float q[4] = {0.0f, 0.0f, 0.0f, 1.0f};  // orientation (x, y, z, w); loaded for ring_mini only
};

// Owned bodies of this rank under the equal-cell ORB (the decomposition initMpi builds).
static std::vector<int> ownedOf(const std::vector<Body>& b, double lo, double box, int gx,
                                bool periodic, int rank, int size) {
  peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{gx, gx, gx});
  peclet::core::halo::DomainMap<3> map;
  for (int i = 0; i < 3; ++i) {
    map.origin[i] = lo;
    map.cellSize[i] = box / gx;
    map.periodic[i] = periodic;
  }
  peclet::core::halo::ParticleMigrator<3> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);
  std::vector<int> gids;
  for (int g = 0; g < static_cast<int>(b.size()); ++g)
    if (mig.ownerOf(peclet::core::Vec<3>{b[g].x[0], b[g].x[1], b[g].x[2]}) == rank)
      gids.push_back(g);
  return gids;
}

// Polydisperse spheres: mass ~ scale^3, isotropic inertia 2/5 m (scale RAD)^2.
static float invMassOf(float s) {
  return 1.0f / (s * s * s);
}
static float invInertiaOf(float s) {
  return 1.0f / (0.4f * s * s * s * (s * RAD) * (s * RAD));
}

static void load(Simulation& sim, const std::vector<Body>& b, const std::vector<int>& gids) {
  std::vector<float> p, v, w, s, im, ii;
  for (int g : gids) {
    p.insert(p.end(), {b[g].x[0], b[g].x[1], b[g].x[2]});
    v.insert(v.end(), {b[g].v[0], b[g].v[1], b[g].v[2]});
    w.insert(w.end(), {b[g].w[0], b[g].w[1], b[g].w[2]});
    s.push_back(b[g].scale);
    im.push_back(invMassOf(b[g].scale));
    const float i = invInertiaOf(b[g].scale);
    ii.insert(ii.end(), {i, i, i});
  }
  sim.setPositions(p);  // first: resets every per-body field to its default
  sim.setScales(s);
  sim.setVelocities(v);
  sim.setAngularVelocities(w);
  sim.setInvMass(im);
  sim.setInvInertia(ii);
}

// Non-spherical bodies: the shape registry's own mass and inertia (setPositions stamps unit mass
// and shape 0's inverse inertia), plus the orientations.
static void loadShaped(Simulation& sim, const std::vector<Body>& b, const std::vector<int>& gids) {
  std::vector<float> p, v, w, s, q;
  for (int g : gids) {
    p.insert(p.end(), {b[g].x[0], b[g].x[1], b[g].x[2]});
    v.insert(v.end(), {b[g].v[0], b[g].v[1], b[g].v[2]});
    w.insert(w.end(), {b[g].w[0], b[g].w[1], b[g].w[2]});
    s.push_back(b[g].scale);
    q.insert(q.end(), {b[g].q[0], b[g].q[1], b[g].q[2], b[g].q[3]});
  }
  sim.setPositions(p);
  sim.setScales(s);
  sim.setVelocities(v);
  sim.setAngularVelocities(w);
  sim.setQuaternions(q);
}

// A dense random cluster: jittered cubic lattice (spacing = base diameter, so the polydisperse
// grains start overlapping) inside a ball centred just off the origin, the common corner of every
// ORB layout,
// random velocities + a net drift + an inward radial component (keeps it colliding), optional
// random spins. Deterministic (fixed seed); every rank builds the same global set.
// `periodic`: centre the ball on the box corner (-16, -16, -16) instead and wrap every position
// into [-16, 16), so the cluster straddles every periodic face.
static std::vector<Body> makeCluster(double ballRadius, bool spins, bool periodic = false) {
  std::mt19937 rng(20260925u);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  const D3 c = periodic ? D3{LO, LO, LO} : D3{0.3, -0.2, 0.1};
  const float drift[3] = {0.7f, -0.4f, 0.3f};
  const double h = 2.0 * RAD;
  const int n = static_cast<int>(std::ceil(ballRadius / h));
  std::vector<Body> b;
  for (int k = -n; k <= n; ++k)
    for (int j = -n; j <= n; ++j)
      for (int i = -n; i <= n; ++i) {
        const D3 r{i * h, j * h, k * h};
        const double rr = norm(r);
        if (rr > ballRadius)
          continue;
        Body q{};
        for (int d = 0; d < 3; ++d) {
          q.x[d] = static_cast<float>(c[d] + r[d] + 0.05 * uni(rng));
          q.v[d] = drift[d] + 1.5f * gauss(rng) - static_cast<float>(1.0 * r[d] / ballRadius);
          q.w[d] = spins ? 3.0f * gauss(rng) : 0.0f;
        }
        q.scale = 1.0f + 0.1f * uni(rng);
        if (periodic)
          for (int d = 0; d < 3; ++d) {
            double x = q.x[d];
            if (x < LO)
              x += L;
            if (x >= LO + L)
              x -= L;
            q.x[d] = static_cast<float>(x);
          }
        b.push_back(q);
      }
  return b;
}

// Defect 1 scene (FOLLOWUPS.md): a grain of scale `hubScale` at the cluster centre with a shell of
// unit grains on Fibonacci points, each overlapping the hub by ~0.02-0.07, moving inward at
// 1 + 0.3 N(0,1) on top of a common drift. Deterministic; every rank builds the same set.
static std::vector<Body> makeHub(float hubScale) {
  std::mt19937 rng(20260926u);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  const D3 c{0.3, -0.2, 0.1};
  const float drift[3] = {0.7f, -0.4f, 0.3f};
  const double rs = hubScale * RAD + RAD - 0.02;  // shell centre distance
  const int ns = static_cast<int>(2.5 * (rs / RAD) * (rs / RAD));
  std::vector<Body> b;
  Body h{};
  for (int d = 0; d < 3; ++d) {
    h.x[d] = static_cast<float>(c[d]);
    h.v[d] = drift[d];
  }
  h.scale = hubScale;
  b.push_back(h);
  const double golden = 3.14159265358979 * (3.0 - std::sqrt(5.0));
  for (int k = 0; k < ns; ++k) {
    const double z = 1.0 - 2.0 * (k + 0.5) / ns, r = std::sqrt(1.0 - z * z), ph = golden * k;
    const D3 dir{r * std::cos(ph), r * std::sin(ph), z};
    const float speed = 1.0f + 0.3f * gauss(rng);
    Body q{};
    for (int d = 0; d < 3; ++d) {
      q.x[d] = static_cast<float>(c[d] + rs * dir[d]);
      q.v[d] = drift[d] - speed * static_cast<float>(dir[d]);
    }
    q.scale = 1.0f + 0.1f * uni(rng);
    b.push_back(q);
  }
  return b;
}

// The review's tri scene (docs/contact_evidence/review): A1 and A2 approach B along `axis` at 1;
// gids 0 (A1), 1 (B), 2 (A2). The ORB faces at 0 put A1/A2 and B on different ranks at np >= 2.
static std::vector<Body> makeTri(int axis) {
  const int k = axis, t = (k + 1) % 3;
  auto mk = [&](float a, float b) {
    Body q{};
    q.x[k] = a;
    q.x[t] = b;
    q.scale = 1.0f;
    return q;
  };
  Body A1 = mk(-0.42f, 0.45f), B = mk(0.45f, 0.0f), A2 = mk(-0.42f, -0.45f);
  A1.v[k] = A2.v[k] = 1.0f;
  return {A1, B, A2};
}

// ring_mini: hollow cylinders (outer diameter 1, height 1.5, wall 0.18) on a 3 x 3 x 3 lattice at
// spacing 0.9 about the cluster centre, jittered by 0.05 uniform per axis; velocities by the
// makeCluster recipe (drift + 1.5 N(0,1) - r / r_max, r the lattice offset, r_max its corner
// distance), no spins; positions and velocities from makeCluster's stream, the orientations
// (uniformly random unit quaternions: normalised 4-D Gaussians) from std::mt19937(11).
static constexpr float kRingD = 1.0f, kRingH = 1.5f, kRingWall = 0.18f, kRingSpacing = 0.9f;
static std::vector<Body> makeRingMini() {
  std::mt19937 rng(20260925u), qrng(11u);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::normal_distribution<float> gauss(0.0f, 1.0f), qgauss(0.0f, 1.0f);
  const D3 c{0.3, -0.2, 0.1};
  const float drift[3] = {0.7f, -0.4f, 0.3f};
  const double h = kRingSpacing, rMax = h * std::sqrt(3.0);
  std::vector<Body> b;
  for (int k = -1; k <= 1; ++k)
    for (int j = -1; j <= 1; ++j)
      for (int i = -1; i <= 1; ++i) {
        const D3 r{i * h, j * h, k * h};
        Body q{};
        for (int d = 0; d < 3; ++d) {
          q.x[d] = static_cast<float>(c[d] + r[d] + 0.05 * kRingD * uni(rng));
          q.v[d] = drift[d] + 1.5f * gauss(rng) - static_cast<float>(r[d] / rMax);
        }
        q.scale = 1.0f;
        float nn = 0.0f;
        for (float& e : q.q) {
          e = qgauss(qrng);
          nn += e * e;
        }
        nn = std::sqrt(nn);
        for (float& e : q.q)
          e /= nn;
        b.push_back(q);
      }
  return b;
}

// Rank-local colouring diagnostics of the last step: over the contacts / manifolds the step
// coloured (colour >= -1; the caller filled both colour arrays with -3 before the step, so the
// entries the step did not colour -- the non-owned visible ones under MPI -- are skipped), the
// largest per-body degree and the same-colour pairs, sum over (body, colour) of (count - 1).
struct ColorDiag {
  int degC = 0, degM = 0, confC = 0, confM = 0;
  int degPt = 0;             // largest per-body degree of the per-POINT contact graph
  int colC = 0, colM = 0;    // colours used (largest colour + 1)
  int leftC = 0, leftM = 0;  // items the step left uncoloured (-1): the count-averaged fallback
};
static void markColors(ProbeSim& sim) {
  Kokkos::deep_copy(sim.parts().contactColor, -3);
  Kokkos::deep_copy(sim.parts().unitColor, -3);
  Kokkos::deep_copy(sim.parts().manifoldColor, -3);
}
static ColorDiag colorDiag(const ProbeSim& sim) {
  const peclet::dem::Particles& P = sim.parts();
  int nc = 0, nm = 0;
  Kokkos::deep_copy(nc, P.contactCount);
  Kokkos::deep_copy(nm, P.manifoldCount);
  auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.contacts);
  auto hcc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.contactColor);
  auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.manifolds);
  auto hmc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.manifoldColor);
  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.realIndices);
  ColorDiag g;
  auto tally = [](std::vector<std::pair<int, int>>& e, int& deg, int& conf, int& col, int& left) {
    std::sort(e.begin(), e.end());
    std::vector<int> d;
    for (std::size_t i = 0; i < e.size(); ++i) {
      const int b = e[i].first;
      if (b >= static_cast<int>(d.size()))
        d.resize(b + 1, 0);
      ++d[b];
      if (i > 0 && e[i].second >= 0 && e[i] == e[i - 1])
        ++conf;
      col = std::max(col, e[i].second + 1);
    }
    left = 0;
    for (const auto& x : e)
      left += x.second == -1;
    for (int x : d)
      deg = std::max(deg, x);
  };
  std::vector<std::pair<int, int>> ec, em, ep;  // (body slot, colour)
  // Position graph = the position units (docs/contact_solve_framework.md §4.3): one edge per unit,
  // the leader contact's bodies; ep keeps the per-point graph for its degree only.
  const int nu = P.numPosUnits;
  auto hus = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.unitStart);
  auto hul = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.unitContacts);
  auto huc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.unitColor);
  for (int u = 0; u < nu; ++u) {
    const int col = huc(u);
    if (col < -1)
      continue;
    const int i = hul(hus(u));
    ec.push_back({hc(i).bodyA, col});
    if (hc(i).bodyB >= 0)
      ec.push_back({hc(i).bodyB, col});
  }
  for (int i = 0; i < nc; ++i) {
    const int col = hcc(i);
    if (col < -1)
      continue;
    ep.push_back({hc(i).bodyA, col});
    if (hc(i).bodyB >= 0)
      ep.push_back({hc(i).bodyB, col});
  }
  for (int i = 0; i < nm; ++i) {
    const int col = hmc(i);
    if (col < -1)
      continue;
    em.push_back({hr(hm(i).bodyA), col});
    if (hm(i).bodyB >= 0)
      em.push_back({hr(hm(i).bodyB), col});
  }
  tally(ec, g.degC, g.confC, g.colC, g.leftC);
  {
    int c0 = 0, c1 = 0, l0 = 0;
    tally(ep, g.degPt, c0, c1, l0);
  }
  tally(em, g.degM, g.confM, g.colM, g.leftM);
  return g;
}

struct Sums {
  double m = 0, P[3] = {0, 0, 0}, mx[3] = {0, 0, 0}, Lo[3] = {0, 0, 0};
};
struct State {
  std::vector<float> x, v, w, m, invI, q;
};
static State readState(const Simulation& s) {
  State st{s.getPositions(), s.getVelocities(), s.getAngularVelocities(),
           s.getMasses(),    s.getInvInertia(), s.getQuaternions()};
  return st;
}
static D3 at(const std::vector<float>& a, int i) {
  return {a[3 * i], a[3 * i + 1], a[3 * i + 2]};
}
// World-frame angular momentum I w of body i of `orient` (its inverse body inertia and
// orientation) spinning at w. Isotropic inertia (spheres): the scalar w / invI, bit for bit the
// historical formula. Otherwise R diag(1/invI) R^T w with R the rotation of the unit quaternion
// (x, y, z, w) -- the frame the solver applies invInertia in (rotateVector).
static D3 spinOf(const State& orient, int i, const D3& w) {
  const double i0 = orient.invI[3 * i], i1 = orient.invI[3 * i + 1], i2 = orient.invI[3 * i + 2];
  if (i0 == i1 && i1 == i2)
    return i0 > 0 ? D3{w[0] / i0, w[1] / i0, w[2] / i0} : D3{0, 0, 0};
  const double qx = orient.q[4 * i], qy = orient.q[4 * i + 1], qz = orient.q[4 * i + 2],
               qw = orient.q[4 * i + 3];
  const double R[3][3] = {
      {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)},
      {2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)},
      {2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)}};
  const double inv[3] = {i0, i1, i2};
  D3 b{0, 0, 0}, out{0, 0, 0};
  for (int k = 0; k < 3; ++k) {  // body frame: R^T w, scaled by the body inertia
    for (int d = 0; d < 3; ++d)
      b[k] += R[d][k] * w[d];
    b[k] = inv[k] > 0 ? b[k] / inv[k] : 0.0;
  }
  for (int d = 0; d < 3; ++d)
    for (int k = 0; k < 3; ++k)
      out[d] += R[d][k] * b[k];
  return out;
}
static D3 spin(const State& st, int i) {  // I w
  return spinOf(st, i, at(st.w, i));
}

// Global sums over owned bodies (host double, Allreduce).
static Sums globalSums(const State& st) {
  double loc[10] = {0};
  const int n = static_cast<int>(st.m.size());
  for (int i = 0; i < n; ++i) {
    const double m = st.m[i];
    const D3 x = at(st.x, i), v = at(st.v, i);
    const D3 xv = cross(x, v), s = spin(st, i);
    loc[0] += m;
    for (int d = 0; d < 3; ++d) {
      loc[1 + d] += m * v[d];
      loc[4 + d] += m * x[d];
      loc[7 + d] += m * xv[d] + s[d];
    }
  }
  double g[10];
  MPI_Allreduce(loc, g, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  Sums S;
  S.m = g[0];
  for (int d = 0; d < 3; ++d) {
    S.P[d] = g[1 + d];
    S.mx[d] = g[4 + d];
    S.Lo[d] = g[7 + d];
  }
  return S;
}
// Angular momentum about a point X moving at V: sum m (x - X) x (v - V) + I w.
static D3 angularAbout(const State& st, const D3& X, const D3& V) {
  double loc[3] = {0, 0, 0};
  for (int i = 0; i < static_cast<int>(st.m.size()); ++i) {
    const D3 x = at(st.x, i), v = at(st.v, i), s = spin(st, i);
    const D3 r{x[0] - X[0], x[1] - X[1], x[2] - X[2]}, u{v[0] - V[0], v[1] - V[1], v[2] - V[2]};
    const D3 c = cross(r, u);
    for (int d = 0; d < 3; ++d)
      loc[d] += st.m[i] * c[d] + s[d];
  }
  D3 g;
  MPI_Allreduce(loc, g.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}
// The velocity phase's angular impulse of one step about X_pred: sum m (x_pred - X_pred) x
// (dv - g dt) + I dw, with the lever arms at the predicted positions the phase solves at. Local
// order is fixed (no migration).
static D3 velocityPhaseTorque(const State& a, const State& b, const D3& g, double dt,
                              const D3& Xpred) {
  double loc[3] = {0, 0, 0};
  for (int i = 0; i < static_cast<int>(a.m.size()); ++i) {
    const D3 x = at(a.x, i), v0 = at(a.v, i), v1 = at(b.v, i);
    D3 xp, dv;
    for (int d = 0; d < 3; ++d) {
      xp[d] = x[d] + (v0[d] + g[d] * dt) * dt - Xpred[d];
      dv[d] = v1[d] - v0[d] - g[d] * dt;
    }
    // Both at the orientation the phase solves at (frozen in the velocity phase).
    const D3 c = cross(xp, dv), s0 = spin(a, i), s1 = spinOf(a, i, at(b.w, i));
    for (int d = 0; d < 3; ++d)
      loc[d] += a.m[i] * c[d] + (s1[d] - s0[d]);
  }
  D3 out;
  MPI_Allreduce(loc, out.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return out;
}

struct Mode {
  std::string name;
  bool friction = false, spins = false, gravity = false, hertz = false, periodic = false;
  bool jacobi = false;  // velocityUseGS off: the legacy count-averaged Jacobi solves
  int velIters = 8;
  int syncEvery = 1;
  bool forwardRotation = true;
  std::string dump;  // --dump=<path>: final-state dump (empty = none)
  // FOLLOWUPS report-only options
  float hubScale = 0.0f;  // > 0: the hub scene (makeHub) instead of the cluster
  bool solo = false;      // --solo: Simulation::step (single-rank demStep), np = 1
  float dt = 0.0f;        // --dt: override the mode's dt (steps scaled to keep the duration)
  int posIters = 20;      // --posit
  float delta = 0.05f;    // --delta (friction_pair): overlap / R
  // WO-0 report-only options (docs/contact_solve_framework.md §8)
  float restitution = 0.5f;  // --e
  int steps = 0;             // --steps: > 0 overrides the mode's step count
  int axis = 0;              // --axis (tri)
  unsigned relabel = 0;      // --relabel: 0 = identity
  bool tri = false, ring = false, poisson = false;
  bool reportOnly = false;  // the WO-0 modes: never fail today
  std::string stab;         // stabilization mode set after the gravity rule's 'off' (empty = keep)
};

// Gather every rank's owned bodies to rank 0 and write them sorted by global body index (see the
// file note for the record layout).
static void dumpState(const Simulation& sim, const std::vector<int>& gids, const std::string& path,
                      int rank, int size) {
  constexpr int kRec = 14;  // int32 gid + 13 float32
  const std::vector<float> x = sim.getPositions(), v = sim.getVelocities(),
                           w = sim.getAngularVelocities(), q = sim.getQuaternions();
  const int n = static_cast<int>(gids.size());
  std::vector<std::uint32_t> loc(static_cast<std::size_t>(n) * kRec);
  for (int i = 0; i < n; ++i) {
    std::uint32_t* r = &loc[static_cast<std::size_t>(i) * kRec];
    const std::int32_t g = gids[i];
    std::memcpy(&r[0], &g, 4);
    std::memcpy(&r[1], &x[3 * i], 12);
    std::memcpy(&r[4], &v[3 * i], 12);
    std::memcpy(&r[7], &w[3 * i], 12);
    std::memcpy(&r[10], &q[4 * i], 16);
  }
  const int cnt = n * kRec;
  std::vector<int> cnts(size), offs(size, 0);
  MPI_Gather(&cnt, 1, MPI_INT, cnts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  int tot = 0;
  if (rank == 0)
    for (int r = 0; r < size; ++r) {
      offs[r] = tot;
      tot += cnts[r];
    }
  std::vector<std::uint32_t> all(rank == 0 ? static_cast<std::size_t>(tot) : 0);
  MPI_Gatherv(loc.data(), cnt, MPI_UINT32_T, all.data(), cnts.data(), offs.data(), MPI_UINT32_T, 0,
              MPI_COMM_WORLD);
  if (rank != 0)
    return;
  const int nrec = tot / kRec;
  std::vector<int> order(nrec);
  for (int k = 0; k < nrec; ++k)
    order[k] = k;
  auto gidOf = [&](int k) {
    std::int32_t g;
    std::memcpy(&g, &all[static_cast<std::size_t>(k) * kRec], 4);
    return g;
  };
  std::sort(order.begin(), order.end(), [&](int a, int b) { return gidOf(a) < gidOf(b); });
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    return;
  }
  for (int k : order)
    std::fwrite(&all[static_cast<std::size_t>(k) * kRec], 4, kRec, f);
  std::fclose(f);
}

static int runCluster(const Mode& md, int rank, int size) {
  const bool hub = md.hubScale > 0.0f;
  std::vector<Body> bodies = md.tri    ? makeTri(md.axis)
                             : md.ring ? makeRingMini()
                             : hub     ? makeHub(md.hubScale)
                                       : makeCluster(6.0, md.spins, md.periodic);
  if (md.relabel != 0) {  // another serial order of the same scene: permute before the gids
    std::mt19937 perm(md.relabel);
    std::shuffle(bodies.begin(), bodies.end(), perm);
  }
  const int n = static_cast<int>(bodies.size());
  const float dt0 = md.hertz ? 1e-4f : 1e-2f;
  const float dt = md.dt > 0.0f ? md.dt : dt0;
  const int steps0 = md.hertz ? 40 : (hub ? 20 : (md.tri ? 3 : (md.ring ? 10 : 50)));
  const int steps =
      md.steps > 0 ? md.steps : static_cast<int>(std::lround(steps0 * dt0 / dt));  // same duration
  const int sub = md.hertz ? 25 : 1;  // Hertz: substeps per recorded step
  const D3 g = md.gravity ? D3{0.0, 0.0, -10.0} : D3{0, 0, 0};

  const bool per3 = md.periodic;
  std::vector<int> gids;
  if (md.solo)
    for (int i = 0; i < n; ++i)
      gids.push_back(i);
  else
    gids = ownedOf(bodies, LO, L, GX, per3, rank, size);
  // A periodic box carries a ghost layer on every face the cluster straddles.
  ProbeSim sim(per3 ? 4 * n + 64 : 2 * n + 64);
  sim.setDomain(L, L, L, per3, per3, per3);
  sim.setDomainMinMax(peclet::dem::F3{-16.0f, -16.0f, -16.0f},
                      peclet::dem::F3{16.0f, 16.0f, 16.0f});
  sim.setGlobalScale(1.0f);
  if (md.ring)  // outer radius D/2; unit mass, the shape's own inertia (setPositions stamps both)
    sim.initializeShape(peclet::dem::HOLLOW_CYLINDER, 0.5f * kRingD, kRingH, kRingWall);
  else
    sim.setSphereShape(RAD);
  sim.setDt(dt);
  sim.setGravity(static_cast<float>(g[0]), static_cast<float>(g[1]), static_cast<float>(g[2]));
  sim.setSolverIterations(md.posIters, md.velIters);
  sim.setMaterialParams(md.restitution, 0.0f, md.ring ? 0.02f : (md.friction ? 0.4f : 0.0f));
  if (md.jacobi)
    sim.setVelocityUseGS(false);
  if (md.gravity)
    sim.setStabilizationMode("off");  // one-sided stabilization is a momentum sink by design
  if (!md.stab.empty())
    sim.setStabilizationMode(md.stab);
  if (md.poisson)
    sim.setRestitutionModel("poisson");
  if (md.hertz) {
    sim.setHertzMaterial(0, 1.0e5f, 0.25f);
  }
  if (md.ring)
    loadShaped(sim, bodies, gids);
  else
    load(sim, bodies, gids);
  if (!md.solo) {
    const std::tuple<double, double, double> origin{LO, LO, LO}, dsize{L, L, L};
    const std::tuple<long, long, long> gsize{GX, GX, GX};
    const std::tuple<bool, bool, bool> per{per3, per3, per3};
    sim.initMpi(origin, dsize, gsize, per, MPI_COMM_WORLD);
    sim.enableMpiStep(0.0, md.syncEvery, md.forwardRotation);
  }
  ColorDiag cd;                // max over steps (rank-local); every XPBD mode
  std::vector<double> keHist;  // CoM-frame kinetic energy after every (recorded) step
  double fricDist =
      0.0;  // cluster_friction: sum over steps of mean |dist| / R of friction contacts
  long fricCount = 0, fricGap = 0;

  State st = readState(sim);
  const Sums S0 = globalSums(st);
  const double M = S0.m;
  const D3 X0{S0.mx[0] / M, S0.mx[1] / M, S0.mx[2] / M}, V0{S0.P[0] / M, S0.P[1] / M, S0.P[2] / M};
  double pScaleLoc = 0.0, lScaleLoc = 0.0;
  for (int i = 0; i < static_cast<int>(st.m.size()); ++i) {
    const D3 x = at(st.x, i), v = at(st.v, i);
    const D3 r{x[0] - X0[0], x[1] - X0[1], x[2] - X0[2]},
        u{v[0] - V0[0], v[1] - V0[1], v[2] - V0[2]};
    pScaleLoc += st.m[i] * norm(v);
    lScaleLoc += st.m[i] * norm(r) * norm(u) + norm(spin(st, i));
  }
  double pScale = 0, lScale = 0;
  MPI_Allreduce(&pScaleLoc, &pScale, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&lScaleLoc, &lScale, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // L about the CoM; about the origin with g = 0 (under free fall the CoM one again: no gravity
  // torque there).
  auto angularCm = [&](const State& s, const Sums& S) {
    return angularAbout(s, D3{S.mx[0] / M, S.mx[1] / M, S.mx[2] / M},
                        D3{S.P[0] / M, S.P[1] / M, S.P[2] / M});
  };
  auto angular = [&](const State& s, const Sums& S) {
    return md.gravity ? angularCm(s, S) : D3{S.Lo[0], S.Lo[1], S.Lo[2]};
  };
  const D3 Lstart = angular(st, S0), LcmStart = angularCm(st, S0);

  double dP = 0, dX = 0, dXpos = 0, dL = 0, dLcm = 0, dLvel = 0, ovl = 0;
  D3 xposAcc{0, 0, 0}, lvelAcc{0, 0, 0};
  Sums Sprev = S0;
  int fail = 0;
  const int nOwned = static_cast<int>(gids.size());
  for (int s = 1; s <= steps; ++s) {
    if (!md.hertz)
      markColors(sim);  // only the items the step colours carry a colour >= -1 afterwards
    if (md.hertz)
      sim.stepHertzMpi(sub, 0.3f);
    else if (md.solo)
      sim.step(1);
    else
      sim.stepMpi(1);
    if (!md.hertz) {
      const ColorDiag c = colorDiag(sim);
      cd.degC = std::max(cd.degC, c.degC);
      cd.degM = std::max(cd.degM, c.degM);
      cd.confC = std::max(cd.confC, c.confC);
      cd.confM = std::max(cd.confM, c.confM);
      cd.colC = std::max(cd.colC, c.colC);
      cd.colM = std::max(cd.colM, c.colM);
      cd.leftC = std::max(cd.leftC, c.leftC);
      cd.leftM = std::max(cd.leftM, c.leftM);
      cd.degPt = std::max(cd.degPt, c.degPt);
    }
    if (md.friction &&
        !md.hertz) {  // friction-active body-body contacts (their |dist| = lever gap)
      const peclet::dem::Particles& P = sim.parts();
      int nc = 0;
      Kokkos::deep_copy(nc, P.contactCount);
      auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.contacts);
      double sd = 0.0;
      int k = 0, kg = 0;
      for (int i = 0; i < nc; ++i)
        if (hc(i).bodyB >= 0 && hc(i).friction_lambda_n > 0.0f) {
          sd += std::fabs(hc(i).dist);
          ++k;
          kg += hc(i).dist > 0.0f;  // a gap (speculative) contact inside the broad-phase margin
        }
      if (k > 0)
        fricDist += sd / k / RAD;
      fricCount += k;
      fricGap += kg;
    }
    {
      float o = sim.maxOverlap(), og = o;
      MPI_Allreduce(&o, &og, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      ovl = std::max(ovl, static_cast<double>(og));
    }
    const State nx = readState(sim);
    if (static_cast<int>(nx.m.size()) != nOwned)
      fail = 1;  // the per-body velocity-phase record assumes fixed ownership
    const Sums S = globalSums(nx);
    {  // kinetic energy in the CoM frame (translational + rotational), host double
      double kl = 0.0, kg = 0.0;
      for (int i = 0; i < static_cast<int>(nx.m.size()); ++i) {
        const D3 v = at(nx.v, i), w = at(nx.w, i), sp = spin(nx, i);
        kl += 0.5 * nx.m[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) +
              0.5 * (sp[0] * w[0] + sp[1] * w[1] + sp[2] * w[2]);
      }
      MPI_Allreduce(&kl, &kg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      kg -= 0.5 * (S.P[0] * S.P[0] + S.P[1] * S.P[1] + S.P[2] * S.P[2]) / M;
      keHist.push_back(kg);
    }
    const double t = static_cast<double>(s) * sub * dt;
    D3 ep, ex, el, ec;
    const D3 Ln = angular(nx, S), Lc = angularCm(nx, S);
    for (int d = 0; d < 3; ++d) {
      ep[d] = S.P[d] - S0.P[d] - M * g[d] * t;
      ex[d] = S.mx[d] / M - X0[d] - V0[d] * t - 0.5 * g[d] * t * t;
      el[d] = Ln[d] - Lstart[d];
      ec[d] = Lc[d] - LcmStart[d];
      xposAcc[d] += (S.mx[d] - Sprev.mx[d]) / M - dt * (S.P[d] + Sprev.P[d]) / (2.0 * M);
    }
    if (!md.hertz) {
      D3 Xpred;  // CoM of the predicted positions x + (v + g dt) dt
      for (int d = 0; d < 3; ++d)
        Xpred[d] = (Sprev.mx[d] + (Sprev.P[d] + M * g[d] * dt) * dt) / M;
      const D3 tv = velocityPhaseTorque(st, nx, g, dt, Xpred);
      for (int d = 0; d < 3; ++d)
        lvelAcc[d] += tv[d];
      dXpos = std::max(dXpos, norm(xposAcc) / RAD);
      dLvel = std::max(dLvel, norm(lvelAcc) / lScale);
    }
    dP = std::max(dP, norm(ep) / pScale);
    dX = std::max(dX, norm(ex) / RAD);
    dL = std::max(dL, norm(el) / lScale);
    dLcm = std::max(dLcm, norm(ec) / lScale);
    for (float v : nx.x)
      if (!std::isfinite(v))
        fail = 1;
    st = nx;
    Sprev = S;
  }
  int ghosts = md.solo ? 0 : sim.numGhost(), totGhost = 0;
  MPI_Allreduce(&ghosts, &totGhost, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  const int thr = Kokkos::DefaultHostExecutionSpace().concurrency();
  if (hub) {
    int loc[4] = {cd.degC, cd.degM, cd.confC, cd.confM}, mx[2], sm[2];
    MPI_Allreduce(loc, mx, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(loc + 2, sm, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf(
          "HUB mode=%s np=%d thr=%d solo=%d hubScale=%.1f N=%d maxDegContacts=%d "
          "maxDegManifolds=%d posConf=%d velConf=%d\n",
          md.name.c_str(), size, thr, md.solo ? 1 : 0, md.hubScale, n, mx[0], mx[1], sm[0], sm[1]);
    if (kFollowupGate && (sm[0] > 0 || sm[1] > 0))
      fail = 1;
  }
  if (!md.hertz) {  // max over steps (above) and ranks
    // Position columns count UNITS (degPos = units per body, leftPos = uncoloured units);
    // degPosPt is the per-point contact degree (the pre-unit degPos).
    const int loc[9] = {cd.confM, cd.confC, cd.degM,  cd.degC, cd.colM,
                        cd.colC,  cd.leftM, cd.leftC, cd.degPt};
    int mx[9];
    MPI_Allreduce(loc, mx, 9, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf(
          "CONFLICTS mode=%s np=%d thr=%d vel=%d pos=%d degVel=%d degPos=%d colVel=%d "
          "colPos=%d leftVel=%d leftPos=%d degPosPt=%d\n",
          md.name.c_str(), size, thr, mx[0], mx[1], mx[2], mx[3], mx[4], mx[5], mx[6], mx[7],
          mx[8]);
  }
  if (rank == 0) {
    std::printf("KE mode=%s np=%d thr=%d", md.name.c_str(), size, thr);
    for (std::size_t k = 0; k < keHist.size(); ++k)
      std::printf(" s%zu=%.9e", k + 1, keHist[k]);
    std::printf("\n");
  }
  if (md.friction && !md.hertz && rank == 0 && size == 1)
    std::printf(
        "FRIC mode=%s dt=%.4g posit=%d meanFricDist/R=%.3e meanFricContacts=%.1f "
        "gapFraction=%.3f\n",
        md.name.c_str(), dt, md.posIters, fricDist / steps, static_cast<double>(fricCount) / steps,
        fricCount > 0 ? static_cast<double>(fricGap) / fricCount : 0.0);
  if (rank == 0) {
    if (md.hertz)
      std::printf(
          "MOMENTUM mode=%s np=%d thr=%d N=%d steps=%d dP=%.3e dX=%.3e dXpos=n/a dL=%.3e "
          "dLcm=%.3e dLvel=n/a ghosts=%d ovl=%.3e\n",
          md.name.c_str(), size, thr, n, steps * sub, dP, dX, dL, dLcm, totGhost, ovl);
    else if (md.periodic)
      std::printf(
          "MOMENTUM mode=%s np=%d thr=%d N=%d steps=%d dP=%.3e dX=n/a dXpos=n/a dL=%.3e "
          "dLcm=%.3e dLvel=n/a ghosts=%d ovl=%.3e\n",
          md.name.c_str(), size, thr, n, steps, dP, dL, dLcm, totGhost, ovl);
    else
      std::printf(
          "MOMENTUM mode=%s np=%d thr=%d N=%d steps=%d dP=%.3e dX=%.3e dXpos=%.3e dL=%.3e "
          "dLcm=%.3e dLvel=%.3e ghosts=%d ovl=%.3e\n",
          md.name.c_str(), size, thr, n, steps, dP, dX, dXpos, dL, dLcm, dLvel, totGhost, ovl);
  }
  if (!md.dump.empty())
    dumpState(sim, gids, md.dump, rank, size);
  const Tol tol = tolOf(md.name);
  if (kGate && !(within(dP, tol.dP) && within(dX, tol.dX) && within(dXpos, tol.dXpos) &&
                 within(dL, tol.dL) && within(dLvel, tol.dLvel))) {
    fail = 1;
    if (rank == 0)
      std::fprintf(stderr, "GATE: %s exceeds its conservation thresholds\n", md.name.c_str());
  }
  if ((hub || md.reportOnly) && !kFollowupGate)
    fail = 0;  // report-only (a non-finite state is printed above, not failed)
  return fail;
}

// ---- FOLLOWUPS defect 2: the legacy friction pass's couple on one sliding pair (np = 1) ----
// Unit spheres A (gid 0) and B (gid 1) on the x axis, approaching along x at vn and sliding along
// y at vt, placed so that at the predicted positions x + v dt (where the narrow phase and the
// velocity phase work) they overlap by delta. One step of Simulation::step. The friction impulse
// J_t acts at p_A on A and at p_B = p_A - dist n on B (dist = -delta, n from B to A), so the
// velocity phase changes the angular momentum by (p_A - p_B) x J_t = -delta n x J_t. The normal
// impulse is central (the manifold carries the common midpoint), so this couple is the whole
// velocity-phase dL. friction_pair_pgs runs the same under free fall: the PGS cone, which applies
// J_t at the manifold midpoint on both bodies.
static int runFrictionPair(const Mode& md, int rank, int size) {
  if (size != 1) {
    if (rank == 0)
      std::printf("FRICPAIR mode=%s: np = 1 only, skipped\n", md.name.c_str());
    return 0;
  }
  const float dt = md.dt > 0.0f ? md.dt : 1e-2f;
  const float vn = 0.2f, vt = 1.0f, mu = 0.4f;
  const D3 g = md.gravity ? D3{0.0, 0.0, -10.0} : D3{0, 0, 0};
  const double delta = md.delta * RAD;
  const double d0 = 2.0 * RAD - delta + vn * dt;  // separation now; 2R - delta after the predict
  std::vector<Body> b(2);
  b[0] = Body{
      {static_cast<float>(-0.5 * d0), 0.0f, 0.0f}, {0.5f * vn, 0.5f * vt, 0.0f}, {0, 0, 0}, 1.0f};
  b[1] = Body{
      {static_cast<float>(0.5 * d0), 0.0f, 0.0f}, {-0.5f * vn, -0.5f * vt, 0.0f}, {0, 0, 0}, 1.0f};
  ProbeSim sim(64);
  sim.setDomain(L, L, L, false, false, false);
  sim.setDomainMinMax(peclet::dem::F3{-16.0f, -16.0f, -16.0f},
                      peclet::dem::F3{16.0f, 16.0f, 16.0f});
  sim.setGlobalScale(1.0f);
  sim.setSphereShape(RAD);
  sim.setDt(dt);
  sim.setGravity(static_cast<float>(g[0]), static_cast<float>(g[1]), static_cast<float>(g[2]));
  sim.setSolverIterations(md.posIters, 8);
  sim.setMaterialParams(0.5f, 0.0f, mu);
  if (md.gravity)
    sim.setStabilizationMode("off");
  load(sim, b, {0, 1});
  const State s0 = readState(sim);
  sim.step(1);
  const State s1 = readState(sim);
  // Contact actually seen (dist at the predicted positions).
  float dist = 0.0f;
  {
    const peclet::dem::Particles& P = sim.parts();
    int nc = 0;
    Kokkos::deep_copy(nc, P.contactCount);
    auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.contacts);
    for (int i = 0; i < nc; ++i)
      if (hc(i).bodyB >= 0)
        dist = hc(i).dist;
  }
  // Velocity-phase angular impulse about the origin, lever arms at the predicted positions.
  D3 dL{0, 0, 0}, xp[2], dv[2];
  for (int i = 0; i < 2; ++i) {
    const D3 x = at(s0.x, i), v0 = at(s0.v, i), v1 = at(s1.v, i);
    for (int d = 0; d < 3; ++d) {
      xp[i][d] = x[d] + (v0[d] + g[d] * dt) * dt;
      dv[i][d] = v1[d] - v0[d] - g[d] * dt;
    }
    const D3 c = cross(xp[i], dv[i]), sp0 = spin(s0, i), sp1 = spin(s1, i);
    for (int d = 0; d < 3; ++d)
      dL[d] += s0.m[i] * c[d] + (sp1[d] - sp0[d]);
  }
  // n from B to A; J on A, tangential part.
  D3 n{xp[0][0] - xp[1][0], xp[0][1] - xp[1][1], xp[0][2] - xp[1][2]};
  const double sep = norm(n);
  for (int d = 0; d < 3; ++d)
    n[d] /= sep;
  D3 J{s0.m[0] * dv[0][0], s0.m[0] * dv[0][1], s0.m[0] * dv[0][2]};
  const double jn = J[0] * n[0] + J[1] * n[1] + J[2] * n[2];
  D3 Jt{J[0] - jn * n[0], J[1] - jn * n[1], J[2] - jn * n[2]};
  const D3 nxJ = cross(n, Jt);
  const D3 pred{static_cast<double>(dist) * nxJ[0], static_cast<double>(dist) * nxJ[1],
                static_cast<double>(dist) * nxJ[2]};  // (p_A - p_B) x J_t, p_A - p_B = dist n
  const double pp = pred[0] * pred[0] + pred[1] * pred[1] + pred[2] * pred[2];
  const double ratio = pp > 0 ? (dL[0] * pred[0] + dL[1] * pred[1] + dL[2] * pred[2]) / pp : 0.0;
  std::printf(
      "FRICPAIR mode=%s dt=%.4g delta/R=%.4f dist/R=%.4e |Jn|=%.4e |Jt|=%.4e |dL|=%.4e "
      "|pred|=%.4e dL.pred/|pred|^2=%.4f\n",
      md.name.c_str(), dt, md.delta, dist / RAD, std::fabs(jn), norm(Jt), norm(dL), std::sqrt(pp),
      ratio);
  if (kFrictionPairGate && md.name == "friction_pair" && !(norm(dL) <= 1e-2 * std::sqrt(pp))) {
    std::fprintf(stderr, "GATE: friction_pair couple ratio |dL|/|pred| = %.3e > 1e-2\n",
                 pp > 0 ? norm(dL) / std::sqrt(pp) : 0.0);
    return 1;
  }
  return 0;
}

// ---- timing (not a ctest): the distributed XPBD step at N = 20000, fully periodic ----
static int runPerf(bool pgs, int rank, int size) {
  const int G = 27;  // 27^3 = 19683 ~ 20000
  const double spacing = 1.02 * 2.0 * RAD, box = G * spacing;
  const int gx = 32;
  std::mt19937 rng(7u);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::vector<Body> b;
  for (int k = 0; k < G; ++k)
    for (int j = 0; j < G; ++j)
      for (int i = 0; i < G; ++i) {
        Body q{};
        const int idx[3] = {i, j, k};
        for (int d = 0; d < 3; ++d) {
          q.x[d] = static_cast<float>((idx[d] + 0.5) * spacing + 0.05 * uni(rng));
          q.v[d] = gauss(rng);
        }
        q.scale = 1.0f;
        b.push_back(q);
      }
  const int n = static_cast<int>(b.size());
  const std::vector<int> gids = ownedOf(b, 0.0, box, gx, true, rank, size);
  Simulation sim(3 * n + 64);
  sim.setDomain(box, box, box, true, true, true);
  sim.setGlobalScale(1.0f);
  sim.setSphereShape(RAD);
  sim.setDt(2e-3f);
  sim.setGravity(0.0f, 0.0f, pgs ? -10.0f : 0.0f);
  sim.setSolverIterations(8, 4);
  sim.setMaterialParams(0.5f, 0.0f, 0.4f);
  load(sim, b, gids);
  const std::tuple<double, double, double> origin{0, 0, 0}, dsize{box, box, box};
  const std::tuple<long, long, long> gsize{gx, gx, gx};
  const std::tuple<bool, bool, bool> per{true, true, true};
  sim.initMpi(origin, dsize, gsize, per, MPI_COMM_WORLD);
  sim.enableMpiStep(0.0, 1, true);
  const int warm = 10, steps = 50;
  sim.stepMpi(warm);
  MPI_Barrier(MPI_COMM_WORLD);
  const auto t0 = std::chrono::steady_clock::now();
  sim.stepMpi(steps);
  MPI_Barrier(MPI_COMM_WORLD);
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() /
      steps;
  const int thr = Kokkos::DefaultHostExecutionSpace().concurrency();
  if (rank == 0)
    std::printf("PERF mode=%s np=%d thr=%d N=%d ms_per_step=%.3f\n", pgs ? "perf_pgs" : "perf_gas",
                size, thr, n, ms);
  return 0;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::string mode = (argc > 1) ? argv[1] : "cluster";
    Mode md;
    md.name = mode;
    int velItersFlag = -1;  // --vel-iters: applied after the mode's own default below
    bool restitutionFlag = false;
    for (int a = 2; a < argc; ++a) {
      if (std::strncmp(argv[a], "--dump=", 7) == 0)
        md.dump = argv[a] + 7;
      else if (std::strncmp(argv[a], "--dt=", 5) == 0)
        md.dt = std::stof(argv[a] + 5);
      else if (std::strncmp(argv[a], "--posit=", 8) == 0)
        md.posIters = std::stoi(argv[a] + 8);
      else if (std::strncmp(argv[a], "--hub=", 6) == 0)
        md.hubScale = std::stof(argv[a] + 6);
      else if (std::strncmp(argv[a], "--delta=", 8) == 0)
        md.delta = std::stof(argv[a] + 8);
      else if (std::strcmp(argv[a], "--solo") == 0)
        md.solo = true;
      else if (std::strncmp(argv[a], "--e=", 4) == 0) {
        md.restitution = std::stof(argv[a] + 4);
        restitutionFlag = true;
      } else if (std::strncmp(argv[a], "--steps=", 8) == 0)
        md.steps = std::stoi(argv[a] + 8);
      else if (std::strncmp(argv[a], "--vel-iters=", 12) == 0)
        velItersFlag = std::stoi(argv[a] + 12);
      else if (std::strncmp(argv[a], "--axis=", 7) == 0)
        md.axis = std::stoi(argv[a] + 7);
      else if (std::strncmp(argv[a], "--relabel=", 10) == 0)
        md.relabel = static_cast<unsigned>(std::stoul(argv[a] + 10));
    }
    if (md.axis < 0 || md.axis > 2) {
      if (rank == 0)
        std::fprintf(stderr, "--axis must be 0, 1 or 2\n");
      md.axis = 0;
    }

    if (md.solo && size != 1) {
      if (rank == 0)
        std::fprintf(stderr, "--solo needs np = 1\n");
      md.solo = false;
    }
    if (mode == "cluster") {
    } else if (mode == "cluster_friction") {
      md.friction = md.spins = true;
    } else if (mode == "cluster_pgs") {
      md.friction = md.spins = md.gravity = true;
    } else if (mode == "cluster_posonly") {
      md.velIters = 0;
    } else if (mode == "cluster_jacobi") {
      md.jacobi = true;
    } else if (mode == "hertz") {
      md.friction = md.spins = md.hertz = true;
    } else if (mode == "cluster_sync3") {
      md.friction = md.spins = true;
      md.syncEvery = 3;
    } else if (mode == "cluster_norot") {
      md.friction = md.spins = true;
      md.forwardRotation = false;
    } else if (mode == "cluster_periodic") {
      md.periodic = true;
    } else if (mode == "hub" || mode == "hub_posonly") {
      if (md.hubScale <= 0.0f)
        md.hubScale = 10.0f;
      if (mode == "hub_posonly")
        md.velIters = 0;
    } else if (mode == "friction_pair_pgs") {
      md.gravity = true;
    } else if (mode == "tri") {
      md.tri = true;
    } else if (mode == "tri_pgs") {
      md.tri = md.gravity = true;
    } else if (mode == "cluster_e09" || mode == "cluster_e10") {
      if (!restitutionFlag)
        md.restitution = mode == "cluster_e09" ? 0.9f : 1.0f;
    } else if (mode == "cluster_poisson") {
      md.friction = md.spins = md.gravity = md.poisson = true;
    } else if (mode == "cluster_multilevel" || mode == "cluster_escalate" ||
               mode == "cluster_ordered" || mode == "cluster_onesided") {
      md.friction = md.spins = md.gravity = true;
      md.stab = mode.substr(8);  // "multilevel" | "escalate" | "ordered" | "onesided"
    } else if (mode == "hub_pgs") {
      if (md.hubScale <= 0.0f)
        md.hubScale = 10.0f;
      md.gravity = true;
    } else if (mode == "ring_mini") {
      md.ring = true;
    }
    if (velItersFlag >= 0)
      md.velIters = velItersFlag;
    md.reportOnly = mode == "tri" || mode == "tri_pgs" || mode == "cluster_e09" ||
                    mode == "cluster_e10" || mode == "cluster_poisson" ||
                    mode == "cluster_multilevel" || mode == "cluster_escalate" ||
                    mode == "cluster_ordered" || mode == "cluster_onesided" || mode == "hub_pgs" ||
                    mode == "ring_mini";
    if (mode == "perf_gas" || mode == "perf_pgs")
      fail = runPerf(mode == "perf_pgs", rank, size);
    else if (mode == "friction_pair" || mode == "friction_pair_pgs")
      fail = runFrictionPair(md, rank, size);
    else if (mode == "hub" || mode == "hub_posonly" || mode == "hub_pgs")
      fail = runCluster(md, rank, size);
    else if (mode == "tri" || mode == "tri_pgs" || mode == "cluster_e09" || mode == "cluster_e10" ||
             mode == "cluster_poisson" || mode == "cluster_multilevel" ||
             mode == "cluster_escalate" || mode == "cluster_ordered" ||
             mode == "cluster_onesided" || mode == "ring_mini")
      fail = runCluster(md, rank, size);
    else if (mode == "cluster" || mode == "cluster_friction" || mode == "cluster_pgs" ||
             mode == "cluster_posonly" || mode == "hertz" || mode == "cluster_sync3" ||
             mode == "cluster_norot" || mode == "cluster_periodic" || mode == "cluster_jacobi")
      fail = runCluster(md, rank, size);
    else {
      if (rank == 0)
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
      fail = 1;
    }
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d)\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d)\n", size);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
