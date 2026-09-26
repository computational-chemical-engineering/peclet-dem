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
//   cluster_pgs_e     GATED (docs/contact_physics_followups.md §6 G-C1, WO-C1): cluster_pgs with
//                     mu = 0, no spins, stabilization off, --e=<e>, --rest-target=<newton|moreau>,
//                     ONE step with every adaptive stop off and 2000 velocity iterations (the
//                     converged PGS). Prints KEGATE (the CoM-frame KE before and after the step).
//                     Under 'moreau': KE after <= KE before (1 + 1e-5), and at e = 1 also
//                     >= KE before (1 - 2e-3); under 'newton' the ratio is reported only.
//   cluster_multilevel, cluster_escalate, cluster_ordered, cluster_onesided   cluster_pgs with that
//                     stabilization mode instead of 'off'
//   hub_pgs           hub under free fall (g != 0)
//   ring_mini         27 hollow cylinders (outer diameter 1, height 1.5, wall 0.18, unit mass
//                     through the shape registry) on a jittered 3 x 3 x 3 lattice at spacing 0.9
//                     about the cluster centre, orientations uniformly random (mt19937(11)),
//                     cluster-recipe velocities, g = 0, friction 0.02, pos/vel iterations 20/8,
//                     dt 1e-2, 10 steps: per-point position-graph degrees far above 64. A
//                     CONSERVATION scene only: it starts tunnelled (85 of 106 contacting pairs
//                     have a linearised overlap problem with no solution), so its ovl cannot
//                     converge under any solver and is not a convergence metric
//                     (docs/contact_physics_followups.md §3.2); the convergence gate is
//                     ring_collide
//   ring_collide      the same tubes on a jittered 3 x 3 x 3 lattice at spacing 1.5, orientations
//                     REJECTION-SAMPLED so the scene starts overlap-free (mt19937(26): a
//                     normalised 4-D Gaussian quaternion per body, up to 2000 draws; the host
//                     oracle rejects any shell point of the new body inside a placed body's
//                     analytic SDF or vice versa, genCylinderShell spacing 0.09), makeRingMini's
//                     velocities, g = 0, friction 0.02, e 0.5, pos/vel iterations 20/8, dt 1e-2,
//                     10 steps (docs/contact_physics_followups.md §3.4, WO-B2): GATES
//                     conservation, and with --gate-pos-cap / --gate-overlap the feasibility
//                     (G-B2)
//   ring_collide_posonly   ring_collide with the velocity solve off, 4 steps: the np-agreement
//                     scene of position_agreement.sh (--no-stop --pos-iters=2000)
//
// GATED hub modes of docs/contact_solve_framework.md §13 (WO-4b; np = 1 gated, np >= 2 report-only
// until WO-5): makeHubLast -- the hub LAST (highest gid), mass scale^3 leaf masses.
//   hub_static        every leaf (scale 1) overlaps the hub by delta = 0.04 R, all at rest, g = 0,
//                     velocity iterations 0, position iterations 64, 1 step: max leaf-hub gap
//                     <= 0.15 delta (§12 S11), residual overlap <= 1e-4 R, dXpos <= 3e-5, |P|
//                     exactly 0 (an over-relaxed projection leaves every leaf ~0.5 delta clear)
//   hub_ml            the dense shell 3.0 (rs/R)^2 (§12 S10; N = 181), the leaves with dir.z < 0
//                     approaching the hub at 0.3 (1 + 0.1 N(0,1)), free
//                     fall, frictionless, stabilization multilevel, velocity iterations 1, 10
//                     steps: dP <= 5e-6, dX, dXpos <= 3e-5, dLvel <= 1e-6 (rigid aggregates,
//                     docs/contact_physics_followups.md §2), and the positive controls (velocity
//                     hub copies > 0 at np = 1, >= 1 multilevel level, split_stats.mlHubAggregated
//                     >= 1, each in at least one step; required at np 1 and 2 only, §12 S18)
//                     -- the coarse cycle at a folded hub
// Both print HUBGAP (maxGap / delta, residual / R), hub_ml also MLCTRL (the controls).
// Options (after the mode): --dump=<path>, --dt=<dt>, --posit=<n> / --pos-iters=<n> (position
// iterations),
// --hub=<scale>, --delta=<overlap / R>, --solo (single-rank demStep via Simulation::step instead
// of step_mpi; np = 1 only), --e=<normal restitution> (default 0.5; cluster_e09/_e10 set theirs),
// --steps=<n> (overrides the mode's step count), --vel-iters=<n> (velocity iterations),
// --stab=<mode> (overrides the mode's stabilization: off|onesided|multilevel|escalate|ordered),
// --fused=<auto|on|off> (diagnostics.set_fused_sweeps; 'on' with --solo runs the device-side
// loops on CUDA, whose ITERS come from the device counter of §12 S12),
// --gate-pos-cap (fail if the main position loop of any step ran its cap: with stops on and a
// large --pos-iters, the feasibility gate of ring_collide, G-B2), --gate-overlap=<tol> (np 1:
// fail if the committed overlap Simulation::computeOverlaps after any step exceeds tol),
// --axis=<0|1|2> (tri: the approach axis), --relabel=<seed> (seed 0 = identity; otherwise the body
// list is std::shuffle'd with std::mt19937(seed) before the gids are assigned, which samples
// another serial Gauss-Seidel order of the same physical scene), --no-stop (every adaptive stop
// of the contact solve off, so each loop runs exactly its cap: the convergence gates G7a / G7c,
// §12 S13; Simulation::debugNoAdaptiveStop, test-only), --rest-target=<newton|moreau>
// (diagnostics.set_restitution_target: the PGS restitution target law, WO-C1).
// dLvel measures the velocity phase from the PREDICTED angular velocity (§12 S16): the predict's
// explicit gyroscopic term is frame rotation, not a contact impulse.
//
// Output: parseable lines per run,
//   MOMENTUM mode=.. np=.. thr=.. N=.. steps=.. dP=.. dX=.. dXpos=.. dL=.. dLcm=.. dLvel=.. ovl=..
//   KE mode=.. np=.. thr=.. s1=.. s2=..      the centre-of-mass-frame kinetic energy (translational
//            + rotational, host double, Allreduced) after every (recorded) step
//   KEROT mode=.. np=.. thr=.. ke0=.. s1=..  its rotational part (spin KE) alone; ke0 = the
//            initial CoM-frame kinetic energy
//   CONFLICTS mode=.. np=.. thr=.. vel=.. pos=.. degVel=.. degPos=.. colVel=.. colPos=..
//            (not hertz) the same-colour pairs of the rank-local velocity (manifold) and position
//            (contact) colourings of each step -- sum over (body slot, colour) of (count - 1), the
//            quantity Simulation::debugColoringConflicts counts, tallied on the host over the items
//            the step coloured -- the largest per-body-slot degree and the colour count, each the
//            maximum over steps and ranks; leftVel / leftPos count the items the step left
//            uncoloured (-1, the count-averaged fallback's set), maximum over steps and ranks
//   ORPHAN mode=.. np=.. thr=.. orphanClamps=..   (Poisson modes) split_stats.orphanClamps summed
//            over steps and ranks: the owner apply's orphan-balance clamp hits (GATE: must be 0)
//   ITERS mode=.. np=.. thr=.. vel=.. pos=.. posMax=.. posMed=..   the iterations the LAST
//            step's main velocity and position loops ran (split_stats velItersUsed /
//            posItersUsed; G7f), and the position loop's maximum and median over the steps
//   OVERLAP mode=.. np=1 committed=..   (--gate-overlap) the largest committed overlap over the
//            steps
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
#include <unordered_map>
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
static constexpr bool kFollowupGate = true;  // WO-10: the hub modes are gates
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
  if (mode == "cluster_pgs_e")  // WO-C1: the §1 free-fall conservation gates (plus KEGATE)
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
  if (mode == "hertz_shear" || mode == "hertz_shear_frictionless")  // WO-7: dP <= 1e-6 under drift
    return {1e-6, -1, -1, -1, -1};
  // docs/contact_solve_framework.md §13.6 G1 additions. hub_static: velocities unchanged, the
  // ABSOLUTE |P| must stay exactly 0 (sum m|v| = 0, no normalized dP).
  // WO-10 (§9, gates on): every mode the framework makes conservative. Measured maxima over np 1..8
  // x OMP 1/8 in IMPL_A.md; thresholds sit ~10-20x above them and far below the broken values.
  if (mode == "hub" || mode == "hub_pgs" || mode == "cluster_poisson" ||
      mode == "cluster_escalate" || mode == "cluster_ordered")  // dP <= 8e-7 (free-fall floor)
    return {5e-6, 1e-5, 1e-5, -1, 1e-6};
  // cluster_multilevel / hub_ml: the coarse bodies are rigid 6-DOF aggregates
  // (docs/contact_physics_followups.md §2, WO-A2), so dLvel is exact to float (translation-only
  // aggregates gave hub_ml 3.0e-3, cluster_multilevel 1.5e-8 at np 1).
  if (mode == "cluster_multilevel")
    return {5e-6, 1e-5, 1e-5, -1, 1e-6};
  if (mode == "hub_posonly")
    return {1e-6, 1e-5, 1e-5, -1, 1e-6};
  if (mode == "cluster_shear" || mode == "cluster_e09" || mode == "cluster_e10" || mode == "tri")
    return {1e-6, 1e-5, 1e-5, -1, 1e-6};
  if (mode == "ring_mini")  // dP <= 3.5e-8, dXpos <= 5.5e-6 R, dLvel <= 4e-8 (after WO-5b)
    return {1e-6, 3e-5, 3e-5, -1, 1e-6};
  if (mode == "ring_collide")  // docs/contact_physics_followups.md §6 G-B2
    return {1e-6, -1, 3e-5, -1, 1e-6};
  if (mode == "hub_static")
    return {0.0, -1, 3e-5, -1, -1};
  if (mode == "hub_ml")
    return {5e-6, 3e-5, 3e-5, -1, 1e-6};
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

// The WO-4b hub scenes (docs/contact_solve_framework.md §13.2, §13.6): makeHub's geometry with the
// hub moved LAST (the highest gid, so at np >= 2 every leaf-hub contact is owned by the leaf's
// owner and the hub is mass-split across ranks) and its mass scale^3 leaf masses (load():
// invMassOf). The random stream is makeHub's (one gauss then one uni per leaf, in order), so the
// leaves sit where makeHub puts them.
//   hub_static  every leaf of scale exactly 1, all velocities zero: every leaf overlaps the hub by
//               delta = rs - (r_h + R) = 0.04 R.
//   hub_ml      the dense shell ns = 3.0 (rs/R)^2 (§12 S10), only the leaves with dir.z < 0 (about
//               half; N = 181 bodies at scale 10), scale 1 +- 0.1 as makeHub; hub and
//               leaves carry makeHub's drift, the leaves a radial approach speed 0.3 (1 + 0.1
//               N(0,1)) (the leaf's own gauss draw).
static constexpr double kHubDelta = 0.02;  // makeHub's overlap rs - (r_h + R) = 0.04 R
static std::vector<Body> makeHubLast(float hubScale, bool ml) {
  std::mt19937 rng(20260926u);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  const D3 c{0.3, -0.2, 0.1};
  const float drift[3] = {0.7f, -0.4f, 0.3f};
  const double rs = hubScale * RAD + RAD - kHubDelta;  // shell centre distance (makeHub's)
  // hub_ml: the DENSE shell 3.0 (rs/R)^2 (§12 S10; N = 181 at scale 10), whose leaves touch each
  // other, so the multilevel matching can aggregate the hub (makeHub's 2.5 leaves a star graph
  // that never builds a level). hub_static keeps makeHub's 2.5.
  const int ns = static_cast<int>((ml ? 3.0 : 2.5) * (rs / RAD) * (rs / RAD));
  std::vector<Body> b;
  const double golden = 3.14159265358979 * (3.0 - std::sqrt(5.0));
  for (int k = 0; k < ns; ++k) {
    const double z = 1.0 - 2.0 * (k + 0.5) / ns, r = std::sqrt(1.0 - z * z), ph = golden * k;
    const D3 dir{r * std::cos(ph), r * std::sin(ph), z};
    const float gk = gauss(rng);
    const float sk = 1.0f + 0.1f * uni(rng);
    if (ml && !(dir[2] < 0.0))
      continue;
    const float speed = ml ? 0.3f * (1.0f + 0.1f * gk) : 0.0f;
    Body q{};
    for (int d = 0; d < 3; ++d) {
      q.x[d] = static_cast<float>(c[d] + rs * dir[d]);
      q.v[d] = ml ? drift[d] - speed * static_cast<float>(dir[d]) : 0.0f;
    }
    q.scale = ml ? sk : 1.0f;
    b.push_back(q);
  }
  Body h{};
  for (int d = 0; d < 3; ++d) {
    h.x[d] = static_cast<float>(c[d]);
    h.v[d] = ml ? drift[d] : 0.0f;
  }
  h.scale = hubScale;
  b.push_back(h);
  return b;
}

// The hub scenes' leaf-gap metric (§13.6): every rank's owned bodies gathered to every rank; the
// hub is the last gid. maxGap = the largest leaf-hub separation |x_l - x_h| - (r_l + r_h) (> 0: the
// leaf is clear of the hub), residual = the largest overlap over every pair (leaf-hub and
// leaf-leaf), 0 if none.
struct HubGap {
  double maxGap = 0.0, residual = 0.0;
};
static HubGap hubGap(const Simulation& sim, const std::vector<int>& gids,
                     const std::vector<Body>& bodies) {
  const std::vector<float> x = sim.getPositions();
  const int n = static_cast<int>(gids.size());
  std::vector<float> loc(static_cast<std::size_t>(4 * n));
  for (int i = 0; i < n; ++i) {
    loc[4 * i] = static_cast<float>(gids[i]);
    for (int d = 0; d < 3; ++d)
      loc[4 * i + 1 + d] = x[3 * i + d];
  }
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int cnt = 4 * n;
  std::vector<int> cnts(size), offs(size, 0);
  MPI_Allgather(&cnt, 1, MPI_INT, cnts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  int tot = 0;
  for (int r = 0; r < size; ++r) {
    offs[r] = tot;
    tot += cnts[r];
  }
  std::vector<float> all(static_cast<std::size_t>(tot));
  MPI_Allgatherv(loc.data(), cnt, MPI_FLOAT, all.data(), cnts.data(), offs.data(), MPI_FLOAT,
                 MPI_COMM_WORLD);
  const int nb = static_cast<int>(bodies.size());
  std::vector<D3> p(static_cast<std::size_t>(nb), D3{0, 0, 0});
  for (int k = 0; k < tot / 4; ++k)
    p[static_cast<int>(all[4 * k])] = D3{all[4 * k + 1], all[4 * k + 2], all[4 * k + 3]};
  HubGap h;
  h.maxGap = -1e30;
  const int hub = nb - 1;
  for (int a = 0; a < nb; ++a)
    for (int b = a + 1; b < nb; ++b) {
      const D3 d{p[a][0] - p[b][0], p[a][1] - p[b][1], p[a][2] - p[b][2]};
      const double gap = norm(d) - (bodies[a].scale + bodies[b].scale) * RAD;
      if (b == hub)
        h.maxGap = std::max(h.maxGap, gap);
      h.residual = std::max(h.residual, -gap);
    }
  return h;
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

// ring_collide (docs/contact_physics_followups.md §3.4): makeRingMini's lattice at spacing 1.5,
// its position jitter and velocities (same stream, same recipe), and orientations rejection-
// sampled from std::mt19937(26) so that no tube starts inside another: a draw is rejected when
// the host oracle finds a shell point of the new body inside a placed body's analytic SDF, or a
// placed body's shell point inside the new body's (dist < 0). No fallback: 2000 failed draws
// abort the setup.
static constexpr float kRingCollideSpacing = 1.5f;
static std::vector<Body> makeRingCollide() {
  using peclet::dem::F3;
  using peclet::dem::F4;
  std::mt19937 rng(20260925u), qrng(26u);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::normal_distribution<float> gauss(0.0f, 1.0f), qgauss(0.0f, 1.0f);
  const D3 c{0.3, -0.2, 0.1};
  const float drift[3] = {0.7f, -0.4f, 0.3f};
  const double h = kRingCollideSpacing, rMax = h * std::sqrt(3.0);
  const F4 params{0.5f * kRingD, kRingH, kRingWall, 0.0f};
  const std::vector<F3> shell =
      peclet::dem::genCylinderShell(0.5f * kRingD, kRingH, kRingWall, 0.09f);
  auto toWorld = [](const Body& b, F3 s) {
    const F3 r = peclet::dem::rotateVector(F4{b.q[0], b.q[1], b.q[2], b.q[3]}, s);
    return F3{b.x[0] + r.x, b.x[1] + r.y, b.x[2] + r.z};
  };
  auto inside = [&](const Body& b, F3 p) {  // p (world) inside b's analytic SDF
    const F3 d{p.x - b.x[0], p.y - b.x[1], p.z - b.x[2]};
    const F3 local = peclet::dem::invRotateVector(F4{b.q[0], b.q[1], b.q[2], b.q[3]}, d);
    return peclet::dem::sdfHollowCylinder(local, params) < 0.0f;
  };
  auto overlaps = [&](const Body& a, const Body& b) {  // either shell inside the other's SDF
    for (const F3& s : shell)
      if (inside(b, toWorld(a, s)) || inside(a, toWorld(b, s)))
        return true;
    return false;
  };
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
        bool placed = false;
        for (int draw = 0; draw < 2000 && !placed; ++draw) {
          float nn = 0.0f;
          for (float& e : q.q) {
            e = qgauss(qrng);
            nn += e * e;
          }
          nn = std::sqrt(nn);
          for (float& e : q.q)
            e /= nn;
          placed = std::none_of(b.begin(), b.end(), [&](const Body& o) { return overlaps(q, o); });
        }
        if (!placed) {
          std::fprintf(stderr,
                       "ring_collide: no overlap-free orientation for body %zu in 2000 draws\n",
                       b.size());
          MPI_Abort(MPI_COMM_WORLD, 2);
        }
        b.push_back(q);
      }
  return b;
}

// Rank-local colouring diagnostics of the last step: over the contacts / manifolds the step
// coloured (colour >= -1 or -3; the caller filled the colour arrays with kNotColoured before the
// step, so the entries the step did not colour -- the non-owned visible ones under MPI -- are
// skipped), the largest per-body degree and the same-colour pairs, sum over (body, colour) of
// (count - 1).
struct ColorDiag {
  int degC = 0, degM = 0, confC = 0, confM = 0;
  int degPt = 0;                     // largest per-body degree of the per-POINT contact graph
  int colC = 0, colM = 0;            // colours used (largest colour + 1)
  int leftC = 0, leftM = 0;          // items the step left uncoloured (-1, or -3: no free colour)
  int confMl = 0;                    // multilevel: same-colour pairs at an aggregate (all levels)
  int copiesVel = 0, copiesPos = 0;  // hub copy slots (§4.4)
};
// The "not coloured by this step" marker; distinct from every colour a step writes (>= 0, -1,
// -2, and -3 = kColorUncolourable).
static constexpr int kNotColoured = -100;
static void markColors(ProbeSim& sim) {
  Kokkos::deep_copy(sim.parts().contactColor, kNotColoured);
  Kokkos::deep_copy(sim.parts().unitColor, kNotColoured);
  Kokkos::deep_copy(sim.parts().manifoldColor, kNotColoured);
}
static ColorDiag colorDiag(ProbeSim& sim) {
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
      left += (x.second == -1 || x.second == -3);  // -3: no free colour (after copies: invariant)
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
  // Hub copies (docs/contact_solve_framework.md §4.4): an edge's vertex is its override slot.
  auto slotOf = [](const auto& ov, int e, int def) {
    return (ov.extent(0) > 0 && ov(e) >= 0) ? ov(e) : def;
  };
  const bool pH = P.posCopies.nHubs > 0, vH = P.velCopies.nHubs > 0;
  auto hpa = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.posCopies.slotA);
  auto hpb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.posCopies.slotB);
  auto hva = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.velCopies.slotA);
  auto hvb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.velCopies.slotB);
  for (int u = 0; u < nu; ++u) {
    const int col = huc(u);
    if (col < -1 && col != -3)  // skip inactive (-2) and not-coloured-here entries
      continue;
    const int i = hul(hus(u));
    ec.push_back({pH ? slotOf(hpa, u, hc(i).bodyA) : hc(i).bodyA, col});
    if (hc(i).bodyB >= 0)
      ec.push_back({pH ? slotOf(hpb, u, hc(i).bodyB) : hc(i).bodyB, col});
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
    if (col < -1 && col != -3)  // skip inactive (-2) and not-coloured-here entries
      continue;
    em.push_back({vH ? slotOf(hva, i, hr(hm(i).bodyA)) : hr(hm(i).bodyA), col});
    if (hm(i).bodyB >= 0)
      em.push_back({vH ? slotOf(hvb, i, hr(hm(i).bodyB)) : hr(hm(i).bodyB), col});
  }
  tally(ec, g.degC, g.confC, g.colC, g.leftC);
  {
    int c0 = 0, c1 = 0, l0 = 0;
    tally(ep, g.degPt, c0, c1, l0);
  }
  tally(em, g.degM, g.confM, g.colM, g.leftM);
  g.confMl = sim.debugMultilevelColoringConflicts();
  g.copiesVel = P.velCopies.nCopies;
  g.copiesPos = P.posCopies.nCopies;
  return g;
}

struct Sums {
  double m = 0, P[3] = {0, 0, 0}, mx[3] = {0, 0, 0}, Lo[3] = {0, 0, 0};
};
struct State {
  std::vector<float> x, v, w, m, invI, q;
  std::vector<int> gid;  // pairs a body's states across a step (a drift migration moves bodies)
};
static State readState(const ProbeSim& s) {
  State st{s.getPositions(),
           s.getVelocities(),
           s.getAngularVelocities(),
           s.getMasses(),
           s.getInvInertia(),
           s.getQuaternions(),
           {}};
  const auto& P = s.parts();
  auto hg = Kokkos::create_mirror_view(P.gid);
  Kokkos::deep_copy(hg, P.gid);
  st.gid.resize(st.m.size());
  for (std::size_t i = 0; i < st.gid.size(); ++i)
    st.gid[i] = hg(static_cast<int>(i));
  return st;
}
// Gather a per-body vector (k values per body) of every rank onto rank 0, in rank order.
template <class T>
static std::vector<T> gatherBodies(const std::vector<T>& v, MPI_Datatype t) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int n = static_cast<int>(v.size());
  std::vector<int> cnt(size), off(size, 0);
  MPI_Gather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  int tot = 0;
  if (rank == 0)
    for (int r = 0; r < size; ++r) {
      off[r] = tot;
      tot += cnt[r];
    }
  std::vector<T> all(rank == 0 ? tot : 0);
  MPI_Gatherv(v.data(), n, t, all.data(), cnt.data(), off.data(), t, 0, MPI_COMM_WORLD);
  return all;
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
// The angular velocity predictVelocityKokkos hands the velocity phase (no external torque in these
// scenes): w + R [-dt invI (wb x I wb)] with wb = R^T w, in double. Isotropic or massless bodies
// and bodies without inverse inertia keep w exactly (the kernel's own guards).
static D3 predictedOmega(const State& st, int i, double dt) {
  const D3 w = at(st.w, i);
  const double i0 = st.invI[3 * i], i1 = st.invI[3 * i + 1], i2 = st.invI[3 * i + 2];
  if (!(st.m[i] > 0) || !(i0 > 0 || i1 > 0 || i2 > 0) || (i0 == i1 && i1 == i2))
    return w;
  const double qx = st.q[4 * i], qy = st.q[4 * i + 1], qz = st.q[4 * i + 2], qw = st.q[4 * i + 3];
  const double R[3][3] = {
      {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)},
      {2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)},
      {2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)}};
  const double inv[3] = {i0, i1, i2};
  double wb[3] = {0, 0, 0}, Lb[3];
  for (int k = 0; k < 3; ++k)
    for (int d = 0; d < 3; ++d)
      wb[k] += R[d][k] * w[d];
  for (int k = 0; k < 3; ++k)
    Lb[k] = inv[k] > 1e-9 ? wb[k] / inv[k] : 0.0;
  const double wxL[3] = {wb[1] * Lb[2] - wb[2] * Lb[1], wb[2] * Lb[0] - wb[0] * Lb[2],
                         wb[0] * Lb[1] - wb[1] * Lb[0]};
  for (int k = 0; k < 3; ++k)
    wb[k] -= inv[k] * wxL[k] * dt;
  D3 out{0, 0, 0};
  for (int d = 0; d < 3; ++d)
    out[d] = R[d][0] * wb[0] + R[d][1] * wb[1] + R[d][2] * wb[2];
  return out;
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
// sum_i m_i (x_i(b) - x_i(a)), minimum image in a periodic box of side `box`, with a body's two
// states paired by gid on rank 0 (a drift migration moves bodies between ranks within a step).
static D3 periodicDisplacement(const State& aLocal, const State& bLocal, double box) {
  const std::vector<float> ax = gatherBodies(aLocal.x, MPI_FLOAT),
                           bx = gatherBodies(bLocal.x, MPI_FLOAT),
                           bm = gatherBodies(bLocal.m, MPI_FLOAT);
  const std::vector<int> ag = gatherBodies(aLocal.gid, MPI_INT),
                         bg = gatherBodies(bLocal.gid, MPI_INT);
  std::unordered_map<int, int> where;
  for (int j = 0; j < static_cast<int>(ag.size()); ++j)
    where[ag[j]] = j;
  D3 out{0, 0, 0};
  for (int i = 0; i < static_cast<int>(bg.size()); ++i) {
    const int j = where.at(bg[i]);
    for (int d = 0; d < 3; ++d) {
      double dx = static_cast<double>(bx[3 * i + d]) - ax[3 * j + d];
      dx -= box * std::round(dx / box);
      out[d] += bm[i] * dx;
    }
  }
  MPI_Bcast(out.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return out;
}
static D3 velocityPhaseTorque(const State& aLocal, const State& bLocal, const D3& g, double dt,
                              const D3& Xpred) {
  // Pair each body's before / after states by gid, on rank 0: a drift migration inside the step
  // (docs/contact_solve_framework.md §5.1) moves bodies between ranks and reorders them, so the
  // local index does not identify a body across a step.
  State a, b;
  a.x = gatherBodies(aLocal.x, MPI_FLOAT);
  a.v = gatherBodies(aLocal.v, MPI_FLOAT);
  a.w = gatherBodies(aLocal.w, MPI_FLOAT);
  a.m = gatherBodies(aLocal.m, MPI_FLOAT);
  a.invI = gatherBodies(aLocal.invI, MPI_FLOAT);
  a.q = gatherBodies(aLocal.q, MPI_FLOAT);
  a.gid = gatherBodies(aLocal.gid, MPI_INT);
  const std::vector<float> bv = gatherBodies(bLocal.v, MPI_FLOAT),
                           bw = gatherBodies(bLocal.w, MPI_FLOAT);
  const std::vector<int> bg = gatherBodies(bLocal.gid, MPI_INT);
  b.v.assign(a.v.size(), 0.0f);
  b.w.assign(a.w.size(), 0.0f);
  {
    std::unordered_map<int, int> where;
    for (int j = 0; j < static_cast<int>(bg.size()); ++j)
      where[bg[j]] = j;
    for (int i = 0; i < static_cast<int>(a.gid.size()); ++i) {
      const int j = where.at(a.gid[i]);
      for (int d = 0; d < 3; ++d) {
        b.v[3 * i + d] = bv[3 * j + d];
        b.w[3 * i + d] = bw[3 * j + d];
      }
    }
  }
  double loc[3] = {0, 0, 0};
  for (int i = 0; i < static_cast<int>(a.m.size()); ++i) {
    const D3 x = at(a.x, i), v0 = at(a.v, i), v1 = at(b.v, i);
    D3 xp, dv;
    for (int d = 0; d < 3; ++d) {
      xp[d] = x[d] + (v0[d] + g[d] * dt) * dt - Xpred[d];
      dv[d] = v1[d] - v0[d] - g[d] * dt;
    }
    // Both at the orientation the phase solves at (frozen in the velocity phase). The baseline
    // is the PREDICTED omega (docs/contact_solve_framework.md §12 S16): predictVelocityKokkos's
    // explicit gyroscopic Euler term -dt invI (w x I w) (integration.hpp), replayed in double, is
    // the free body's frame-rotation compensation, not a contact impulse, so it is not booked to
    // the velocity phase. Isotropic inertia (spheres) has w x I w = 0: the baseline is w itself.
    const D3 c = cross(xp, dv), s0 = spinOf(a, i, predictedOmega(a, i, dt)),
             s1 = spinOf(a, i, at(b.w, i));
    for (int d = 0; d < 3; ++d)
      loc[d] += a.m[i] * c[d] + (s1[d] - s0[d]);
  }
  D3 out{loc[0], loc[1], loc[2]};  // rank 0 holds the sum; the others summed nothing
  MPI_Bcast(out.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
  float omega = -1.0f;    // --omega: the position over-relaxation (WO-12 scan; < 0 = default)
  float delta = 0.05f;    // --delta (friction_pair): overlap / R
  // WO-0 report-only options (docs/contact_solve_framework.md §8)
  float restitution = 0.5f;  // --e
  int steps = 0;             // --steps: > 0 overrides the mode's step count
  int axis = 0;              // --axis (tri)
  unsigned relabel = 0;      // --relabel: 0 = identity
  bool tri = false, ring = false, poisson = false;
  bool ringCollide = false;   // ring (tube shape) on the overlap-free ring_collide scene
  bool ringPosOnly = false;   // ring_collide_posonly: velocity solve off, 4 steps
  bool gatePosCap = false;    // --gate-pos-cap: fail if a step's main position loop ran its cap
  float gateOverlap = -1.0f;  // --gate-overlap=<tol>: np 1, committed overlap after every step
  bool hubStatic = false, hubMl = false;  // the WO-4b hub scenes (makeHubLast)
  bool shear = false;       // WO-7: v_x += kShearRate z (bodies drift out of their owners' blocks)
  bool reportOnly = false;  // the WO-0 modes: never fail today
  std::string stab;         // stabilization mode set after the gravity rule's 'off' (empty = keep)
  std::string fused;        // --fused=auto|on|off: diagnostics.set_fused_sweeps (empty = default)
  bool noStop = false;  // --no-stop: every adaptive stop off, each loop runs its cap (§12 S13, G7)
  std::string restTarget;  // --rest-target=newton|moreau (empty = the default, newton)
  bool keGate = false;     // cluster_pgs_e: the G-C1 KE gate (KEGATE line)
};

// Gather every rank's owned bodies to rank 0 and write them sorted by global body index (see the
// file note for the record layout).
// The scene index ("lattice" / body index of the generator) of every body, keyed by its
// simulation gid. Ownership is not fixed (the drift vote migrates bodies, WO-7), so a dump must
// identify a body by its gid, which migration carries, not by the setup's owned list.
static std::unordered_map<int, int> sceneIndexByGid(const ProbeSim& sim,
                                                    const std::vector<int>& setupGids) {
  const auto& P = sim.parts();
  auto hg = Kokkos::create_mirror_view(P.gid);
  Kokkos::deep_copy(hg, P.gid);
  std::vector<int> pairs;  // (sim gid, scene index) in the setup order (setPositions order)
  for (int i = 0; i < static_cast<int>(setupGids.size()); ++i)
    pairs.insert(pairs.end(), {static_cast<int>(hg(i)), setupGids[i]});
  int n = static_cast<int>(pairs.size()), size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<int> cnt(size), off(size, 0);
  MPI_Allgather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
  int tot = 0;
  for (int r = 0; r < size; ++r) {
    off[r] = tot;
    tot += cnt[r];
  }
  std::vector<int> all(tot);
  MPI_Allgatherv(pairs.data(), n, MPI_INT, all.data(), cnt.data(), off.data(), MPI_INT,
                 MPI_COMM_WORLD);
  std::unordered_map<int, int> m;
  for (int k = 0; k + 1 < tot; k += 2)
    m[all[k]] = all[k + 1];
  return m;
}

static void dumpState(const ProbeSim& sim, const std::unordered_map<int, int>& sceneOf,
                      const std::string& path, int rank, int size) {
  constexpr int kRec = 14;  // int32 gid + 13 float32
  const std::vector<float> x = sim.getPositions(), v = sim.getVelocities(),
                           w = sim.getAngularVelocities(), q = sim.getQuaternions();
  std::vector<int> gids(x.size() / 3);  // the scene index of every body owned NOW
  {
    auto hg = Kokkos::create_mirror_view(sim.parts().gid);
    Kokkos::deep_copy(hg, sim.parts().gid);
    for (std::size_t i = 0; i < gids.size(); ++i)
      gids[i] = sceneOf.at(hg(static_cast<int>(i)));
  }
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
  std::vector<Body> bodies = md.tri                       ? makeTri(md.axis)
                             : md.ringCollide             ? makeRingCollide()
                             : md.ring                    ? makeRingMini()
                             : (md.hubStatic || md.hubMl) ? makeHubLast(md.hubScale, md.hubMl)
                             : hub                        ? makeHub(md.hubScale)
                                                          : makeCluster(6.0, md.spins, md.periodic);
  if (md.relabel != 0) {  // another serial order of the same scene: permute before the gids
    std::mt19937 perm(md.relabel);
    std::shuffle(bodies.begin(), bodies.end(), perm);
  }
  // WO-7 drift scenes: a shear flow carries bodies several radii beyond their owners' blocks
  // over the run (docs/contact_solve_framework.md §5.1 / §5.5, the drift vote).
  constexpr float kShearRate = 2.0f;
  if (md.shear)
    for (Body& b : bodies)
      b.v[0] += kShearRate * b.x[2];
  const int n = static_cast<int>(bodies.size());
  const float dt0 = md.hertz ? 1e-4f : 1e-2f;
  const float dt = md.dt > 0.0f ? md.dt : dt0;
  const int steps0 = md.shear         ? 200
                     : md.hertz       ? 40
                     : md.hubStatic   ? 1
                     : md.hubMl       ? 10
                     : md.ringPosOnly ? 4
                                      : (hub ? 20 : (md.tri ? 3 : (md.ring ? 10 : 50)));
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
  if (md.omega > 0.0f)
    sim.parts().positionOmega = md.omega;  // WO-12 omega scan (internal knob)
  sim.debugIterationCounters(true);  // ITERS also from a device-side loop (§12 S12; no numerics)
  if (md.noStop)
    sim.debugNoAdaptiveStop(true);  // G7a / G7c run with N forced (§12 S13)
  if (!md.fused.empty())
    sim.setFusedSweeps(md.fused);  // bit-identical submission policy (CUDA); exercises S12
  sim.setMaterialParams(md.restitution, 0.0f, md.ring ? 0.02f : (md.friction ? 0.4f : 0.0f));
  if (md.jacobi)
    sim.setVelocityUseGS(false);
  if (md.gravity)
    sim.setStabilizationMode("off");  // one-sided stabilization is a momentum sink by design
  if (!md.stab.empty())
    sim.setStabilizationMode(md.stab);
  if (md.poisson)
    sim.setRestitutionModel("poisson");
  if (!md.restTarget.empty())
    sim.setRestitutionTarget(md.restTarget);  // WO-C1 A/B (default newton: never called)
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
    sim.ensureGlobalGids();  // the final (Exscan-rebased) ids before the first snapshot
  }
  const std::unordered_map<int, int> sceneOf = sceneIndexByGid(sim, gids);  // for --dump
  ColorDiag cd;                // max over steps (rank-local); every XPBD mode
  std::vector<double> keHist;  // CoM-frame kinetic energy after every (recorded) step
  // ... and its rotational part (spin KE; R-A1 of docs/contact_physics_followups.md)
  std::vector<double> keRotHist;
  double fricDist =
      0.0;  // cluster_friction: sum over steps of mean |dist| / R of friction contacts
  long fricCount = 0, fricGap = 0;

  State st = readState(sim);
  const Sums S0 = globalSums(st);
  const double M = S0.m;
  // Kinetic energy in the CoM frame (translational + rotational), host double, Allreduced.
  auto keCm = [&](const State& s, const Sums& S) {
    double kl = 0.0, kg = 0.0;
    for (int i = 0; i < static_cast<int>(s.m.size()); ++i) {
      const D3 v = at(s.v, i), w = at(s.w, i), sp = spin(s, i);
      kl += 0.5 * s.m[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) +
            0.5 * (sp[0] * w[0] + sp[1] * w[1] + sp[2] * w[2]);
    }
    MPI_Allreduce(&kl, &kg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    kg -= 0.5 * (S.P[0] * S.P[0] + S.P[1] * S.P[1] + S.P[2] * S.P[2]) / M;
    return kg;
  };
  const double ke0 = keCm(st, S0);  // before the first step (KEGATE)
  const D3 X0{S0.mx[0] / M, S0.mx[1] / M, S0.mx[2] / M}, V0{S0.P[0] / M, S0.P[1] / M, S0.P[2] / M};
  double ke0 = 0.0;  // the initial CoM-frame kinetic energy (the KEROT line's reference)
  {
    double kl = 0.0;
    for (int i = 0; i < static_cast<int>(st.m.size()); ++i) {
      const D3 v = at(st.v, i), w = at(st.w, i), sp = spin(st, i);
      kl += 0.5 * st.m[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) +
            0.5 * (sp[0] * w[0] + sp[1] * w[1] + sp[2] * w[2]);
    }
    MPI_Allreduce(&kl, &ke0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ke0 -= 0.5 * (S0.P[0] * S0.P[0] + S0.P[1] * S0.P[1] + S0.P[2] * S0.P[2]) / M;
  }
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
  // hub_static starts at rest (sum m|v| = 0): its dP is the ABSOLUTE |P(t) - P(0)| (§13.6).
  const double pNorm = pScale > 0.0 ? pScale : 1.0;
  // ITERS: the iterations the last step's main velocity / position loops ran (split_stats; the
  // stops are Allreduce-MAXed, so every rank ran the same count); the positive controls of hub_ml
  // (§13.2): hub copies, multilevel levels, mlHubAggregated, each the max over steps and ranks.
  int itVel = 0, itPos = 0, ctlLevels = 0, ctlAgg = 0, orphanClamps = 0;
  std::vector<int> itPosSteps;    // every step's main position-loop iterations (max over ranks)
  double committedOverlap = 0.0;  // --gate-overlap: max over steps of computeOverlaps (np 1)
  D3 xposAcc{0, 0, 0}, lvelAcc{0, 0, 0};
  // Periodic box: the CoM of wrapped positions jumps at a wrap, so the position-phase drift is
  // accumulated from each body's minimum-image displacement (fixed ownership, asserted below).
  D3 xposAccPer{0, 0, 0};
  double dXposPer = 0.0;
  Sums Sprev = S0;
  int fail = 0;
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
      cd.confMl = std::max(cd.confMl, c.confMl);
      cd.copiesVel = std::max(cd.copiesVel, c.copiesVel);
      cd.copiesPos = std::max(cd.copiesPos, c.copiesPos);
      const peclet::dem::SplitStats ss = sim.debugSplitStats();
      itVel = ss.velItersUsed;
      itPos = ss.posItersUsed;
      itPosSteps.push_back(ss.posItersUsed);
      ctlLevels = std::max(ctlLevels, sim.parts().mlLast.numLevels);
      ctlAgg = std::max(ctlAgg, ss.mlHubAggregated);
      orphanClamps += ss.orphanClamps;  // this step call's clamp hits (§13.3)
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
    // The committed overlap re-measures the committed state with a single-rank probe (no ghosts),
    // so it is taken at np 1 only.
    if (md.gateOverlap >= 0.0f && size == 1)
      committedOverlap = std::max(committedOverlap, static_cast<double>(sim.computeOverlaps()));
    const State nx = readState(sim);
    // Ownership is NOT fixed since WO-7 (the drift vote migrates bodies inside a step): every
    // per-body comparison across a step pairs the two states by gid (velocityPhaseTorque,
    // periodicDisplacement).
    const Sums S = globalSums(nx);
    {  // kinetic energy in the CoM frame (translational + rotational), host double
      double kl[2] = {0.0, 0.0}, kg[2] = {0.0, 0.0};
      for (int i = 0; i < static_cast<int>(nx.m.size()); ++i) {
        const D3 v = at(nx.v, i), w = at(nx.w, i), sp = spin(nx, i);
        const double kr = 0.5 * (sp[0] * w[0] + sp[1] * w[1] + sp[2] * w[2]);
        kl[0] += 0.5 * nx.m[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + kr;
        kl[1] += kr;
      }
      MPI_Allreduce(kl, kg, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      kg[0] -= 0.5 * (S.P[0] * S.P[0] + S.P[1] * S.P[1] + S.P[2] * S.P[2]) / M;
      keHist.push_back(kg[0]);
      keRotHist.push_back(kg[1]);
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
    if (md.periodic) {
      const D3 gl = periodicDisplacement(st, nx, 2.0 * -LO);
      for (int d = 0; d < 3; ++d)
        xposAccPer[d] += gl[d] / M - dt * (S.P[d] + Sprev.P[d]) / (2.0 * M);
      dXposPer = std::max(dXposPer, norm(xposAccPer) / RAD);
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
    dP = std::max(dP, norm(ep) / pNorm);
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
    // Vertices are colouring vertices: hub copy slots included (§4.4), so degVel / degPos report
    // the per-copy degree (<= 32 at a split hub); ml = multilevel same-colour pairs at an
    // aggregate; copiesVel / copiesPos = hub copy slots (max over steps and ranks).
    const int loc[12] = {cd.confM, cd.confC, cd.degM,  cd.degC,   cd.colM,      cd.colC,
                         cd.leftM, cd.leftC, cd.degPt, cd.confMl, cd.copiesVel, cd.copiesPos};
    int mx[12];
    MPI_Allreduce(loc, mx, 12, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf(
          "CONFLICTS mode=%s np=%d thr=%d vel=%d pos=%d degVel=%d degPos=%d colVel=%d "
          "colPos=%d leftVel=%d leftPos=%d degPosPt=%d ml=%d copiesVel=%d copiesPos=%d\n",
          md.name.c_str(), size, thr, mx[0], mx[1], mx[2], mx[3], mx[4], mx[5], mx[6], mx[7], mx[8],
          mx[9], mx[10], mx[11]);
  }
  if (md.poisson) {  // §13.3: the per-copy shares bound every draw by the balance
    int clampsAll = 0;
    MPI_Allreduce(&orphanClamps, &clampsAll, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf("ORPHAN mode=%s np=%d thr=%d orphanClamps=%d\n", md.name.c_str(), size, thr,
                  clampsAll);
    if (clampsAll != 0) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "GATE: %s orphan balance clamped (overdraw)\n", md.name.c_str());
    }
  }
  if (!md.hertz) {
    int loc[4] = {itVel, itPos, ctlLevels, ctlAgg}, mx[4];
    MPI_Allreduce(loc, mx, 4, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    int cv = cd.copiesVel, cvMax = 0;
    MPI_Allreduce(&cv, &cvMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    // Per step, the stops are Allreduce-MAXed, so every rank ran the same count; MAX anyway.
    std::vector<int> ps(itPosSteps.size());
    if (!itPosSteps.empty())
      MPI_Allreduce(itPosSteps.data(), ps.data(), static_cast<int>(ps.size()), MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
    const int posMax = ps.empty() ? 0 : *std::max_element(ps.begin(), ps.end());
    std::vector<int> sorted = ps;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t ns = sorted.size();
    const double posMed = ns == 0       ? 0.0
                          : ns % 2 == 1 ? sorted[ns / 2]
                                        : 0.5 * (sorted[ns / 2 - 1] + sorted[ns / 2]);
    if (rank == 0)
      std::printf("ITERS mode=%s np=%d thr=%d vel=%d pos=%d posMax=%d posMed=%.1f\n",
                  md.name.c_str(), size, thr, mx[0], mx[1], posMax, posMed);
    // G-B2 feasibility (docs/contact_physics_followups.md §6): with the stops on, a feasible
    // overlap problem converges below its cap in EVERY step; ring_mini (tunnelled) runs to it.
    if (md.gatePosCap && posMax >= md.posIters) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "GATE: %s position loop ran its cap %d (posMax %d)\n", md.name.c_str(),
                     md.posIters, posMax);
    }
    if (md.hubMl) {
      if (rank == 0)
        std::printf("MLCTRL mode=%s np=%d thr=%d velCopies=%d mlLevels=%d mlHubAggregated=%d\n",
                    md.name.c_str(), size, thr, cvMax, mx[2], mx[3]);
      // The positive controls (§13.2): the mode fails if it does not test what it claims. They
      // are required at np 1 and 2 only (§12 S18): at np 4 / 8 every rank sees the whole 181-body
      // shell but owns too few eligible contacts for the matching to build a level, so there
      // only conservation is gated.
      if (size <= 2 && ((size == 1 && cvMax <= 0) || mx[2] < 1 || mx[3] < 1)) {
        fail = 1;
        if (rank == 0)
          std::fprintf(stderr, "GATE: hub_ml positive control failed\n");
      }
    }
  }
  if (md.gateOverlap >= 0.0f && size == 1) {
    if (rank == 0)
      std::printf("OVERLAP mode=%s np=%d committed=%.3e (gate %.1e)\n", md.name.c_str(), size,
                  committedOverlap, static_cast<double>(md.gateOverlap));
    if (!(committedOverlap <= md.gateOverlap)) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "GATE: %s committed overlap %.3e > %.1e\n", md.name.c_str(),
                     committedOverlap, static_cast<double>(md.gateOverlap));
    }
  }
  if (md.hubStatic || md.hubMl) {
    const HubGap hg = hubGap(sim, gids, bodies);
    if (rank == 0)
      std::printf("HUBGAP mode=%s np=%d thr=%d maxGap/delta=%.4e residual/R=%.4e\n",
                  md.name.c_str(), size, thr, hg.maxGap / kHubDelta, hg.residual / RAD);
    // hub_static (§13.6, bound §12 S11): every leaf ends in contact with the hub and the overlaps
    // are gone. The coupling gaps are ~ k m_leaf / m_hub delta, plus the random-walk residual of
    // ~32 sequential ~0.01 delta pushes per copy before the fold (~0.06 delta), which the
    // non-retractable projection keeps; omega 1.5 leaves ~0.58 delta. Bound 0.15 delta.
    if (md.hubStatic && !(hg.maxGap <= 0.15 * kHubDelta && hg.residual <= 1e-4 * RAD)) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "GATE: hub_static leaf gap / residual overlap\n");
    }
  }
  if (rank == 0) {
    std::printf("KE mode=%s np=%d thr=%d", md.name.c_str(), size, thr);
    for (std::size_t k = 0; k < keHist.size(); ++k)
      std::printf(" s%zu=%.9e", k + 1, keHist[k]);
    std::printf("\n");
    std::printf("KEROT mode=%s np=%d thr=%d ke0=%.9e", md.name.c_str(), size, thr, ke0);
    for (std::size_t k = 0; k < keRotHist.size(); ++k)
      std::printf(" s%zu=%.9e", k + 1, keRotHist[k]);
    std::printf("\n");
  }
  // G-C1 (docs/contact_physics_followups.md §6, WO-C1): the converged one-step PGS in the dense
  // frictionless cluster. Moreau's target is energy-consistent for a uniform e (§4.1): KE may
  // not grow, and at e = 1 it is conserved up to the sub-threshold (e = 0) contacts. Newton's
  // ratio is reported (it exceeds 1 at e >= 0.9: pre-separating loaded contacts create energy).
  if (md.keGate && !keHist.empty()) {
    const bool moreau = md.restTarget == "moreau";
    const double ratio = keHist.back() / ke0;
    if (rank == 0)
      std::printf("KEGATE mode=%s np=%d thr=%d target=%s e=%.3f ke0=%.9e ke1=%.9e ratio=%.9f\n",
                  md.name.c_str(), size, thr, moreau ? "moreau" : "newton",
                  static_cast<double>(md.restitution), ke0, keHist.back(), ratio);
    if (moreau && !(ratio <= 1.0 + 1e-5 && (md.restitution != 1.0f || ratio >= 1.0 - 2e-3))) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "GATE: Moreau KE ratio %.9f outside the G-C1 bounds (e = %.3f)\n",
                     ratio, static_cast<double>(md.restitution));
    }
  }
  // G2 (docs/contact_solve_framework.md §9, the review's 3-body face scene): under policy X the
  // distributed g = 0 one-shot is a legal serial Gauss-Seidel order, so the kinetic energy after
  // the collision equals np 1's on every axis. References: np 1 (every axis, every order).
  if (md.tri && !md.gravity && !keHist.empty()) {  // the g = 0 one-shot scene only
    const double ref = md.restitution == 0.5f   ? 0.1379044264
                       : md.restitution == 0.8f ? 0.2459279908
                                                : -1.0;
    if (ref > 0.0 && std::fabs(keHist.back() - ref) > 1e-6 * ref) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "GATE: tri KE %.9e differs from np 1 %.9e\n", keHist.back(), ref);
    }
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
          "MOMENTUM mode=%s np=%d thr=%d N=%d steps=%d dP=%.3e dX=n/a dXpos=%.3e dL=%.3e "
          "dLcm=%.3e dLvel=n/a ghosts=%d ovl=%.3e\n",
          md.name.c_str(), size, thr, n, steps, dP, dXposPer, dL, dLcm, totGhost, ovl);
    else
      std::printf(
          "MOMENTUM mode=%s np=%d thr=%d N=%d steps=%d dP=%.3e dX=%.3e dXpos=%.3e dL=%.3e "
          "dLcm=%.3e dLvel=%.3e ghosts=%d ovl=%.3e\n",
          md.name.c_str(), size, thr, n, steps, dP, dX, dXpos, dL, dLcm, dLvel, totGhost, ovl);
  }
  if (rank == 0)  // WO-7: collective drift migrations (the same count on every rank)
    std::printf("DRIFT mode=%s np=%d migrations=%lld\n", md.name.c_str(), size,
                static_cast<long long>(sim.parts().splitStats.driftMigrations));
  if (!md.dump.empty())
    dumpState(sim, sceneOf, md.dump, rank, size);
  const Tol tol = tolOf(md.name);
  if (kGate && !(within(dP, tol.dP) && within(dX, tol.dX) && within(dXpos, tol.dXpos) &&
                 within(dL, tol.dL) && within(dLvel, tol.dLvel))) {
    fail = 1;
    if (rank == 0)
      std::fprintf(stderr, "GATE: %s exceeds its conservation thresholds\n", md.name.c_str());
  }
  // hub_static / hub_ml are GATED at every np since WO-5 landed the rank-level mass split
  // (docs/contact_solve_framework.md §13.5 WO-5 acceptance 3).
  const bool gatedHub = md.hubStatic || md.hubMl;
  if (((hub && !gatedHub) || md.reportOnly) && !kFollowupGate)
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
static int gPerfLattice = 27;  // perf_* lattice side (--perf-g=G; WO-11 size scan)
static int runPerf(bool pgs, int rank, int size) {
  const int G = gPerfLattice;  // default 27: 27^3 = 19683 ~ 20000 (--perf-g=G)
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
  ProbeSim sim(3 * n + 64);
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
  {  // WO-11 cost diagnostics (read after the timed loop; no effect on the timing)
    const auto& st = sim.parts().splitStats;
    int g = sim.numGhost(), gs = 0, gmax = 0;
    MPI_Allreduce(&g, &gs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&g, &gmax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf(
          "PERFDIAG mode=%s np=%d migrations=%lld ghostsTotal=%d ghostsMax=%d "
          "velItersLast=%d posItersLast=%d rebuilds=%ld gathers=%ld\n",
          pgs ? "perf_pgs" : "perf_gas", size, static_cast<long long>(st.driftMigrations), gs, gmax,
          st.velItersUsed, st.posItersUsed, sim.mpiRebuilds(), sim.mpiGathers());
  }
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
    int posItersFlag = -1;  // --pos-iters (alias --posit): likewise
    std::string stabFlag;   // --stab=<mode>: override the mode's stabilization (test-only)
    bool restitutionFlag = false;
    for (int a = 2; a < argc; ++a) {
      if (std::strncmp(argv[a], "--dump=", 7) == 0)
        md.dump = argv[a] + 7;
      else if (std::strncmp(argv[a], "--dt=", 5) == 0)
        md.dt = std::stof(argv[a] + 5);
      else if (std::strncmp(argv[a], "--posit=", 8) == 0)
        posItersFlag = std::stoi(argv[a] + 8);
      else if (std::strncmp(argv[a], "--pos-iters=", 12) == 0)
        posItersFlag = std::stoi(argv[a] + 12);
      else if (std::strncmp(argv[a], "--omega=", 8) == 0)
        md.omega = std::stof(argv[a] + 8);
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
      else if (std::strncmp(argv[a], "--stab=", 7) == 0)
        stabFlag = argv[a] + 7;
      else if (std::strncmp(argv[a], "--fused=", 8) == 0)
        md.fused = argv[a] + 8;
      else if (std::strcmp(argv[a], "--no-stop") == 0)
        md.noStop = true;
      else if (std::strncmp(argv[a], "--rest-target=", 14) == 0)
        md.restTarget = argv[a] + 14;
      else if (std::strcmp(argv[a], "--gate-pos-cap") == 0)
        md.gatePosCap = true;
      else if (std::strncmp(argv[a], "--gate-overlap=", 15) == 0)
        md.gateOverlap = std::stof(argv[a] + 15);
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
    } else if (mode == "cluster_pgs_e") {  // G-C1: mu = 0, no spins, one converged step
      md.gravity = md.keGate = true;
      md.velIters = 2000;
      md.noStop = true;
      if (md.steps <= 0)
        md.steps = 1;
    } else if (mode == "cluster_posonly") {
      md.velIters = 0;
    } else if (mode == "cluster_jacobi") {
      md.jacobi = true;
    } else if (mode == "hertz") {
      md.friction = md.spins = md.hertz = true;
    } else if (mode == "hertz_shear") {  // WO-7 acceptance: Hertz under drift, 200 steps
      md.friction = md.spins = md.hertz = md.shear = true;
    } else if (mode == "hertz_shear_frictionless") {  // isolates the Mindlin history carry
      md.spins = md.hertz = md.shear = true;
    } else if (mode == "cluster_shear") {  // the XPBD counterpart (drift vote in demStepMpi)
      md.friction = md.spins = md.shear = true;
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
    } else if (mode == "ring_collide") {
      md.ring = md.ringCollide = true;
    } else if (mode == "ring_collide_posonly") {
      md.ring = md.ringCollide = md.ringPosOnly = true;
      md.velIters = 0;
    } else if (mode == "hub_static") {  // §13.6: g = 0, velocity solve off, 64 position iterations
      if (md.hubScale <= 0.0f)
        md.hubScale = 10.0f;
      md.hubStatic = true;
      md.velIters = 0;
      md.posIters = 64;
    } else if (mode ==
               "hub_ml") {  // §13.2: free fall, frictionless, multilevel, 1 velocity iteration
      if (md.hubScale <= 0.0f)
        md.hubScale = 10.0f;
      md.hubMl = md.gravity = true;
      md.velIters = 1;
      md.stab = "multilevel";
    }
    if (velItersFlag >= 0)
      md.velIters = velItersFlag;
    if (posItersFlag >= 0)
      md.posIters = posItersFlag;
    if (!stabFlag.empty())
      md.stab = stabFlag;
    md.reportOnly = mode == "tri" || mode == "tri_pgs" || mode == "cluster_e09" ||
                    mode == "cluster_e10" || mode == "cluster_poisson" ||
                    mode == "cluster_multilevel" || mode == "cluster_escalate" ||
                    mode == "cluster_ordered" || mode == "cluster_onesided" || mode == "hub_pgs" ||
                    mode == "ring_mini";
    if (mode == "perf_gas" || mode == "perf_pgs") {
      for (int a = 2; a < argc; ++a)
        if (std::strncmp(argv[a], "--perf-g=", 9) == 0)
          gPerfLattice = std::atoi(argv[a] + 9);
      fail = runPerf(mode == "perf_pgs", rank, size);
    } else if (mode == "friction_pair" || mode == "friction_pair_pgs")
      fail = runFrictionPair(md, rank, size);
    else if (mode == "hub" || mode == "hub_posonly" || mode == "hub_pgs" || mode == "hub_static" ||
             mode == "hub_ml")
      fail = runCluster(md, rank, size);
    else if (mode == "tri" || mode == "tri_pgs" || mode == "cluster_e09" || mode == "cluster_e10" ||
             mode == "cluster_poisson" || mode == "cluster_multilevel" ||
             mode == "cluster_escalate" || mode == "cluster_ordered" ||
             mode == "cluster_onesided" || mode == "ring_mini" || mode == "ring_collide" ||
             mode == "ring_collide_posonly")
      fail = runCluster(md, rank, size);
    else if (mode == "cluster" || mode == "cluster_friction" || mode == "cluster_pgs" ||
             mode == "cluster_pgs_e" || mode == "cluster_posonly" || mode == "hertz" ||
             mode == "cluster_sync3" || mode == "cluster_norot" || mode == "cluster_periodic" ||
             mode == "cluster_jacobi" || mode == "hertz_shear" || mode == "cluster_shear" ||
             mode == "hertz_shear_frictionless")
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
