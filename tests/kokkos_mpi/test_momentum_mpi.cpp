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
//   hertz             the force-based Hertz-Mindlin engine (step_hertz_mpi), mu = 0.4, spins
//   perf_gas / perf_pgs  timing only (not a ctest): N = 20000, fully periodic, ms/step
//
// Output: one parseable line per run,
//   MOMENTUM mode=.. np=.. thr=.. N=.. steps=.. dP=.. dX=.. dXpos=.. dL=.. dLcm=.. dLvel=..
// Report-only for now: the thresholds below are the future gate (kGate = false: never fails on
// drift; a non-finite state still fails).
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
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

// ---- the future conservation gate (switch kGate on once the distributed solve conserves) ----
static constexpr bool kGate = false;
static constexpr double kTolP = 1e-6;  // relative linear momentum
static constexpr double kTolX = 1e-5;  // CoM drift / radius
static constexpr double kTolL = 1e-5;  // relative velocity-phase angular momentum (dLvel)

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

// A dense random cluster: jittered cubic lattice (spacing = base diameter, so the polydisperse
// grains start overlapping) inside a ball centred just off the origin, the common corner of every
// ORB layout,
// random velocities + a net drift + an inward radial component (keeps it colliding), optional
// random spins. Deterministic (fixed seed); every rank builds the same global set.
static std::vector<Body> makeCluster(double ballRadius, bool spins) {
  std::mt19937 rng(20260925u);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  const D3 c{0.3, -0.2, 0.1};
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
        b.push_back(q);
      }
  return b;
}

struct Sums {
  double m = 0, P[3] = {0, 0, 0}, mx[3] = {0, 0, 0}, Lo[3] = {0, 0, 0};
};
struct State {
  std::vector<float> x, v, w, m, invI;
};
static State readState(const Simulation& s) {
  State st{s.getPositions(), s.getVelocities(), s.getAngularVelocities(), s.getMasses(),
           s.getInvInertia()};
  return st;
}
static D3 at(const std::vector<float>& a, int i) {
  return {a[3 * i], a[3 * i + 1], a[3 * i + 2]};
}
static D3 spin(const State& st, int i) {  // I w for isotropic inertia
  const double iI = st.invI[3 * i];
  const D3 w = at(st.w, i);
  return iI > 0 ? D3{w[0] / iI, w[1] / iI, w[2] / iI} : D3{0, 0, 0};
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
    const D3 c = cross(xp, dv), s0 = spin(a, i), s1 = spin(b, i);
    for (int d = 0; d < 3; ++d)
      loc[d] += a.m[i] * c[d] + (s1[d] - s0[d]);
  }
  D3 out;
  MPI_Allreduce(loc, out.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return out;
}

struct Mode {
  std::string name;
  bool friction = false, spins = false, gravity = false, hertz = false;
  int velIters = 8;
};

static int runCluster(const Mode& md, int rank, int size) {
  const std::vector<Body> bodies = makeCluster(6.0, md.spins);
  const int n = static_cast<int>(bodies.size());
  const float dt = md.hertz ? 1e-4f : 1e-2f;
  const int steps = md.hertz ? 40 : 50;
  const int sub = md.hertz ? 25 : 1;  // Hertz: substeps per recorded step
  const D3 g = md.gravity ? D3{0.0, 0.0, -10.0} : D3{0, 0, 0};

  const std::vector<int> gids = ownedOf(bodies, LO, L, GX, false, rank, size);
  Simulation sim(2 * n + 64);
  sim.setDomain(L, L, L, false, false, false);
  sim.setDomainMinMax(peclet::dem::F3{-16.0f, -16.0f, -16.0f},
                      peclet::dem::F3{16.0f, 16.0f, 16.0f});
  sim.setGlobalScale(1.0f);
  sim.setSphereShape(RAD);
  sim.setDt(dt);
  sim.setGravity(static_cast<float>(g[0]), static_cast<float>(g[1]), static_cast<float>(g[2]));
  sim.setSolverIterations(20, md.velIters);
  sim.setMaterialParams(0.5f, 0.0f, md.friction ? 0.4f : 0.0f);
  if (md.gravity)
    sim.setStabilizationMode("off");  // one-sided stabilization is a momentum sink by design
  if (md.hertz) {
    sim.setHertzMaterial(0, 1.0e5f, 0.25f);
  }
  load(sim, bodies, gids);
  const std::tuple<double, double, double> origin{LO, LO, LO}, dsize{L, L, L};
  const std::tuple<long, long, long> gsize{GX, GX, GX};
  const std::tuple<bool, bool, bool> per{false, false, false};
  sim.initMpi(origin, dsize, gsize, per, MPI_COMM_WORLD);
  sim.enableMpiStep(0.0, 1, /*forward_rotation=*/true);

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

  double dP = 0, dX = 0, dXpos = 0, dL = 0, dLcm = 0, dLvel = 0;
  D3 xposAcc{0, 0, 0}, lvelAcc{0, 0, 0};
  Sums Sprev = S0;
  int fail = 0;
  const int nOwned = static_cast<int>(gids.size());
  for (int s = 1; s <= steps; ++s) {
    if (md.hertz)
      sim.stepHertzMpi(sub, 0.3f);
    else
      sim.stepMpi(1);
    const State nx = readState(sim);
    if (static_cast<int>(nx.m.size()) != nOwned)
      fail = 1;  // the per-body velocity-phase record assumes fixed ownership
    const Sums S = globalSums(nx);
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
  int ghosts = sim.numGhost(), totGhost = 0;
  MPI_Allreduce(&ghosts, &totGhost, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  const int thr = Kokkos::DefaultHostExecutionSpace().concurrency();
  if (rank == 0) {
    if (md.hertz)
      std::printf(
          "MOMENTUM mode=%s np=%d thr=%d N=%d steps=%d dP=%.3e dX=%.3e dXpos=n/a dL=%.3e "
          "dLcm=%.3e dLvel=n/a ghosts=%d\n",
          md.name.c_str(), size, thr, n, steps * sub, dP, dX, dL, dLcm, totGhost);
    else
      std::printf(
          "MOMENTUM mode=%s np=%d thr=%d N=%d steps=%d dP=%.3e dX=%.3e dXpos=%.3e dL=%.3e "
          "dLcm=%.3e dLvel=%.3e ghosts=%d\n",
          md.name.c_str(), size, thr, n, steps, dP, dX, dXpos, dL, dLcm, dLvel, totGhost);
  }
  if (kGate && (!(dP < kTolP) || !(dX < kTolX) || (!md.hertz && !(dLvel < kTolL))))
    fail = 1;
  return fail;
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
    if (mode == "cluster") {
    } else if (mode == "cluster_friction") {
      md.friction = md.spins = true;
    } else if (mode == "cluster_pgs") {
      md.friction = md.spins = md.gravity = true;
    } else if (mode == "cluster_posonly") {
      md.velIters = 0;
    } else if (mode == "hertz") {
      md.friction = md.spins = md.hertz = true;
    }
    if (mode == "perf_gas" || mode == "perf_pgs")
      fail = runPerf(mode == "perf_pgs", rank, size);
    else if (mode == "cluster" || mode == "cluster_friction" || mode == "cluster_pgs" ||
             mode == "cluster_posonly" || mode == "hertz")
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
