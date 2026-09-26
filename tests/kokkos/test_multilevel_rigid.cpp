// Oracle of the multilevel coarse cycle's rigid 6-DOF aggregates (docs/contact_physics_followups.md
// §2.3-§2.4, gate G-A2). 40 bodies (spheres + tubes with random inertia frames), 2 hand-built
// levels (pairs at level 1, pairs of pairs at level 2, singletons at both), one wall, random v and
// omega and a random warm accumulator. ONE coarse cycle on the device (the level geometry of
// buildMlLevelGeometryKokkos + multilevelCoarseCycleKokkos) against a host DOUBLE reference of the
// same algebra:
//   rigid      : per-group V, Omega and member v, omega within 1e-5 (relative to the largest);
//                |dP| <= 1e-6 sum|J|, |dL| <= 1e-6 sum|J| R_g,max (wall impulses subtracted);
//                KE_after <= KE_before (1 + 1e-6); a singleton row's w and vn equal the fine PGS
//                row's within 1e-6 relative.
//   invI_g = 0 : the translation-only cycle (the release fallback of a singular group, applied to
//                every group after the build); it must REPRODUCE dL = sum (X_A - X_B) x J (plus the
//                wall's lever error) -- which proves the dL measurement discriminates.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "solver_multilevel.hpp"

using namespace peclet::dem;

namespace {

struct Vd {
  double x = 0, y = 0, z = 0;
};
Vd vd(double x, double y, double z) {
  return Vd{x, y, z};
}
Vd operator+(Vd a, Vd b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vd operator-(Vd a, Vd b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vd operator*(Vd a, double s) {
  return {a.x * s, a.y * s, a.z * s};
}
double dot(Vd a, Vd b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vd cross(Vd a, Vd b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double norm(Vd a) {
  return std::sqrt(dot(a, a));
}
struct M3 {
  double a[3][3] = {};
};
Vd mul(const M3& m, Vd v) {
  return {m.a[0][0] * v.x + m.a[0][1] * v.y + m.a[0][2] * v.z,
          m.a[1][0] * v.x + m.a[1][1] * v.y + m.a[1][2] * v.z,
          m.a[2][0] * v.x + m.a[2][1] * v.y + m.a[2][2] * v.z};
}
M3 inv3(const M3& m) {
  const auto& a = m.a;
  M3 r;
  const double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                     a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                     a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  r.a[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
  r.a[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
  r.a[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
  r.a[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
  r.a[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
  r.a[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
  r.a[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
  r.a[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
  r.a[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
  return r;
}
// active rotation matrix of a unit quaternion (x, y, z, w)
M3 rotm(const float* q) {
  const double x = q[0], y = q[1], z = q[2], w = q[3];
  M3 R;
  R.a[0][0] = 1 - 2 * (y * y + z * z);
  R.a[0][1] = 2 * (x * y - z * w);
  R.a[0][2] = 2 * (x * z + y * w);
  R.a[1][0] = 2 * (x * y + z * w);
  R.a[1][1] = 1 - 2 * (x * x + z * z);
  R.a[1][2] = 2 * (y * z - x * w);
  R.a[2][0] = 2 * (x * z - y * w);
  R.a[2][1] = 2 * (y * z + x * w);
  R.a[2][2] = 1 - 2 * (x * x + y * y);
  return R;
}

constexpr int N = 40;
constexpr int kSweeps = 2;

struct Scene {
  std::vector<float> invMass, invI, quat, pos, vel, ang, lambda;  // flat: [N], [N*3], [N*4], ...
  std::vector<ManifoldC> man;
  std::vector<int> parent;  // level 1: [0, N); level 2: [N, N + ng1)
  int ng1 = 0, ng2 = 0;
  std::vector<long long> colorPacked;
  int nCol[2] = {0, 0};
  int wallIdx = -1, singletonIdx = -1;
  std::vector<float> massG, invMassG;  // pooled [ng1 + ng2]
};

// World spin inertia (true inertia) of body i.
M3 worldInertia(const Scene& s, int i) {
  const M3 R = rotm(&s.quat[4 * i]);
  M3 J;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) {
      double acc = 0;
      for (int k = 0; k < 3; ++k)
        acc += R.a[r][k] * (1.0 / s.invI[3 * i + k]) * R.a[c][k];
      J.a[r][c] = acc;
    }
  return J;
}

Scene makeScene() {
  Scene s;
  std::mt19937 rng(20260926);
  std::uniform_real_distribution<float> u(-1.f, 1.f), up(0.5f, 2.0f);
  std::normal_distribution<float> nq(0.f, 1.f);
  s.invMass.resize(N);
  s.invI.resize(3 * N);
  s.quat.resize(4 * N);
  s.pos.resize(3 * N);
  s.vel.resize(3 * N);
  s.ang.resize(3 * N);
  for (int i = 0; i < N; ++i) {
    s.invMass[i] = up(rng);
    const float m = 1.0f / s.invMass[i];
    if (i % 2 == 0) {  // sphere r = 0.5: I = 0.4 m r^2 (isotropic)
      const float ii = 1.0f / (0.4f * m * 0.25f);
      s.invI[3 * i] = s.invI[3 * i + 1] = s.invI[3 * i + 2] = ii;
    } else {  // tube-like: distinct principal moments
      s.invI[3 * i] = 1.0f / (m * (0.2f + 0.1f * up(rng)));
      s.invI[3 * i + 1] = 1.0f / (m * (0.35f + 0.1f * up(rng)));
      s.invI[3 * i + 2] = 1.0f / (m * (0.2f + 0.05f * up(rng)));
    }
    float a = nq(rng), b = nq(rng), c = nq(rng), d = nq(rng);
    const float nr = std::sqrt(a * a + b * b + c * c + d * d);
    s.quat[4 * i] = a / nr;
    s.quat[4 * i + 1] = b / nr;
    s.quat[4 * i + 2] = c / nr;
    s.quat[4 * i + 3] = d / nr;
    for (int k = 0; k < 3; ++k) {
      s.pos[3 * i + k] = 3.0f * u(rng);
      s.vel[3 * i + k] = u(rng);
      s.ang[3 * i + k] = u(rng);
    }
  }
  // levels: bodies 2k, 2k+1 (k < 16) -> level-1 group k; bodies 32..39 singletons (groups 16..23).
  // level 2: level-1 groups 2k, 2k+1 (k < 7) -> group k; level-1 groups 14..23 singletons.
  s.ng1 = 24;
  s.ng2 = 17;
  s.parent.resize(N + s.ng1);
  for (int i = 0; i < N; ++i)
    s.parent[i] = i < 32 ? i / 2 : 16 + (i - 32);
  for (int g = 0; g < s.ng1; ++g)
    s.parent[N + g] = g < 14 ? g / 2 : 7 + (g - 14);
  // manifolds: random body pairs (one or two contact points meeting at a midpoint), one wall
  std::uniform_int_distribution<int> ub(0, N - 1);
  auto addPoint = [&](ManifoldC& m, int a, int b, int t) {
    const Vd xa = vd(s.pos[3 * a], s.pos[3 * a + 1], s.pos[3 * a + 2]);
    const Vd xb = vd(s.pos[3 * b], s.pos[3 * b + 1], s.pos[3 * b + 2]);
    const Vd p = (xa + xb) * 0.5 + vd(0.3 * u(rng), 0.3 * u(rng), 0.3 * u(rng));
    Vd n = (xb - xa) + vd(0.4 * u(rng), 0.4 * u(rng), 0.4 * u(rng));
    n = n * (1.0 / norm(n));
    if (t)
      n = n * -1.0;  // either orientation: the row's sign comes from the alignment
    const float nx = float(n.x), ny = float(n.y), nz = float(n.z);
    const float rax = float(p.x - xa.x), ray = float(p.y - xa.y), raz = float(p.z - xa.z);
    const float rbx = float(p.x - xb.x), rby = float(p.y - xb.y), rbz = float(p.z - xb.z);
    m.normal_sum.x += nx;
    m.normal_sum.y += ny;
    m.normal_sum.z += nz;
    m.rA_sum.x += rax;
    m.rA_sum.y += ray;
    m.rA_sum.z += raz;
    m.rB_sum.x += rbx;
    m.rB_sum.y += rby;
    m.rB_sum.z += rbz;
    m.torque_armA_sum.x += ray * nz - raz * ny;  // rA x n
    m.torque_armA_sum.y += raz * nx - rax * nz;
    m.torque_armA_sum.z += rax * ny - ray * nx;
    m.torque_armB_sum.x -= rby * nz - rbz * ny;  // -(rB x n)
    m.torque_armB_sum.y -= rbz * nx - rbx * nz;
    m.torque_armB_sum.z -= rbx * ny - rby * nx;
    m.num_points += 1;
  };
  auto blank = [](int a, int b) {
    ManifoldC m{};
    m.bodyA = a;
    m.bodyB = b;
    m.num_points = 0;
    return m;
  };
  for (int k = 0; k < 110; ++k) {
    int a = ub(rng), b = ub(rng);
    if (a == b)
      continue;
    if (a > b)
      std::swap(a, b);
    ManifoldC m = blank(a, b);
    const int t = static_cast<int>(rng() & 1);
    addPoint(m, a, b, t);
    if ((rng() & 3) == 0)
      addPoint(m, a, b, t);
    s.man.push_back(m);
  }
  {  // the singleton row: bodies 34 and 35 (singletons at both levels)
    ManifoldC m = blank(34, 35);
    addPoint(m, 34, 35, 0);
    s.singletonIdx = static_cast<int>(s.man.size());
    s.man.push_back(m);
  }
  {  // the wall: under body 0 (a member of the level-1 pair {0, 1}), normal +z, contact 0.5 below
    ManifoldC m = blank(0, -1);
    m.normal_sum = F4{0.f, 0.f, 1.f, 0.f};
    m.rA_sum = F4{0.1f, -0.05f, -0.5f, 0.f};
    const float rx = 0.1f, ry = -0.05f, rz = -0.5f;
    m.torque_armA_sum = F4{ry * 1.f - rz * 0.f, rz * 0.f - rx * 1.f, 0.f, 0.f};
    m.num_points = 1;
    s.wallIdx = static_cast<int>(s.man.size());
    s.man.push_back(m);
    s.vel[2] = -1.5f;  // body 0 approaching the wall
  }
  const int M = static_cast<int>(s.man.size());
  s.lambda.assign(M, 0.0f);
  for (int k = 0; k < M; ++k)
    if (rng() & 1)
      s.lambda[k] = 0.05f * (u(rng) + 1.0f);
  // colouring per level: greedy lowest free colour over the group endpoints (group-disjoint)
  s.colorPacked.assign(M, ~0ll);
  for (int lvl = 1; lvl <= 2; ++lvl) {
    const int ng = lvl == 1 ? s.ng1 : s.ng2;
    std::vector<unsigned long long> used(ng, 0);
    int maxc = -1;
    for (int k = 0; k < M; ++k) {
      auto grpOf = [&](int b) {
        int g = s.parent[b];
        if (lvl == 2)
          g = s.parent[N + g];
        return g;
      };
      const int gA = grpOf(s.man[k].bodyA);
      const int gB = s.man[k].bodyB >= 0 ? grpOf(s.man[k].bodyB) : -1;
      if (gA == gB)
        continue;
      const unsigned long long f = used[gA] | (gB >= 0 ? used[gB] : 0ull);
      int c = 0;
      while ((f >> c) & 1ull)
        ++c;
      const int sh = 6 * (lvl - 1);
      s.colorPacked[k] = (s.colorPacked[k] & ~(63ll << sh)) | (static_cast<long long>(c) << sh);
      used[gA] |= 1ull << c;
      if (gB >= 0)
        used[gB] |= 1ull << c;
      maxc = std::max(maxc, c);
    }
    s.nCol[lvl - 1] = maxc + 1;
  }
  // group masses (the build's float sums; order immaterial here)
  s.massG.assign(s.ng1 + s.ng2, 0.0f);
  for (int i = 0; i < N; ++i) {
    const float m = 1.0f / std::max(s.invMass[i], 1e-30f);
    const int g1 = s.parent[i], g2 = s.parent[N + g1];
    s.massG[g1] += m;
    s.massG[s.ng1 + g2] += m;
  }
  s.invMassG.resize(s.massG.size());
  for (std::size_t g = 0; g < s.massG.size(); ++g)
    s.invMassG[g] = 1.0f / s.massG[g];
  return s;
}

// ---------------------------------------------------------------- host double reference
struct RefOut {
  std::vector<Vd> v, w;  // members after the cycle
  std::vector<Vd> V, W;  // per pooled group: after its level's sweeps
  double sumJ = 0;       // sum of |d| |N| over every coarse update
  double rgMax = 0;      // largest member offset |d|
  Vd wallP, wallL;       // the wall's linear / angular impulse on the bodies (true lever)
  Vd lvlErrL;            // translation-only: sum of the lever errors (X_A - X_B) x J, ...
};

RefOut reference(const Scene& s, bool rot) {
  RefOut R;
  const int M = static_cast<int>(s.man.size());
  std::vector<double> lam(s.lambda.begin(), s.lambda.end());
  R.v.resize(N);
  R.w.resize(N);
  for (int i = 0; i < N; ++i) {
    R.v[i] = vd(s.vel[3 * i], s.vel[3 * i + 1], s.vel[3 * i + 2]);
    R.w[i] = vd(s.ang[3 * i], s.ang[3 * i + 1], s.ang[3 * i + 2]);
  }
  R.V.resize(s.ng1 + s.ng2);
  R.W.resize(s.ng1 + s.ng2);
  auto X = [&](int i) { return vd(s.pos[3 * i], s.pos[3 * i + 1], s.pos[3 * i + 2]); };
  for (int lvl = 1; lvl <= 2; ++lvl) {
    const int ng = lvl == 1 ? s.ng1 : s.ng2, off = lvl == 1 ? 0 : s.ng1;
    std::vector<int> grp(N);
    for (int i = 0; i < N; ++i)
      grp[i] = lvl == 1 ? s.parent[i] : s.parent[N + s.parent[i]];
    std::vector<double> Mg(ng, 0);
    std::vector<Vd> Xg(ng);
    for (int i = 0; i < N; ++i) {
      const double m = 1.0 / s.invMass[i];
      Mg[grp[i]] += m;
      Xg[grp[i]] = Xg[grp[i]] + X(i) * m;
    }
    for (int g = 0; g < ng; ++g)
      Xg[g] = Xg[g] * (1.0 / Mg[g]);
    std::vector<M3> Ig(ng), invIg(ng);
    std::vector<Vd> Lg(ng), Vg(ng);
    for (int i = 0; i < N; ++i) {
      const double m = 1.0 / s.invMass[i];
      const int g = grp[i];
      const Vd d = X(i) - Xg[g];
      R.rgMax = std::max(R.rgMax, norm(d));
      const M3 J = worldInertia(s, i);
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
          const double dr = r == 0 ? d.x : r == 1 ? d.y : d.z;
          const double dc = c == 0 ? d.x : c == 1 ? d.y : d.z;
          Ig[g].a[r][c] += J.a[r][c] + m * ((r == c ? dot(d, d) : 0.0) - dr * dc);
        }
      Vg[g] = Vg[g] + R.v[i] * m;
      Lg[g] = Lg[g] + cross(d, R.v[i]) * m + mul(J, R.w[i]);
    }
    std::vector<Vd> V0(ng), W(ng), W0(ng);
    for (int g = 0; g < ng; ++g) {
      invIg[g] = inv3(Ig[g]);
      Vg[g] = Vg[g] * (1.0 / Mg[g]);
      V0[g] = Vg[g];
      W[g] = rot ? mul(invIg[g], Lg[g]) : Vd{};
      W0[g] = W[g];
    }
    const int sh = 6 * (lvl - 1);
    const int nCol = s.nCol[lvl - 1];
    for (int sw = 0; sw < kSweeps; ++sw)
      for (int col = 0; col < nCol; ++col)
        for (int k = 0; k < M; ++k) {
          if (static_cast<int>((s.colorPacked[k] >> sh) & 63) != col)
            continue;
          const ManifoldC& m = s.man[k];
          const int a = m.bodyA, b = m.bodyB;
          const int gA = grp[a], gB = b >= 0 ? grp[b] : -1;
          const double invN = 1.0 / m.num_points;
          const Vd Nn = vd(m.normal_sum.x, m.normal_sum.y, m.normal_sum.z);
          const Vd rA = vd(m.rA_sum.x, m.rA_sum.y, m.rA_sum.z) * invN;
          const Vd rB = vd(m.rB_sum.x, m.rB_sum.y, m.rB_sum.z) * invN;
          const Vd diff = b < 0 ? rA : rA - rB;
          const double sgn = dot(Nn, diff) > 0 ? 1.0 : -1.0;
          const Vd TauA = vd(m.torque_armA_sum.x, m.torque_armA_sum.y, m.torque_armA_sum.z);
          const Vd TauB = vd(m.torque_armB_sum.x, m.torque_armB_sum.y, m.torque_armB_sum.z);
          const Vd TA = TauA + cross(X(a) - Xg[gA], Nn);
          const Vd TB = b >= 0 ? TauB - cross(X(b) - Xg[gB], Nn) : Vd{};
          const double iMA = 1.0 / Mg[gA], iMB = b >= 0 ? 1.0 / Mg[gB] : 0.0;
          double vn = dot(Vg[gA] - (b >= 0 ? Vg[gB] : Vd{}), Nn);
          double w = dot(Nn, Nn) * (iMA + iMB);
          if (rot) {
            vn += dot(TA, W[gA]) + (b >= 0 ? dot(TB, W[gB]) : 0.0);
            w += dot(TA, mul(invIg[gA], TA)) + (b >= 0 ? dot(TB, mul(invIg[gB], TB)) : 0.0);
          }
          const double pOld = lam[k];
          const double pNew = std::max(0.0, pOld + sgn * vn / w);
          const double d = pNew - pOld;
          if (d == 0.0)
            continue;
          lam[k] = pNew;
          R.sumJ += std::fabs(d) * norm(Nn);
          const double lp = -sgn * d;
          const Vd J = Nn * lp;
          Vg[gA] = Vg[gA] + J * iMA;
          if (rot)
            W[gA] = W[gA] + mul(invIg[gA], TA * lp);
          if (b >= 0) {
            Vg[gB] = Vg[gB] - J * iMB;
            if (rot)
              W[gB] = W[gB] + mul(invIg[gB], TB * lp);
            R.lvlErrL = R.lvlErrL + cross(Xg[gA] - Xg[gB], J);
          } else {
            R.wallP = R.wallP + J;
            const Vd trueL = cross(X(a), J) + TauA * lp;
            R.wallL = R.wallL + trueL;
            R.lvlErrL = R.lvlErrL + cross(Xg[gA], J) - trueL;
          }
        }
    for (int i = 0; i < N; ++i) {
      const int g = grp[i];
      const Vd dW = W[g] - W0[g];
      R.v[i] = R.v[i] + (Vg[g] - V0[g]) + cross(dW, X(i) - Xg[g]);
      R.w[i] = R.w[i] + dW;
    }
    for (int g = 0; g < ng; ++g) {
      R.V[off + g] = Vg[g];
      R.W[off + g] = W[g];
    }
  }
  return R;
}

// ---------------------------------------------------------------- device run
struct DevOut {
  std::vector<float> vel, ang, lambda, velG, angG;
  float rowVn = 0, rowW = 0;  // the singleton row at level 1 (rigid run only)
  std::vector<float> rowVel;  // the level-1 group states the row read (V, Omega of 34 / 35)
};

DevOut runDevice(const Scene& s, bool rot) {
  const int M = static_cast<int>(s.man.size());
  const int ngTot = s.ng1 + s.ng2;
  auto up1 = [](const std::vector<float>& h, const char* name) {
    Kokkos::View<float*, CpMem> d(name, h.size());
    auto hm = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < h.size(); ++i)
      hm(i) = h[i];
    Kokkos::deep_copy(d, hm);
    return d;
  };
  auto upN = [](const std::vector<float>& h, auto d) {
    auto hm = Kokkos::create_mirror_view(d);
    const std::size_t w = d.extent(1);
    for (std::size_t i = 0; i < d.extent(0); ++i)
      for (std::size_t c = 0; c < w; ++c)
        hm(i, c) = h[i * w + c];
    Kokkos::deep_copy(d, hm);
    return d;
  };
  auto invMass = up1(s.invMass, "invMass");
  auto invI = upN(s.invI, Kokkos::View<float* [3], CpMem>("invI", N));
  auto quat = upN(s.quat, Kokkos::View<float* [4], CpMem>("quat", N));
  auto pos = upN(s.pos, Kokkos::View<float* [3], CpMem>("pos", N));
  auto vel = upN(s.vel, Kokkos::View<float* [3], CpMem>("vel", N));
  auto ang = upN(s.ang, Kokkos::View<float* [3], CpMem>("ang", N));
  auto lambda = up1(s.lambda, "lambda");
  Kokkos::View<ManifoldC*, CpMem> man("man", M);
  {
    auto hm = Kokkos::create_mirror_view(man);
    for (int k = 0; k < M; ++k)
      hm(k) = s.man[k];
    Kokkos::deep_copy(man, hm);
  }
  Kokkos::View<int*, CpMem> realIdx("realIdx", N);
  Kokkos::parallel_for(
      "ri", Kokkos::RangePolicy<CpExec>(0, N), KOKKOS_LAMBDA(int i) { realIdx(i) = i; });
  MlScratch S;
  S.colorPacked = Kokkos::View<long long*, CpMem>("cp", M);
  {
    auto hm = Kokkos::create_mirror_view(S.colorPacked);
    for (int k = 0; k < M; ++k)
      hm(k) = s.colorPacked[k];
    Kokkos::deep_copy(S.colorPacked, hm);
  }
  S.parent = Kokkos::View<int*, CpMem>("parent", N + s.ng1);
  {
    auto hm = Kokkos::create_mirror_view(S.parent);
    for (int k = 0; k < N + s.ng1; ++k)
      hm(k) = s.parent[k];
    Kokkos::deep_copy(S.parent, hm);
  }
  S.massG = up1(s.massG, "massG");
  S.invMassG = up1(s.invMassG, "invMassG");
  S.velG = Kokkos::View<float* [3], CpMem>("velG", ngTot);
  S.velG0 = Kokkos::View<float* [3], CpMem>("velG0", ngTot);
  S.grp = Kokkos::View<int*, CpMem>("grp", N);
  S.mate = Kokkos::View<int*, CpMem>("mate", N);
  S.originG = Kokkos::View<float* [3], CpMem>("O", ngTot);
  S.comOffG = Kokkos::View<float* [3], CpMem>("C", ngTot);
  S.angG = Kokkos::View<float* [3], CpMem>("angG", ngTot);
  S.angG0 = Kokkos::View<float* [3], CpMem>("angG0", ngTot);
  S.invIG = Kokkos::View<float* [6], CpMem>("invIG", ngTot);
  S.accD = Kokkos::View<double* [7], CpMem>("accD", N);
  ContactHierarchy H;
  H.numLevels = 2;
  H.groupOff = {0, s.ng1};
  H.numGroups = {s.ng1, s.ng2};
  H.parentOff = {0, N};
  H.numColors = {s.nCol[0], s.nCol[1]};
  // the level geometry, over the composed map of each level (as the hierarchy build does)
  const Kokkos::View<const float* [3], CpMem> posC(pos);
  const MlBodyViews body{posC, ang, Kokkos::View<const float* [4], CpMem>(quat),
                         Kokkos::View<const float* [3], CpMem>(invI)};
  int bad = 0;
  for (int lvl = 1; lvl <= 2; ++lvl) {
    auto grp = S.grp;
    auto parent = S.parent;
    const int pOff = H.parentOff[lvl - 1];
    Kokkos::parallel_for(
        "compose", Kokkos::RangePolicy<CpExec>(0, N),
        KOKKOS_LAMBDA(int i) { grp(i) = parent(pOff + (lvl == 1 ? i : grp(i))); });
    bad += buildMlLevelGeometryKokkos(S, H.groupOff[lvl - 1], H.numGroups[lvl - 1], N, posC,
                                      Kokkos::View<const float*, CpMem>(invMass), body.quat,
                                      body.invIc);
  }
  if (bad)
    std::printf("  WARNING: %d singular groups\n", bad);
  if (!rot)
    Kokkos::deep_copy(S.invIG, 0.0f);  // every group translation-only (the release fallback)
  Kokkos::View<float, CpMem> maxApp("maxApp");
  multilevelCoarseCycleKokkos(man, M, realIdx, invMass, vel, lambda, maxApp, N, H, S, kSweeps,
                              body);
  Kokkos::fence();
  DevOut o;
  auto down = [](auto d, std::vector<float>& h) {
    auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
    const std::size_t w = d.rank() > 1 ? d.extent(1) : 1;
    h.resize(d.extent(0) * w);
    for (std::size_t i = 0; i < d.extent(0); ++i)
      for (std::size_t c = 0; c < w; ++c) {
        if constexpr (decltype(hm)::rank() > 1)
          h[i * w + c] = hm(i, c);
        else
          h[i] = hm(i);
      }
  };
  down(vel, o.vel);
  down(ang, o.ang);
  down(lambda, o.lambda);
  down(S.velG, o.velG);
  down(S.angG, o.angG);
  if (rot) {
    // the singleton row at level 1 (grp = the level-1 map), on the coarse state V = v, Omega =
    // omega of the two singleton bodies' INITIAL state (a converged row has vn ~ 0)
    auto grp = S.grp;
    auto parent = S.parent;
    auto velG = S.velG;
    auto angG = S.angG;
    const int bA = 34, bB = 35;
    Kokkos::parallel_for(
        "compose1", Kokkos::RangePolicy<CpExec>(0, N),
        KOKKOS_LAMBDA(int i) { grp(i) = parent(i); });
    const float sv[12] = {s.vel[3 * bA], s.vel[3 * bA + 1], s.vel[3 * bA + 2],
                          s.ang[3 * bA], s.ang[3 * bA + 1], s.ang[3 * bA + 2],
                          s.vel[3 * bB], s.vel[3 * bB + 1], s.vel[3 * bB + 2],
                          s.ang[3 * bB], s.ang[3 * bB + 1], s.ang[3 * bB + 2]};
    const int gA1 = s.parent[bA], gB1 = s.parent[bB];
    Kokkos::parallel_for(
        "setRow", Kokkos::RangePolicy<CpExec>(0, 1), KOKKOS_LAMBDA(int) {
          for (int c = 0; c < 3; ++c) {
            velG(gA1, c) = sv[c];
            angG(gA1, c) = sv[3 + c];
            velG(gB1, c) = sv[6 + c];
            angG(gB1, c) = sv[9 + c];
          }
        });
    const MlCoarseSweep f = makeMlCoarseSweep(man, realIdx, S, lambda, maxApp,
                                              Kokkos::View<const float*, CpMem>(), body);
    Kokkos::View<float[2], CpMem> out("rowOut");
    const int idx = s.singletonIdx;
    Kokkos::parallel_for(
        "row", Kokkos::RangePolicy<CpExec>(0, 1), KOKKOS_LAMBDA(int) {
          MlCoarseSweep::Row r;
          if (f.row(idx, 0, r)) {
            out(0) = r.vn;
            out(1) = r.w;
          }
        });
    auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    o.rowVn = ho(0);
    o.rowW = ho(1);
  }
  return o;
}

// ---------------------------------------------------------------- measurements
struct Budget {
  Vd P, L;
  double KE = 0;
};
Budget budget(const Scene& s, const std::vector<float>& v, const std::vector<float>& w) {
  Budget b;
  for (int i = 0; i < N; ++i) {
    const double m = 1.0 / s.invMass[i];
    const Vd x = vd(s.pos[3 * i], s.pos[3 * i + 1], s.pos[3 * i + 2]);
    const Vd vi = vd(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
    const Vd wi = vd(w[3 * i], w[3 * i + 1], w[3 * i + 2]);
    const M3 J = worldInertia(s, i);
    b.P = b.P + vi * m;
    b.L = b.L + cross(x, vi) * m + mul(J, wi);
    b.KE += 0.5 * m * dot(vi, vi) + 0.5 * dot(wi, mul(J, wi));
  }
  return b;
}

double relErrV(const std::vector<float>& dev, const std::vector<Vd>& ref, std::size_t n) {
  double e = 0, sc = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Vd d = vd(dev[3 * i], dev[3 * i + 1], dev[3 * i + 2]);
    e = std::max(e, norm(d - ref[i]));
    sc = std::max(sc, norm(ref[i]));
  }
  return e / std::max(sc, 1e-30);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const Scene s = makeScene();
    const int ngTot = s.ng1 + s.ng2;
    const Budget b0 = budget(s, s.vel, s.ang);
    auto wallImpulse = [&](const DevOut& o, Vd& P, Vd& L) {
      const ManifoldC& m = s.man[s.wallIdx];
      const Vd Nn = vd(m.normal_sum.x, m.normal_sum.y, m.normal_sum.z);
      const double sgn = (m.normal_sum.z * m.rA_sum.z > 0) ? 1.0 : -1.0;  // alignment N.rA
      const double lp = -sgn * (double(o.lambda[s.wallIdx]) - double(s.lambda[s.wallIdx]));
      const Vd x0 = vd(s.pos[0], s.pos[1], s.pos[2]);
      P = Nn * lp;
      L = cross(x0, P) + vd(m.torque_armA_sum.x, m.torque_armA_sum.y, m.torque_armA_sum.z) * lp;
    };

    // ------------------------------------------------ the rigid coarse space
    {
      const RefOut R = reference(s, true);
      const DevOut o = runDevice(s, true);
      const double eV = relErrV(o.velG, R.V, ngTot), eW = relErrV(o.angG, R.W, ngTot);
      const double ev = relErrV(o.vel, R.v, N), ew = relErrV(o.ang, R.w, N);
      const Budget b1 = budget(s, o.vel, o.ang);
      Vd wP, wL;
      wallImpulse(o, wP, wL);
      const double dP = norm(b1.P - b0.P - wP), dL = norm(b1.L - b0.L - wL);
      const double tolP = 1e-6 * R.sumJ, tolL = 1e-6 * R.sumJ * R.rgMax;
      std::printf("rot:   group V %.2e  Omega %.2e  member v %.2e  omega %.2e (tol 1e-5)\n", eV, eW,
                  ev, ew);
      std::printf("rot:   |dP| %.3e (tol %.3e)  |dL| %.3e (tol %.3e)  sum|J| %.3f  Rg %.3f\n", dP,
                  tolP, dL, tolL, R.sumJ, R.rgMax);
      std::printf("rot:   KE %.9f -> %.9f  (ratio %.3e)\n", b0.KE, b1.KE, b1.KE / b0.KE - 1.0);
      if (!(eV <= 1e-5 && eW <= 1e-5 && ev <= 1e-5 && ew <= 1e-5)) {
        std::printf("FAIL: rigid cycle differs from the double reference\n");
        status = 1;
      }
      if (!(dP <= tolP && dL <= tolL)) {
        std::printf("FAIL: rigid cycle does not conserve P / L\n");
        status = 1;
      }
      if (!(b1.KE <= b0.KE * (1.0 + 1e-6))) {
        std::printf("FAIL: rigid cycle raised the kinetic energy\n");
        status = 1;
      }
      if (R.sumJ <= 0.0 || o.lambda[s.wallIdx] == s.lambda[s.wallIdx]) {
        std::printf("FAIL: the scene exercised no coarse impulse / no wall impulse\n");
        status = 1;
      }
      // singleton row: bodies 34 (level-1 group 18) and 35 (group 19); the fine PGS row (no
      // growth) on the same state, the bodies' initial (v, omega)
      {
        const ManifoldC& m = s.man[s.singletonIdx];
        const int a = 34, b = 35, gA = 18, gB = 19;
        const Vd Nn = vd(m.normal_sum.x, m.normal_sum.y, m.normal_sum.z);
        const Vd TauA = vd(m.torque_armA_sum.x, m.torque_armA_sum.y, m.torque_armA_sum.z);
        const Vd TauB = vd(m.torque_armB_sum.x, m.torque_armB_sum.y, m.torque_armB_sum.z);
        (void)gA;
        (void)gB;
        auto ld = [](const std::vector<float>& h, int i) {
          return vd(h[3 * i], h[3 * i + 1], h[3 * i + 2]);
        };
        const double t0 = dot(ld(s.vel, a), Nn), t1 = dot(ld(s.ang, a), TauA);
        const double t2 = dot(ld(s.vel, b), Nn), t3 = dot(ld(s.ang, b), TauB);
        const double vn = t0 + t1 - t2 + t3;
        const double vnScale = std::fabs(t0) + std::fabs(t1) + std::fabs(t2) + std::fabs(t3);
        auto gim = [&](int i, Vd tau) {  // genInvMass: tau^T R diag(invI) R^T tau
          const M3 Rm = rotm(&s.quat[4 * i]);
          double acc = 0;
          for (int k = 0; k < 3; ++k) {
            const double t = Rm.a[0][k] * tau.x + Rm.a[1][k] * tau.y + Rm.a[2][k] * tau.z;
            acc += t * t * s.invI[3 * i + k];
          }
          return acc;
        };
        const double w = dot(Nn, Nn) * (double(s.invMass[a]) + double(s.invMass[b])) +
                         gim(a, TauA) + gim(b, TauB);
        const double rvn = std::fabs(o.rowVn - vn) / vnScale;
        const double rw = std::fabs(o.rowW - w) / w;
        std::printf(
            "rot:   singleton row vn %.7g vs fine %.7g (rel %.2e), w %.7g vs %.7g (rel "
            "%.2e)\n",
            double(o.rowVn), vn, rvn, double(o.rowW), w, rw);
        if (!(rw <= 1e-6 && rvn <= 1e-6)) {
          std::printf("FAIL: a singleton group's row differs from the fine PGS row\n");
          status = 1;
        }
      }
    }
    // ------------------------------------------------ invI_g = 0: translation-only
    // (discriminates)
    {
      const RefOut R = reference(s, false);
      const DevOut o = runDevice(s, false);
      const double ev = relErrV(o.vel, R.v, N);
      const Budget b1 = budget(s, o.vel, o.ang);
      Vd wP, wL;
      wallImpulse(o, wP, wL);
      const double dP = norm(b1.P - b0.P - wP);
      const Vd dLv = b1.L - b0.L - wL;
      const double tolL = 1e-6 * R.sumJ * R.rgMax;
      const double mis = norm(dLv - R.lvlErrL);
      std::printf(
          "trans: member v %.2e  |dP| %.3e  |dL| %.3e  predicted (X_A-X_B)xJ %.3e  "
          "|dL - pred| %.3e\n",
          ev, dP, norm(dLv), norm(R.lvlErrL), mis);
      if (!(ev <= 1e-5 && dP <= 1e-6 * R.sumJ)) {
        std::printf("FAIL: translation-only cycle differs from the double reference\n");
        status = 1;
      }
      if (!(norm(R.lvlErrL) > 100.0 * tolL && mis <= 1e-3 * norm(R.lvlErrL) + tolL)) {
        std::printf("FAIL: translation-only dL does not reproduce (X_A - X_B) x J\n");
        status = 1;
      }
    }
  }
  Kokkos::finalize();
  std::printf(status ? "test_multilevel_rigid: FAILED\n" : "test_multilevel_rigid: PASSED\n");
  return status;
}
