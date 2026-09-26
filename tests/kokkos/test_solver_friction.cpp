// Correctness of the Kokkos friction cluster (compute_plane_load -> accumulate_normal_impulse ->
// count_friction_contacts -> solve_contact_friction) against a host replication of the identical
// math, run as the same 4-step sequence. Candidate contacts with a borderline normal approach are
// dropped so the active/inactive decision (and the integer per-body counts) are decisive on both
// host and device. Compares friction_lambda_n, plane-load/count, and the friction delta_vel/
// delta_ang_vel. Runs on whatever backend Kokkos was built for.
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "solver_friction.hpp"

using namespace peclet::dem;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int N = 400;
    const float growthRate = 0.01f;
    const float frictionDynamic = 0.5f;

    std::mt19937 rng(53);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    std::uniform_real_distribution<float> upos(0.5f, 2.0f);
    std::uniform_int_distribution<int> ubody(0, N - 1);

    std::vector<float> invMass(N), iIx(N), iIy(N), iIz(N);
    std::vector<float> vx(N), vy(N), vz(N), wx(N), wy(N), wz(N);
    for (int i = 0; i < N; ++i) {
      invMass[i] = upos(rng);
      iIx[i] = upos(rng);
      iIy[i] = upos(rng);
      iIz[i] = upos(rng);
      vx[i] = uf(rng);
      vy[i] = uf(rng);
      vz[i] = uf(rng);
      wx[i] = uf(rng);
      wy[i] = uf(rng);
      wz[i] = uf(rng);
    }
    auto vP = [&](int i) { return F3{vx[i], vy[i], vz[i]}; };
    auto wP = [&](int i) { return F3{wx[i], wy[i], wz[i]}; };
    auto iI = [&](int i) { return F3{iIx[i], iIy[i], iIz[i]}; };

    // Build a contact list dropping borderline normal approach (|approach| < 0.05).
    std::vector<ContactC> contacts;
    for (int k = 0; k < 4000; ++k) {
      ContactC c{};
      int a = ubody(rng), b;
      if ((rng() & 3) == 0)
        b = -1;
      else {
        do {
          b = ubody(rng);
        } while (b == a);
        if (b < a) {
          int t = a;
          a = b;
          b = t;
        }
      }
      c.bodyA = a;
      c.bodyB = b;
      float nx = uf(rng), ny = uf(rng), nz = uf(rng);
      float nl = std::sqrt(nx * nx + ny * ny + nz * nz) + 1e-12f;
      c.normal = F4{nx / nl, ny / nl, nz / nl, 0};
      c.rA = F4{uf(rng) * 0.5f, uf(rng) * 0.5f, uf(rng) * 0.5f, 0};
      c.rB = F4{uf(rng) * 0.5f, uf(rng) * 0.5f, uf(rng) * 0.5f, 0};
      c.dist = 0;
      c.friction_lambda_n = 0;
      c.weight = 0;
      F3 rA{c.rA.x, c.rA.y, c.rA.z}, rB{c.rB.x, c.rB.y, c.rB.z},
          n{c.normal.x, c.normal.y, c.normal.z};
      float approach;
      if (b < 0) {
        F3 vAc = add3(vP(a), cross3v(wP(a), rA));
        approach = -dot3(vAc, n);
      } else {
        F3 vAc = add3(vP(a), cross3v(wP(a), rA));
        F3 vBc = add3(vP(b), cross3v(wP(b), rB));
        F3 vrel = add3(sub3(vAc, vBc), scale3(sub3(rA, rB), growthRate));
        approach = -dot3(vrel, n);
      }
      if (std::fabs(approach) < 0.05f)
        continue;
      contacts.push_back(c);
    }
    const int M = (int)contacts.size();

    // --- upload ---
    Kokkos::View<ContactC*, CpMem> dC("c", M);
    auto hC = Kokkos::create_mirror_view(dC);
    for (int k = 0; k < M; ++k)
      hC(k) = contacts[k];
    Kokkos::deep_copy(dC, hC);
    Kokkos::View<float*, CpMem> dIM("im", N);
    {
      auto h = Kokkos::create_mirror_view(dIM);
      for (int i = 0; i < N; ++i)
        h(i) = invMass[i];
      Kokkos::deep_copy(dIM, h);
    }
    auto up3 = [&](const char* nm, std::vector<float>& x, std::vector<float>& y,
                   std::vector<float>& z) {
      Kokkos::View<float* [3], CpMem> v(nm, N);
      auto h = Kokkos::create_mirror_view(v);
      for (int i = 0; i < N; ++i) {
        h(i, 0) = x[i];
        h(i, 1) = y[i];
        h(i, 2) = z[i];
      }
      Kokkos::deep_copy(v, h);
      return v;
    };
    auto dInvI = up3("ii", iIx, iIy, iIz);
    auto dVel = up3("v", vx, vy, vz);
    auto dAng = up3("w", wx, wy, wz);
    Kokkos::View<int*, CpMem> dReal("r", N);
    {
      auto h = Kokkos::create_mirror_view(dReal);
      for (int i = 0; i < N; ++i)
        h(i) = i;
      Kokkos::deep_copy(dReal, h);
    }
    FrManifoldCounts dPF("pf", N);
    Kokkos::View<float* [3], CpMem> dDV("dv", N), dDW("dw", N);

    // --- device sequence ---
    computePlaneLoadKokkos(dC, M, dIM, dInvI, dVel, dAng, dPF);
    accumulateNormalImpulseKokkos(dC, M, dIM, dInvI, dVel, dAng, dReal, growthRate);
    countFrictionContactsKokkos(dC, M, dReal, dPF);
    // Identity orientations: the world inverse inertia R diag(invI) R^T is diag(invI), so the
    // serial reference below (body frame = world frame) stays exact for anisotropic inertia.
    Kokkos::View<float* [4], CpMem> dQuat("q", N);
    {
      auto h = Kokkos::create_mirror_view(dQuat);
      for (int i = 0; i < N; ++i) {
        h(i, 0) = h(i, 1) = h(i, 2) = 0.0f;
        h(i, 3) = 1.0f;
      }
      Kokkos::deep_copy(dQuat, h);
    }
    solveContactFrictionKokkos(dC, M, dIM, dInvI, dQuat, dVel, dAng, dReal, dPF, frictionDynamic,
                               dDV, dDW);

    std::vector<float> gln(M), gpx(N), gpy(N), gdv(3 * N), gdw(3 * N);
    {
      auto h = Kokkos::create_mirror_view(dC);
      Kokkos::deep_copy(h, dC);
      for (int k = 0; k < M; ++k)
        gln[k] = h(k).friction_lambda_n;
    }
    {
      auto h = Kokkos::create_mirror_view(dPF);
      Kokkos::deep_copy(h, dPF);
      for (int i = 0; i < N; ++i) {
        gpx[i] = h(i, 0);
        gpy[i] = h(i, 1);
      }
    }
    {
      auto h = Kokkos::create_mirror_view(dDV);
      Kokkos::deep_copy(h, dDV);
      for (int i = 0; i < N; ++i) {
        gdv[3 * i] = h(i, 0);
        gdv[3 * i + 1] = h(i, 1);
        gdv[3 * i + 2] = h(i, 2);
      }
    }
    {
      auto h = Kokkos::create_mirror_view(dDW);
      Kokkos::deep_copy(h, dDW);
      for (int i = 0; i < N; ++i) {
        gdw[3 * i] = h(i, 0);
        gdw[3 * i + 1] = h(i, 1);
        gdw[3 * i + 2] = h(i, 2);
      }
    }

    // --- host replication of the same 4 steps ---
    std::vector<float> ln(M, 0), pfx(N, 0), pfy(N, 0), rdv(3 * N, 0), rdw(3 * N, 0);
    auto cW = [&](F3 r, F3 dir, float invM, F3 invI) {
      F3 rn = cross3v(r, dir);
      return invM + rn.x * rn.x * invI.x + rn.y * rn.y * invI.y + rn.z * rn.z * invI.z;
    };
    // 1. plane load
    for (int k = 0; k < M; ++k) {
      const ContactC& c = contacts[k];
      if (c.bodyB >= 0)
        continue;
      int a = c.bodyA;
      if (invMass[a] <= 0)
        continue;
      F3 rA{c.rA.x, c.rA.y, c.rA.z}, n{c.normal.x, c.normal.y, c.normal.z};
      F3 vAc = add3(vP(a), cross3v(wP(a), rA));
      float ap = -dot3(vAc, n);
      if (ap <= 0)
        continue;
      float wn = cW(rA, n, invMass[a], iI(a));
      ln[k] = (wn > 1e-6f) ? ap / wn : 0.f;
      float load = ap / invMass[a];
      if (load > pfx[a])
        pfx[a] = load;
    }
    // 2. accumulate force-chain load (body-body AND wall)
    for (int k = 0; k < M; ++k) {
      const ContactC& c = contacts[k];
      if (c.bodyB <
          0) {  // wall: transmitted load on the owned grain (A term only, vs wall velocity)
        int a = c.bodyA;
        if (invMass[a] <= 0)
          continue;
        F3 rA{c.rA.x, c.rA.y, c.rA.z}, n{c.normal.x, c.normal.y, c.normal.z};
        F3 vAc = add3(vP(a), cross3v(wP(a), rA));
        F3 vWall{c.boundaryVel.x, c.boundaryVel.y, c.boundaryVel.z};
        float ap = -dot3(sub3(vAc, vWall), n);
        if (ap <= 0)
          continue;
        float wn = cW(rA, n, invMass[a], iI(a));
        if (wn > 1e-6f)
          ln[k] += ap / wn;
        continue;
      }
      int a = c.bodyA, b = c.bodyB;
      if (a > b)
        continue;
      F3 rA{c.rA.x, c.rA.y, c.rA.z}, rB{c.rB.x, c.rB.y, c.rB.z},
          n{c.normal.x, c.normal.y, c.normal.z};
      F3 vAc = add3(vP(a), cross3v(wP(a), rA));
      F3 vBc = add3(vP(b), cross3v(wP(b), rB));
      F3 vrel = add3(sub3(vAc, vBc), scale3(sub3(rA, rB), growthRate));
      float ap = -dot3(vrel, n);
      if (ap <= 0)
        continue;
      float wn = cW(rA, n, invMass[a], iI(a)) + cW(rB, n, invMass[b], iI(b));
      if (wn > 1e-6f)
        ln[k] += ap / wn;
    }
    // 3. count
    for (int k = 0; k < M; ++k) {
      if (ln[k] <= 0)
        continue;
      pfy[contacts[k].bodyA] += 1.f;
      if (contacts[k].bodyB >= 0)
        pfy[contacts[k].bodyB] += 1.f;
    }
    // 4. solve
    for (int k = 0; k < M; ++k) {
      const ContactC& c = contacts[k];
      float lam = ln[k];
      if (lam <= 0)
        continue;
      int a = c.bodyA, b = c.bodyB;
      float invMA = invMass[a], invMB = (b >= 0) ? invMass[b] : 0.f;
      F3 invIA = iI(a), invIB = (b >= 0) ? iI(b) : F3{0, 0, 0};
      F3 rA{c.rA.x, c.rA.y, c.rA.z}, rB{c.rB.x, c.rB.y, c.rB.z},
          n{c.normal.x, c.normal.y, c.normal.z};
      F3 vAc = add3(vP(a), cross3v(wP(a), rA));
      F3 vBc{0, 0, 0};
      if (b >= 0)
        vBc = add3(vP(b), cross3v(wP(b), rB));
      F3 vrel = sub3(vAc, vBc);
      float vn = dot3(vrel, n);
      F3 vt = sub3(vrel, scale3(n, vn));
      float vl = std::sqrt(dot3(vt, vt));
      if (vl < 1e-7f)
        continue;
      F3 t = scale3(vt, 1.f / vl);
      F3 rnA = cross3v(rA, t), rnB = cross3v(rB, t);
      float wt = invMA + invMB + rnA.x * rnA.x * invIA.x + rnA.y * rnA.y * invIA.y +
                 rnA.z * rnA.z * invIA.z + rnB.x * rnB.x * invIB.x + rnB.y * rnB.y * invIB.y +
                 rnB.z * rnB.z * invIB.z;
      if (wt < 1e-6f)
        continue;
      float bound =
          lam;  // accumulated force-chain load bounds friction for both body-body and wall
      float nA = pfy[a], nB = (b >= 0) ? pfy[b] : 0.f;
      float invn = 1.f / std::fmax(std::fmax(nA, nB), 1.f);
      float lt = -vl / wt;
      float mf = frictionDynamic * bound;
      if (lt < -mf)
        lt = -mf;
      lt *= invn;
      rdv[3 * a] += t.x * lt * invMA;
      rdv[3 * a + 1] += t.y * lt * invMA;
      rdv[3 * a + 2] += t.z * lt * invMA;
      rdw[3 * a] += rnA.x * invIA.x * lt;
      rdw[3 * a + 1] += rnA.y * invIA.y * lt;
      rdw[3 * a + 2] += rnA.z * invIA.z * lt;
      if (b >= 0) {
        rdv[3 * b] += -t.x * lt * invMB;
        rdv[3 * b + 1] += -t.y * lt * invMB;
        rdv[3 * b + 2] += -t.z * lt * invMB;
        rdw[3 * b] += -rnB.x * invIB.x * lt;
        rdw[3 * b + 1] += -rnB.y * invIB.y * lt;
        rdw[3 * b + 2] += -rnB.z * invIB.z * lt;
      }
    }

    int bad = 0;
    auto tol = [&](float g, float r) { return std::fabs(g - r) <= 1e-4f * (1 + std::fabs(r)); };
    for (int k = 0; k < M; ++k)
      if (!tol(gln[k], ln[k]))
        ++bad;
    for (int i = 0; i < N; ++i) {
      if (!tol(gpx[i], pfx[i]))
        ++bad;
      if (gpy[i] != pfy[i])
        ++bad;
    }
    for (int i = 0; i < 3 * N; ++i) {
      if (!tol(gdv[i], rdv[i]))
        ++bad;
      if (!tol(gdw[i], rdw[i]))
        ++bad;
    }
    if (bad) {
      std::fprintf(stderr, "FAIL: %d friction quantities differ\n", bad);
      status = 1;
    } else
      std::printf(
          "[solver_friction] PASS: %d contacts, lambda_n/plane-load/count/deltas match host (exec: "
          "%s)\n",
          M, CpExec::name());
  }
  // ---- One ring-ring friction impulse, worked by hand (docs/contact_solve_framework.md §12 S15)
  // -- Two hollow cylinders (body-frame inverse inertia (3.628, 3.628, 5.675), the ring_mini rings)
  // at arbitrary orientations, one sliding contact whose midpoint arms meet at a common point c.
  // The pass applies J = lt t on A and -J on B at c, so by hand: m_A dv_A = J, I_A,world dw_A = r_A
  // x J (r_A = c - x_A), likewise for B with -J, and the pair's angular momentum about the origin,
  // sum x_i x m_i dv_i + I_i,world dw_i = (x_A + r_A - x_B - r_B) x J, is exactly 0. A body-frame
  // diagonal applied to the world torque (the pre-S15 kernel) misses that by O(1) of |r_A x J|.
  {
    using D3 = std::array<double, 3>;
    auto qn = [](double x, double y, double z, double w) {
      const double l = std::sqrt(x * x + y * y + z * z + w * w);
      return std::array<double, 4>{x / l, y / l, z / l, w / l};
    };
    const std::array<double, 4> qd[2] = {qn(0.3, -0.5, 0.2, 0.78), qn(-0.6, 0.1, 0.4, 0.68)};
    const D3 invIb{3.628, 3.628, 5.675};
    const double invM[2] = {1.0 / 0.41, 1.0 / 0.37};
    const D3 x[2] = {D3{0.0, 0.0, 0.0}, D3{0.9, 0.2, -0.1}};
    const D3 v[2] = {D3{0.3, -0.2, 0.5}, D3{-0.1, 0.4, 0.2}};
    const D3 w[2] = {D3{1.0, 0.5, -0.7}, D3{-0.3, 0.8, 0.2}};
    const D3 cpt{0.46, 0.13, -0.02};  // the common contact point
    D3 n{0.9, 0.2, -0.1};
    {
      const double l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      for (auto& e : n)
        e /= l;
    }
    const double dist = -0.02;  // overlap; the surface points are c -/+ dist n / 2
    ContactC c{};
    c.bodyA = 0;
    c.bodyB = 1;
    c.normal = F4{float(n[0]), float(n[1]), float(n[2]), 0.0f};
    c.rA =
        F4{float(cpt[0] + 0.5 * dist * n[0] - x[0][0]), float(cpt[1] + 0.5 * dist * n[1] - x[0][1]),
           float(cpt[2] + 0.5 * dist * n[2] - x[0][2]), 0.0f};
    c.rB =
        F4{float(cpt[0] - 0.5 * dist * n[0] - x[1][0]), float(cpt[1] - 0.5 * dist * n[1] - x[1][1]),
           float(cpt[2] - 0.5 * dist * n[2] - x[1][2]), 0.0f};
    c.dist = float(dist);
    c.friction_lambda_n = 10.0f;  // a Coulomb bound that does not clamp
    Kokkos::View<ContactC*, CpMem> dC("c1", 1);
    Kokkos::deep_copy(dC, c);
    Kokkos::View<float*, CpMem> dIM("im1", 2);
    Kokkos::View<float* [3], CpMem> dII("ii1", 2), dV("v1", 2), dW("w1", 2), dDV("dv1", 2),
        dDW("dw1", 2);
    Kokkos::View<float* [4], CpMem> dQ("q1", 2);
    Kokkos::View<int*, CpMem> dR("r1", 2);
    FrManifoldCounts dPF("pf1", 2);
    {
      auto hIM = Kokkos::create_mirror_view(dIM);
      auto hII = Kokkos::create_mirror_view(dII);
      auto hV = Kokkos::create_mirror_view(dV);
      auto hW = Kokkos::create_mirror_view(dW);
      auto hQ = Kokkos::create_mirror_view(dQ);
      auto hR = Kokkos::create_mirror_view(dR);
      auto hPF = Kokkos::create_mirror_view(dPF);
      for (int i = 0; i < 2; ++i) {
        hIM(i) = float(invM[i]);
        hR(i) = i;
        hPF(i, 0) = 0.0f;
        hPF(i, 1) = 1.0f;
        for (int d = 0; d < 3; ++d) {
          hII(i, d) = float(invIb[d]);
          hV(i, d) = float(v[i][d]);
          hW(i, d) = float(w[i][d]);
        }
        for (int d = 0; d < 4; ++d)
          hQ(i, d) = float(qd[i][d]);
      }
      Kokkos::deep_copy(dIM, hIM);
      Kokkos::deep_copy(dII, hII);
      Kokkos::deep_copy(dV, hV);
      Kokkos::deep_copy(dW, hW);
      Kokkos::deep_copy(dQ, hQ);
      Kokkos::deep_copy(dR, hR);
      Kokkos::deep_copy(dPF, hPF);
    }
    solveContactFrictionKokkos(dC, 1, dIM, dII, dQ, dV, dW, dR, dPF, 0.5f, dDV, dDW);
    auto hDV = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dDV);
    auto hDW = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dDW);
    // Host, double: R of the float quaternion the kernel read, I_world = R diag(1/invI) R^T.
    auto Rot = [&](int i) {
      const double qx = float(qd[i][0]), qy = float(qd[i][1]), qz = float(qd[i][2]),
                   qw = float(qd[i][3]);
      return std::array<D3, 3>{
          D3{1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)},
          D3{2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)},
          D3{2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)}};
    };
    // M diag(s) M^T y with M = R: the world tensor applied to y.
    auto worldApply = [&](int i, const D3& y, bool inverse) {
      const auto R = Rot(i);
      D3 b{0, 0, 0}, out{0, 0, 0};
      for (int k = 0; k < 3; ++k) {
        for (int d = 0; d < 3; ++d)
          b[k] += R[d][k] * y[d];
        b[k] = inverse ? b[k] * double(float(invIb[k])) : b[k] / double(float(invIb[k]));
      }
      for (int d = 0; d < 3; ++d)
        for (int k = 0; k < 3; ++k)
          out[d] += R[d][k] * b[k];
      return out;
    };
    auto cross = [](const D3& a, const D3& b) {
      return D3{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    };
    auto norm = [](const D3& a) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); };
    D3 J, JB, rA, rB, dL{0, 0, 0}, dLold{0, 0, 0};
    for (int d = 0; d < 3; ++d) {
      J[d] = hDV(0, d) / double(float(invM[0]));
      JB[d] = hDV(1, d) / double(float(invM[1]));
      rA[d] = cpt[d] - x[0][d];
      rB[d] = cpt[d] - x[1][d];
    }
    const D3 tA = cross(rA, J);
    D3 LdwA, LdwB;
    for (int i = 0; i < 2; ++i) {
      const D3 dw{hDW(i, 0), hDW(i, 1), hDW(i, 2)};
      const D3 Ldw = worldApply(i, dw, false);  // I_world dw
      (i == 0 ? LdwA : LdwB) = Ldw;
      const D3 xJ = cross(x[i], i == 0 ? J : JB);
      for (int d = 0; d < 3; ++d)
        dL[d] += xJ[d] + Ldw[d];
    }
    // The pre-S15 kernel for comparison: dw = invI (component-wise) (r x J) on the world torque.
    for (int i = 0; i < 2; ++i) {
      const D3 tq = cross(i == 0 ? rA : rB, i == 0 ? J : D3{-J[0], -J[1], -J[2]});
      const D3 dwOld{tq[0] * float(invIb[0]), tq[1] * float(invIb[1]), tq[2] * float(invIb[2])};
      const D3 Ldw = worldApply(i, dwOld, false);
      const D3 xJ = cross(x[i], i == 0 ? J : D3{-J[0], -J[1], -J[2]});
      for (int d = 0; d < 3; ++d)
        dLold[d] += xJ[d] + Ldw[d];
    }
    const D3 JsumV{J[0] + JB[0], J[1] + JB[1], J[2] + JB[2]};
    const D3 eA{LdwA[0] - tA[0], LdwA[1] - tA[1], LdwA[2] - tA[2]};
    const D3 tB = cross(rB, D3{-J[0], -J[1], -J[2]});
    const D3 eB{LdwB[0] - tB[0], LdwB[1] - tB[1], LdwB[2] - tB[2]};
    const double tn = norm(tA);
    std::printf(
        "[solver_friction] ring-ring impulse: |J| %.4e |r_A x J| %.4e | |J_A + J_B| / |J| "
        "%.2e | |I_A dw_A - r_A x J| / |r_A x J| %.2e (B %.2e) | |dL| / |r_A x J| %.2e "
        "(pre-S15 kernel: %.2e)\n",
        norm(J), tn, norm(JsumV) / norm(J), norm(eA) / tn, norm(eB) / norm(tB), norm(dL) / tn,
        norm(dLold) / tn);
    if (!(norm(J) > 0 && norm(JsumV) <= 1e-6 * norm(J) && norm(eA) <= 1e-5 * tn &&
          norm(eB) <= 1e-5 * norm(tB) && norm(dL) <= 1e-5 * tn)) {
      std::fprintf(stderr, "FAIL: the ring-ring friction impulse does not conserve L\n");
      status = 1;
    }
  }
  Kokkos::finalize();
  return status;
}
