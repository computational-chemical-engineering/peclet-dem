// Correctness of the Kokkos manifold velocity solve (dem::solveVelocityKokkos) against a host
// replication of the identical impulse math. Random bodies + canonical manifolds (bodyA<bodyB, or
// bodyB=-1 boundary); identity real-index map. Compares the accumulated delta_vel / delta_ang_vel
// per body within tolerance (device atomic vs host sequential accumulation differ only in float
// summation order). Runs on whatever backend Kokkos was built for.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "solver_velocity.hpp"

using namespace peclet::dem;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int N = 400;
    const int M = 3000;
    const float growthRate = 0.01f;
    const float restitution = 0.2f;

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    std::uniform_real_distribution<float> upos(0.2f, 2.0f);
    std::uniform_int_distribution<int> ubody(0, N - 1);
    std::normal_distribution<float> nq(0.f, 1.f);

    std::vector<float> invMass(N), iIx(N), iIy(N), iIz(N);
    std::vector<float> qx(N), qy(N), qz(N), qw(N);
    std::vector<float> vx(N), vy(N), vz(N), wx(N), wy(N), wz(N);
    for (int i = 0; i < N; ++i) {
      invMass[i] = upos(rng);
      iIx[i] = upos(rng);
      iIy[i] = upos(rng);
      iIz[i] = upos(rng);
      float a = nq(rng), b = nq(rng), c = nq(rng), d = nq(rng);
      float nrm = std::sqrt(a * a + b * b + c * c + d * d) + 1e-12f;
      qx[i] = a / nrm;
      qy[i] = b / nrm;
      qz[i] = c / nrm;
      qw[i] = d / nrm;
      vx[i] = uf(rng);
      vy[i] = uf(rng);
      vz[i] = uf(rng);
      wx[i] = uf(rng);
      wy[i] = uf(rng);
      wz[i] = uf(rng);
    }

    std::vector<ManifoldC> man(M);
    for (int k = 0; k < M; ++k) {
      int a = ubody(rng), b;
      if ((rng() & 7) == 0) {
        b = -1;
      } else {
        do {
          b = ubody(rng);
        } while (b == a);
        if (b < a) {
          int t = a;
          a = b;
          b = t;
        }  // canonical bodyA<bodyB
      }
      ManifoldC m{};
      m.bodyA = a;
      m.bodyB = b;
      m.normal_sum = F4{uf(rng), uf(rng), uf(rng), 0};
      m.torque_armA_sum = F4{uf(rng), uf(rng), uf(rng), 0};
      m.torque_armB_sum = F4{uf(rng), uf(rng), uf(rng), 0};
      m.rA_sum = F4{uf(rng), uf(rng), uf(rng), 0};
      m.rB_sum = F4{uf(rng), uf(rng), uf(rng), 0};
      m.num_points = 1 + (int)(rng() % 5);
      // Boundary (idB<0) moving-wall extension: half the walls carry an explicit per-wall restitution
      // + surface velocity (summed over num_points, count-averaged by the solve), half use the < 0
      // sentinel = the global material + a static wall. Body-body manifolds never read these.
      if (b < 0) {
        if (rng() & 1u) {
          const float wr = 0.5f * (uf(rng) + 1.0f);  // per-wall restitution in [0,1]
          m.restitution_sum = wr * m.num_points;
          m.wallVel_sum = F4{uf(rng), uf(rng), uf(rng), 0};
        } else {
          m.restitution_sum = -1.0f * m.num_points;  // sentinel -> global restitution, static wall
          m.wallVel_sum = F4{0, 0, 0, 0};
        }
      }
      man[k] = m;
    }

    // --- upload ---
    auto up3 = [&](const char* nm, const std::vector<float>& x, const std::vector<float>& y,
                   const std::vector<float>& z) {
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
    Kokkos::View<float*, CpMem> dInvMass("invMass", N);
    {
      auto h = Kokkos::create_mirror_view(dInvMass);
      for (int i = 0; i < N; ++i)
        h(i) = invMass[i];
      Kokkos::deep_copy(dInvMass, h);
    }
    auto dInvI = up3("invI", iIx, iIy, iIz);
    Kokkos::View<float* [4], CpMem> dQuat("quat", N);
    {
      auto h = Kokkos::create_mirror_view(dQuat);
      for (int i = 0; i < N; ++i) {
        h(i, 0) = qx[i];
        h(i, 1) = qy[i];
        h(i, 2) = qz[i];
        h(i, 3) = qw[i];
      }
      Kokkos::deep_copy(dQuat, h);
    }
    auto dVel = up3("vel", vx, vy, vz);
    auto dAng = up3("ang", wx, wy, wz);
    Kokkos::View<int*, CpMem> dReal("real", N);
    {
      auto h = Kokkos::create_mirror_view(dReal);
      for (int i = 0; i < N; ++i)
        h(i) = i;
      Kokkos::deep_copy(dReal, h);
    }
    Kokkos::View<ManifoldC*, CpMem> dMan("man", M);
    {
      auto h = Kokkos::create_mirror_view(dMan);
      for (int k = 0; k < M; ++k)
        h(k) = man[k];
      Kokkos::deep_copy(dMan, h);
    }
    Kokkos::View<float* [3], CpMem> dDV("dv", N), dDW("dw", N);  // zero-initialised

    solveVelocityKokkos(dMan, M, dInvMass, dInvI, dQuat, dVel, dAng, dReal, growthRate, restitution,
                        dDV, dDW);

    std::vector<float> gdv(3 * N), gdw(3 * N);
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

    // --- host reference: identical math, sequential accumulation ---
    std::vector<float> rdv(3 * N, 0.f), rdw(3 * N, 0.f);
    auto qOf = [&](int i) { return F4{qx[i], qy[i], qz[i], qw[i]}; };
    for (int k = 0; k < M; ++k) {
      const ManifoldC m = man[k];
      if (m.num_points <= 0)
        continue;
      int idA = m.bodyA, idB = m.bodyB, realA = idA, realB = idB;
      if (idB >= 0) {
        realB = idB;
        if (realA > realB)
          continue;
      }
      float invMA = invMass[realA], invMB = (idB >= 0) ? invMass[realB] : 0.f;
      F3 invIA{iIx[realA], iIy[realA], iIz[realA]};
      F3 invIB = (idB >= 0) ? F3{iIx[realB], iIy[realB], iIz[realB]} : F3{0, 0, 0};
      F4 qA = qOf(realA), qB = (idB >= 0) ? qOf(realB) : F4{0, 0, 0, 1};
      F3 vA{vx[realA], vy[realA], vz[realA]}, wA{wx[realA], wy[realA], wz[realA]};
      F3 vB{0, 0, 0}, wB{0, 0, 0};
      float rest = restitution;
      if (idB >= 0) {
        vB = F3{vx[realB], vy[realB], vz[realB]};
        wB = F3{wx[realB], wy[realB], wz[realB]};
      } else {
        const float in = 1.f / (float)m.num_points;
        vB = scale3(F3{m.wallVel_sum.x, m.wallVel_sum.y, m.wallVel_sum.z}, in);
        const float ra = m.restitution_sum * in;
        if (ra >= 0.f)
          rest = ra;
      }
      F3 Nsum{m.normal_sum.x, m.normal_sum.y, m.normal_sum.z};
      F3 TauA{m.torque_armA_sum.x, m.torque_armA_sum.y, m.torque_armA_sum.z};
      F3 TauB{m.torque_armB_sum.x, m.torque_armB_sum.y, m.torque_armB_sum.z};
      float invN = 1.f / (float)m.num_points;
      F3 rAavg = scale3(F3{m.rA_sum.x, m.rA_sum.y, m.rA_sum.z}, invN);
      F3 rBavg = scale3(F3{m.rB_sum.x, m.rB_sum.y, m.rB_sum.z}, invN);
      float lenN = std::sqrt(dot3(Nsum, Nsum));
      if (lenN < 1e-9f)
        continue;
      F3 diff = (idB < 0) ? rAavg : sub3(rAavg, rBavg);  // boundary: rBavg is absolute, use rAavg
      F3 vG = scale3(diff, growthRate);
      float vn = dot3(vA, Nsum) + dot3(wA, TauA) + dot3(vB, F3{-Nsum.x, -Nsum.y, -Nsum.z}) +
                 dot3(wB, TauB);
      vn += dot3(vG, Nsum);
      float align = dot3(Nsum, diff);
      if (align > 0) {
        if (vn < 0)
          continue;
      } else {
        if (vn > 0)
          continue;
      }
      float Nsq = dot3(Nsum, Nsum);
      float wAn = Nsq * invMA + detail::genInvMass(TauA, invIA, qA);
      float wBn = Nsq * invMB + detail::genInvMass(TauB, invIB, qB);
      float wT = wAn + wBn;
      if (wT <= 0)
        continue;
      float lambda = (-rest * vn - vn) / wT;
      F3 Jlin = scale3(Nsum, lambda), JangA = scale3(TauA, lambda), JangB = scale3(TauB, lambda);
      rdv[3 * realA] += Jlin.x * invMA;
      rdv[3 * realA + 1] += Jlin.y * invMA;
      rdv[3 * realA + 2] += Jlin.z * invMA;
      {
        F3 Jl = invRotateVector(qA, JangA);
        F3 dwl{Jl.x * invIA.x, Jl.y * invIA.y, Jl.z * invIA.z};
        F3 dww = rotateVector(qA, dwl);
        rdw[3 * realA] += dww.x;
        rdw[3 * realA + 1] += dww.y;
        rdw[3 * realA + 2] += dww.z;
      }
      if (idB >= 0) {
        rdv[3 * realB] += -Jlin.x * invMB;
        rdv[3 * realB + 1] += -Jlin.y * invMB;
        rdv[3 * realB + 2] += -Jlin.z * invMB;
        F3 Jl = invRotateVector(qB, JangB);
        F3 dwl{Jl.x * invIB.x, Jl.y * invIB.y, Jl.z * invIB.z};
        F3 dww = rotateVector(qB, dwl);
        rdw[3 * realB] += dww.x;
        rdw[3 * realB + 1] += dww.y;
        rdw[3 * realB + 2] += dww.z;
      }
    }

    int bad = 0;
    for (int i = 0; i < 3 * N; ++i) {
      float t = 1e-4f * (1.0f + std::fabs(rdv[i]));
      if (std::fabs(gdv[i] - rdv[i]) > t)
        ++bad;
      float tw = 1e-4f * (1.0f + std::fabs(rdw[i]));
      if (std::fabs(gdw[i] - rdw[i]) > tw)
        ++bad;
    }
    if (bad) {
      std::fprintf(stderr, "FAIL: %d/%d delta components differ\n", bad, 6 * N);
      status = 1;
    } else {
      std::printf("[solver_velocity] PASS: delta_vel/delta_ang_vel match host (exec: %s)\n",
                  CpExec::name());
    }
  }
  Kokkos::finalize();
  return status;
}
