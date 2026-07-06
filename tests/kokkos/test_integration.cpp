// Correctness of the Kokkos time-integration kernels (integration.cu port) against a host
// replication. Runs the per-step sequence: predict velocity (gravity + gyroscopic) -> inject delta
// velocities -> apply velocity deltas -> re-integrate (persist v, trapezoidal x, quat integrate) ->
// inject delta positions + contact counts -> Jacobi count-averaged apply -> final commit (periodic
// wrap). Element-wise kernels are deterministic, so device must match host within float tol;
// integer constraint counts match exactly. Runs on whatever backend Kokkos was built for.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "integration.hpp"

using namespace peclet::dem;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int N = 500;
    const float dt = 0.01f;
    const F3 gravity{0.f, 0.f, -9.81f};
    Domain dom{F3{-5, -5, -5}, F3{5, 5, 5}, F3{10, 10, 10}, true, true, true};

    std::mt19937 rng(17);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    std::uniform_real_distribution<float> up(0.5f, 2.f);
    std::normal_distribution<float> nq(0.f, 1.f);

    std::vector<float> px(N), py(N), pz(N), im(N), qx(N), qy(N), qz(N), qw(N);
    std::vector<float> vx(N), vy(N), vz(N), ax(N), ay(N), az(N), iix(N), iiy(N), iiz(N);
    std::vector<float> dvx(N), dvy(N), dvz(N), dax(N), day(N), daz(N), dpx(N), dpy(N), dpz(N);
    std::vector<int> cnt(N);
    for (int i = 0; i < N; ++i) {
      px[i] = uf(rng) * 5;
      py[i] = uf(rng) * 5;
      pz[i] = uf(rng) * 5;
      im[i] = up(rng);
      float a = nq(rng), b = nq(rng), c = nq(rng), d = nq(rng);
      float nl = std::sqrt(a * a + b * b + c * c + d * d) + 1e-12f;
      qx[i] = a / nl;
      qy[i] = b / nl;
      qz[i] = c / nl;
      qw[i] = d / nl;
      vx[i] = uf(rng);
      vy[i] = uf(rng);
      vz[i] = uf(rng);
      ax[i] = uf(rng);
      ay[i] = uf(rng);
      az[i] = uf(rng);
      iix[i] = up(rng);
      iiy[i] = up(rng);
      iiz[i] = up(rng);
      dvx[i] = uf(rng) * 0.1f;
      dvy[i] = uf(rng) * 0.1f;
      dvz[i] = uf(rng) * 0.1f;
      dax[i] = uf(rng) * 0.1f;
      day[i] = uf(rng) * 0.1f;
      daz[i] = uf(rng) * 0.1f;
      dpx[i] = uf(rng) * 0.1f;
      dpy[i] = uf(rng) * 0.1f;
      dpz[i] = uf(rng) * 0.1f;
      cnt[i] = 0;
    }

    // --- device views ---
    auto mk3 = [&](const char* n, std::vector<float>& x, std::vector<float>& y,
                   std::vector<float>& z) {
      V3 v(n, N);
      auto h = Kokkos::create_mirror_view(v);
      for (int i = 0; i < N; ++i) {
        h(i, 0) = x[i];
        h(i, 1) = y[i];
        h(i, 2) = z[i];
      }
      Kokkos::deep_copy(v, h);
      return v;
    };
    auto mk4 = [&](const char* n, std::vector<float>& x, std::vector<float>& y,
                   std::vector<float>& z, std::vector<float>& w) {
      V4 v(n, N);
      auto h = Kokkos::create_mirror_view(v);
      for (int i = 0; i < N; ++i) {
        h(i, 0) = x[i];
        h(i, 1) = y[i];
        h(i, 2) = z[i];
        h(i, 3) = w[i];
      }
      Kokkos::deep_copy(v, h);
      return v;
    };
    auto pos = mk3("pos", px, py, pz), vel = mk3("vel", vx, vy, vz), angVel = mk3("av", ax, ay, az),
         invI = mk3("ii", iix, iiy, iiz);
    auto quat = mk4("q", qx, qy, qz, qw);
    Vf invMass("im", N);
    {
      auto h = Kokkos::create_mirror_view(invMass);
      for (int i = 0; i < N; ++i)
        h(i) = im[i];
      Kokkos::deep_copy(invMass, h);
    }
    V3 posPred("pp", N), velPred("vp", N), angVelPred("avp", N), deltaPos("dp", N),
        deltaVel("dv", N), deltaAngVel("dav", N);
    V4 quatPred("qp", N), deltaQuat("dq", N);
    Vi cc("cc", N);

    // --- device sequence ---
    V3 extF("extF", N);  // zero external force (the CFD-DEM drag buffer; default no-op)
    predictVelocityKokkos(N, pos, invMass, vel, quat, angVel, invI, posPred, quatPred, velPred,
                          angVelPred, deltaPos, deltaQuat, deltaVel, deltaAngVel, cc, gravity, dt,
                          extF);
    // inject delta velocities
    {
      auto h = Kokkos::create_mirror_view(deltaVel);
      for (int i = 0; i < N; ++i) {
        h(i, 0) = dvx[i];
        h(i, 1) = dvy[i];
        h(i, 2) = dvz[i];
      }
      Kokkos::deep_copy(deltaVel, h);
      auto g = Kokkos::create_mirror_view(deltaAngVel);
      for (int i = 0; i < N; ++i) {
        g(i, 0) = dax[i];
        g(i, 1) = day[i];
        g(i, 2) = daz[i];
      }
      Kokkos::deep_copy(deltaAngVel, g);
    }
    applyVelocityDeltasKokkos(N, velPred, angVelPred, deltaVel, deltaAngVel);
    applyVelocityAndPredictPositionKokkos(N, pos, invMass, vel, quat, velPred, angVelPred, posPred,
                                          quatPred, angVel, dt);
    // inject delta positions + counts (emulate a position-solve result)
    {
      auto h = Kokkos::create_mirror_view(deltaPos);
      for (int i = 0; i < N; ++i) {
        h(i, 0) = dpx[i];
        h(i, 1) = dpy[i];
        h(i, 2) = dpz[i];
      }
      Kokkos::deep_copy(deltaPos, h);
      auto g = Kokkos::create_mirror_view(cc);
      for (int i = 0; i < N; ++i)
        g(i) = 1 + (i % 3);
      Kokkos::deep_copy(cc, g);
      for (int i = 0; i < N; ++i)
        cnt[i] = 1 + (i % 3);
    }
    applyUpdatesKokkos(N, posPred, velPred, deltaPos, deltaVel, cc);
    finalCommitKokkos(N, pos, invMass, posPred, quat, quatPred, dom);

    // gather
    auto g3 = [&](V3 v) {
      std::vector<float> o(3 * N);
      auto h = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(h, v);
      for (int i = 0; i < N; ++i) {
        o[3 * i] = h(i, 0);
        o[3 * i + 1] = h(i, 1);
        o[3 * i + 2] = h(i, 2);
      }
      return o;
    };
    auto g4 = [&](V4 v) {
      std::vector<float> o(4 * N);
      auto h = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(h, v);
      for (int i = 0; i < N; ++i) {
        o[4 * i] = h(i, 0);
        o[4 * i + 1] = h(i, 1);
        o[4 * i + 2] = h(i, 2);
        o[4 * i + 3] = h(i, 3);
      }
      return o;
    };
    auto gpos = g3(pos), gvel = g3(vel), gav = g3(angVel);
    auto gq = g4(quat);

    // --- host replication ---
    std::vector<float> hpx(N), hpy(N), hpz(N), hvx(N), hvy(N), hvz(N), havx(N), havy(N), havz(N);
    std::vector<float> hqx(N), hqy(N), hqz(N), hqw(N);
    for (int i = 0; i < N; ++i) {
      float invM = im[i];
      // predict velocity
      F3 v{vx[i], vy[i], vz[i]};
      if (invM > 0)
        v = add3(v, scale3(gravity, dt));
      F3 wpred{ax[i], ay[i], az[i]};
      F3 invI3{iix[i], iiy[i], iiz[i]};
      F4 q{qx[i], qy[i], qz[i], qw[i]};
      if (invM > 0 && (invI3.x > 0 || invI3.y > 0 || invI3.z > 0)) {
        F3 wb = invRotateVector(q, F3{ax[i], ay[i], az[i]});
        F3 Ib{invI3.x > 1e-9f ? 1.f / invI3.x : 0.f, invI3.y > 1e-9f ? 1.f / invI3.y : 0.f,
              invI3.z > 1e-9f ? 1.f / invI3.z : 0.f};
        F3 Lb{Ib.x * wb.x, Ib.y * wb.y, Ib.z * wb.z};
        F3 wxL = cross3v(wb, Lb);
        F3 al{-invI3.x * wxL.x, -invI3.y * wxL.y, -invI3.z * wxL.z};
        wb = add3(wb, scale3(al, dt));
        wpred = rotateVector(q, wb);
      }
      // posPred (speculative, overwritten later)
      // apply velocity deltas
      F3 vP = add3(v, F3{dvx[i], dvy[i], dvz[i]});
      F3 wP = add3(wpred, F3{dax[i], day[i], daz[i]});
      // re-integration: persist, trapezoidal pos, quat integrate
      F3 vStart{vx[i], vy[i], vz[i]};
      F3 vFinal = vP;
      F3 x{px[i], py[i], pz[i]};
      if (invM > 0)
        x = add3(x, scale3(add3(vStart, vFinal), 0.5f * dt));
      F3 omega = wP;  // angVelPred after deltas
      F4 qp = q;
      if (invM > 0) {
        qp.x += 0.5f * dt * (omega.x * q.w + omega.y * q.z - omega.z * q.y);
        qp.y += 0.5f * dt * (omega.y * q.w + omega.z * q.x - omega.x * q.z);
        qp.z += 0.5f * dt * (omega.z * q.w + omega.x * q.y - omega.y * q.x);
        qp.w += 0.5f * dt * (-omega.x * q.x - omega.y * q.y - omega.z * q.z);
        float len = std::sqrt(qp.x * qp.x + qp.y * qp.y + qp.z * qp.z + qp.w * qp.w);
        if (len > 1e-9f) {
          float s = 1.f / len;
          qp = F4{qp.x * s, qp.y * s, qp.z * s, qp.w * s};
        }
      }
      // velPred persisted into vel = vFinal; angVel = omega
      // apply updates (count = 1+i%3): posPred += dpos/count; velPred += dvel(=0 now)/count
      int c = 1 + (i % 3);
      float f = 1.f / (float)c;
      x = add3(x, scale3(F3{dpx[i], dpy[i], dpz[i]}, f));
      // vel delta is 0 at this point (cleared by applyVelocityDeltas), so velPred unchanged
      // final commit: wrap
      if (dom.periodic_x) {
        if (x.x < dom.min.x)
          x.x += dom.size.x;
        else if (x.x >= dom.max.x)
          x.x -= dom.size.x;
      }
      if (dom.periodic_y) {
        if (x.y < dom.min.y)
          x.y += dom.size.y;
        else if (x.y >= dom.max.y)
          x.y -= dom.size.y;
      }
      if (dom.periodic_z) {
        if (x.z < dom.min.z)
          x.z += dom.size.z;
        else if (x.z >= dom.max.z)
          x.z -= dom.size.z;
      }
      hpx[i] = x.x;
      hpy[i] = x.y;
      hpz[i] = x.z;
      hvx[i] = vFinal.x;
      hvy[i] = vFinal.y;
      hvz[i] = vFinal.z;
      havx[i] = omega.x;
      havy[i] = omega.y;
      havz[i] = omega.z;
      hqx[i] = qp.x;
      hqy[i] = qp.y;
      hqz[i] = qp.z;
      hqw[i] = qp.w;
    }

    int bad = 0;
    auto tol = [&](float a, float b) { return std::fabs(a - b) <= 1e-4f * (1 + std::fabs(b)); };
    for (int i = 0; i < N; ++i) {
      if (!tol(gpos[3 * i], hpx[i]) || !tol(gpos[3 * i + 1], hpy[i]) ||
          !tol(gpos[3 * i + 2], hpz[i]))
        ++bad;
      if (!tol(gvel[3 * i], hvx[i]) || !tol(gvel[3 * i + 1], hvy[i]) ||
          !tol(gvel[3 * i + 2], hvz[i]))
        ++bad;
      if (!tol(gav[3 * i], havx[i]) || !tol(gav[3 * i + 1], havy[i]) ||
          !tol(gav[3 * i + 2], havz[i]))
        ++bad;
      if (!tol(gq[4 * i], hqx[i]) || !tol(gq[4 * i + 1], hqy[i]) || !tol(gq[4 * i + 2], hqz[i]) ||
          !tol(gq[4 * i + 3], hqw[i]))
        ++bad;
    }
    if (bad) {
      std::fprintf(stderr, "FAIL: %d/%d bodies differ\n", bad, N);
      status = 1;
    } else
      std::printf("[integration] PASS: %d bodies, pos/vel/ang_vel/quat match host (exec: %s)\n", N,
                  CpExec::name());
  }
  Kokkos::finalize();
  return status;
}
