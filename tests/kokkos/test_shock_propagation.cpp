// Physics of the shock-propagation path (Guendelman et al. 2003; docs/DECISIONS "grounded
// rise-gated inelastic shock") in BOTH velocity solvers: dem::solveVelocity (atomic-accumulate
// Jacobi) and dem::solveVelocityColoredGS (in-place Gauss–Seidel). No production caller passes
// shock inputs today, so the step's state_hash never reaches this path; this test pins it.
//
// One body-body manifold per case, two spheres of diameter 1 with a central contact (zero torque
// arms), gravity along -z. Expectations:
//   (a) A above a grounded, non-rising B  -> B unchanged, A's approach removed, no bounce (e = 0)
//   (b) the mirror (B above A)            -> A unchanged, B's approach removed, no bounce
//   (c) a rising support (-v.gHat > 4 vRest), (d) an ungrounded support, (e) a near-horizontal
//       pair (|up| <= 0.3 |dx|), (f) a non-persistent contact -> the symmetric impulse: momentum
//       conserved, relative normal velocity reversed with the material restitution
//   (g) both solvers give the same post-solve velocities in every case
//   (h) a resting approach (|vn| < vRest) -> symmetric with e = 0
// The pair geometry is read at the manifold's (ghost-aware) body ids, the velocities, masses and
// grounded levels at the real slots: ids 2, 3 map to real slots 0, 1, and the id-indexed and
// real-indexed copies disagree, so a mixed-up index flips the outcome. A bitwise FNV-1a hash of
// every output is printed for base/head comparison; `--dump` also prints each output's bits.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "solver_velocity.hpp"

using namespace peclet::dem;

namespace {

constexpr int kSlots = 4;          // real slots 0 (A), 1 (B); id slots 2 (A), 3 (B)
constexpr float kRest = 0.5f;      // global restitution
constexpr float kVRest = 0.1f;     // resting threshold; the rise gate is 4 * kVRest
constexpr float kInvMassA = 1.0f;  // unequal masses: momentum conservation is a real check
constexpr float kInvMassB = 0.5f;
constexpr float kTol = 2e-6f;

enum Expect { kStaticB, kStaticA, kSymmetric, kSymmetricResting };

struct Case {
  const char* name;
  F3 pA, pB, vA, vB;
  unsigned char persistent, groundedA, groundedB;
  Expect expect;
};

struct Outputs {  // post-solve body velocities at the real slots + everything the hash covers
  std::vector<float> floats;
  std::vector<int> ints;
  F3 vA, vB, wA, wB;
};

std::uint64_t fnv1a(std::uint64_t h, const void* p, std::size_t n) {
  const auto* b = static_cast<const unsigned char*>(p);
  for (std::size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}

template <class V>
V upload3(const char* label, const F3 (&x)[kSlots]) {
  V v(label, kSlots);
  auto h = Kokkos::create_mirror_view(v);
  for (int i = 0; i < kSlots; ++i) {
    h(i, 0) = x[i].x;
    h(i, 1) = x[i].y;
    h(i, 2) = x[i].z;
  }
  Kokkos::deep_copy(v, h);
  return v;
}

template <class T>
Kokkos::View<T*, CpMem> upload1(const char* label, const T (&x)[kSlots]) {
  Kokkos::View<T*, CpMem> v(label, kSlots);
  auto h = Kokkos::create_mirror_view(v);
  for (int i = 0; i < kSlots; ++i)
    h(i) = x[i];
  Kokkos::deep_copy(v, h);
  return v;
}

template <class V>
void append3(std::vector<float>& out, const V& v) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  for (int i = 0; i < kSlots; ++i)
    for (int c = 0; c < 3; ++c)
      out.push_back(h(i, c));
}

F3 at(const std::vector<float>& f, int offset, int slot) {
  return F3{f[offset + 3 * slot], f[offset + 3 * slot + 1], f[offset + 3 * slot + 2]};
}

/// The case's bodies, manifold and shock inputs on the device; `solver` 0 = solveVelocity, 1 = GS.
Outputs runCase(const Case& c, int solver) {
  const F3 gHat{0.0f, 0.0f, -1.0f};
  const F3 d = sub3(c.pB, c.pA);
  const float len = std::sqrt(dot3(d, d));
  const F3 nrm = scale3(d, 1.0f / len);  // A -> B
  ManifoldC m{};
  m.bodyA = 2;
  m.bodyB = 3;
  m.normal_sum = F4{nrm.x, nrm.y, nrm.z, 0};
  m.torque_armA_sum = F4{0, 0, 0, 0};  // central contact
  m.torque_armB_sum = F4{0, 0, 0, 0};
  m.rA_sum = F4{0.5f * d.x, 0.5f * d.y, 0.5f * d.z, 0};  // contact at the midpoint
  m.rB_sum = F4{-0.5f * d.x, -0.5f * d.y, -0.5f * d.z, 0};
  m.num_points = 1;
  m.restitution_sum = -1.0f;  // global material
  m.friction_sum = -1.0f;

  // Real slots carry the bodies; the id slots carry garbage that only an index mix-up would read
  // (velocities, masses), the TRUE geometry (posPred is ghost-aware), and the complement of the
  // grounded levels.
  const F3 vel[kSlots] = {c.vA, c.vB, F3{9.0f, -9.0f, 9.0f}, F3{-7.0f, 7.0f, -7.0f}};
  const F3 zero3[kSlots] = {};
  const F3 pos[kSlots] = {c.pB, c.pA, c.pA, c.pB};
  const F3 invI[kSlots] = {F3{1, 1, 1}, F3{1, 1, 1}, F3{3, 3, 3}, F3{3, 3, 3}};
  const float invMass[kSlots] = {kInvMassA, kInvMassB, 7.0f, 7.0f};
  const int real[kSlots] = {0, 1, 0, 1};
  const unsigned char grounded[kSlots] = {c.groundedA, c.groundedB,
                                          static_cast<unsigned char>(!c.groundedA),
                                          static_cast<unsigned char>(!c.groundedB)};
  const unsigned char persistent[1] = {c.persistent};

  Kokkos::View<ManifoldC*, CpMem> dMan("man", 1);
  Kokkos::deep_copy(dMan, m);
  Kokkos::View<unsigned char*, CpMem> dPersist("persistent", 1);
  Kokkos::deep_copy(dPersist, persistent[0]);
  Kokkos::View<float* [4], CpMem> dQuat("quat", kSlots);
  {
    auto h = Kokkos::create_mirror_view(dQuat);
    for (int i = 0; i < kSlots; ++i) {
      h(i, 0) = 0.0f;
      h(i, 1) = 0.0f;
      h(i, 2) = 0.0f;
      h(i, 3) = 1.0f;
    }
    Kokkos::deep_copy(dQuat, h);
  }
  auto dVel = upload3<Kokkos::View<float* [3], CpMem>>("vel", vel);
  auto dAng = upload3<Kokkos::View<float* [3], CpMem>>("ang", zero3);
  auto dPos = upload3<Kokkos::View<float* [3], CpMem>>("pos", pos);
  auto dInvI = upload3<Kokkos::View<float* [3], CpMem>>("invI", invI);
  auto dInvMass = upload1("invMass", invMass);
  auto dReal = upload1("real", real);
  auto dGrounded = upload1("grounded", grounded);

  Outputs o;
  if (solver == 0) {
    Kokkos::View<float* [3], CpMem> dDV("dv", kSlots), dDW("dw", kSlots);
    Kokkos::View<int*, CpMem> dCounts("counts", kSlots);
    solveVelocity(
        dMan, 1, dInvMass, dInvI, dQuat, dVel, dAng, dReal, /*growthRate=*/0.0f, kRest, kVRest, dDV,
        dDW, dCounts, {}, 0,
        ShockPropagation{
            .persistent = dPersist, .posPred = dPos, .gHat = gHat, .grounded = dGrounded});
    append3(o.floats, dDV);
    append3(o.floats, dDW);
    auto hc = Kokkos::create_mirror_view(dCounts);
    Kokkos::deep_copy(hc, dCounts);
    for (int i = 0; i < kSlots; ++i)
      o.ints.push_back(hc(i));
    // Single manifold, count <= 1 per body: the averaged apply is a plain add.
    o.vA = F3{c.vA.x + o.floats[0], c.vA.y + o.floats[1], c.vA.z + o.floats[2]};
    o.vB = F3{c.vB.x + o.floats[3], c.vB.y + o.floats[4], c.vB.z + o.floats[5]};
    o.wA = at(o.floats, 3 * kSlots, 0);
    o.wB = at(o.floats, 3 * kSlots, 1);
  } else {
    Kokkos::View<int*, CpMem> dColor("color", 1);  // zero: one colour
    Kokkos::View<float, CpMem> dMaxApproach("maxApproach");
    solveVelocityColoredGS(
        ColoredManifolds{.manifolds = dMan, .numManifolds = 1, .mColor = dColor, .numColors = 1},
        VelocitySweepBodies{.invMass = dInvMass,
                            .invInertia = dInvI,
                            .quat = dQuat,
                            .velPred = dVel,
                            .angVelPred = dAng,
                            .realIdx = dReal},
        /*growthRate=*/0.0f, kRest, kVRest, dMaxApproach, {},
        ShockPropagation{
            .persistent = dPersist, .posPred = dPos, .gHat = gHat, .grounded = dGrounded});
    append3(o.floats, dVel);
    append3(o.floats, dAng);
    float maxApproach = 0.0f;
    Kokkos::deep_copy(maxApproach, dMaxApproach);
    o.floats.push_back(maxApproach);
    o.vA = at(o.floats, 0, 0);
    o.vB = at(o.floats, 0, 1);
    o.wA = at(o.floats, 3 * kSlots, 0);
    o.wB = at(o.floats, 3 * kSlots, 1);
  }
  return o;
}

bool near(float a, float b) {
  return std::fabs(a - b) <= kTol * (1.0f + std::fabs(b));
}
bool near3(F3 a, F3 b) {
  return near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z);
}
bool same3(F3 a, F3 b) {
  return std::memcmp(&a, &b, sizeof(F3)) == 0;
}

/// The physics expectations of one case for one solver's outputs; prints each violation.
int check(const Case& c, const char* solver, const Outputs& o) {
  int bad = 0;
  auto fail = [&](const char* what) {
    std::fprintf(stderr, "FAIL case %s (%s): %s\n", c.name, solver, what);
    ++bad;
  };
  const F3 d = sub3(c.pB, c.pA);
  const F3 nrm = scale3(d, 1.0f / std::sqrt(dot3(d, d)));
  const float vnPre = dot3(sub3(c.vA, c.vB), nrm);  // > 0: approaching
  const float vnPost = dot3(sub3(o.vA, o.vB), nrm);
  if (!(vnPre > 0.0f))
    fail("case set-up: the pair does not approach");
  if (!same3(o.wA, F3{0, 0, 0}) || !same3(o.wB, F3{0, 0, 0}))
    fail("central contact produced an angular velocity");
  switch (c.expect) {
    case kStaticB:
      if (!same3(o.vB, c.vB))
        fail("the static lower body B changed");
      if (same3(o.vA, c.vA))
        fail("the upper body A was not corrected");
      if (!near(vnPost, 0.0f))
        fail("A bounced off the static support (e != 0)");
      break;
    case kStaticA:
      if (!same3(o.vA, c.vA))
        fail("the static lower body A changed");
      if (same3(o.vB, c.vB))
        fail("the upper body B was not corrected");
      if (!near(vnPost, 0.0f))
        fail("B bounced off the static support (e != 0)");
      break;
    case kSymmetric:
    case kSymmetricResting: {
      const F3 dpA = scale3(sub3(o.vA, c.vA), 1.0f / kInvMassA);
      const F3 dpB = scale3(sub3(o.vB, c.vB), 1.0f / kInvMassB);
      if (!near3(F3{dpA.x + dpB.x, dpA.y + dpB.y, dpA.z + dpB.z}, F3{0, 0, 0}))
        fail("momentum not conserved");
      if (same3(o.vA, c.vA) || same3(o.vB, c.vB))
        fail("one side of the symmetric impulse was not applied");
      const float e = (c.expect == kSymmetric) ? kRest : 0.0f;
      if (!near(vnPost, -e * vnPre))
        fail("post-solve normal velocity is not -e vn");
      break;
    }
  }
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  bool dump = false;
  for (int i = 1; i < argc; ++i)
    dump = dump || std::strcmp(argv[i], "--dump") == 0;
  {
    const F3 top{0.0f, 0.0f, 1.0f}, bottom{0.0f, 0.0f, 0.0f};
    const F3 falling{0.0f, 0.0f, -1.0f}, support{0.05f, 0.0f, 0.1f};  // support rises 0.1 <= 0.4
    const Case cases[] = {
        {"a_A_above_grounded_B", top, bottom, falling, support, 1, 1, 1, kStaticB},
        {"b_B_above_grounded_A", bottom, top, support, falling, 1, 1, 1, kStaticA},
        {"c_rising_support", top, bottom, falling, F3{0.0f, 0.0f, 0.5f}, 1, 1, 1, kSymmetric},
        {"d_ungrounded_support", top, bottom, falling, support, 1, 1, 0, kSymmetric},
        {"e_near_horizontal", F3{0.0f, 0.0f, 0.2f}, F3{1.0f, 0.0f, 0.0f}, F3{1.0f, 0.0f, 0.0f},
         F3{0.0f, 0.0f, 0.0f}, 1, 1, 1, kSymmetric},
        {"f_not_persistent", top, bottom, falling, support, 0, 1, 1, kSymmetric},
        {"h_resting_approach", top, bottom, F3{0.0f, 0.0f, -0.05f}, F3{0.0f, 0.0f, 0.0f}, 0, 1, 1,
         kSymmetricResting},
    };
    std::uint64_t all = 1469598103934665603ull;
    for (const Case& c : cases) {
      const Outputs jac = runCase(c, 0), gs = runCase(c, 1);
      int bad = check(c, "solveVelocity", jac) + check(c, "solveVelocityColoredGS", gs);
      // (g) the two solvers agree; the Jacobi side's count bookkeeping matches the applied sides.
      if (!near3(jac.vA, gs.vA) || !near3(jac.vB, gs.vB)) {
        std::fprintf(stderr, "FAIL case %s: (g) the solvers disagree\n", c.name);
        ++bad;
      }
      const int wantA = (c.expect == kStaticA) ? 0 : 1, wantB = (c.expect == kStaticB) ? 0 : 1;
      if (jac.ints[0] != wantA || jac.ints[1] != wantB || jac.ints[2] != 0 || jac.ints[3] != 0) {
        std::fprintf(stderr, "FAIL case %s: velCounts %d %d %d %d\n", c.name, jac.ints[0],
                     jac.ints[1], jac.ints[2], jac.ints[3]);
        ++bad;
      }
      std::uint64_t h = 1469598103934665603ull;
      for (const Outputs* o : {&jac, &gs}) {
        h = fnv1a(h, o->floats.data(), o->floats.size() * sizeof(float));
        h = fnv1a(h, o->ints.data(), o->ints.size() * sizeof(int));
      }
      all = fnv1a(all, &h, sizeof(h));
      std::printf("case %-22s %s hash %016llx\n", c.name, bad ? "FAIL" : "ok",
                  static_cast<unsigned long long>(h));
      if (dump) {
        const char* names[2] = {"jacobi", "gs"};
        const Outputs* outs[2] = {&jac, &gs};
        for (int s = 0; s < 2; ++s)
          for (std::size_t i = 0; i < outs[s]->floats.size(); ++i) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &outs[s]->floats[i], sizeof(bits));
            std::printf("dump %s %s %zu %08x\n", c.name, names[s], i, bits);
          }
      }
      status |= bad ? 1 : 0;
    }
    std::printf("[shock_propagation] %s: hash %016llx (exec: %s)\n", status ? "FAIL" : "PASS",
                static_cast<unsigned long long>(all), CpExec::name());
  }
  Kokkos::finalize();
  return status;
}
