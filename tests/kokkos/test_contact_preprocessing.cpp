// Correctness of the Kokkos contact->manifold reduction (peclet::dem::reduceContactsToManifoldsKokkos)
// against a host reference that groups contacts by canonical pair and sums the same per-contact
// transform. Checks: manifold count == unique pairs; per-pair summed quantities match (float tol);
// num_points and canonical (bodyA,bodyB) match exactly. Runs on whatever backend Kokkos was built
// for (CUDA locally; OpenMP for CI).
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "contact_preprocessing.hpp"

using peclet::dem::ContactC;
using peclet::dem::F4;
using peclet::dem::ManifoldC;

static bool close(float a, float b) { return std::fabs(a - b) <= 1e-3f * (1.0f + std::fabs(b)); }
static bool close4(const F4& a, const F4& b) {
  return close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int Nbodies = 500;
    const int Ncontacts = 8000;

    std::mt19937 rng(2024);
    std::uniform_int_distribution<int> ubody(0, Nbodies - 1);
    std::uniform_real_distribution<float> uf(-1.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    std::vector<ContactC> h(Ncontacts);
    for (int i = 0; i < Ncontacts; ++i) {
      int a = ubody(rng);
      int b;
      if (u01(rng) < 0.2f) {
        b = -1;  // boundary
      } else {
        do {
          b = ubody(rng);
        } while (b == a);
      }
      h[i].bodyA = a;
      h[i].bodyB = b;
      h[i].normal = F4{uf(rng), uf(rng), uf(rng), 0.0f};
      h[i].rA = F4{uf(rng), uf(rng), uf(rng), 0.0f};
      h[i].rB = F4{uf(rng), uf(rng), uf(rng), 0.0f};
      h[i].dist = uf(rng) * 0.5f;  // ~half inactive (dist>0), half active
      h[i].friction_lambda_n = 0.0f;
      h[i].weight = 1.0f;
    }

    // --- device reduction ---
    Kokkos::View<ContactC*, peclet::dem::CpMem> dContacts("contacts", Ncontacts);
    {
      auto hc = Kokkos::create_mirror_view(dContacts);
      for (int i = 0; i < Ncontacts; ++i) hc(i) = h[i];
      Kokkos::deep_copy(dContacts, hc);
    }
    Kokkos::View<ManifoldC*, peclet::dem::CpMem> dMan("manifolds", Ncontacts);  // upper bound
    Kokkos::View<int, peclet::dem::CpMem> dCount("count");
    const int nman = peclet::dem::reduceContactsToManifoldsKokkos(dContacts, Ncontacts, dMan, dCount);

    std::vector<ManifoldC> gotMan(nman);
    {
      auto hm = Kokkos::create_mirror_view(dMan);
      Kokkos::deep_copy(hm, dMan);
      for (int i = 0; i < nman; ++i) gotMan[i] = hm(i);
    }

    // --- host reference: group by key, sum the same transform ---
    std::map<std::uint64_t, ManifoldC> ref;
    for (int i = 0; i < Ncontacts; ++i) {
      const std::uint64_t key = peclet::dem::pairKey(h[i]);
      auto it = ref.find(key);
      if (it == ref.end()) {
        ManifoldC m{};
        peclet::dem::decodeKey(key, m.bodyA, m.bodyB);
        it = ref.emplace(key, m).first;
      }
      const ManifoldC t = peclet::dem::transformContact(h[i]);
      ManifoldC& m = it->second;
      m.num_points += t.num_points;
      F4* md[5] = {&m.normal_sum, &m.torque_armA_sum, &m.torque_armB_sum, &m.rA_sum, &m.rB_sum};
      const F4* td[5] = {&t.normal_sum, &t.torque_armA_sum, &t.torque_armB_sum, &t.rA_sum, &t.rB_sum};
      for (int q = 0; q < 5; ++q) {
        md[q]->x += td[q]->x;
        md[q]->y += td[q]->y;
        md[q]->z += td[q]->z;
      }
    }

    if (static_cast<int>(ref.size()) != nman) {
      std::fprintf(stderr, "FAIL: manifold count %d != unique pairs %zu\n", nman, ref.size());
      status = 1;
    }

    // Match device manifolds to reference by canonical key.
    int mismatches = 0;
    for (const auto& m : gotMan) {
      const std::uint64_t key =
          (m.bodyB < 0)
              ? ((static_cast<std::uint64_t>(static_cast<unsigned>(m.bodyA)) << 32) | 0xFFFFFFFFu)
              : ((static_cast<std::uint64_t>(static_cast<unsigned>(m.bodyA)) << 32) |
                 static_cast<unsigned>(m.bodyB));
      auto it = ref.find(key);
      if (it == ref.end()) {
        ++mismatches;
        continue;
      }
      const ManifoldC& r = it->second;
      if (m.num_points != r.num_points || !close4(m.normal_sum, r.normal_sum) ||
          !close4(m.torque_armA_sum, r.torque_armA_sum) ||
          !close4(m.torque_armB_sum, r.torque_armB_sum) || !close4(m.rA_sum, r.rA_sum) ||
          !close4(m.rB_sum, r.rB_sum)) {
        ++mismatches;
      }
    }
    if (mismatches) {
      std::fprintf(stderr, "FAIL: %d manifold mismatches\n", mismatches);
      status = 1;
    }
    if (!status) {
      std::printf("[contact_preprocessing] PASS: %d manifolds match host reference (exec: %s)\n",
                  nman, peclet::dem::CpExec::name());
    }
  }
  Kokkos::finalize();
  return status;
}
