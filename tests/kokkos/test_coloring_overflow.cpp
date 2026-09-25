// dem — colour-mask overflow of the contact / manifold graph colourings (REPORT-ONLY; the future
// gate). docs/contact_evidence/FOLLOWUPS.md, defect 1.
//
// Both greedy colourings (colorManifoldsKokkos: the velocity sweeps; colorContactsKokkos: the
// position sweeps) pick the lowest colour free at both endpoints with
//     while (c < 62 && (forbidden & (1 << c))) ++c;
// so once colours 0..61 are all taken at a body, every further contact of that body gets colour
// 62, whether or not 62 is taken too. Colours are 0..62 (63 values), so a body with D > 63 contacts
// carries at least D - 63 same-colour pairs: the colour classes are no longer independent sets,
// and the in-place read-modify-write of one colour class races on that body (multi-thread, GPU).
// The comment's intent -- leave the contact uncoloured (-1) for the count-averaged fallback -- is
// never reached: the arbitration colours at least one contact per round, so the stall break does
// not fire and `leftover` stays 0.
//
// The test builds a star graph: one hub body touching D leaf bodies (the leaves touch nothing
// else), for D across the 63 threshold, runs both colourings, and counts
//     conflicts = sum over (body, colour) of (number of contacts with that colour at that body - 1)
// A valid colouring has 0. It prints one line per (colouring, D):
//     COLOR_OVERFLOW graph=<contacts|manifolds> D=.. colours=.. leftover=.. conflicts=.. at62=..
// at62 = contacts coloured 62. It never fails (kGate = false); when the colourings are fixed,
// kGate = true makes conflicts > 0 a failure.
#include <cstdint>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <map>
#include <utility>
#include <vector>

#include "solver_position.hpp"
#include "solver_velocity.hpp"

using namespace peclet::dem;

// ---- the future gate: a valid colouring has no same-colour pair at any body ----
static constexpr bool kGate = false;

// sum over (body, colour) of (count - 1), for coloured (>= 0) edges.
static int conflictsOf(const std::vector<std::pair<int, int>>& edges, const std::vector<int>& col) {
  std::map<std::pair<int, int>, int> n;
  for (std::size_t e = 0; e < edges.size(); ++e) {
    if (col[e] < 0)
      continue;
    ++n[{edges[e].first, col[e]}];
    if (edges[e].second >= 0)
      ++n[{edges[e].second, col[e]}];
  }
  int c = 0;
  for (const auto& [k, v] : n)
    c += v - 1;
  return c;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    for (int D : {32, 62, 63, 64, 65, 100, 300}) {
      // body 0 = hub, bodies 1..D = leaves; edge e = (0, e + 1)
      const int nb = D + 1, ne = D;
      std::vector<std::pair<int, int>> edges;
      for (int e = 0; e < ne; ++e)
        edges.push_back({0, e + 1});

      Kokkos::View<long long*, CpMem> winner("winner", nb);
      Kokkos::View<std::uint64_t*, CpMem> mask("mask", nb);

      // position graph: one ContactC per edge
      {
        Kokkos::View<ContactC*, CpMem> c("c", ne);
        auto hc = Kokkos::create_mirror_view(c);
        for (int e = 0; e < ne; ++e) {
          hc(e) = ContactC{};
          hc(e).bodyA = edges[e].first;
          hc(e).bodyB = edges[e].second;
          hc(e).dist = -0.01f;
        }
        Kokkos::deep_copy(c, hc);
        Kokkos::View<int*, CpMem> col("col", ne);
        int leftover = -1;
        const int nc = colorContactsKokkos(c, ne, nb, col, winner, mask, leftover);
        auto hcol = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), col);
        std::vector<int> v(hcol.data(), hcol.data() + ne);
        int at62 = 0;
        for (int x : v)
          at62 += (x == 62);
        const int conf = conflictsOf(edges, v);
        std::printf(
            "COLOR_OVERFLOW graph=contacts  D=%3d colours=%2d leftover=%d conflicts=%3d at62=%3d\n",
            D, nc, leftover, conf, at62);
        if (kGate && conf > 0)
          fail = 1;
      }
      // velocity graph: one ManifoldC per edge, identity real-index map
      {
        Kokkos::View<ManifoldC*, CpMem> m("m", ne);
        Kokkos::View<int*, CpMem> realIdx("realIdx", nb);
        auto hm = Kokkos::create_mirror_view(m);
        auto hr = Kokkos::create_mirror_view(realIdx);
        for (int e = 0; e < ne; ++e) {
          hm(e) = ManifoldC{};
          hm(e).bodyA = edges[e].first;
          hm(e).bodyB = edges[e].second;
          hm(e).num_points = 1;
        }
        for (int i = 0; i < nb; ++i)
          hr(i) = i;
        Kokkos::deep_copy(m, hm);
        Kokkos::deep_copy(realIdx, hr);
        Kokkos::View<int*, CpMem> col("col", ne);
        int leftover = -1;
        const int nc = colorManifoldsKokkos(m, ne, realIdx, nb, col, winner, mask, leftover);
        auto hcol = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), col);
        std::vector<int> v(hcol.data(), hcol.data() + ne);
        int at62 = 0;
        for (int x : v)
          at62 += (x == 62);
        const int conf = conflictsOf(edges, v);
        std::printf(
            "COLOR_OVERFLOW graph=manifolds D=%3d colours=%2d leftover=%d conflicts=%3d at62=%3d\n",
            D, nc, leftover, conf, at62);
        if (kGate && conf > 0)
          fail = 1;
      }
    }
  }
  Kokkos::finalize();
  std::printf(fail ? "FAILED\n" : "OK (report-only: kGate = %s)\n", kGate ? "true" : "false");
  return fail;
}
