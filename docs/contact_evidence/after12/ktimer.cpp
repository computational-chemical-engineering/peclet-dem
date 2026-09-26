// Minimal Kokkos Tools kernel timer: accumulates wall time per kernel name, prints on finalize
// (rank 0 only if OMPI_COMM_WORLD_RANK is set). For the session's WO-11 cost attribution.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
namespace {
using clk = std::chrono::steady_clock;
struct Rec { double t = 0; long n = 0; };
std::map<std::string, Rec> recs;
struct Fr { std::string n; clk::time_point t; double cov = 0; };
std::vector<Fr> stack;
clk::time_point dcT;
std::uint64_t nextId = 0;
std::map<std::uint64_t, std::pair<std::string, clk::time_point>> open;
void begin(const char* name, std::uint64_t* id) { *id = nextId++; open[*id] = {name, clk::now()}; }
void end(std::uint64_t id) {
  auto it = open.find(id); if (it == open.end()) return;
  auto& r = recs[it->second.first];
  const double d = std::chrono::duration<double>(clk::now() - it->second.second).count();
  r.t += d; ++r.n; open.erase(it); if (!stack.empty()) stack.back().cov += d;
}
}
extern "C" {
void kokkosp_init_library(const int, const std::uint64_t, const std::uint32_t, void*) {}
void kokkosp_finalize_library() {
  const char* r = std::getenv("OMPI_COMM_WORLD_RANK");
  if (r && std::atoi(r) != 0) return;
  std::vector<std::pair<std::string, Rec>> v(recs.begin(), recs.end());
  std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.t > b.second.t; });
  double tot = 0; for (auto& x : v) tot += x.second.t;
  std::fprintf(stderr, "KTIMER total_kernel_s=%.4f\n", tot);
  int k = 0;
  for (auto& x : v) { if (k++ >= 400) break; std::fprintf(stderr, "KT %9.4f s %8ld  %s\n", x.second.t, x.second.n, x.first.c_str()); }
}
void kokkosp_begin_parallel_for(const char* n, const std::uint32_t, std::uint64_t* id) { begin(n, id); }
void kokkosp_end_parallel_for(const std::uint64_t id) { end(id); }
void kokkosp_begin_parallel_reduce(const char* n, const std::uint32_t, std::uint64_t* id) { begin(n, id); }
void kokkosp_end_parallel_reduce(const std::uint64_t id) { end(id); }
void kokkosp_begin_parallel_scan(const char* n, const std::uint32_t, std::uint64_t* id) { begin(n, id); }
void kokkosp_end_parallel_scan(const std::uint64_t id) { end(id); }
void kokkosp_push_profile_region(const char* n) { stack.push_back({std::string("REGION ") + n, clk::now(), 0}); }
void kokkosp_pop_profile_region() {
  if (stack.empty()) return;
  const double d = std::chrono::duration<double>(clk::now() - stack.back().t).count();
  auto& r = recs[stack.back().n]; r.t += d; ++r.n;
  auto& x = recs["XHOST " + stack.back().n.substr(7)]; x.t += d - stack.back().cov; ++x.n;
  stack.pop_back(); if (!stack.empty()) stack.back().cov += d;
}
void kokkosp_begin_deep_copy(void*, const char*, const void*, void*, const char*, const void*, const std::uint64_t) { dcT = clk::now(); }
void kokkosp_end_deep_copy() { const double d = std::chrono::duration<double>(clk::now() - dcT).count(); auto& r = recs["deep_copy(all)"]; r.t += d; ++r.n; if (!stack.empty()) stack.back().cov += d; }
}
