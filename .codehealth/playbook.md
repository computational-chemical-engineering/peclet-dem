# dem playbook (overrides the cce-code-health seed by id)

Read the seed first (`plugins/cce-code-health/skills/refactor-loop/playbook-seed.md`, tool pin c5ee4df), then this file.
Read `CLAUDE.md` (settled decisions) before touching any solver file; a refactor never reverses one.
Gate: `PECLET_SUITE=<suite> codehealth gate`; bitwise = `state_hash` on the host/OpenMP build at ONE thread.

## Rules for new code

1. Pass `Kokkos::View`s by value and capture them by value in `KOKKOS_LAMBDA`; never capture `this` in a device
   lambda (copy members to locals first; it is illegal on CUDA only, so the host build will not catch it).
2. Never change the order of kernel launches, fences, or MPI calls; never change a colouring or the Gauss–Seidel
   sweep order. Any of these changes results, at more than one thread or on the GPU even when state_hash agrees.
3. Never introduce or reorder atomics or reductions in a refactor. state_hash runs at one thread, so it cannot see
   ordering changes that only show at more threads.
4. A moved kernel keeps its execution space and memory space (`CpMem`, the same policy); never move work to the host.
5. Name a thing by what it is, never by where it runs: no `Kokkos`/`Device` qualifiers in identifiers.
6. The MPI schedule never branches on rank-local state; every rank makes the same collective calls (CLAUDE.md).
7. Local order is never MPI arrival order (CLAUDE.md, reproducibility trap); keep the existing re-orderings.
8. No environment variable may change a result; algorithm choices are `Simulation` setters.
9. Keep `src/` header-only with `inline` free functions or private structs in the same header; do not add a `.cpp`.
10. Format with the suite venv's clang-format 18.1.8 (dem's CI format job blocks).

## Refactor recipes

| id | smell | when | when not | recipe |
|---|---|---|---|---|
| extract-kernel-phase | long function (`demSolveContacts`, 1160 lines, CCN 261) | a contiguous phase (colouring setup, position projection, velocity solve, restitution, rank-level M/X) reads and writes a nameable set of Views and scalars | the phase interleaves with another phase's launches, or shares many locals that the next phase mutates; the phase needs 10+ inputs (do `launch-params-struct` first) | move the phase verbatim into an `inline` free function (or a private member of the `Hooks` template) in the same header; same launches in the same order, same Views, same fences, same `if constexpr` branches; pass the needed Views by value and locals by reference; the caller becomes a readable sequence of phases |
| launch-params-struct | many parameters (launchers with 14–33 params, mostly Views) | the same group of 4+ Views/scalars travels together through several launchers and forms one concept (e.g. body state, manifold arrays, PGS accumulators) | the group is used by only one function, or a member would need a different memory space | a private aggregate struct holding the Views BY VALUE (trivially copyable, captured by value into device lambdas); default member values reproduce today's default arguments exactly |
| dedup-colouring | duplication (`colorManifolds*` vs `colorContactsIncremental*`) | two colouring routines differ only in the edge source / key | they differ in iteration order, tie-breaking or colour cap handling | one helper parameterized by the edge accessor; prove identical colourings by state_hash and `coloring_conflicts()`; keep the 64-colour cap and hub copies |
| functor-from-lambda | a kernel lambda carrying most of a launcher's CCN | the lambda body has its own branching and a name would explain it | the extraction only moves complexity (see seed: same complexity, new address) or nvcc rejects it | a named functor or `KOKKOS_INLINE_FUNCTION` with explicit by-value View members; build_cuda guards nvcc; the reviewer must agree it reads better |
| guard-clauses | (seed) | host-side setup code only | inside device code where it changes divergence or FP order | as seed |
| replace-temp-with-query | (seed) | host-side scalars only | device code, or anything in a solver inner loop (recomputing can change FP results under contraction) | as seed |
| split-flag-argument | (seed) | a private launcher whose flag selects two disjoint kernels | the flag is threaded from a `Simulation` setter through several levels | as seed |

Evidence (filled by consolidation from the run log, never by hand):
- extract-kernel-phase:
- launch-params-struct:
- dedup-colouring:
- functor-from-lambda:

## Anti-patterns (review rejects)

- **Reordering for tidiness**: any change to kernel launch order, fence placement, MPI call order, colouring or
  sweep order, or lambda capture semantics, even when state_hash agrees (it runs at one thread on the host only).
- **Touching atomics or reductions** (type, order, fusion, split): reject unless the step is explicitly an
  `optimize`-class step with a tolerance gate. That is never part of this refactor campaign.
- **Space migration**: a moved kernel that runs in another execution space or reads another memory space, or a
  `create_mirror_view`/`deep_copy` added or removed.
- **Settled-decision drift**: any change touching a bullet in dem `CLAUDE.md` "Settled decisions" (staged
  symmetric + one-sided contacts, `val − residual` wall sign, SoA Views, copies/mass split, ω_pos = 1.5, Moreau
  restitution, K = 64, the 8·g·dt slip gate, owner-exclusive contacts with reverse accumulation, ...).
- **Defaulted-argument drift**: a parameter struct whose defaults differ from the old default arguments.
- **Public surface change**: renaming or re-signing anything reachable from `src/dem_bindings.cpp`.
- **A phase helper with a parameter list as long as the phase** (seed c5ee4df): S1 step 3 moved a phase into a
  19-parameter helper and gained 0.9 (332.1 -> 331.2). Group the inputs first. The `[&]`-capturing lambdas passed
  into a helper (`voteInvariant`, `withConsensus`) carry caller locals the metric does not count; review checks them.
- **Debt by lambda shuffling**: cpp@0 counts lambda bodies in the enclosing function, so moving a kernel into a
  functor lowers "debt" without lowering complexity. Accept it only when the reviewer agrees the result reads better.
