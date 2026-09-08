# Archive — dated campaign records

These are **dated investigation records and superseded notes**, moved out of the live documentation
set on 2026-09-08 (suite `docs/QUALITY_PLAN.md` decision D7: *docs describe the code that exists*).
Each was written to steer a piece of work that has since landed or been folded into a reference
document. Nothing here is maintained: file/line citations, measured tables, "next steps" and status
lines are snapshots of their date — the authority on how `peclet.dem` behaves today is `src/` plus
the reference docs in the parent directory ([solver_details.md](../solver_details.md),
[mpi.md](../mpi.md), [multi_gpu_testing.md](../multi_gpu_testing.md),
[visualization.md](../visualization.md)) and the repo `CLAUDE.md`. Nothing here is deleted.

| note | date | what it is |
|---|---|---|
| [packing_investigation.md](packing_investigation.md) | 2026-06/07 | The five-phase investigation of why periodic monodisperse sphere packing did not reach random close packing, and the fixes that closed it (Phase 0 measurement → Phase 4 friction-model analysis). The narrative behind today's validated packing protocol; the shipped meter is `examples/pack_meter.py`. Its "the shipped `docs/solver_details.md` is out of date" note (Phase 1) refers to a revision that has since been rewritten. |
| [velocity_solver_algorithm.md](velocity_solver_algorithm.md) | 2026-07 | A focused summary of the velocity solve, written to retract an earlier CUDA-era conclusion about "unweighted summation". Superseded by [solver_details.md](../solver_details.md) §4, which describes the shipped colored-Gauss–Seidel / warm-started-PGS velocity phase; this note still describes the pre-PGS one-shot Jacobi form. |
