# Packing investigation — why periodic sphere packing doesn't reach random close packing

Systematic investigation of why periodic monodisperse sphere packing fails to reach the expected random
close packing density (φ ≈ 0.64, coordination Z ≈ 6). Reference benchmark: frictionless monodisperse RCP
φ ≈ 0.64, isostatic Z ≈ 6, residual overlap ≪ 1 %.

Artifacts: `pack_meter.py` (trustworthy meter), `phase0_sphere_repro.py` (reproduction sweep).

## Phase 0 — trustworthy measurement + reproduction

### Finding 0.1 — the shipped meter is wrong
`verify_packing_spheres.py` computes `phi = N·(4/3)π·radius³ / domain³` with `radius` = the *base* 0.5 and
a domain sized for `phi_ref = 0.6`, **ignoring the grown per-particle `scale`**. It therefore reports the
design φ (0.6) by construction, regardless of the actual packing. `pack_meter.py` replaces it: effective
radii `r_i = base·scale_i·global_scale`, true periodic box volume, overlap-corrected solid fraction,
coordination number Z, rattler count, max overlap, and g(r), all with a periodic KD-tree.

### Finding 0.2 — the protocol never searches for jamming
The growth model grows every particle to a *preset* target size (`growth_factor` 0.05 → 1.0) that fills the
box to the chosen design φ, then stops (`growth_rate` is zeroed at completion). There is no gravity and no
continued drive, so nothing compacts the particles toward jamming. Reproduction (N=2000, frictionless,
periodic, single-phase growth + settle):

| design φ | φ_corrected | max overlap | Z | rattlers |
|---|---|---|---|---|
| 0.55 | 0.550 | 3.7e-4 | 0.00 | 2000 |
| 0.60 | 0.600 | 9.8e-5 | 0.00 | 2000 |
| 0.62 | 0.620 | 9.1e-5 | 0.00 | 2000 |
| 0.64 | 0.640 | **9.5e-2** | 3.42 | 210 |
| 0.66 | 0.658 | **3.6e-1** | 5.60 | 20 |
| 0.68 | 0.675 | **3.8e-1** | 6.41 | 6 |

Below ≈0.62 the particles form **zero contacts** (a gapped, contact-free arrangement that merely fills the
box); at/above 0.64 the box is too small and they **interpenetrate grossly** (9–38 %). No design φ produces
the signature of a real jammed packing (Z ≈ 6 with overlap ≪ 1 %).

### Finding 0.3 — the engine does not resolve contacts (primary defect)
A *gentle* approach (slow growth `rate=0.3`, 100 solver iterations, 4000 settle steps) at φ=0.64, run to
**full rest** (KE ≈ 4e-8):

```
φ_corrected = 0.640   max_overlap = 8.8 %   Z = 2.08   rattlers = 997 / 2000
```

8.8 % interpenetration **and** 50 % rattlers **at zero kinetic energy** is physically impossible for a real
packing — deeply overlapped pairs stay stuck while other particles never engage, and neither more
iterations nor more settling time fixes it. The XPBD **position solver is not driving contacts to
resolution**. This is an engine-level defect, and it is the primary cause: even a correct jamming-search
protocol cannot produce a valid packing on top of a solver that leaves ~9 % overlaps unresolved.

## Conclusions and next steps

Two distinct problems, the second dominant:
1. **Protocol:** grows to a preset φ instead of searching for the jamming density (needs an adaptive
   growth / overlap-criterion search, as in the RingBed reference protocol).
2. **Engine (primary):** the position solver leaves large persistent overlaps with simultaneous rattlers
   even at full rest — contacts are not being resolved.

Planned next (Phase 1): (a) localize the engine defect by diffing/comparing against the pinned older engine
`RingBed-CFD-Surrogate/extern/packing-gpu@324e4fa` the user reports worked; (b) read the XPBD position
solver (`solver_position.cu`, `xpbd_solver.cu`, `contact_preprocessing.cu`, periodic min-image in the
narrow phase) to find why overlaps are not resolved (Jacobi relaxation / mass weighting / manifold or
periodic-image bug). Then fix the engine, validate a clean RCP (Z ≈ 6, overlap ≪ 1 %, isotropic g(r)),
and only then layer the adaptive jamming-search protocol and extend to rings.
