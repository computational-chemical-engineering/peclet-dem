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

## Phase 1 — localize the engine defect

Artifact: `phase1_protocol_spheres.py` (the RingBed annealing protocol ported to spheres).

### Finding 1.1 — not a regression
The solver translation units (`solver_position.cu`, `solver_velocity.cu`, `xpbd_solver.cu`,
`narrowphase.cu`, `contact_preprocessing.cu`) are **byte-identical** to the pinned older engine
`extern/packing-gpu@324e4fa` the user reports worked; only the MPI halo code differs. The older engine
reproduces the same broken sphere result (φ=0.64 gentle → 13 % overlap, Z=2.1, 990 rattlers). The defect
is intrinsic and long-standing, not introduced by recent edits.

### Finding 1.2 — the pipeline is velocity-solve + position post-stabilization (docs were stale)
The shipped `docs/solver_details.md` is out of date. The real `step()` runs: predict (gravity) → ghosts →
broad/narrow phase → **Phase A velocity solve** (restitution / friction / growth-separation) → predict
position from the resolved velocity → **Phase B position solve** (Jacobi post-stabilization to remove
penetration) → commit (`d_pos = d_pos_pred`, velocity NOT re-derived from the position change). So the
contact *dynamics* live in the velocity solve, which does nothing at rest with no relative velocity — the
scheme is designed around **agitation** (the thermostat) + elastic collisions during growth, then cooling.
Phase 0's naive test (no thermostat, restitution 0) therefore froze; that was a missing-protocol artifact,
not proof of an engine bug.

### Finding 1.3 — the overlap query defeats the protocol (the engine defect)
Running the full RingBed annealing protocol (MB seed, elastic + thermostat during adaptive growth, then
dissipative cooling) the engine grows **unchecked** to a collapsed state — φ_protocol=0.72, but
φ_corrected=0.465, max_overlap=90 %, Z=5.7 — because **`get_max_overlap()` reads ≈0 throughout growth**, so
the overlap-criterion feedback never backs the growth off. Direct comparison on a φ=0.68 grown state:

```
engine get_max_overlap() = 0.035   <- the protocol's control signal (under-reports ~10-30x)
engine compute_overlaps() = 0.95   <- a CORRECT signal exists, sees the deep overlap
meter: max_overlap 0.41,  phi_corr 0.674,  Z 6.17,  rattlers 4/1000
```

`get_max_overlap()` returns the **position solver's mid-iteration residual** (`solver_position.cu:240`,
zeroed every iteration), not the committed overlap — it under-reports by ~10-30×. `compute_overlaps()`
re-runs detection and reports the real value. The packing at φ=0.68 actually has a healthy network
(Z≈6, 4 rattlers) but a few **pathologically deep stuck overlaps** the position solver never resolves, and
the control signal is blind to them.

### Phase 1 conclusion
The engine defect is twofold and actionable:
1. **`get_max_overlap()` is not a valid overlap measure** (position-solver residual, under-reports ~10-30×).
   Any protocol that controls growth off it steers blind → unchecked inflation → collapse. A correct
   signal (`compute_overlaps()`) already exists.
2. **A few contacts are left deeply unresolved** (stuck deep overlaps) even when the bulk network is fine —
   to be traced to the once-per-step stale contact list, the periodic min-image in the per-contact `C`, or
   Jacobi under-relaxation in `solver_position.cu`.

Planned Phase 2: (a) fix the overlap measurement so the criterion is meaningful (report the committed
detection-time overlap; have the protocol use it), (b) trace and fix the stuck deep-overlap pairs, then
(c) re-run the adaptive protocol and validate a clean RCP (Z≈6, overlap ≪ 1 %, isotropic g(r)).
