# Restitution target law: Newton vs Moreau (G-C3 report, WO-C1)

Report for the user's choice of the PGS restitution target (`docs/contact_physics_followups.md`
§4, R-C1). Branch `restlaw`, base dem `77ba917`. Newton stays the default; Moreau is
`sim.diagnostics.set_restitution_target('moreau')`. Nothing here changes a default.

**The two laws.** Both act only on the g ≠ 0 path, which runs the projected Gauss–Seidel. Each
closed contact gets a separation-velocity target from its pre-solve normal velocity u⁻.

- **Newton**: target −e u⁻ when the pair was approaching. A pair that was already separating
  gets target 0, which means it may not approach at all.
- **Moreau**: target −e u⁻ on every closed contact with |u⁻| ≥ v_rest = 2 dt |g|. A separating
  pair may drift back together at up to e|u⁻| before the contact pushes.

Below v_rest, and on one-sided contacts, e = 0 under both laws.

## 1. Energy in a dense kinetic cluster: the reason for the question

The ratio is KE after / KE before, in the centre-of-mass frame. Each case is one converged step
(2000 PGS iterations, no adaptive stop) with frictionless spheres. Newton adds energy whenever a
pre-separating contact ends up loaded (§4.1). Moreau has a theorem: KE cannot rise when e is
uniform.

| scene | contacts | e = 0.5 N / M | e = 0.9 N / M | e = 1.0 N / M |
|---|---|---|---|---|
| model A.4 of the note (numpy) | 670 | 0.743 / 0.664 | **1.204** / 0.915 | **1.364** / 1.0000 |
| pytest scene: A.4 in `peclet.dem`, v_rest = 2e-4 | 803 | 0.7256 / 0.6627 | **1.1490** / 0.9146 | **1.2979** / 0.99999991 |
| harness `cluster_pgs_e` (925 spheres), v_rest = 0.2, np 1 | – | 0.7288 / 0.6998 | **1.0274** / 0.9241 | **1.1298** / 1.000126 |
| the same, np 2 / 4 / 8 | – | agrees with np 1 to ≤ 3e-8 | ≤ 3e-8 | ≤ 3e-8 |
| the same on CUDA: np 1 fused device loop, np 2 step_mpi | – | – | 1.02737 / 0.92406 | 1.12978 / 1.000126 |

Kinetic energy over 10 steps in the harness cluster (KE₀ = 3536.7, np 1):

| e | law | step 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|
| 0.5 | N | 2577.5 | 2007.3 | 1680.5 | 1448.9 | 1267.7 |
| 0.5 | M | 2475.1 | 1924.4 | 1607.9 | 1376.9 | 1201.9 |
| 0.9 | N | 3633.5 | 3403.0 | 3199.9 | 3000.6 | 2824.3 |
| 0.9 | M | 3268.1 | 3044.0 | 2855.2 | 2663.6 | 2505.6 |
| 1.0 | N | **3995.7** (+13 %) | 3996.3 | 3996.2 | 3995.7 | 3995.5 |
| 1.0 | M | 3537.1 | 3537.0 | 3536.7 | 3536.0 | 3535.6 |

At e = 1, Newton adds 13 % of the energy in the first step. After that step it neither gains
nor loses: once the pre-separating pairs are released, the cluster behaves as if elastic. Moreau
stays within 0.03 % of KE₀ over the whole run.

**The resting-threshold channel is measured, not only bounded (R-C3).** At the harness's dt =
1e-2, v_rest = 0.2, while contact velocities are about 1.5. So some loaded contacts sit below
v_rest and are held at e = 0. The e = 1 Moreau step then gains **1.26e-4**. That is above
G-C1's 1e-5 bound, although the note says no creation was observed in its model. Shrinking v_rest
makes the gain vanish:

| dt (so v_rest = 20 dt) | 1e-2 | 3e-3 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| Moreau KE₁/KE₀ − 1, e = 1 | +1.26e-4 | −2.1e-5 | +6.9e-6 | +3e-9 |
| Newton KE₁/KE₀ − 1, e = 1 | +0.130 | +0.126 | +0.107 | +0.115 |

## 2. Where the two laws agree bit for bit (G-C2)

Method: every `get_*` array read by the tests was hashed at OMP 1 under each law, and the hashes
compared.

- **Identical under Moreau:**
  - `test_binary_exactness` (e = 0.2, 0.5, 0.8, 0.95);
  - `test_restitution.py` (normal restitution, sliding friction);
  - `test_bounce.py` and `test_bounce_gravity.py`;
  - `test_cone_friction.py` (Walton oblique impact, binary restitution with friction, slide to
    roll, kinetic slide, slab stick);
  - the g = 0 Enskog cooling gas: 150 steps, positions and velocities `array_equal`.
- **Different: the statics battery's column and pour-collapse.** The note expected them unchanged
  because they are resting beds. They are not resting while they settle.
  - The column first differs at substep 33 (t = 0.33 s). At that point 524 of 2132 closed pairs
    are kinetically separating (u⁻ < −0.2). The differing bodies sit in inter-layer pairs, for
    example 357–501 (z 2.47/3.48, u⁻ = −0.67, gap +0.010) and 791–935 (z 5.46/6.46, u⁻ = −0.39,
    gap −0.0001). These are layers rebounding from each other inside the contact margin.
  - The pour first differs at substep 123 (t = 1.23 s), in the impact. Examples: pair 1–2 on the
    floor (u⁻ = −3.87, gap +0.042) and 146–290 (u⁻ = −3.32, gap +0.031).
  - Both tests still pass under Moreau (table 3).

## 3. Flows and beds (single runs; CUDA runs are not bitwise reproducible, so small differences include chaotic divergence)

| observable | Newton | Moreau | change |
|---|---|---|---|
| **Dosta silo** (100k M1 spheres, large orifice, e = 0.5, CUDA): discharge rate 0.5–2.5 s | 27 913 /s | 28 314 /s | +1.4 % |
| silo: grains discharged in 0–0.5 s | 25 624 | 27 379 | +6.8 % |
| silo: grains left at t = 2 s | 28 389 | 25 836 | −9.0 % |
| **Dosta 25k impact** (steel ball at 5 m/s into an M1 bed, CUDA): lowest ball displacement | −0.1399 m | −0.1401 m | +0.1 % |
| impact: displacement at t = 0.1 s | −0.1053 m | −0.1124 m | |
| impact: rebound (end − lowest) | 0.0346 m | 0.0277 m | **−20 %** |
| **pytest rotating drum** (264 grains, e = 0.1, ω = 0.5): bed tilt, mean ± sd over the last 1000 steps | +38.4 ± 4.8° | +37.4 ± 4.2° | within noise |
| drum: surface speed (mean \|v\| of the 10 % of grains nearest the axis, same window) | 6.11 ± 1.29 | 5.46 ± 2.38 | within noise |
| drum: static-bed tilt (the fill phase differs) | +1.8° | +2.6° | |
| **statics column** (1728 grains, e = 0.5, 800 steps): z95 / mean\|vz\| / max overlap | 7.61 / 0.0042 / 5e-4 | 7.63 / 0.0009 / 3e-4 | comes to rest faster |
| **statics pour-collapse** (1500 steps): z95 / mean\|vz\| / zmin | 7.66 / 0.0007 / 0.382 | 7.68 / 0.0005 / 0.431 | |
| **Enskog cooling**, φ = 0.35, e = 0.8, OMP 1, measured slope / Enskog: g = 0 (one-shot path) | 1.8044 | 1.8044 | bitwise identical |
| cooling, periodic free fall g = 0.01 (PGS path, v_rest = 4e-4) | 1.7827 | 1.8135 | +1.7 % |
| cooling, periodic free fall g = 10 (PGS path, v_rest = 0.4) | 1.8781 | 1.9166 | +2.0 % |
| cooling, T(t = 1)/T₀ at g = 0.01 | 0.3093 | 0.3018 | −2.4 % |

The same cooling gas at OMP 2 scatters by about ±1.3 % in the slope from thread order alone.
The +1.7–2.0 % above is at OMP 1, where each run is deterministic, but it is a small effect.

The Dosta harness `~/Codes/dem-bench/peclet/case{1_silo,3_impact}.py` still uses the pre-1.0 API.
These runs used scratch copies ported to 1.0: `initialize_shape`, 3-sequence `set_gravity`,
`set_dt` + `step()`, and a `--rest-target` flag. The note expected the 25k rebound to move by less
than 2 %. It moved by −20 % on the rebound displacement; the depth of penetration is unchanged.

## 4. Reading

- **Moreau removes the energy creation.** Newton creates up to +13 % (harness) or +30 % (model
  scene) of the kinetic energy in one step of a dense e = 1 cluster, and +3 % / +15 % at e = 0.9.
  Moreau is dissipative or conservative in every case, except the resting-threshold channel,
  which is ≤ 1.3e-4 at the harness's coarse dt.
- **Where Moreau changes results:**
  - an impact into a loose bed returns less rebound (−20 %);
  - a pour or collapse settles slightly faster;
  - a dense granular gas cools about 2 % faster;
  - silo discharge moves by about 1 %.
- **Unchanged bit for bit:** single contacts and the g = 0 one-shot.

## Reproduce

- **Harness:**
  `tests/kokkos_mpi/test_momentum_mpi cluster_pgs_e --e=<e> --rest-target=<newton|moreau>`, with
  `[--steps=10] [--dt=...]` (OMP 1).
- **pytest scene:** `tests/python/test_restitution_target.py::one_step_ke_ratio`.
- **G-C2 hashes:** a pytest plugin replaces `peclet.dem.Simulation` with a subclass that sets the
  target in `__init__` and hashes every `get_{positions,velocities,angular_velocities,quaternions}`
  result per test.
- **Cooling:** the scene of `test_colored_gs.run_cooling`, plus `set_gravity((0, 0, g))`,
  `set_stabilization('off')` and `set_sleeping(False)` when g ≠ 0.
- **Drum:** `test_rotating_drum.build` / `settle_and_spin`, with the tilt averaged every 50 steps
  over the last 1000.
