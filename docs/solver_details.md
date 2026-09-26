# Solver implementation details

What `peclet.dem` actually computes, and where each piece lives. The engine is **header-only
Kokkos + ArborX** under `src/` (compiled as C++; the Kokkos launch compiler routes device code
through `nvcc`/`hipcc` — there are no `.cu` files, the CUDA sources were retired in 2026-06). The
Python module is `peclet.dem`; drive it from Python, there is no C++ main.

`peclet.dem` ships **two contact engines over the same particle SoA** (`src/particles.hpp`):

| engine | entry point | driver | contact model | time step |
|---|---|---|---|---|
| **impulse / XPBD** (default) | `Simulation.step(n)`, `relax(n)`, `step_mpi(n)` | `demSolveContacts` (`src/solve_driver.hpp`) | velocity-level impulses + a position-level non-penetration projection; contacts are hard constraints | set by the collision scale, not by a stiffness |
| **force-based** | `Simulation.step_hertz(substeps)`, `step_hertz_mpi(substeps)` | `demStepForce` + `HertzMindlinLaw` (`src/solve_driver_force.hpp`, `src/solver_hertz.hpp`) | explicit soft-sphere Hertz–Mindlin (Dosta 2024 / LIGGGHTS formulas) | Rayleigh-time limited |

Both engines run the **same sequence single-GPU and distributed**: the MPI form is the same driver
with a different `Hooks` policy (`SoloSolveHooks` vs `MpiSolveHooks` / `SoloForceHooks` vs
`MpiForceHooks`), so there are no drifting copies. See [mpi.md](mpi.md).

Everything below assumes `set_dt(dt)` has been called — every stepper raises otherwise, there is no
default time step.

---

## 1. The impulse step (`demStep`, `src/step_solve.hpp`)

```text
growth ramp                    updateGrowthScalesKokkos           (growth_rate > 0)
predict                        predictVelocityKokkos              gravity + gyroscopic + ext. force/torque
freeze sleepers                freezeAsleepKokkos                 (sleeping, |g| > 0, no ext. drive)
ghosts                         generateGhostsKokkos               periodic band = 2 R_max + margin
broad phase                    findCollisionsArborX / ...Verlet   ArborX BVH over real + ghost
narrow phase                   detectContactsKokkos / detectWallSdfKokkos
manifolds                      reduceContactsToManifoldsKokkos    per-pair aggregate
wake / sleep masks             wakeDisturbedKokkos, ...           (sleeping)
SOLVE                          demSolveContacts                   §3-§6, shared with step_mpi
commit                         finalCommitKokkos                  wrap + write pos/quat
sleep detection                updateSleepKokkos                  (sleeping)
thermostat                     applyThermostatKokkos              (thermostat_tau > 0)
```

`relax(n)` runs the same pipeline at `dt = 0` and restores the stored `dt` afterwards — overlap
removal only, no gravity, no velocity update, and it is the one stepper that does **not** require
`set_dt`. It is what `step(0.0)` used to be before 1.0.0.

### Prediction — `src/integration.hpp` (`predictVelocityKokkos`)

From state $\mathbf{x}_n, \mathbf{v}_n$ and step $\Delta t$:

1. $\mathbf{v}_{pred} = \mathbf{v}_n + (\mathbf{g} + m^{-1}\mathbf{F}_{ext})\,\Delta t$
2. angular prediction including the gyroscopic term (Euler's equations in the body frame,
   $\dot{\boldsymbol\omega}_b = -\mathbf{I}^{-1}(\boldsymbol\omega_b \times \mathbf{I}\boldsymbol\omega_b)$),
   rotated back to world
3. $\mathbf{x}_{pred} = \mathbf{x}_n + \mathbf{v}_{pred}\,\Delta t$, with
   $\mathbf{q}_{pred} = \mathbf{q}_n$ (the quaternion is integrated later, from the *solved*
   angular velocity)
4. all delta buffers (`deltaPos`, `deltaQuat`, `deltaVel`, `deltaAngVel`) and the per-body
   constraint counts are cleared

Gravity defaults to **zero** and there is no implicit floor — `set_gravity((0, 0, -9.81))` and
`add_plane(point, normal)` (or an SDF wall) are the caller's job.

### Periodic ghosts — `src/periodicity.hpp` (`generateGhostsKokkos`)

Real particles within `ghostBand = 2 * maxOwnedRadius(P) + margin` (twice the largest **effective**
radius, growth included — not `global_scale` — plus the narrow-phase margin) of a periodic face are
copied, shifted by the box period, carrying the full state including the **predicted** position;
`realIndices` maps each ghost back to its owner so impulses land on the real body. Ghosts are not
integrated, and a ghost's position correction is discarded at commit — which is why the band must
reach the FARTHER partner of every reported pair (`a + b < rA + rB + margin`): with both partners
imaged a wrap pair is two twin manifolds and each body carries half the overlap, exactly like a
pair inside the box. (Until 2026-09-10 the band was one radius: every wrap pair was still detected,
but one with `max(a, b) > R_max` was resolved one-sidedly.) In the distributed step this layer is a
cross-rank gather with the band `rcut + skin` instead.

---

## 2. Collision detection

**Broad phase** — `src/broadphase_arborx.hpp` (`findCollisionsArborX`). An `ArborX` bounding-volume
hierarchy over *every* particle (real + ghost) with half-width $r_i + \text{margin}$
($\text{margin} = 0.1\,r_{max}$); one intersection query per **real** particle emits the
overlapping candidate pairs $(i,j)$, $i<j$, into a buffer guarded by an atomic counter.
`findCollisionsGrow` (`src/solve_driver.hpp`) detects a buffer overflow, grows it 1.5× and re-runs,
so no candidate is silently dropped and the narrow phase can never read past the end.

`set_verlet_skin(f)` (default `0.0`, off; single-GPU, non-periodic only) caches the candidate list
built at $\text{margin} + f\,r_{max}$ and reuses it while nothing has moved more than half the
skin — a **superset** of the true candidates, so the narrow phase yields identical contacts (in a
different order). It composes with sleeping: a frozen bed never rebuilds.

**Narrow phase** — `src/narrowphase.hpp` (`detectContactsKokkos`, `detectBoundaryKokkos`,
`detectWallSdfKokkos`). Candidate pairs and boundary planes/SDF walls are verified against the
shape's surface point-shell and its signed distance field: overlap
$d_{eff} = \mathrm{SDF}(\mathbf{x}) - r$, contact if $d_{eff} < \text{margin}$, normal from the SDF
gradient (central difference). Contacts are then reduced to per-pair **manifolds**
(`src/contact_preprocessing.hpp`, `reduceContactsToManifoldsKokkos`) — aggregate normal, torque
arm, lever arm — which the velocity phase consumes; `contactSlot` keeps the contact→manifold map
the friction bound reads through.

**Detection is decoupled from the solve.** Both phases run **once** per substep, on the
**predicted** state. The solvers iterate on that fixed contact/manifold list: they do not detect new
contacts, but they *do* re-evaluate geometry (distance, normal, lever arms) against the updated
predicted state every iteration.

---

## 3. The shared solve driver (`demSolveContacts`, `src/solve_driver.hpp`)

Two gates decide which of two regimes the driver runs, and both are read off the *simulation*, not
an environment variable:

- `usePersist = |g| > 0` — with gravity off (growth packing, homogeneous cooling) the driver runs
  the original one-shot colored-GS path, bit-identical to its pre-PGS behaviour.
- `usePGS = usePersist && velocity_solver == 'gauss_seidel'` (the default; the Jacobi A/B lives on
  `sim.diagnostics.set_velocity_solver`) — the full warm-started PGS statics path.

`set_solver_iterations(pos, vel)` caps both loops. **`velocityIterations` defaults to 0** — no
velocity solve, hence no restitution — while `positionIterations` defaults to 10.

**Every per-contact update is one of two forms** (`docs/contact_solve_framework.md` §0–§1): a
**projection-form** update whose target is fixed for the phase (PGS normal, cone, Poisson release,
the overlap projection, every stabilization mode, multilevel, the mass-split `'jacobi'`
diagnostic), or the **event-form** `g = 0` one-shot, whose target is the *current* iterate and
which is applied only while the pair is still approaching. A body touched from more than one place
during a substep — several ranks under MPI, or more than 32 contact edges in one colouring even at
np 1 — is represented by copies: a projection-form phase mass-splits them (`m/k`, `I/k`, mean
reconciliation, conservative at every iterate), while the one-shot instead holds each shared body
exclusively on one rank per sync interval, so the run stays a legal serial Gauss–Seidel order. See
[mpi.md](mpi.md#bodies-updated-in-several-places) for the distributed mechanics and the per-phase
policy table.

---

## 4. Velocity phase — contact dynamics

*Files: `src/solver_velocity.hpp`, `src/solver_friction.hpp`, `src/solver_multilevel.hpp`.*

The manifold graph is **graph-colored once** per substep (topology only, reused across sweeps) and
normal restitution runs as **colored Gauss–Seidel**: colour classes are body-disjoint, so a sweep is
a true sequential projection with no count-averaging softening. `set_incremental_coloring(True)`
(default, single-GPU PGS path) carries surviving pairs' colours across substeps and re-arbitrates
only the new manifolds, forcing a full recolour on the first substep and whenever the colour count
creeps past 1.3× the last full one. **This changes results** — the colouring fixes the sweep order.

Per manifold the sweep computes the relative normal velocity along the aggregate normal
$\mathbf{N}_{sum}$, including a **growth-separation** term
$\mathbf{v}_{growth} = (\mathbf{r}_A - \mathbf{r}_B)\dot g$ so inflating particles push apart;
separating contacts are skipped and periodic duplicates de-duplicated by the
$\text{realA} > \text{realB}$ guard. With effective inverse mass
$w = |\mathbf{N}_{sum}|^2 m^{-1} + \boldsymbol\tau^{T}\mathbf{I}_{world}^{-1}\boldsymbol\tau$ per
body and $w_{total} = w_A + w_B$, the impulse is $\lambda = (-e\,v_n - v_n)/w_{total}$, scattered
atomically as $\Delta\mathbf{v} = \lambda\,\mathbf{N}_{sum}m^{-1}$ and the matching angular delta
through $\mathbf{I}_{world}^{-1}$. Its target depends on the *current* iterate (event form): under
MPI a rank-shared body is held exclusively by one rank per sync interval (a `gate` per manifold,
`docs/contact_solve_framework.md` §1.4) rather than mass-split, so the distributed run stays a
legal serial Gauss–Seidel order and every firing is an exact, non-increasing-energy binary
collision — mass-averaging this update instead compounds restitution (measured 14–18 % kinetic
energy lost per substep on a dense cluster) and is rejected.

On top of that, with gravity on (the warm-started PGS statics path, projection form throughout —
mass-split M under MPI):

- **Persistent contacts.** A pair already in contact last substep is *loaded*, not impacting, so it
  gets $e = 0$ (the impulse still cancels the approach — pure inelastic support). A pile's static
  weight then travels through impulse chains and a settling column actually cools; material/wall
  restitution stays reserved for newly formed contacts. Pair keys are built from `realIndices`
  single-rank and from **global ids** under MPI, so the ledger survives halo rebuilds and ownership
  migration.
- **Warm start.** Each manifold's converged push impulse from the previous substep is gathered by
  pair key and applied up front, re-establishing a static force network in roughly one sweep.
- **Restitution model.** `set_restitution_model('newton')` (default) applies the per-substep Newton
  rule. `'poisson'` is event-level: a per-pair compression budget is banked and released as a
  budget-capped separation-velocity target during unloading
  (`updateRestitutionBankKokkos`), with an orphan-transfer pass settling dead pairs' remaining
  budget onto their endpoint bodies. Read the instruments with
  `sim.diagnostics.rest_bank_stats()` / `rest_orphan_stats()`.
- **Resting threshold.** Below roughly the speed one substep of free fall gains
  ($2\,\Delta t\,|g|$) a contact is treated as RESTING and bounces with $e = 0$ — the dense-pile
  energy-bomb guard.
- **Friction.** In the PGS path the Coulomb cone is inside the sweep, bounded by
  $\mu\,($velocity-impulse channel $+$ position-channel load$)$; the position-channel carry
  (`commitPosImpulseKokkos`, end of the step) is what keeps stick from leaking in a jostled bed.
  Outside it (no gravity, or the Jacobi A/B) the legacy four-kernel cluster of
  `src/solver_friction.hpp` runs instead: `computePlaneLoadKokkos` (one-shot wall normal load),
  `accumulateNormalImpulseKokkos` (body-body force-chain load), `countFrictionContactsKokkos` and a
  single count-averaged, Coulomb-clamped `solveContactFrictionKokkos` sweep — a raw sync,
  unaffected by the copies framework, that runs after the phase's own reconciliation on settled
  state. A frictional **wall** drives friction even when the body-body material is frictionless;
  the default body-body material is frictionless. Every per-contact velocity computation in this
  legacy pass acts at the contact's single midpoint (`rA_mid`/`rB_mid`, the same arms the manifold
  path already used), not the two surface points: applying $\pm J_t$ at two different points
  created a spurious couple `dist · n × J_t` (measured ratio 1.000 before the fix), which broke
  angular momentum and changes every `g = 0` run with body–body friction. It also rotates the
  inverse inertia to the world frame ($R\,\mathbf{I}^{-1}\,R^T$) before applying a torque, instead
  of using the body-frame tensor directly — the earlier form changed a hand-worked ring–ring
  impulse's angular momentum by 32 % of the applied torque, against 4.5e-11 after the fix; an
  isotropy guard skips the rotation for isotropic inertia, so spheres stay byte-identical.
- **Adaptive stop.** `maxApproach` records the largest *applied* correction; the loop ends when it
  falls below a tolerance well under the resting floor (meaningful increments are $\sim g\Delta t$,
  one chain link per sweep — stopping at the resting floor starves deep chains, measured: a
  113-layer pile plateaued at $v_z \approx -5$). `positionIterations`/`velocityIterations` are the
  caps. Distributed, the residual is `MPI_Allreduce(MAX)`-ed so every rank breaks together.
- **Colouring never falls back.** The graph coloring never forces a colour past 64: a vertex with
  more than 32 active edges is split into local mass-split copies (round-robin over its edges) and
  the phase is recoloured, which is guaranteed to place every edge (the palette lemma,
  `docs/contact_solve_framework.md` §1.6). A manifold the colouring still cannot place after that
  is an invariant violation and throws — the old count-averaged-Jacobi fallback for
  interpenetration degree > 62 is gone.

### Stabilization pass

The main sweeps are fully **momentum-conserving** (Guendelman staged solve, side flags zero), so
impact, discharge and shear see correct physics. Only if they leave an unconverged residual — a
collapsing column needs about one sweep per layer to carry its weight to the floor, which is
unaffordable — does the stabilization pass run, selected by `set_stabilization(mode)`. Every mode
is a projection-form update (mass-split M under MPI):

| mode | what runs |
|---|---|
| `'off'` | no pass — pure symmetric PGS: exact ballistic response, but a deep column mid-collapse cannot be arrested within the iteration budget |
| `'onesided'` | **default** — grounded held-lower-side impulses: arrests any collapse, at twice the main iteration budget, but is a momentum sink |
| `'multilevel'` | the GraphMG pass (see below): momentum-conserving transport acceleration |

Two further **measurement** modes live on the developer tier, `sim.diagnostics.set_stabilization`:
`'escalate'` (plain symmetric colored sweeps up to 256 — provably correct, and the sweep count it
needs bounds what the other passes must deliver) and `'ordered'` (level-ordered symmetric sweeps
from a height-from-floor BFS; measured insufficient for deep columns, kept for A/B).

**Multilevel (GraphMG)** — `src/solver_multilevel.hpp`. Greedy pairwise aggregation over the
*quasi-static* contact graph builds super-bodies; fine manifolds crossing aggregate boundaries are
re-solved with the **aggregate** masses and inertias, so a supported chain's genuinely huge inertia
plays the role the held lower side faked and one coarse impulse drains a whole column — while every
impulse stays symmetric. The super-bodies are **rigid 6-DOF aggregates**
(`docs/contact_physics_followups.md` §2): the coarse state $(V_g, \Omega_g)$ is the mass/inertia
projection of the fine state onto rigid motions ($V_g = \sum \mu v / M_g$,
$\Omega_g = I_g^{-1} \sum [\mu\, d \times v + J \omega]$, with $d$ a member's offset from the
group's centre of mass and $I_g = \sum [J + \mu(|d|^2 \mathbb{1} - d d^T)]$, built in double once
per hierarchy); a crossing row carries the angular arms $T_A = \tau_A + d_A \times N$,
$T_B = \tau_B - d_B \times N$, so each coarse impulse is the fine manifold's own, applied at its
contact points; and the prolongation is the rigid motion $\delta v = \delta V + \delta\Omega \times d$,
$\delta\omega = \delta\Omega$. Linear and angular momentum are therefore exact (to float) and the
kinetic energy never rises in a coarse step. Translation-only aggregates (before 2026-09-26) are the
limit $I_g^{-1} \to 0$; they lost angular momentum $(X_A - X_B) \times J$ per crossing impulse
(`hub_ml` dLvel 3.0e-3). Named consequences: frictionless sphere spins couple to aggregate rotation
in this mode, and aggregation never crosses a periodic wrap (a periodic image never merges;
crossing manifolds may still wrap). Ballistic pairs
($|v_{n0}| > $ the quasi-static threshold) never aggregate, so an impactor keeps its fine-level
physics and its rebound; coarse $\lambda$ shares the fine accumulator so the ledger stays consistent
for the next warm start and the Coulomb bound. Its stop criterion is the **quasi-static** residual
(fine corrections on contacts with $|v_{n0}| \le 4 v_{rest}$, plus every coarse correction): the
full residual is dominated by ballistic contacts in flowing scenes, and gating on it burns the whole
extra budget every substep (an over-convergence brake on discharge, measured −7 %).

A mass-split hub's `s` copies are folded to one value before the coarse cycle and re-seeded after,
so between folds they move together; the coarse cycle must then treat them as **one** vertex
carrying their *combined* mass `a·m/k` (the true mass `m` at np 1, where `a = k = s`), never each
copy's own split mass `m/k` — else the coarse impulse under-counts the body's inertia by a factor
of `s` and momentum leaks at every cycle (`docs/contact_solve_framework.md` §13.2).

---

## 5. Re-integration

*File: `src/integration.hpp` (`applyVelocityAndPredictPositionKokkos`).*

The solved velocity is **persisted as the new state** and used to re-predict the position:

1. $\mathbf{v}_{n+1} = \mathbf{v}_{pred}^{solved}$ (into both `vel` and `velPred`)
2. $\mathbf{x}_{pred} = \mathbf{x}_n + \tfrac12(\mathbf{v}_n + \mathbf{v}_{n+1})\Delta t$
3. the quaternion is advanced from the *solved* angular velocity and renormalised

So the contact-resolved velocity does feed the position state.

---

## 6. Position phase — overlap removal

*File: `src/solver_position.hpp`.*

Pure geometric projection: no dissipation, no velocity impulses. A contact pair with more than one
contact point (rings, hollow cylinders) is coloured and swept as one **unit** — every point of the
pair, in original-index order, inside one colouring edge and one work item — instead of per point;
a single-point unit is today's exact edge and arithmetic, so spheres and analytic walls stay
bitwise, while a ring bed's colour count drops from 79–613 per-point edges to about 5–20 pairs
(`docs/contact_solve_framework.md` §4.3). Beyond that the **contact** graph (of units) is coloured
once, with the same incremental-colouring policy as the velocity phase, and
`solvePositionColoredGSKokkos` sweeps it as colored Gauss–Seidel. For a contact between $A$ and $B$
(or a static wall) the linearised non-penetration constraint on the surface points is

$$ C(\mathbf{x}) = (\mathbf{p}_A^{surf} - \mathbf{p}_B^{surf})\cdot\mathbf{n} \ge 0 $$

evaluated with the lever arms and normal delta-rotated from the static frame to the predicted one
($C \ge 0$ ⇒ inactive, skipped). Contacts are rigid (compliance 0), so

$$ \Delta\lambda = \frac{-C(\mathbf{x}_{pred})}{\tilde w}, \qquad
   \tilde w = m_A^{-1} + m_B^{-1} $$

applied as $\Delta\mathbf{x} = \mathbf{n}\Delta\lambda\,m^{-1}$ — **translation only**: the phase
never rotates a body (overlap removal stays decoupled from the velocity update), so the diagonal
is the translational effective mass of the solve views ($m^{-1}$ is the mass-split $k\,m^{-1}$ of a
copy; a wall side contributes 0). Before WO-B1 (2026-09-26,
`docs/contact_physics_followups.md` §3.3) $\tilde w$ also carried a rotational term
$(\mathbf{r}\times\mathbf{n})^{T}\mathbf{I}^{-1}(\mathbf{r}\times\mathbf{n})$ for a rotation that was
never applied, built from a world-frame arm and body-frame inertia (not frame-indifferent), which
overstated the stop metric by $\tilde w/(m_A^{-1}+m_B^{-1})$ (~2.4, up to ~5.6, on tube rows). The
fixed point does not depend on $\tilde w$ — it is a per-row relaxation — so converged positions are
unchanged; finite-iteration results of non-spherical bodies and **spinning** spheres (whose
delta-rotated arm gives $|\mathbf{r}\times\mathbf{n}| \sim R\,\omega\,dt$) changed with it, while a
non-spinning sphere's term was below half an ulp and it stays bitwise.

The loop stops once the deepest penetration falls below `posTol = 1e-4 * base_radius *
global_scale` (~0.01 % of a radius), capped at `positionIterations`; distributed, that residual is
Allreduce-MAXed too. Selecting the Jacobi A/B (`set_velocity_solver('jacobi')`) runs the whole
phase through the older **Jacobi** form of the same projection (`solvePositionKokkos` +
`applyUpdatesKokkos`), but this is now **mass-split**, not count-averaged: every contact is its own
copy, solved against `m/count`, and the true-mass deltas are summed with factor 1
(`docs/contact_solve_framework.md` §3.1). The colouring's own count-averaged fallback for
uncoloured leftovers is gone; a vertex above 32 active edges gets local mass-split copies and is
recoloured instead (above), and an edge still uncolourable after that throws.

**Accumulated, retractable projection (WO-12, 2026-09-26; USER decision).** Each contact keeps
its net position push $\Lambda \ge 0$ (`posLambdaContact`, zeroed per substep) and every sweep
projects it, $\Lambda' = \max(0,\ \Lambda - \omega_{pos} C / \tilde w)$, applying the change
$d = \Lambda' - \Lambda$ — which may be negative, so an overshoot is pulled back. This is projected
SOR on the accumulated multiplier (the velocity phase's PGS form, in positions): over-relaxation is
legitimate, $\omega_{pos} = 1.5$ on every contact (np 1 included), and the fixed point is the unique
least-displacement solution, so the converged positions agree across rank counts (measured:
$\max |x_{np N} - x_{np 1}| \le 4.7\times 10^{-5} R$ at np 2/4/8, gated at $10^{-4} R$; the
non-accumulated POCS it replaced differed by $10^{-2} R$). The stop is the position change
$\max |d|\,\tilde w$ below $10^{-4} R$ — the true relative position change of the pair —
(retractions included); `max_overlap` stays the largest
violation seen. Dense clusters converge in 2.9× fewer iterations than the old $\omega = 1$
projection (34 vs 97). The superseded form — applied only while $C < 0$, never retracted, hence
never over-relaxed ($\omega = 1.5$ on it left a permanent gap of $(\omega_{eff}-1)|C|$, §13.1) —
survives only as the kernel unit test's reference. The velocity phase's PGS
normal keeps its over-relaxation hook (its multiplier is accumulated and clamped, so an overshoot
there is legitimately retractable).

> `sim.max_overlap` is this loop's **last-iteration residual**, so it under-reports the committed
> overlap. `sim.compute_overlaps()` re-runs broad + narrow phase on the committed state and is the
> honest measurement.

---

## 7. Commit, sleeping, thermostat

`finalCommitKokkos` (`src/integration.hpp`) wraps the resolved predicted position into the domain on
periodic axes and commits it with the integrated orientation. The velocity is **not** re-derived
from the position change — it was already set in §4/§5.

**Island sleeping** (`src/sleeping.hpp`, `set_sleeping(...)`, default **on**; requires gravity and
no external force/torque, inert under MPI). A grounded real body whose motion stayed below
`threshold_scale ×` the resting floor for `consecutive` substeps is put to sleep (velocity zeroed,
integration skipped). A manifold with both endpoints asleep (a static wall counts as asleep) drops
out of the colouring, the sweeps and the multilevel hierarchy, so a fully settled bed collapses to
the broad/narrow-phase floor. A sleeper is given effective inverse mass `immovable_frac` of its own
(0.01 by default, 0 = exactly immovable) for the solve, which keeps awake–asleep contacts correct
without waking anything; waking is then a physics decision — a fast approaching neighbour, a change
in the contact set, a moving wall, and optionally `wake_on_lost_contact`.

**Berendsen thermostat** (`applyThermostatKokkos`): with `thermostat_tau > 0` the linear and
angular velocities are rescaled at the end of the step (the packing-annealing protocol uses it).

---

## 8. Execution policies — what changes results and what does not

| switch | tier | effect |
|---|---|---|
| `set_cuda_graphs(enabled)` | `diagnostics` | CUDA-graph capture + replay of each solver iteration loop; the step is host-submission-bound (measured ~3300 launches / ~11 ms per step at 25 k), replay collapses that into one. **Bit-identical**, inert off CUDA and on the distributed step. |
| `set_fused_sweeps('auto'\|'on'\|'off')` | `diagnostics` | a whole colour sweep — and where eligible the whole adaptive loop — as ONE kernel behind software grid barriers (`src/solver_fused.hpp`). `'auto'` uses it exactly where graph replay is unavailable. **Bit-identical**. |
| `set_velocity_solver('gauss_seidel'\|'jacobi')` | `diagnostics` | selects the colored-GS or the mass-split Jacobi diagnostic (velocity and position both; conservative, more dissipative than Gauss–Seidel at a fixed iteration count). **Changes results.** |
| `set_incremental_coloring(enabled)` | `Simulation` | warm-started colouring; **changes results** (sweep order), which is why it is public. |
| `set_verlet_skin(frac)` | `Simulation` | broadphase cache; identical contact *set*, different order ⇒ run-to-run scatter at float precision. |
| `set_sleeping(...)`, `set_stabilization(...)`, `set_restitution_model(...)` | `Simulation` | physics/algorithm choices; each documented above. |

No environment variable changes what `peclet.dem` computes (QUALITY_PLAN package E) — the single
remaining `getenv` in `src/` is `PECLET_DEM_HERTZ_PROFILE`, a timing print.

---

## 9. The force-based engine (Hertz–Mindlin)

*Files: `src/solve_driver_force.hpp` (driver), `src/solver_hertz.hpp` (force law).*

`step_hertz(substeps)` runs classical explicit soft-sphere DEM: **viscoelastic Hertz** normal force
with restitution-matched damping ($\beta_d = \ln e/\sqrt{\ln^2 e + \pi^2}$) plus a **no-slip Mindlin
tangential spring with history**, Coulomb-clamped — the LIGGGHTS
`gran model hertz tangential history` formulas, i.e. the Dosta et al. (2024) benchmark model.
Spheres only, non-periodic domains, SDF walls (no planes). Set the material with
`set_hertz_material(mat, youngs, poisson)`.

The driver owns the engine-generic machinery — a cached **Verlet pair list** whose slots carry the
per-pair shear history $\boldsymbol\xi$ (rebuilt when any particle moved skin/2, history carried
across a rebuild by pair key), the wall candidate list, symplectic-Euler integration, displacement
tracking and profiling — while the force law is a **policy**: `HertzMindlinLaw` is the first, and
other explicit laws (linear spring-dashpot, cohesive/JKR, bonded) slot in beside it reusing the same
driver, halo choreography and tests. History keys are global ids, so they survive halo rebuilds and
ownership migration; the migration pack carries each particle's slice.

---

## 10. Distributed form

`step_mpi` / `step_hertz_mpi` run the **same drivers** with the MPI hooks. Cross-rank Hertz pair
lists are **canonically oriented, lower global id first**: the Mindlin spring's tangential
displacement $\boldsymbol\xi$ is the displacement of `pairs(idx, 0)` relative to `pairs(idx, 1)`
and changes sign with the order, so the two owners of a cross-rank pair must agree on it, or a
gid-keyed history carry through migration hands one rank the wrong sign (measured: dP 9.1e-4 with
friction, starting at the first drift migration, before the fix). At np 1, gid equals slot and the
broad phase already emits lower-first, so this is byte-identical there. XPBD (`step_mpi`): every
contact is solved by exactly one rank (the only owner that sees it, else the lower-gid body's
owner), the colouring and sweeps stay rank-local over that rank's owned contacts and owned + ghost
bodies, the partner half of every impulse lands in the ghost slot, and at every sync (every
`sync_every` iterations plus once after every phase) each ghost's change is reverse-accumulated onto
its owner before the owners re-publish — so linear momentum and the centre of mass are conserved to
round-off (`docs/mpi_momentum_conservation.md`). Each adaptive-stop residual is
`MPI_Allreduce(MAX)`-ed so all ranks break together. That is processor-block Gauss–Seidel: the same
physics as single-rank, not bit-exact (a contact across a rank face sees its far body as of the
last reconciliation). The explicit Hertz–Mindlin engine evaluates each cross-rank pair on both
owners from bit-identical state, which is conservative as it stands. Full detail, validation and
the periodicity/capacity rules: [mpi.md](mpi.md); scaling and profiling:
[multi_gpu_testing.md](multi_gpu_testing.md).

---

## Further reading

- [archive/packing_investigation.md](archive/packing_investigation.md) — the (dated) investigation
  behind today's packing protocol and the design principle *the velocity phase owns all dissipation
  and the full resting normal impulse; the position phase only removes geometric overlap*.
- [archive/velocity_solver_algorithm.md](archive/velocity_solver_algorithm.md) — the pre-PGS
  summary of the velocity solve, superseded by §4.
