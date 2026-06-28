# Velocity Solver Algorithm

> **Status note.** An earlier version of this file described the velocity solver against the retired
> CUDA sources (`src/cuda/solver_velocity.cu` / `src/cuda/integration.cu`) and concluded that its
> "unweighted summation" caused micro-jitter / over-correction. That conclusion **predates the
> validated engine** and is no longer accurate: with the current Kokkos solver the packing engine
> reaches a genuine random close packing (φ ≈ 0.64, isostatic Z ≈ 6, max overlap ~0.3 %) and is stable.
> For the full picture see:
> - [`solver_details.md`](solver_details.md) — the current two-pass (velocity → position) pipeline.
> - [`packing_investigation.md`](packing_investigation.md) — the authoritative narrative + validation.
>
> This file is kept as a focused summary of the **velocity solve** only.

The velocity solve is the engine's **primary contact-dynamics pass** (restitution, friction, growth
separation). It runs *before* the position is advanced; the subsequent position solve only removes
residual overlap. It is header-only Kokkos — `src/solver_velocity.hpp` and `src/solver_friction.hpp`,
orchestrated by `demStep` in `src/sim.hpp`. Velocity-delta accumulation/flush lives in
`src/integration.hpp`.

## Pipeline (per step)

```text
[once, if friction] computePlaneLoadKokkos          # wall normal load
for it in [0, velocityIterations):                  # default ~20
    [if friction] accumulateNormalImpulseKokkos     # body-body force-chain normal load
    solveVelocityKokkos                             # normal restitution impulse (per manifold)
    applyVelocityDeltasKokkos                       # fold deltas into velPred/angVelPred, clear
[if friction]
    countFrictionContactsKokkos                     # per-body active-contact count
    solveContactFrictionKokkos                      # Coulomb-clamped, count-averaged tangential sweep
    applyVelocityDeltasKokkos
```

## Step A — normal impulse (`solveVelocityKokkos`)
One thread per **manifold** (a pair's aggregated contact points), not per raw contact. It reads the
predicted velocities (`velPred`, `angVelPred`), computes the relative normal velocity along the
aggregate manifold normal $\mathbf{N}_{sum}$ — plus a **growth-separation** term so inflating particles
push apart — skips separating contacts, and de-duplicates periodic copies via the
$\text{realA} > \text{realB}$ guard.

The impulse with restitution coefficient $e$ is
$$ \lambda = \frac{-e\,v_n - v_n}{w_{total}}, \qquad w_{total} = w_A + w_B, $$
where $w = |\mathbf{N}_{sum}|^2 m^{-1} + \boldsymbol\tau^{T}\mathbf{I}_{world}^{-1}\boldsymbol\tau$. The
linear and angular deltas are scattered onto both bodies' **real** indices with atomic adds.

## Step B — apply deltas (`applyVelocityDeltasKokkos`)
Runs over particles: adds the accumulated `deltaVel`/`deltaAngVel` onto `velPred`/`angVelPred` and
clears the buffers for the next iteration.

## Averaging
The **normal** restitution solve operates one-thread-per-manifold and folds the accumulated deltas
directly (a body in several manifolds receives the sum of their impulses). The **friction** sweep is
Jacobi **count-averaged**: `countFrictionContactsKokkos` records each body's active-contact count and
`solveContactFrictionKokkos` divides the tangential impulse by the larger of the two bodies' counts,
which is what keeps a body squeezed between many contacts from over-correcting. Combined with the
restitution-driven dynamics and the optional Berendsen thermostat, the scheme settles to a stable RCP
rather than oscillating (see `packing_investigation.md`).
