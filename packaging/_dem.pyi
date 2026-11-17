"""
peclet.dem (Kokkos + ArborX): performance-portable XPBD granular dynamics
"""

from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


def finalize() -> None:
    """
    Release every live object and zero-copy array of this module, then Kokkos::finalize() (deterministic teardown; also run automatically at interpreter exit). Idempotent. After it, the module's solver objects and *_view arrays must not be used.
    """

execution_space: str

class Diagnostics:
    """
    Developer tier of a Simulation, reached as `sim.diagnostics`: instruments, ablations and execution-policy switches. Nothing here is needed to set up, run or read out a simulation; the setters that CHANGE RESULTS say so in their docstring.
    """

    def set_stabilization(self, mode: str) -> None:
        """
        The full stabilization mode set: the three production modes of Simulation.set_stabilization plus the two measurement modes -- 'escalate' (extra symmetric sweeps up to 256; diagnostic/fallback) and 'ordered' (level-ordered symmetric sweeps; known insufficient for deep columns).
        """

    def set_velocity_solver(self, name: str) -> None:
        """
        A/B switch of the collision solves: 'gauss_seidel' (default; colored Gauss-Seidel, correct multi-contact dissipation) or 'jacobi' (mass-split Jacobi: every contact is its own copy, solved against masses m/count, true-mass deltas summed -- conservative, and at a fixed iteration budget more dissipative than Gauss-Seidel). CHANGES RESULTS.
        """

    @property
    def velocity_solver(self) -> str:
        """'gauss_seidel' or 'jacobi'."""

    def set_cuda_graphs(self, enabled: bool) -> None:
        """
        CUDA-graph replay of the solver's iteration loops (default True): capture collapses each iteration's launch storm into one replay. Results are bit-identical either way; inert on non-CUDA backends and on the distributed step.
        """

    @property
    def cuda_graphs(self) -> bool:
        """Whether CUDA-graph replay of the solver loops is enabled."""

    def set_fused_sweeps(self, mode: str) -> None:
        """
        Fused color sweeps (CUDA): a whole sweep -- and where eligible the whole adaptive iteration loop -- as ONE kernel behind software grid barriers. 'auto' (default) uses them exactly where graph replay is unavailable (the distributed step, or set_cuda_graphs(False)); 'on'/'off' force. Bit-identical results either way.
        """

    @property
    def fused_sweeps(self) -> str:
        """Fused-sweep policy: 'auto', 'on' or 'off'."""

    def coloring_conflicts(self) -> tuple[int, int]:
        """
        (velocity, position) graph-coloring-invariant violations in the last substep; a valid coloring returns (0, 0). Test hook for the incremental warm-start path.
        """

    def coloring_leftovers(self) -> tuple[int, int]:
        """
        (velocity, position) edges -- manifolds, position units -- the last substep's colourings left uncoloured (single-rank). Non-zero means a colouring exhausted its arbitration rounds or palette.
        """

    def multilevel_coloring_conflicts(self) -> int:
        """
        Same-colour pairs at an aggregate of the last multilevel hierarchy, summed over its levels; a valid coarse colouring returns 0 (0 also when no hierarchy was built).
        """

    def split_stats(self) -> dict:
        """
        Dict of the last substep's body copies (docs/contact_solve_framework.md §4.4): hub copy slots per phase, light hubs (hub mass < 10x its partners' mean), bodies solved through more than one slot per phase (hubs; periodic images in the single-rank position phase), and (0 until implemented) unfired split contacts and drift migrations; ml_hub_aggregated: split velocity vertices in a multi-member level-1 multilevel group (max over the last step call's substeps); vel_iters_used / pos_iters_used: iterations of the last substep's main velocity and position loops (-1: a device-side loop, whose count is read back only with the C++ diagnostics switch debugIterationCounters); orphan_clamps: clamp hits of the distributed Poisson orphan balance over the last step_mpi call (must stay 0).
        """

    def rest_orphan_stats(self) -> tuple[float, float, int]:
        """
        Poisson-restitution instrument: (sum, max, n_bodies>0) of the per-body orphaned event budget (physical impulse units).
        """

    def rest_bank_stats(self) -> tuple[float, float, int]:
        """
        Poisson-restitution instrument: (sum, max, n_pairs>0) of the per-pair owed separation impulse committed last substep (physical impulse units).
        """

    def wall_sdf_at(self, wall_index: int, points: Annotated[NDArray[numpy.float32], dict(order='C')]) -> NDArray[numpy.float32]:
        """
        The wall's signed distance at world points: (M,3) in -> (M,) out, POSITIVE in the void where the grains live -- exactly what the narrow phase reads, so it is the honest check of a set_wall_transform placement and the way to draw a stirrer.
        """

    def profiling_info(self) -> dict:
        """
        Dict of the particle / broad-phase pair / manifold counts and the last-step max overlap (the same four values the Simulation properties carry).
        """

    @property
    def mpi_rebuilds(self) -> int:
        """
        Cumulative halo topology-rebuild count (Verlet-skin path); with mpi_gathers this is the ghost-reuse ratio.
        """

    @property
    def mpi_gathers(self) -> int:
        """Cumulative ghost gather() count across distributed steps."""

class Simulation:
    """
    One granular-dynamics simulation (XPBD impulse engine + explicit Hertz-Mindlin engine, the same particle set). Set up shapes -> domain -> positions -> materials, `set_dt`, then `step(n)`; read out with the `get_*` arrays. Developer instruments live on `.diagnostics`. Units: any consistent set chosen by the caller (SI or not); each docstring states the dimension of its quantities.
    """

    def __init__(self, capacity: int) -> None:
        """
        Allocate for up to `capacity` particles (owned + periodic ghost slots); set_positions refuses more.
        """

    @property
    def capacity(self) -> int:
        """The particle capacity this was built with."""

    @property
    def diagnostics(self) -> Diagnostics:
        """The developer tier (Diagnostics) of this simulation."""

    def initialize_shape(self, shape: str, radius: float, height: float = 0.0, thickness: float = 0.0) -> None:
        """
        Select the single particle shape and its dimensions: 'sphere' (radius), 'hollow_cylinder' (radius, height, thickness) or 'box' (half-extent = radius), all in length units. RESETS the shape registry to this one shape (use add_shape for a mixture) and records the unit-mass inverse inertia that set_positions applies to every particle -- so call it BEFORE set_positions.
        """

    def set_sdf_shape(self, grid: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), order='F')], origin: Sequence[float], spacing: Sequence[float], shell: Annotated[NDArray[numpy.float32], dict(order='C')], inv_inertia: Sequence[float], bounding_radius: float) -> None:
        """
        Import a general particle: `grid` is the body-frame signed distance on an (nx, ny, nz) array indexed [x, y, z] (negative inside), sampled at nodes origin + (x, y, z) * spacing; `shell` the (M,3) surface point shell (EMPTY = sample the field's own zero level set); `inv_inertia` the unit-mass principal diagonal inverse inertia; `bounding_radius` the canonical enclosing radius. `origin`, `spacing`, `grid` and `bounding_radius` are in length units, `inv_inertia` in 1/length^2. RESETS the registry to this one shape. See peclet.dem.build_particle for the SDF -> (grid, shell, inertia) helper.
        """

    def add_shape(self, shape: str, radius: float, height: float = 0.0, thickness: float = 0.0) -> int:
        """
        Append an analytic shape ('sphere' | 'hollow_cylinder' | 'box', dimensions -- radius, height, thickness in length units -- as initialize_shape) to the registry and return its index, for a simulation with a MIXTURE of shapes. initialize_shape stays the single-shape entry point (it RESETS the registry). Assign the returned index with set_shape_ids.
        """

    def add_sdf_shape(self, grid: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), order='F')], origin: Sequence[float], spacing: Sequence[float], shell: Annotated[NDArray[numpy.float32], dict(order='C')], inv_inertia: Sequence[float], bounding_radius: float) -> int:
        """
        Append a grid-SDF shape (arguments as set_sdf_shape: origin, spacing, grid and bounding_radius in length units, inv_inertia in 1/length^2) and return its index. set_sdf_shape stays the single-shape entry point (it RESETS the registry).
        """

    def add_scene_shape(self, node_ints: Annotated[NDArray[numpy.int32], dict(order='C')], node_reals: Annotated[NDArray[numpy.float32], dict(order='C')], root: int, shell: Annotated[NDArray[numpy.float32], dict(order='C')], inv_inertia: Sequence[float], bounding_radius: float) -> int:
        """
        Register a COMPOSED analytic particle shape from core's flat node encoding (the arrays peclet.geom.SceneBuilder.encode() returns; CSG of the full leaf vocabulary): the collision field is the exact tree, evaluated in canonical body space. shell: (M,3) surface probe points (bake the tree and run the shell path -- the point-shell model still needs probes). inv_inertia: unit-mass principal diagonal inverse inertia; bounding_radius: canonical enclosing radius. THE CANONICAL FRAME MUST BE THE PRINCIPAL INERTIA FRAME (SceneBuilder.principal_frame emits exactly that); a non-principal tree runs the diagonal-inertia rotational update on the wrong frame, silently. Returns the shape id for set_shape_ids.
        """

    def set_shape_ids(self, ids: Annotated[NDArray[numpy.int32], dict(order='C')]) -> None:
        """
        Per-particle shape index (one int per particle, each < num_shapes). Refreshes each particle's inverse inertia from its new shape. Call it AFTER set_positions: set_positions resets every particle to shape 0 (and sizes the particle set this call is checked against).
        """

    @property
    def num_shapes(self) -> int:
        """Number of registered shapes."""

    @overload
    def set_domain(self, min: Sequence[float], max: Sequence[float]) -> None:
        """
        Set the domain by (min, max) corner tuples (arbitrary origin; length units); keeps current periodicity.
        """

    @overload
    def set_domain(self, extent: Sequence[float], origin: Sequence[float] = [0.0, 0.0, 0.0], periodic: Sequence[bool] = [True, True, False]) -> None:
        """
        Set the domain the suite-canonical way: `extent` is the box SIZE (not the far corner), `origin` the lower corner (both in length units), `periodic` the per-axis flags. Equivalent to `set_domain(min=origin, max=origin+extent)` plus `set_periodic(*periodic)`. Either form resets the broad-phase skin to 0.1 x the global scale (as set_global_scale does).
        """

    def set_periodic(self, x: bool, y: bool, z: bool) -> None:
        """
        Set periodic boundaries per axis (x, y, z); read back with the `periodic` property (suite/docs/NAMING.md 1.4).
        """

    @property
    def origin(self) -> tuple[float, float, float]:
        """
        The domain's lower corner (x, y, z), in length units — read-only; set it with `set_domain`.
        """

    @property
    def extent(self) -> tuple[float, float, float]:
        """
        The domain's SIZE (Lx, Ly, Lz), in length units — read-only, and note it is a size and not the far corner, which is `origin + extent` (suite/docs/NAMING.md 1.1).
        """

    @property
    def periodic(self) -> tuple[bool, bool, bool]:
        """
        Per-axis periodicity (x, y, z) — read-only; set it with `set_periodic`.
        """

    def set_gravity(self, gravity: Sequence[float]) -> None:
        """
        Set the gravitational acceleration (gx, gy, gz), in length/time^2 units -- any sequence of three floats. The default is ZERO.
        """

    @property
    def gravity(self) -> tuple[float, float, float]:
        """The gravitational acceleration (gx, gy, gz), in length/time^2 units."""

    def set_thermostat(self, temperature: float, tau: float, kB: float = 1.0) -> None:
        """
        Enable a Berendsen-style velocity thermostat: target `temperature` (temperature units, >= 0), coupling time `tau` (time units; tau = 0 disables) and Boltzmann constant `kB` (energy/temperature units; the measured temperature is 2 x kinetic energy / (3N x kB)).
        """

    def set_solver_iterations(self, pos: int, vel: int) -> None:
        """
        Set the XPBD position- and velocity-solve iteration counts. `vel` defaults to 0 (no velocity solve, hence no restitution).
        """

    def set_hertz_material(self, mat: int, youngs: float, poisson: float) -> None:
        """
        Per-material Young's modulus and Poisson ratio for the soft-sphere Hertz-Mindlin engine (material ids as in set_material_ids).
        """

    def set_stabilization(self, mode: str) -> None:
        """
        Stabilization pass of the staged velocity solve: 'off' (pure symmetric PGS -- exact ballistic response, but a deep static column mid-collapse cannot be arrested within the iteration budget), 'onesided' (default: held-lower-side grounded impulses -- arrests any collapse but is a momentum sink), 'multilevel' (GraphMG contact-graph aggregation: coarse inelastic solves at super-body masses -- momentum-conserving transport acceleration).
        """

    @property
    def stabilization(self) -> str:
        """
        The stabilization mode: 'off', 'onesided', 'multilevel' (or a diagnostics mode).
        """

    def set_restitution_model(self, model: str) -> None:
        """
        Restitution model of the PGS velocity solve: 'newton' (default; per-substep restitution with Moreau's target, -e x the pre-solve normal velocity on every closed contact above the resting threshold, which cannot create kinetic energy for a uniform e) or 'poisson' (event-level: each pair banks its kinetic compression impulse and releases e x the bank as a budget-capped separation-velocity target during unloading -- restores the multi-substep-impact rebound per-substep restitution cannot return; it keeps Newton's per-substep targets, -e x the approach on approaching contacts only).
        """

    @property
    def restitution_model(self) -> str:
        """'newton' or 'poisson'."""

    def set_global_scale(self, scale: float) -> None:
        """
        Set a global length scale applied to all particles (effective radius = scale x global_scale x base radius). Resets the broad-phase skin, as set_domain does -- call it BEFORE set_domain if you rely on the skin.
        """

    @property
    def global_scale(self) -> float:
        """The global length scale."""

    def set_dt(self, dt: float) -> None:
        """
        Set the time step (> 0, time units) every stepper uses: step, step_hertz, step_mpi, step_hertz_mpi. There is no default -- a step before set_dt raises.
        """

    @property
    def dt(self) -> float:
        """The time step set by set_dt, in time units."""

    def set_material_params(self, restitution_normal: float, restitution_tangent: float = 0.0, friction: float = 0.0) -> None:
        """
        Set the BODY-BODY normal/tangential restitution and Coulomb friction coefficient. The default friction is ZERO, and add_analytic_wall / add_sdf_wall set the particle-WALL material only -- so a bed more than a few layers deep run with the defaults behaves like a liquid: it transmits full hydrostatic pressure to the container and the position solve squeezes grains through the boundary. That failure is silent and looks like a solver-convergence bug (raising the position iterations and halving dt do not move it). Set a non-zero friction for any deep bed.
        """

    def set_material_ids(self, ids: Sequence[int]) -> None:
        """
        Per-particle material ids (0..7). Pair (e, mu) values come from set_pair_material; without any set_pair_material call the global material applies everywhere.
        """

    def set_pair_material(self, a: int, b: int, restitution: float, friction: float) -> None:
        """
        Symmetric pair material (restitution, friction) for material ids (a, b). The first call seeds every pair from the current global material.
        """

    def set_wall_material_id(self, wid: int, mat: int) -> None:
        """
        Give an SDF wall a material id so particle-wall (e, mu) resolves via the pair table instead of the wall's binary material.
        """

    def add_analytic_wall(self, node_ints: Annotated[NDArray[numpy.int32], dict(order='C')], node_reals: Annotated[NDArray[numpy.float32], dict(order='C')], root: int, invert: bool, restitution: float = 0.0, friction: float = 0.0) -> int:
        """
        Add an ANALYTIC wall from a core shape tree in the flat node encoding (3 ints + 16 reals per node). Exact at every scale, with no voxel grid to replicate per rank. invert=False for a stirrer/obstacle (grains outside the solid); invert=True for a container, built from a SOLID body (a solid cylinder for a drum). The wall is positioned by the node TRANSFORM -- an identity transform sits at the origin. Returns the wall index.
        """

    def add_plane(self, point: Sequence[float], normal: Sequence[float]) -> None:
        """
        Add an infinite boundary wall plane through `point` (length units), with `normal` (dimensionless direction) pointing into the half-space the grains occupy (both 3-sequences). XPBD engine only; the Hertz engine takes SDF walls.
        """

    def add_sdf_wall(self, grid: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), order='F')], origin: Sequence[float], spacing: Sequence[float], restitution: float = 0.0, friction: float = 0.0) -> int:
        """
        Add a static world-space SDF wall/container (drum barrel, hopper, vibrating tray): `grid` is the signed distance (length units) on an (nx, ny, nz) array indexed [x, y, z] at world nodes origin + (x, y, z) * spacing -- POSITIVE in the void where grains live, NEGATIVE in the solid wall; `origin` and `spacing` are in length units; restitution / friction are the binary particle-wall material (dimensionless). Returns the wall index (for set_wall_velocity). See peclet.dem.build_wall_sdf for the SDF -> (grid, origin, spacing) helper.
        """

    def set_wall_velocity(self, wall_index: int, lin_vel: Sequence[float] = [0.0, 0.0, 0.0], ang_vel: Sequence[float] = [0.0, 0.0, 0.0], center: Sequence[float] = [0.0, 0.0, 0.0]) -> None:
        """
        Set a wall's rigid-body surface velocity v(x) = lin_vel + ang_vel × (x − center): lin_vel in length/time units, ang_vel in 1/time (radians per time), center in length units (felt by grains in contact even though the geometry is static). Cheap; call every step for a vibrating wall.
        """

    def set_wall_transform(self, wall_index: int, translation: Sequence[float] = [0.0, 0.0, 0.0], quat: Sequence[float] = [0.0, 0.0, 0.0, 1.0]) -> None:
        """
        Place an ANALYTIC wall (add_analytic_wall) rigidly in the world: quat (x,y,z,w) then translation, composed onto the AUTHORED root transform, so calls are absolute and never compound. This moves the GEOMETRY -- a stirrer blade sweeps -- where set_wall_velocity only gives the static surface a velocity field (enough for an axisymmetric drum). Drive both together each step: integrate ang_vel into the quaternion here and pass the same ang_vel to set_wall_velocity. Grid-SDF walls have no tree and are refused.
        """

    def set_positions(self, positions: Annotated[NDArray[numpy.float32], dict(order='C')]) -> None:
        """
        Set particle positions (length units) from an (N, 3) array, or (N, 4) whose column 3 is the INVERSE mass (1/mass units) (0 is remapped to 1; use set_inv_mass for a fixed body). N may not exceed `capacity`. This (re)sizes the particle set and RESETS every particle's quaternion, scale, inverse mass, shape id (to 0), velocities and gid -- so every other per-particle setter goes AFTER it.
        """

    def set_velocities(self, velocities: Annotated[NDArray[numpy.float32], dict(order='C')]) -> None:
        """Set particle velocities (length/time units) from an (N, 3) array."""

    def set_external_forces(self, forces: Annotated[NDArray[numpy.float32], dict(order='C')]) -> None:
        """
        Set the per-particle external FORCE (e.g. fluid drag; force units, mass x length/time^2) from an (N, 3) array. Applied each step as dv = F*invMass*dt; persists until re-set or cleared.
        """

    def clear_external_forces(self) -> None:
        """Zero all per-particle external forces (force units)."""

    def set_external_torques(self, torques: Annotated[NDArray[numpy.float32], dict(order='C')]) -> None:
        """
        Set the per-particle external TORQUE in the WORLD frame from an (N, 3) array (the resolved-CFD-DEM hydrodynamic torque, a magnetic couple, ...). Applied each step in the angular predictor as Euler's equation in the body frame, dw = invI*(tau_body - w x I w)*dt, alongside the gyroscopic term that is already there; persists until re-set or cleared. Only bodies with a finite inertia respond -- a torque on a body whose invInertia is zero is inert, exactly as the gyroscopic term is. Sleeping is disabled while a torque is set, as it is for external forces.
        """

    def clear_external_torques(self) -> None:
        """
        Zero all per-particle external torques (and re-enable island sleeping).
        """

    def set_quaternions(self, quaternions: Annotated[NDArray[numpy.float32], dict(order='C')]) -> None:
        """
        Set particle orientation quaternions (x, y, z, w) from an (N, 4) array.
        """

    def set_angular_velocities(self, angular_velocities: Annotated[NDArray[numpy.float32], dict(order='C')]) -> None:
        """
        Set particle angular velocities (body frame; 1/time units, radians per time) from an (N, 3) array.
        """

    def set_inv_inertia(self, inv_inertia: Annotated[NDArray[numpy.float32], dict(order='C')]) -> None:
        """
        Set the per-particle principal-frame diagonal inverse inertia from an (N, 3) array.
        """

    def set_inv_mass(self, inv_mass: Annotated[NDArray[numpy.float32], dict(order='C')]) -> None:
        """Set per-particle inverse mass, an (N,) array (0 = fixed body)."""

    def set_scales_uniform(self, scale: float) -> None:
        """Set a single uniform scale for all particles."""

    def set_scales(self, scales: Annotated[NDArray[numpy.float32], dict(order='C')]) -> None:
        """Set per-particle scales from an (N,) array (the growth target)."""

    def set_growth_params(self, rate: float, new_factor: float = -1.0) -> None:
        """
        Set the particle growth rate (factor *= exp(rate*dt) per step, capped at 1) and, if new_factor > 0, the current growth factor (a negative value keeps it; 0.01 if growth was inactive).
        """

    @property
    def growth_factor(self) -> float:
        """The current particle growth factor (-1 = growth inactive)."""

    @property
    def growth_rate(self) -> float:
        """The particle growth rate."""

    def get_masses(self) -> NDArray[numpy.float32]:
        """
        Return per-particle masses (1 / inverse mass; 0 for fixed bodies) as an (N,) array.
        """

    def get_positions(self) -> NDArray[numpy.float32]:
        """Return particle positions (length units) as an (N, 3) numpy array."""

    def get_velocities(self) -> NDArray[numpy.float32]:
        """
        Return particle velocities (length/time units) as an (N, 3) numpy array.
        """

    def get_angular_velocities(self) -> NDArray[numpy.float32]:
        """
        Return particle angular velocities (body frame; 1/time units) as an (N, 3) numpy array.
        """

    def get_inv_inertia(self) -> NDArray[numpy.float32]:
        """
        Return the per-particle principal-frame diagonal inverse inertia as an (N, 3) array.
        """

    def get_quaternions(self) -> NDArray[numpy.float32]:
        """
        Return particle orientation quaternions (x, y, z, w) as an (N, 4) numpy array.
        """

    def get_scales(self) -> NDArray[numpy.float32]:
        """Return per-particle scales as an (N,) numpy array."""

    def get_positions_view(self) -> NDArray[numpy.float32]:
        """
        Zero-copy (N,3) device array of positions (NumPy view on host, DLPack/CuPy on GPU).
        """

    def get_velocities_view(self) -> NDArray[numpy.float32]:
        """
        Zero-copy (N,3) device array of velocities (NumPy view on host, DLPack/CuPy on GPU).
        """

    def get_external_forces_view(self) -> NDArray[numpy.float32]:
        """
        Zero-copy (N,3) device array of the per-particle external force (NumPy view on host, DLPack/CuPy on GPU) — write fluid drag here directly to avoid a host round-trip.
        """

    def get_inv_mass_view(self) -> NDArray[numpy.float32]:
        """
        Zero-copy (N,) device array of per-particle inverse mass (NumPy view on host, DLPack/CuPy on GPU) — read-only use; needed for stiff-safe drag integration.
        """

    def step(self, n: int = 1) -> None:
        """
        Advance `n` XPBD substeps of the time step set by set_dt (raises if set_dt was never called).
        """

    def relax(self, n: int = 1) -> None:
        """
        Run `n` dynamics-free RELAXATION substeps: overlap removal only, no gravity and no velocity update -- the growth-packing protocol's settle move (dt = 0 inside; the stored dt is untouched and set_dt is not required).
        """

    def step_hertz(self, substeps: int = 1, skin_frac: float = 0.30000001192092896) -> None:
        """
        Advance `substeps` explicit soft-sphere Hertz-Mindlin steps of the time step set by set_dt (spheres, SDF walls, non-periodic): viscoelastic Hertz normal force + Mindlin shear-history spring, Coulomb-clamped; (e, mu) from the pair-material tables, stiffness from set_hertz_material. skin_frac is the Verlet pair-list skin as a fraction of the radius.
        """

    @property
    def num_particles(self) -> int:
        """The number of particles."""

    @property
    def num_contacts(self) -> int:
        """Broad-phase candidate pairs found in the last step (ArborX BVH query)."""

    @property
    def num_manifolds(self) -> int:
        """
        Narrow-phase contact manifolds (touching pairs) resolved in the last step.
        """

    @property
    def max_overlap(self) -> float:
        """
        Maximum pair interpenetration recorded by the position solver in the last step (its last-iteration residual, so it under-reports the committed overlap -- see docs/archive/packing_investigation.md). compute_overlaps() measures the committed state.
        """

    def compute_overlaps(self) -> float:
        """
        Measure and return the maximum pair interpenetration of the current committed state.
        """

    def get_sdf_grid(self, resolution: tuple[int, int, int]) -> NDArray[numpy.float32]:
        """
        Reconstruct the packed-bed SDF (negative inside solid) over the domain on an (rx, ry, rz) grid; returns an (rx, ry, rz) array indexed [x, y, z].
        """

    def export_sdf(self, filename: str, resolution: tuple[int, int, int]) -> None:
        """
        Reconstruct and write the packed-bed SDF on an (rx, ry, rz) grid to a VTI file.
        """

    def write_vtp(self, filename: str) -> None:
        """Write particle state to a VTP file (ParaView/Ovito)."""

    def export_lammps(self, filename: str, step: int) -> None:
        """Export particle state to a LAMMPS dump file."""

    def set_sleeping(self, enabled: bool, threshold_scale: float = 2.0, consecutive: int = 64, wake_scale: float = 40.0, wake_on_lost_contact: bool = False, immovable_frac: float = 0.009999999776482582) -> None:
        """
        Island sleeping (single-GPU statics, default ON): freeze grounded bodies whose motion stays below threshold_scale x the resting floor for `consecutive` substeps; wake only above wake_scale x that floor (hysteresis vs residual jitter). wake_on_lost_contact additionally wakes a sleeper whose support disappeared; immovable_frac is a sleeper's effective inverse-mass fraction in the solve (0 = exactly immovable). Requires gravity and no external force; inert under MPI.
        """

    @property
    def sleeping(self) -> bool:
        """Whether island sleeping is enabled."""

    @property
    def num_asleep(self) -> int:
        """Number of currently-sleeping real bodies."""

    def set_verlet_skin(self, skin_frac: float) -> None:
        """
        Enable the Verlet-cached impulse broadphase (single-GPU, non-periodic, default OFF, i.e. skin_frac = 0): skip the ArborX rebuild while nothing moved more than skin/2 (skin = skin_frac x max grain radius).
        """

    @property
    def verlet_skin(self) -> float:
        """
        Broadphase-skin fraction of the max grain radius (0 = rebuild every step).
        """

    def set_incremental_coloring(self, enabled: bool) -> None:
        """
        Incremental (warm-started) graph coloring of the contact manifolds (default True): reuse last substep's colors and repair only the conflicts. This CHANGES RESULTS -- the coloring fixes the Gauss-Seidel sweep order -- so False reproduces the pre-incremental behaviour rather than merely running slower.
        """

    @property
    def incremental_coloring(self) -> bool:
        """Whether incremental (warm-started) coloring is enabled."""

    def set_shell_detection(self, mode: str) -> None:
        """
        Narrow-phase shell detection of the XPBD engine (step, step_mpi, relax, compute_overlaps; step_hertz is unaffected). 'one_way' (default): only the lower-index body's shell points are tested against the other body's SDF, so a thin rim, wall or edge pressed into a flat face between those points can go unseen. 'two_way': shell pairs are also probed the other way round, and a sphere-shell pair always uses the exact sphere probe. Enable it for thin-walled or non-convex shells (tubes, rings), sharp edges and corners against faces, and large size ratios where the smaller body's features are finer than the larger body's shell spacing; not needed for spheres or smooth convex bodies of similar size with dense shells. Costs up to 2x the narrow phase and the contacts of shell pairs (docs/solver_details.md). CHANGES RESULTS for non-sphere bodies.
        """

    @property
    def shell_detection(self) -> str:
        """The narrow-phase shell detection: 'one_way' or 'two_way'."""

    def init_mpi(self, origin: tuple[float, float, float], extent: tuple[float, float, float], cells: tuple[int, int, int], periodic: tuple[bool, bool, bool]) -> None:
        """
        Set up the ORB block decomposition + core particle halo for the distributed step on the global domain (`origin` lower corner and `extent` SIZE in length units, `cells` per axis for the ORB grid, `periodic` per axis -- the suite's domain quartet).
        """

    def enable_mpi_step(self, rcut: float = 0.0, sync_every: int = 1, forward_rotation: bool = True, rebalance_every: int = 0, verlet_skin: float = 0.0) -> None:
        """
        Enable the distributed step: ghost-band width rcut (a lower bound -- step_mpi widens it to the contact reach plus a drift slack and the prediction allowance, max(rcut, 2.1 x the global maximum radius + 0.25 x it + P), so the default 0 means exactly that), sync_every (the owner/ghost reconciliation interval; conservation is exact at any value -- a body updated from several places is solved through copies, mass-split for the projection-form phases and exclusively held by one rank per sync interval for the g = 0 one-shot restitution sweep -- larger sync_every means more lag, fewer messages), rotation forwarding, the load-rebalance interval in steps (0 = fixed decomposition; every step also votes, in the same reduction as the band, whether any body has drifted more than the slack outside its owner's block, and migrates it if so -- ownership is therefore not fixed across a run, and ghost selection uses the domain-clamped position on non-periodic axes), and the Verlet ghost-reuse skin (0 = rebuild the halo topology every substep; >0 = reuse it until a particle moves > skin, and P above is the skin instead of the predicted displacement).
        """

    def step_mpi(self, n: int = 1) -> None:
        """
        Advance the distributed (MPI) simulation by `n` steps of the time step set by set_dt, with halo exchange.
        """

    def step_hertz_mpi(self, substeps: int = 1, skin_frac: float = 0.30000001192092896) -> None:
        """
        Advance `substeps` distributed explicit Hertz-Mindlin (force-based) steps of the time step set by set_dt — the MPI counterpart of step_hertz on the init_mpi/enable_mpi_step decomposition. rebalance_every counts CALLS of this method; migration carries the Mindlin history.
        """

    def rebalance(self) -> int:
        """
        Re-decompose by particle count and migrate ownership now; returns this rank's new owned count.
        """

    def migrate_to_weights(self, weights: Sequence[float], align: int = 1) -> int:
        """
        Co-rebalance: migrate particle ownership onto the weighted ORB of per-cell weights (global x-fastest over the init_mpi cell grid), so each rank owns the particles in a block of near-equal total weight -- the SAME partition the coupled flow solver redistributes onto from the same weight field. `align` (a power of two, default 1) puts every block boundary on a multiple of `align` cells; 1 is the plain weighted ORB. A coupled run passes the alignment flow's rebalance_by_weights returned, which it chose for its pressure multigrid; dem alone has no reason to align. Raises ValueError when `align` is not a power of two, does not divide the cell grid, or `weights` does not cover it. Returns this rank's new owned count.
        """

    @property
    def rank(self) -> int:
        """This rank's MPI index."""

    @property
    def num_ghost(self) -> int:
        """The number of ghost particles on this rank."""

def export_lammps(filename: str, step: int, pos: Annotated[NDArray[numpy.float32], dict(order='C')], vel: Annotated[NDArray[numpy.float32], dict(order='C')], quats: Annotated[NDArray[numpy.float32], dict(order='C')], radii: Annotated[NDArray[numpy.float32], dict(order='C')], box_min: Sequence[float] | None = None, box_max: Sequence[float] | None = None, periodic: bool = False) -> None:
    """
    Module-level LAMMPS dump writer from raw arrays (filename, step, pos, vel, quats, radii, box corners, periodic flag), written as given: pos, radii and the box corners in length units, vel in length/time units, quats dimensionless.
    """
