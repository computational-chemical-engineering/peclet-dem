/// @file
/// @brief dem — portable (Kokkos) particle SoA container: the storage the dem flip pivots on.
///
/// Replaces ParticleSystemData's float4* arrays with Kokkos SoA Views (backend-default layout:
/// coalesced on GPU, cache-friendly on CPU; packed .w scalars split into their own arrays). Holds
/// the per-particle state, the predicted/delta buffers, the collision/contact/manifold buffers, the
/// atomic counters (rank-0 Views), and the static shape/plane data — everything the ported kernels
/// (broadphase_arborx / narrowphase / contact_preprocessing / solver_* / integration / periodicity)
/// operate on. allocate() sizes them for a given capacity.
#ifndef DEM_PARTICLES_HPP
#define DEM_PARTICLES_HPP

#include <cstdint>
#include <Kokkos_Core.hpp>
#include <vector>

#include "contact_preprocessing.hpp"  // ContactC, ManifoldC
#include "integration.hpp"            // Domain, V3/V4/Vf/Vi
#include "narrowphase.hpp"            // ShapeDesc, PlaneP

namespace peclet::dem {

/// TEST-ONLY host capture of one distributed substep's contact ownership (C++ only; filled by
/// demStepMpi when Particles::debugCapture is set, never read by the step): the owned contacts
/// [0, ncOwned) after the partition and the owned predicted state they were detected on.
struct DebugContactCapture {
  // per owned contact: global ids of bodyA / bodyB (-1 for a wall), the periodic image of each
  // body's slot (integer box lengths per axis; 0 for an owned row and on a closed axis) and dist
  std::vector<int> gidA, gidB, imageA, imageB;  // image*: 3 per contact
  std::vector<float> dist;
  // per owned body: global id, predicted position (3), world radius
  std::vector<int> gid;
  std::vector<float> posPred, rad;
};

/// One solve phase's body copies (docs/contact_solve_framework.md §4.4, §4.5, §WO-4 item 2): the
/// hub copies of the colouring (extra slots [slotBase, slotBase + nCopies) appended to the SoA,
/// per-edge slot overrides) and the fold GROUPS -- one per body updated through more than one slot
/// in the phase (a hub's base + copies; in demStep's position phase also the periodic images of a
/// body). Group g's members are groupSlot[groupStart(g), groupStart(g + 1)), base (the body's own
/// slot) first; groupShift is each member's periodic shift (0 but for images and their copies),
/// groupK the active-copy count k, seedX / seedW the phase seed sigma of the base. Grow-only views;
/// nGroups == 0 = the phase has no copies (every sweep runs on today's indices and masses).
struct PhaseCopies {
  int nHubs = 0, nCopies = 0, nGroups = 0, slotBase = 0;
  int maxHubDegree = 0;                       // diagnostics: the largest hub degree
  Kokkos::View<int*, CpMem> slotA, slotB;     // per edge: override slot of the A / B end, -1
  Kokkos::View<int*, CpMem> copyBase;         // per copy slot (slot - slotBase): its hub's vertex
  Kokkos::View<int*, CpMem> hubVertex, hubS;  // per hub: base vertex, copy count s (incl. base)
  Kokkos::View<int*, CpMem> groupStart, groupSlot, groupK;
  Kokkos::View<float* [3], CpMem> groupShift;       // per member
  Kokkos::View<unsigned char*, CpMem> groupActive;  // per member: >= 1 active edge (counts in k)
  Kokkos::View<float* [3], CpMem> seedX, seedW;  // per group: sigma (velPred / posPred; angVelPred)
};

/// diagnostics.split_stats() (docs/contact_solve_framework.md §WO-4 item 4, §13.5 WO-4b): the last
/// substep's copies. unfiredSplitContacts (WO-6) and driftMigrations (WO-7) stay 0 until those
/// land. mlHubAggregated: the mass-split velocity vertices (k > 1: a hub's base; a rank-split
/// body's slot) that sit in a level-1 multilevel group of >= 2 members, the max over the substeps
/// of the last step call (§13.2's positive control). velItersUsed / posItersUsed: the iterations
/// the last substep's main velocity loop and position loop ran (a device-side loop reports its
/// count only with Particles::iterCounters on, §12 S12; -1 otherwise). orphanClamps (§13.3, the
/// distributed step): the owner-apply clamp hits of the Poisson orphan balance (B_new < 0 before
/// max(0, .)), summed over the substeps of the last step call; must stay 0 (the per-copy shares
/// B / k bound every draw by the balance).
struct SplitStats {
  int velHubCopies = 0, posHubCopies = 0, lightHubs = 0, splitBodiesVel = 0, splitBodiesPos = 0;
  long long unfiredSplitContacts = 0, driftMigrations = 0;
  int mlHubAggregated = 0, velItersUsed = 0, posItersUsed = 0;
  int orphanClamps = 0;
};

struct Particles {
  // --- per-particle state (size = capacity) ---
  V3 pos;
  Vf invMass;
  V4 quat;
  V3 vel;
  V3 angVel;
  V3 invInertia;
  Vf scale;
  Vf targetScale;
  Vi shapeId;
  V3 posPred;
  V4 quatPred;
  V3 velPred;
  V3 angVelPred;
  V3 deltaPos;
  V4 deltaQuat;
  V3 deltaVel;
  V3 deltaAngVel;
  Vi constraintCounts;
  Vi realIndices;
  // Global particle id (stable across halo rebuilds and MPI ownership migration; identity == local
  // index on the single-GPU path). The distributed step builds the persistent-contact pair keys
  // from it — local slots are not stable identities there. Ghost slots carry the owner's gid.
  Vi gid;
  Kokkos::View<float* [2], CpMem> planeFriction;
  Vf rad;       // effective broadphase radius scratch (scale * globalScale)
  V3 extForce;  // per-particle external FORCE (e.g. fluid drag); F=ma => dv = extForce*invMass*dt
  // Per-particle external TORQUE in the WORLD frame (resolved CFD-DEM hydrodynamic torque, a
  // magnetic couple, ...). Rotated into the body frame in the predictor, where the gyroscopic
  // Euler term already lives:  dw_body = invI * (tau_body - w x I w) * dt.  World frame is the
  // caller-facing convention because that is what every force/torque source produces; the body
  // frame is an internal detail of the principal-axis inertia representation.
  V3 extTorque;

  // --- collision/contact/manifold buffers ---
  Kokkos::View<int* [2], CpMem> pairs;        // broadphase candidates (maxPairs)
  Kokkos::View<ContactC*, CpMem> contacts;    // narrowphase output (maxContacts)
  Kokkos::View<ManifoldC*, CpMem> manifolds;  // reduction output (maxContacts)
  // Graph-colouring scratch for the single-GPU colored Gauss–Seidel velocity solve: per-manifold
  // colour (maxContacts; -2 inactive, -1 uncoloured, >=0 colour), plus per-body arbitration winner
  // and committed-colour bitmask (both indexed by REAL body index, sized capacity).
  Kokkos::View<int*, CpMem> manifoldColor;  // per-manifold colour (velocity solve)
  // Incremental (warm-started) colouring (single-GPU PGS path): the previous substep's per-manifold
  // colour, sorted alongside prevPairKeys (committed by pair key exactly like prevLambda). A
  // surviving pair keeps its colour so the Jones-Plassmann arbitration only re-runs over the NEW
  // manifolds; velLastFullColors is the colour count at the last FULL recolour (creep-recompaction
  // reference). See colorManifoldsIncrementalKokkos.
  Kokkos::View<int*, CpMem> prevManifoldColor;
  int velLastFullColors = 0;
  // Persistent-contact restitution (gravity-gated): pair keys of this/last substep's manifolds +
  // the per-manifold "existed last substep" flag. A persistent contact is LOADED, not a fresh
  // impact — it gets e = 0 (the impulse still cancels the approach: pure inelastic support), which
  // is how the velocity solve carries a pile's static weight through impulse chains. Restitution
  // stays reserved for newly formed contacts (genuine impacts). |g| = 0 leaves all of this idle.
  Kokkos::View<unsigned long long*, CpMem> pairKeys;
  Kokkos::View<unsigned long long*, CpMem> prevPairKeys;
  Kokkos::View<unsigned char*, CpMem> manifoldPersistent;
  int prevPairCount = 0;
  // Grounded level per REAL body (Guendelman support levels, warm-started + decayed): 255 at a
  // wall/plane contact, propagated lower -> upper through the contact graph a few sweeps per
  // substep. One-sided (shock-propagation) impulses require the LOWER body grounded > 0, so a
  // gas-borne emulsion or lifted slug (no path to the floor) keeps momentum-conserving impulses
  // and its weight stays on the gas -- only genuinely supported chains drain into the ground.
  Kokkos::View<unsigned char*, CpMem> groundedLevel;
  // Warm-started PGS (projected Gauss-Seidel) velocity solve: per-manifold accumulated push
  // impulse (this substep), the previous substep's converged impulses (sorted alongside
  // prevPairKeys for the warm-start gather), and the pre-solve approach velocity (restitution
  // bias). At convergence lambdaAcc IS the contact force network (x dt).
  Kokkos::View<float*, CpMem> lambdaAcc;
  Kokkos::View<float*, CpMem> prevLambda;
  Kokkos::View<float*, CpMem> vn0;
  Kokkos::View<float* [3], CpMem> vt0;  // pre-solve tangential surface velocity (beta reference)
  // Friction-cone PGS: per-manifold accumulated tangential impulse (world frame, kept in the
  // contact tangent plane by projection) and the previous substep's converged values (sorted
  // alongside prevPairKeys for the warm-start gather). |lambdaT| <= mu * lambdaAcc at all times.
  Kokkos::View<float* [3], CpMem> lambdaT;
  Kokkos::View<float* [3], CpMem> prevLambdaT;
  // Position-channel normal load (see solvePositionColoredGSKokkos): per-contact positional
  // lambda this substep, its per-manifold impulse-equivalent, the previous substep's value
  // (warm-gathered) that tops up the friction cone's Coulomb bound, and the sorted store.
  Kokkos::View<float*, CpMem> hertzSnPair;  // per cached pair: last step's patch stiffness sum
  Kokkos::View<float*, CpMem> hertzSnWall;  // per (particle, wall): same, for wall patches
  Kokkos::View<float*, CpMem> posLambdaContact;
  Kokkos::View<float*, CpMem> posImpulse;      // gathered: LAST substep's position-channel load
  Kokkos::View<float*, CpMem> prevPosImpulse;  // sorted alongside prevPairKeys
  // Event-level (Poisson) restitution (restitutionModel == 1): per-pair OWED separation impulse
  // (physical units) — e x the event's banked kinetic-compression impulse, minus what has already
  // been returned. Carried by pair key like lambdaT (gathered into restBank, committed to
  // prevRestBank); restRel is the per-substep release accumulator (clamped 0..owed inside the PGS
  // sweep — the friction-cone-shaped budget cap). See updateRestitutionBankKokkos.
  Kokkos::View<float*, CpMem> restBank;
  Kokkos::View<float*, CpMem> prevRestBank;  // sorted alongside prevPairKeys
  Kokkos::View<float*, CpMem> restRel;       // this substep's released impulse (lambda units)
  // Event state: peak physical approach speed of the pair's current impact event (> 0 = event
  // active). Set/refreshed by kinetic approaches, decays 1/256 per substep, cleared once it ages
  // below the resting threshold — so a buried/absorbed event's bank evaporates instead of popping.
  // While active the contact banks its applied normal-impulse flux EVERY substep (the co-moving
  // compression plateau has vn0 ~ 0, which a per-substep kinetic gate would miss), and the release
  // separation velocity is capped at e x vPeak (sustained unloading push, never an impulsive dump).
  Kokkos::View<float*, CpMem> restVPeak;
  Kokkos::View<float*, CpMem> prevRestVPeak;  // sorted alongside prevPairKeys
  // Orphan transfer (pair-churn fix): a pair that DIES with owed budget credits it to its
  // endpoint bodies mass-weighted (per-body balance + the event peak speed it carried, both
  // decayed 1/256 per substep so stranded credit evaporates). Live releasing pairs draw the
  // balance back on demand inside the PGS sweep — a penetrating impactor's event budget then
  // survives the ~20-substep turnover of its contact partners instead of dying with each pair.
  // Indexed by REAL body slot (owned range authoritative; MPI ghosts mirrored owner->ghost).
  Kokkos::View<float*, CpMem> bodyOrphan;
  Kokkos::View<float*, CpMem> bodyOrphanVPeak;
  Kokkos::View<unsigned char*, CpMem> prevMatched;  // scratch: prev-ledger entry seen this substep
  // Side flags for the STABILIZATION pass (0 = symmetric): zeroed for the main momentum-
  // conserving sweeps, filled from persistence+grounding only if statics fail to converge.
  Kokkos::View<unsigned char*, CpMem> sideFlags;
  // Level-ordered ("ordered") stabilization: per-REAL-body height-from-floor BFS level
  // (recomputed fresh each time the pass triggers) + the per-manifold (level, colour) bucket
  // key and permutation the ordered sweeps iterate through.
  Kokkos::View<int*, CpMem> heightLevel;
  Kokkos::View<int*, CpMem> levelKey;
  Kokkos::View<int*, CpMem> levelPerm;
  // Multilevel (GraphMG) stabilization scratch (solver_multilevel.hpp): packed per-manifold
  // 6-bit colour per level, pooled parent maps / group mass / group velocity arrays (levels
  // shrink >= 10% each, so the pools hold every level), the composed body -> group map, and
  // the matching scratch.
  Kokkos::View<long long*, CpMem> mlColorPacked;
  Kokkos::View<int*, CpMem> mlParent;
  Kokkos::View<float*, CpMem> mlInvMassG;
  Kokkos::View<float* [3], CpMem> mlVelG;
  Kokkos::View<float* [3], CpMem> mlVelG0;
  Kokkos::View<float*, CpMem> mlMassG;
  Kokkos::View<int*, CpMem> mlGrp;
  Kokkos::View<int*, CpMem> mlMate;
  // --- Hertz-Mindlin soft-sphere engine (solver_hertz.hpp): cached Verlet pair list state ---
  Kokkos::View<float* [3], CpMem> hertzXi;                 // per cached pair: Mindlin shear history
  Kokkos::View<unsigned long long*, CpMem> hertzKeys;      // keys of the cached pairs
  Kokkos::View<unsigned long long*, CpMem> hertzPrevKeys;  // sorted keys of the PREVIOUS list
  Kokkos::View<float* [3], CpMem> hertzPrevXi;             // xi aligned with hertzPrevKeys
  Kokkos::View<float* [3], CpMem> hertzXiWall;             // per (particle, wall) shear history
  Kokkos::View<int*, CpMem> hertzWallCand;  // near-wall candidate slots (i * 8 + wallIdx)
  Kokkos::View<int, CpMem> hertzWallCandCount;
  int hertzNumWallCand = 0;
  // Effective Hertz contact-curvature radius for non-spherical shapes, as a fraction of the
  // bounding radius (true curvature is undefined at faces/edges; spheres use their real radius).
  float hertzContactRadiusFrac = 0.5f;
  Kokkos::View<float* [3], CpMem> hertzRefPos;  // positions at the last pair build
  Kokkos::View<float, CpMem> hertzDispMax;      // max |pos-ref|^2 since the build
  Kokkos::View<float*, CpMem> hertzE, hertzNu;  // per-material Young / Poisson
  int hertzNumPairs = -1;                       // -1: no valid cached list
  int hertzPrevCount = 0;
  static constexpr int kHertzMaxWalls = 4;
  // Stabilization mode of the staged velocity solve (Phase B; see sim.hpp): 0 = off (pure
  // symmetric PGS), 1 = one-sided grounded pass (default), 2 = multilevel (level-ordered
  // symmetric sweeps -- momentum is transported, never deleted), 3 = escalate (extra symmetric
  // sweeps; diagnostic/fallback).
  int stabilizationMode = 1;
  // Restitution model of the PGS velocity solve: 0 = newton (default; per-substep restitution on
  // the pre-solve approach — unchanged behaviour), 1 = poisson (event-level: kinetic compression
  // impulse banked per pair, released as a budget-capped separation-velocity target during
  // unloading — restores the multi-substep-impact rebound per-substep Newton cannot return).
  int restitutionModel = 0;
  // Per-particle material id + flat pair-material table [kMaxMaterials^2 * 2] of (restitution,
  // friction) rows; zero-length pairMaterials = feature off (global material everywhere).
  Kokkos::View<unsigned char*, CpMem> materialId;
  Kokkos::View<float*, CpMem> pairMaterials;
  Kokkos::View<int*, CpMem> contactSlot;   // contact -> manifold slot (PGS friction bound)
  Kokkos::View<int*, CpMem> contactColor;  // per-contact colour (position solve; = its unit's)
  // Position units (docs/contact_solve_framework.md §4.3): the contacts of one body pair, or of one
  // body and one wall, swept sequentially in one work item. CSR over the solved contacts
  // [0, nc): unitStart (numPosUnits + 1), unitContacts (nc; ascending inside a unit), and the
  // per-unit colour. Rebuilt every substep on the Gauss-Seidel path; numPosUnits = 0 otherwise.
  Kokkos::View<int*, CpMem> unitStart;
  Kokkos::View<int*, CpMem> unitContacts;
  Kokkos::View<int*, CpMem> unitColor;
  int numPosUnits = 0;
  // Incremental (warm-started) position colouring (single-GPU PGS path): this substep's per-contact
  // pair keys + the previous substep's (sorted keys, colour) ledger, so a surviving contact keeps
  // its colour and only NEW contacts re-arbitrate. Unlike the manifold graph a pair CAN own several
  // contacts (non-spherical multi-point patches), so carried colours are conflict-CHECKED (a shared
  // colour on a shared body -> full recolour that substep). posLastFullColors = colours at the last
  // full recolour (creep-recompaction reference). See colorContactsIncrementalKokkos.
  Kokkos::View<unsigned long long*, CpMem> contactKeys;
  Kokkos::View<unsigned long long*, CpMem> prevContactKeys;
  Kokkos::View<int*, CpMem> prevContactColor;
  Kokkos::View<int*, CpMem> posCommitPerm;
  int posPrevContactCount = 0;
  int posLastFullColors = 0;
  // Per-body round-winner key for the colouring arbitration. 64-bit: hashed-random priority in the
  // high word (splitmix32 of the edge index), the unique edge index in the low word. Random
  // priorities give O(log n) arbitration rounds w.h.p.; RAW indices are adversarial for poured
  // lattice beds (monotone index chains -> O(chain) rounds -> minutes per step at 1M grains).
  // Dense colour-bucket scratch (buildColorBucketsKokkos): per-manifold permutation for the PGS
  // velocity sweeps, the pooled commit permutation, a 64-int cursor, and the lazily-grown
  // per-level coarse-cycle permutation (numLevels x numManifolds segments).
  Kokkos::View<int*, CpMem> velPerm;
  Kokkos::View<int*, CpMem> commitPerm;
  Kokkos::View<int*, CpMem> bucketCursor;
  Kokkos::View<int*, CpMem> mlBucketPerm;
  // Fused colour sweeps (solver_fused.hpp): position-colour bucket permutation, pooled device
  // copies of the host colour offsets (velocity / position / flat multilevel), and the software
  // grid-barrier arrival counter (memset per fused launch, shared by all fused kernels — they
  // are stream-ordered).
  Kokkos::View<int*, CpMem> posPerm;
  Kokkos::View<int*, CpMem> velOffsDev;
  Kokkos::View<int*, CpMem> posOffsDev;
  Kokkos::View<int*, CpMem> mlOffsDev;
  Kokkos::View<unsigned*, CpMem> fusedBar;
  // Cross-step CUDA-graph executable cache (opaque cudaGraphExec_t per iteration loop:
  // velocity / onesided / multilevel / position). Owned here; leaked at teardown by design
  // (freeing needs the CUDA context, which Kokkos may already have torn down).
  void* graphCache[4] = {nullptr, nullptr, nullptr, nullptr};
  Kokkos::View<long long*, CpMem> bodyWinner;
  Kokkos::View<std::uint64_t*, CpMem> bodyColorMask;  // per-body committed-colour bitmask
  // --- island sleeping / freezing (single-GPU statics path; see sleeping.hpp) ---
  // Per-REAL-body asleep flag + low-motion counter; effective inverse mass for the solve (asleep
  // bodies + their ghosts -> 0, swapped in for the solve so a sleeper is immovable); the
  // moving-wall "never sleep" flag + this-substep live contact count (contact-set-change wake); the
  // stored count. Per-manifold / per-contact "both endpoints asleep" masks exclude a frozen island
  // from the colouring / sweeps / multilevel hierarchy (the ledger still carries their force
  // network).
  Kokkos::View<unsigned char*, CpMem> asleep;
  Kokkos::View<unsigned char*, CpMem> sleepCounter;
  Kokkos::View<unsigned char*, CpMem> sleepMovingWall;
  Kokkos::View<int*, CpMem> sleepCurCount;
  Kokkos::View<int*, CpMem> sleepPrevCount;
  Kokkos::View<float*, CpMem> invMassEff;
  Kokkos::View<unsigned char*, CpMem> manifoldSleep;
  Kokkos::View<unsigned char*, CpMem> contactSleep;
  bool sleepingEnabled = true;  // set_sleeping(enabled=False) disables; default ON
  float sleepScale = 2.0f;      // cSleep: sleep threshold = sleepScale * vRest
  float wakeScale = 40.0f;      // cWake: wake if an awake neighbour exceeds wakeScale * vRest
                                // (hysteresis: >> the residual settling jitter so a frozen bed
                                // stays frozen; only a genuine impact/disturbance wakes it)
  int sleepK = 64;              // substeps below threshold before sleeping (high enough that an
                    // impact's unloading/rebound completes before the network re-sleeps)
  // rule (b): wake on a LOST contact (support removed); set_sleeping(wake_on_lost_contact=)
  bool sleepWakeLostContact = false;
  // Effective inverse-mass fraction of a sleeper for the solve (0 = exactly immovable). A small
  // POSITIVE value keeps the sleeper very heavy but not infinitely rigid, so an awake body wedged
  // at a frozen-pocket boundary can relieve against it instead of the PGS normal impulse diverging
  // (trapped-between-two-rigid-constraints blow-up that a settling column reliably hit, ejected via
  // the friction cone to NaN); the sleeper's velocity is re-zeroed each substep so no momentum
  // accumulates and both-asleep interior manifolds are still fully excluded (the speed win). 0.01 =
  // sleeper 100x a grain's mass: stable through the 96k column + violent pour, case3 penetration
  // and the settled-bed freeze both preserved. set_sleeping(immovable_frac=) overrides.
  float sleepImmovableFrac = 0.01f;
  bool extForceActive = false;   // CFD-DEM drag present -> sleeping disabled this step
  bool extTorqueActive = false;  // external couple present -> sleeping disabled this step
  // --- Verlet-cached broadphase for the impulse step (single-GPU, non-periodic; see demStep) ---
  // The impulse broadphase rebuilds the ArborX pair list every step; between rebuilds no new pair
  // can appear if no particle has moved more than skin/2 (with the list built at margin + skin).
  // impRefPos = posPred at the last build; impNumPairs = the cached candidate count (-1 = invalid,
  // forces a build). Periodic ghosts are regenerated per step (unstable slot ids), so this is used
  // only when the domain is non-periodic. Off by default (rebuild every step).
  Kokkos::View<float* [3], CpMem> impRefPos;
  Kokkos::View<float, CpMem> impDispMax;
  int impNumPairs = -1;
  float impRefMaxRad = 0.0f;    // max effective radius at the last build (growth bound)
  float verletSkinFrac = 0.0f;  // set_verlet_skin; 0 = off (rebuild every step)

  // --- solver execution policy (Simulation setters; see solve_driver.hpp / solver_fused.hpp) ---
  // cudaGraphs and fusedSweeps only choose HOW the same arithmetic is submitted to the GPU (both
  // paths are bit-identical, and both are inert on a non-CUDA backend). incrementalColoring does
  // change results: the colouring fixes the Gauss-Seidel sweep order.
  bool cudaGraphs = true;           // set_cuda_graphs: capture+replay the sweep loops (CUDA)
  int fusedSweeps = -1;             // set_fused_sweeps: -1 auto, 0 off, 1 on (CUDA)
  bool incrementalColoring = true;  // set_incremental_coloring: warm-started recolouring

  // --- atomic counters / scalars (rank-0 Views) ---
  Kokkos::View<int, CpMem> pairCount, contactCount, manifoldCount, topGhost;
  Kokkos::View<float, CpMem> maxOverlap;
  // The accumulated position projection's stop residual max |d| w per iteration (WO-12), and its
  // over-relaxation (internal; kPositionOmega by default, a test hook for the omega scan).
  Kokkos::View<float, CpMem> posResidual;
  float positionOmega = 1.5f;
  // Max physical approach speed among approaching manifolds in the last velocity sweep — drives the
  // colored-GS velocity loop's adaptive stop (converged once no pair approaches above the resting
  // threshold). maxOverlap plays the same role for the position loop.
  Kokkos::View<float, CpMem> maxApproach;
  // Quasi-static share of maxApproach (corrections on contacts with |vn0| <= 4 vRest): the
  // multilevel stabilization loop's stop criterion -- flowing scenes keep ballistic churn out
  // of it, so the pass ends after ~one cycle instead of burning its full budget as an
  // over-convergence brake on discharge.
  Kokkos::View<float, CpMem> maxApproachQS;
  // The largest consensus correction since the last stop vote (docs/contact_solve_framework.md
  // §12 S14): |mean - copy| of any active copy at a local fold (velocity: velPred; position:
  // posPred) or a rank-level M reconciliation, absolute like the phase's residual. A phase with
  // copies folds it into its stop vote (the same Allreduce-MAX) and zeroes it after the read; a
  // phase without copies never writes or reads it.
  Kokkos::View<float, CpMem> maxConsensus;

  // --- static geometry ---
  Kokkos::View<ShapeDesc*, CpMem> shapes;
  Kokkos::View<float* [3], CpMem> shell;
  Kokkos::View<PlaneP*, CpMem> planes;
  Kokkos::View<float*, CpMem> sdfGrid;  // concatenated grid-SDF samples (imported shapes)
  // static world-space SDF walls (drum barrel, hopper, vibrating tray) + their concatenated
  // samples.
  Kokkos::View<WallSdf*, CpMem> walls;
  Kokkos::View<float*, CpMem> wallGrid;
  // shape nodes backing ANALYTIC walls (Layer 1); WallSdf::nodes points into this
  Kokkos::View<peclet::core::geom::ShapeNode<float>*, CpMem> wallNodes;
  // SHAPE_SCENE particle trees: the pooled node table (base pointer + absolute indices)
  Kokkos::View<peclet::core::geom::ShapeNode<float>*, CpMem> shapeNodes;

  // --- sizes & params (host) ---
  int capacity = 0, numReal = 0, numParticles = 0;
  int maxPairs = 0, maxContacts = 0, numPlanes = 0, numWalls = 0;
  // LARGEST surface-shell size across the registered shapes, mirrored from Simulation so the
  // free-function contact-buffer growth (growContactBuffers) can size itself without reading the
  // device shape descriptors back to the host every step.
  int shellPoints = 0;
  // max wall friction (host) — gates the friction path so a frictional wall works even with a
  // frictionless body-body material (global frictionDynamic == 0).
  float wallFrictionMax = 0.0f;
  Domain domain{};
  F3 gravity{0, 0, 0};
  float dt = 1e-3f, globalScale = 1.0f, growthRate = 0.0f, growthFactor = -1.0f;
  float baseRadius =
      1.0f;  // shape canonical radius; effective radius = scale * globalScale * baseRadius
  float thermostatTau = 0.0f, thermostatTemp = 0.0f,
        thermostatKB = 1.0f;  // Berendsen (tau>0 enables)
  float frictionDynamic = 0.0f, restitutionNormal = 0.0f, skin = 0.1f;
  // Walton tangential restitution beta (0 = tangentially dead stick, the pre-beta behaviour;
  // > 0 reverses the pre-collision tangential surface velocity of COLLIDING contacts).
  float restitutionTangent = 0.0f;
  int positionIterations = 10, velocityIterations = 0;
  // Single-GPU collision solves: true = colored Gauss–Seidel for BOTH the restitution (velocity)
  // and the overlap (position) solve — correct coupled multi-contact impulses + non-penetration,
  // default; false = count-averaged Jacobi (the legacy robust path, still used by step_mpi).
  bool velocityUseGS = true;

  // --- contact-solve copies (docs/contact_solve_framework.md §4.4, §4.5) ---
  PhaseCopies velCopies, posCopies;
  // Solve views (§4.5): per slot k x invMass / k x invInertia of its body while the phase has
  // copies (else the sweeps take invMass / invInertia themselves), and splitSlot = (k > 1).
  // Grow-only, sized to the slots the phase touches.
  Kokkos::View<float*, CpMem> invMassSolve;
  Kokkos::View<float* [3], CpMem> invInertiaSolve;
  Kokkos::View<unsigned char*, CpMem> splitSlot;
  Kokkos::View<int*, CpMem> vertexDegree;  // scratch: per colouring vertex, active edges
  // Rank-level M (docs/contact_solve_framework.md §13.3; the distributed step only, recomputed
  // every substep): per slot the local active copy counts aVel / aPos (the activity pass: 0 or 1
  // at a vertex outside any group, the group's count at a hub base) and the global active copy
  // counts kVel / kPos (owner: max(1, a(own) + sum of the ghosts' a), forwarded to the ghosts);
  // activityHit the pass's per-slot scratch; invMassCoarse the multilevel coarse vertex's inverse
  // mass invMass k / max(1, a) (§13.2). Grow-only.
  Kokkos::View<int*, CpMem> aVel, aPos, kVel, kPos;
  Kokkos::View<unsigned char*, CpMem> activityHit;
  // Rank-level X (docs/contact_solve_framework.md §1.4, §13.5 WO-6; the g = 0 one-shot on an
  // exchanging rank): per slot the velocity activity mask (bit col(r) of every rank on which the
  // body is active, made global by the g = 0 opening), the per-manifold fire gate of the current
  // sync interval, and the solve counter that starts each substep's holder cycle one step later.
  // solveEpoch is incremented by every demSolveContacts, so it is identical on all ranks.
  Kokkos::View<unsigned long long*, CpMem> velMask;
  Kokkos::View<unsigned char*, CpMem> xGate;
  long long solveEpoch = 0;
  Kokkos::View<float*, CpMem> invMassCoarse;
  // split_stats.orphanClamps, accumulated on the device by the owner apply (read by
  // Simulation::debugSplitStats, reset by the step entry points; no fence in the step).
  Kokkos::View<int, CpMem> orphanClampCount;
  Kokkos::View<float* [3], CpMem>
      imageShift;  // demStep: posPred(image) - posPred(real) at generation
  // The velocity incremental colouring carries colours by pair key; after a substep coloured
  // through hub copies those colours are per COPY and may repeat at the body, so the next substep
  // recolours in full (§4.2 item 4).
  bool velCopiesLastSubstep = false;
  SplitStats splitStats;
  // The last multilevel hierarchy's shape (host), for the coarse-colouring validity check
  // (Simulation::debugMultilevelColoringConflicts). numLevels = 0: none built last substep.
  struct MlLast {
    int numLevels = 0, numManifolds = 0, numBodies = 0;
    std::vector<int> parentOff, numGroups;
  } mlLast;

  // Diagnostics (Simulation::debugIterationCounters; docs/contact_solve_framework.md §12 S12):
  // with iterCounters on, a single-rank device-side (fused) main velocity / position loop writes
  // its iteration count to iterCountDev(0) / (1), read back into splitStats.velItersUsed /
  // posItersUsed. Off (the default): nothing is written or read back, the counters read -1.
  bool iterCounters = false;
  // TEST-ONLY (Simulation::debugNoAdaptiveStop; §12 S13): every adaptive stop of the contact solve
  // is disabled, so each loop runs exactly its iteration cap (the votes still run: they are
  // collective and carry the colouring invariant). Off (the default) changes nothing.
  bool noAdaptiveStop = false;
  Kokkos::View<int*, CpMem> iterCountDev;

  // TEST-ONLY contact capture (Simulation::debugCaptureContacts; see DebugContactCapture).
  bool debugCapture = false;
  DebugContactCapture debugCaptured;

  // nPlanes is the plane-array CAPACITY; numPlanes (the live count) stays 0 until planes are added.
  void allocate(int cap, int maxPairs_, int maxContacts_, int nShapes, int nShell, int nPlanes) {
    capacity = cap;
    maxPairs = maxPairs_;
    maxContacts = maxContacts_;
    numPlanes = 0;
    pos = V3("pos", cap);
    invMass = Vf("invMass", cap);
    quat = V4("quat", cap);
    vel = V3("vel", cap);
    angVel = V3("angVel", cap);
    invInertia = V3("invInertia", cap);
    scale = Vf("scale", cap);
    targetScale = Vf("targetScale", cap);
    shapeId = Vi("shapeId", cap);
    posPred = V3("posPred", cap);
    quatPred = V4("quatPred", cap);
    velPred = V3("velPred", cap);
    angVelPred = V3("angVelPred", cap);
    deltaPos = V3("deltaPos", cap);
    deltaQuat = V4("deltaQuat", cap);
    deltaVel = V3("deltaVel", cap);
    deltaAngVel = V3("deltaAngVel", cap);
    constraintCounts = Vi("constraintCounts", cap);
    realIndices = Vi("realIndices", cap);
    gid = Vi("gid", cap);
    planeFriction = Kokkos::View<float* [2], CpMem>("planeFriction", cap);
    rad = Vf("rad", cap);
    extForce = V3("extForce", cap);  // zero-initialised => no external force by default
    extTorque = V3("extTorque", cap);
    pairs = Kokkos::View<int* [2], CpMem>("pairs", maxPairs);
    contacts = Kokkos::View<ContactC*, CpMem>("contacts", maxContacts);
    manifolds = Kokkos::View<ManifoldC*, CpMem>("manifolds", maxContacts);
    manifoldColor = Kokkos::View<int*, CpMem>("manifoldColor", maxContacts);
    prevManifoldColor = Kokkos::View<int*, CpMem>("prevManifoldColor", maxContacts);
    pairKeys = Kokkos::View<unsigned long long*, CpMem>("pairKeys", maxContacts);
    prevPairKeys = Kokkos::View<unsigned long long*, CpMem>("prevPairKeys", maxContacts);
    manifoldPersistent = Kokkos::View<unsigned char*, CpMem>("manifoldPersistent", maxContacts);
    prevPairCount = 0;
    contactColor = Kokkos::View<int*, CpMem>("contactColor", maxContacts);
    unitStart = Kokkos::View<int*, CpMem>("unitStart", maxContacts + 1);
    unitContacts = Kokkos::View<int*, CpMem>("unitContacts", maxContacts);
    unitColor = Kokkos::View<int*, CpMem>("unitColor", maxContacts);
    contactKeys = Kokkos::View<unsigned long long*, CpMem>("contactKeys", maxContacts);
    prevContactKeys = Kokkos::View<unsigned long long*, CpMem>("prevContactKeys", maxContacts);
    prevContactColor = Kokkos::View<int*, CpMem>("prevContactColor", maxContacts);
    posCommitPerm = Kokkos::View<int*, CpMem>("posCommitPerm", maxContacts);
    velPerm = Kokkos::View<int*, CpMem>("velPerm", maxContacts);
    commitPerm = Kokkos::View<int*, CpMem>("commitPerm", maxContacts);
    bucketCursor = Kokkos::View<int*, CpMem>("bucketCursor", 64);
    mlBucketPerm = Kokkos::View<int*, CpMem>("mlBucketPerm", 0);
    posPerm = Kokkos::View<int*, CpMem>("posPerm", maxContacts);
    velOffsDev = Kokkos::View<int*, CpMem>("velOffsDev", 65);
    posOffsDev = Kokkos::View<int*, CpMem>("posOffsDev", 65);
    mlOffsDev = Kokkos::View<int*, CpMem>("mlOffsDev", 65 * 10);   // 10 = kMlMaxLevels
    fusedBar = Kokkos::View<unsigned*, CpMem>("fusedBar", 32769);  // 4096 blocks x 8 + 1
    bodyWinner = Kokkos::View<long long*, CpMem>("bodyWinner", cap);
    bodyColorMask = Kokkos::View<std::uint64_t*, CpMem>("bodyColorMask", cap);
    asleep = Kokkos::View<unsigned char*, CpMem>("asleep", cap);
    sleepCounter = Kokkos::View<unsigned char*, CpMem>("sleepCounter", cap);
    sleepMovingWall = Kokkos::View<unsigned char*, CpMem>("sleepMovingWall", cap);
    sleepCurCount = Kokkos::View<int*, CpMem>("sleepCurCount", cap);
    sleepPrevCount = Kokkos::View<int*, CpMem>("sleepPrevCount", cap);
    invMassEff = Kokkos::View<float*, CpMem>("invMassEff", cap);
    manifoldSleep = Kokkos::View<unsigned char*, CpMem>("manifoldSleep", maxContacts);
    contactSleep = Kokkos::View<unsigned char*, CpMem>("contactSleep", maxContacts);
    impRefPos = Kokkos::View<float* [3], CpMem>("impRefPos", cap);
    impDispMax = Kokkos::View<float, CpMem>("impDispMax");
    groundedLevel = Kokkos::View<unsigned char*, CpMem>("groundedLevel", cap);
    materialId = Kokkos::View<unsigned char*, CpMem>("materialId", cap);
    lambdaAcc = Kokkos::View<float*, CpMem>("lambdaAcc", maxContacts);
    lambdaT = Kokkos::View<float* [3], CpMem>("lambdaT", maxContacts);
    posLambdaContact = Kokkos::View<float*, CpMem>("posLambdaContact", maxContacts);
    posImpulse = Kokkos::View<float*, CpMem>("posImpulse", maxContacts);
    prevPosImpulse = Kokkos::View<float*, CpMem>("prevPosImpulse", maxContacts);
    restBank = Kokkos::View<float*, CpMem>("restBank", maxContacts);
    prevRestBank = Kokkos::View<float*, CpMem>("prevRestBank", maxContacts);
    restRel = Kokkos::View<float*, CpMem>("restRel", maxContacts);
    restVPeak = Kokkos::View<float*, CpMem>("restVPeak", maxContacts);
    prevRestVPeak = Kokkos::View<float*, CpMem>("prevRestVPeak", maxContacts);
    bodyOrphan = Kokkos::View<float*, CpMem>("bodyOrphan", cap);
    bodyOrphanVPeak = Kokkos::View<float*, CpMem>("bodyOrphanVPeak", cap);
    prevMatched = Kokkos::View<unsigned char*, CpMem>("prevMatched", maxContacts);
    sideFlags = Kokkos::View<unsigned char*, CpMem>("sideFlags", maxContacts);
    heightLevel = Kokkos::View<int*, CpMem>("heightLevel", cap);
    levelKey = Kokkos::View<int*, CpMem>("levelKey", maxContacts);
    levelPerm = Kokkos::View<int*, CpMem>("levelPerm", maxContacts);
    mlColorPacked = Kokkos::View<long long*, CpMem>("mlColorPacked", maxContacts);
    mlParent = Kokkos::View<int*, CpMem>("mlParent", 5 * cap);
    mlInvMassG = Kokkos::View<float*, CpMem>("mlInvMassG", 4 * cap);
    mlVelG = Kokkos::View<float* [3], CpMem>("mlVelG", 4 * cap);
    mlVelG0 = Kokkos::View<float* [3], CpMem>("mlVelG0", 4 * cap);
    mlMassG = Kokkos::View<float*, CpMem>("mlMassG", 4 * cap);
    mlGrp = Kokkos::View<int*, CpMem>("mlGrp", cap);
    mlMate = Kokkos::View<int*, CpMem>("mlMate", cap);
    hertzXiWall = Kokkos::View<float* [3], CpMem>("hertzXiWall", cap * kHertzMaxWalls);
    hertzWallCand = Kokkos::View<int*, CpMem>("hertzWallCand", cap * 2);
    hertzSnWall = Kokkos::View<float*, CpMem>("hertzSnWall", cap * kHertzMaxWalls);
    hertzWallCandCount = Kokkos::View<int, CpMem>("hertzWallCandCount");
    hertzRefPos = Kokkos::View<float* [3], CpMem>("hertzRefPos", cap);
    hertzDispMax = Kokkos::View<float, CpMem>("hertzDispMax");
    hertzE = Kokkos::View<float*, CpMem>("hertzE", 8);
    hertzNu = Kokkos::View<float*, CpMem>("hertzNu", 8);
    {
      auto he = Kokkos::create_mirror_view(hertzE);
      auto hn = Kokkos::create_mirror_view(hertzNu);
      for (int i = 0; i < 8; ++i) {
        he(i) = 1.0e9f;
        hn(i) = 0.25f;
      }
      Kokkos::deep_copy(hertzE, he);
      Kokkos::deep_copy(hertzNu, hn);
    }
    prevLambdaT = Kokkos::View<float* [3], CpMem>("prevLambdaT", maxContacts);
    contactSlot = Kokkos::View<int*, CpMem>("contactSlot", maxContacts);
    prevLambda = Kokkos::View<float*, CpMem>("prevLambda", maxContacts);
    vn0 = Kokkos::View<float*, CpMem>("vn0", maxContacts);
    vt0 = Kokkos::View<float* [3], CpMem>("vt0", maxContacts);
    pairCount = Kokkos::View<int, CpMem>("pairCount");
    contactCount = Kokkos::View<int, CpMem>("contactCount");
    manifoldCount = Kokkos::View<int, CpMem>("manifoldCount");
    topGhost = Kokkos::View<int, CpMem>("topGhost");
    maxOverlap = Kokkos::View<float, CpMem>("maxOverlap");
    posResidual = Kokkos::View<float, CpMem>("posResidual");
    maxApproach = Kokkos::View<float, CpMem>("maxApproach");
    maxApproachQS = Kokkos::View<float, CpMem>("maxApproachQS");
    maxConsensus = Kokkos::View<float, CpMem>("maxConsensus");
    shapes = Kokkos::View<ShapeDesc*, CpMem>("shapes", nShapes > 0 ? nShapes : 1);
    shell = Kokkos::View<float* [3], CpMem>("shell", nShell > 0 ? nShell : 1);
    planes = Kokkos::View<PlaneP*, CpMem>("planes", nPlanes > 0 ? nPlanes : 1);
    sdfGrid = Kokkos::View<float*, CpMem>("sdfGrid", 1);    // resized by setSdfShape
    walls = Kokkos::View<WallSdf*, CpMem>("walls", 1);      // resized by addSdfWall
    wallGrid = Kokkos::View<float*, CpMem>("wallGrid", 1);  // concatenated wall samples
    wallNodes = Kokkos::View<peclet::core::geom::ShapeNode<float>*, CpMem>("wallNodes", 1);
    shapeNodes = Kokkos::View<peclet::core::geom::ShapeNode<float>*, CpMem>("shapeNodes", 1);
    numWalls = 0;
  }

  // Grow the per-particle SoA to hold at least `newCap` particles (real + periodic-ghost headroom),
  // preserving the existing [0,numReal) state (Kokkos::resize copies the overlapping subextent).
  // The single-GPU step sizes this from the domain before generating ghosts; without the headroom
  // every ghost slot overflows `capacity` and cross-boundary contacts silently vanish. A no-op when
  // the SoA is already large enough (e.g. the MPI path, whose caller pre-sizes capacity for the
  // worst-case ghost band). The collision buffers (pairs/contacts/manifolds) keep their
  // construction-time sizing — each real particle still issues one broad-phase query.
  void ensureCapacity(int newCap) {
    if (newCap <= capacity)
      return;
    Kokkos::resize(pos, newCap);
    Kokkos::resize(invMass, newCap);
    Kokkos::resize(quat, newCap);
    Kokkos::resize(vel, newCap);
    Kokkos::resize(angVel, newCap);
    Kokkos::resize(invInertia, newCap);
    Kokkos::resize(scale, newCap);
    Kokkos::resize(targetScale, newCap);
    Kokkos::resize(shapeId, newCap);
    Kokkos::resize(posPred, newCap);
    Kokkos::resize(quatPred, newCap);
    Kokkos::resize(velPred, newCap);
    Kokkos::resize(angVelPred, newCap);
    Kokkos::resize(deltaPos, newCap);
    Kokkos::resize(deltaQuat, newCap);
    Kokkos::resize(deltaVel, newCap);
    Kokkos::resize(deltaAngVel, newCap);
    Kokkos::resize(constraintCounts, newCap);
    Kokkos::resize(realIndices, newCap);
    Kokkos::resize(gid, newCap);
    Kokkos::resize(bodyWinner, newCap);
    Kokkos::resize(bodyColorMask, newCap);
    Kokkos::resize(groundedLevel, newCap);
    Kokkos::resize(heightLevel, newCap);
    Kokkos::resize(planeFriction, newCap);
    Kokkos::resize(rad, newCap);
    Kokkos::resize(extForce, newCap);
    Kokkos::resize(extTorque, newCap);
    // materialId is written per GHOST slot by generateGhostsKokkos (guarded by `capacity`),
    // so it MUST track the padded capacity like every other per-slot array. Its absence here
    // was a silent out-of-bounds write into the neighbouring allocation — harmless or
    // catastrophic depending on the device allocator's layout (the 2026-08 H100 packing
    // corruption: contacts silently unresolved, phi_voxel 0.40 instead of 0.50).
    Kokkos::resize(materialId, newCap);
    // Not ghost-indexed today, but per-particle and cap-sized at construction — keep them in
    // lockstep so a future ghost-slot consumer cannot reintroduce the same class of bug.
    Kokkos::resize(asleep, newCap);
    Kokkos::resize(sleepCounter, newCap);
    Kokkos::resize(sleepMovingWall, newCap);
    // The rest of the capacity-sized per-body arrays. These were the LAST twelve left behind, and
    // the same class of bug as materialId above: every one of them is written up to numParticles
    // (or numReal) by some kernel, so any that keeps the pre-growth extent is an out-of-bounds
    // write. Found via peclet-examples/pall-ring-packing, where a 48-ring pour reported
    // "corrupted double-linked list" from inside step() -- mlGrp/mlMate (the multilevel
    // stabiliser's per-body group and mate arrays) and invMassEff (written for numParticles by
    // buildInvMassEffKokkos) are the ones that bit. A cap-sized array that ensureCapacity does not
    // resize is a latent heap corruption, full stop; there is no reason to keep any of them out.
    Kokkos::resize(invMassEff, newCap);
    Kokkos::resize(sleepCurCount, newCap);
    Kokkos::resize(sleepPrevCount, newCap);
    Kokkos::resize(bodyOrphan, newCap);
    Kokkos::resize(bodyOrphanVPeak, newCap);
    Kokkos::resize(mlGrp, newCap);
    Kokkos::resize(mlMate, newCap);
    Kokkos::resize(impRefPos, newCap);
    Kokkos::resize(hertzRefPos, newCap);
    Kokkos::resize(hertzWallCand, newCap);
    Kokkos::resize(hertzSnWall, newCap);
    Kokkos::resize(hertzXiWall, newCap);
    capacity = newCap;
  }

  // Const views for the read-only kernel inputs.
  Kokkos::View<const float* [3], CpMem> cpos() const { return pos; }
  Kokkos::View<const float*, CpMem> crad() const { return rad; }
};

}  // namespace peclet::dem

#endif  // DEM_PARTICLES_HPP
