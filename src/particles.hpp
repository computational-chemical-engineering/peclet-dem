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

#include "contact_preprocessing.hpp"  // ContactC, ManifoldC
#include "integration.hpp"            // Domain, V3/V4/Vf/Vi
#include "narrowphase.hpp"            // ShapeDesc, PlaneP

namespace peclet::dem {

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
  Kokkos::View<int*, CpMem> contactColor;  // per-contact colour (position solve)
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
  bool sleepingEnabled = true;  // set_sleeping / PECLET_DEM_SLEEP=0 disables; default ON
  float sleepScale = 2.0f;       // cSleep: sleep threshold = sleepScale * vRest
  float wakeScale = 40.0f;       // cWake: wake if an awake neighbour exceeds wakeScale * vRest
                                 // (hysteresis: >> the residual settling jitter so a frozen bed
                                 // stays frozen; only a genuine impact/disturbance wakes it)
  int sleepK = 64;               // substeps below threshold before sleeping (high enough that an
                    // impact's unloading/rebound completes before the network re-sleeps)
  bool sleepWakeLostContact = false;  // rule (b): wake on a LOST contact (support removed)
  // Effective inverse-mass fraction of a sleeper for the solve (0 = exactly immovable). A small
  // POSITIVE value keeps the sleeper very heavy but not infinitely rigid, so an awake body wedged at
  // a frozen-pocket boundary can relieve against it instead of the PGS normal impulse diverging
  // (trapped-between-two-rigid-constraints blow-up that a settling column reliably hit, ejected via
  // the friction cone to NaN); the sleeper's velocity is re-zeroed each substep so no momentum
  // accumulates and both-asleep interior manifolds are still fully excluded (the speed win). 0.01 =
  // sleeper 100x a grain's mass: stable through the 96k column + violent pour, case3 penetration
  // and the settled-bed freeze both preserved. PECLET_DEM_SLEEP_INVMASS_FRAC overrides.
  float sleepImmovableFrac = 0.01f;
  bool extForceActive = false;        // CFD-DEM drag present -> sleeping disabled this step
  bool extTorqueActive = false;       // external couple present -> sleeping disabled this step
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
  float verletSkinFrac = 0.0f;  // PECLET_DEM_VERLET_SKIN; 0 = off

  // --- atomic counters / scalars (rank-0 Views) ---
  Kokkos::View<int, CpMem> pairCount, contactCount, manifoldCount, topGhost;
  Kokkos::View<float, CpMem> maxOverlap;
  // Max physical approach speed among approaching manifolds in the last velocity sweep — drives the
  // colored-GS velocity loop's adaptive stop (converged once no pair approaches above the resting
  // threshold). maxOverlap plays the same role for the position loop.
  Kokkos::View<float, CpMem> maxApproach;
  // Quasi-static share of maxApproach (corrections on contacts with |vn0| <= 4 vRest): the
  // multilevel stabilization loop's stop criterion -- flowing scenes keep ballistic churn out
  // of it, so the pass ends after ~one cycle instead of burning its full budget as an
  // over-convergence brake on discharge.
  Kokkos::View<float, CpMem> maxApproachQS;

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
    maxApproach = Kokkos::View<float, CpMem>("maxApproach");
    maxApproachQS = Kokkos::View<float, CpMem>("maxApproachQS");
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
    capacity = newCap;
  }

  // Const views for the read-only kernel inputs.
  Kokkos::View<const float* [3], CpMem> cpos() const { return pos; }
  Kokkos::View<const float*, CpMem> crad() const { return rad; }
};

}  // namespace peclet::dem

#endif  // DEM_PARTICLES_HPP
