from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
from functools import partial
import jax
from jax import numpy as jnp, jit
import jax_dataclasses as jdc

from wpsim import spatial, maths
from wpsim.soft import state, libgeo

"""as a layer on top / adjacent to the kinematic sim, we design a softbody
tet sim to model muscles, fascia, fat and skin in one continuum.

The sim is also differentiable to allow tracing target shapes as close as 
possible back to sim parametres.

this is a bit more intense than the rigid case, so instead of xpbd we use a 
matrix-free fem solution as suggested in diffpd

"""


def computeLbs(params:state.SimStaticParams, frameData:state.FrameStaticData):
	"""
	Computes Linear Blend Skinning on the GPU to get targetKinematicPos.
	Uses translation and quaternion (orient) for joints.
	"""
	# Expand weights and indices
	# skinWeights: (nV, nSW), skinIndices: (nV, nSW)
	w = maths.uint8ToFloat16(params.skinWeights)  # float16
	idx = params.skinIndices  # int16

	# Gather joint data
	t = frameData.jointTranslates[idx]  # (nV, nSW, 3)
	q = frameData.jointOrients[idx]  # (nV, nSW, 4)

	# Simple GPU-Quat-Vector multiplication: q * p_rest * inv(q) + t
	def applyTransform(pos, q, t):
		# Standard quaternion rotation logic
		uv = 2.0 * jnp.cross(q[..., :3], pos)
		uuv = 2.0 * jnp.cross(q[..., :3], uv)
		return pos + q[..., 3] * uv + uuv + t

	# Apply to all weight slots
	pRest = params.restPositions[:, None, :]  # (nV, 1, 3)
	transformed = jax.vmap(jax.vmap(applyTransform))(pRest, q, t)

	# Weighted average
	return jnp.sum(transformed * w[..., None], axis=1)


@jax.vmap
def interpolateMuscleDm(activation, neutralDm, flexedDms, keys):
	"""
	Performs piece-wise linear interpolation of DmInv matrices.
	activation: scalar
	neutralDm: (3, 3)
	flexedDms: (nKeys, 3, 3)
	keys: (nKeys,) uint8 activation levels for each sculpt
	"""
	# We find the interval. i.e., keys[i] <= activation < keys[i+1]
	# We treat the neutralDm as Key -1 at activation 0.0

	# Search for the right interval
	idx = jnp.searchsorted(keys, activation) - 1

	# Define bounds for interpolation
	# Lower bound
	lowKey = jnp.where(idx == -1, 0.0, keys[idx])
	lowDm = jnp.where(idx == -1, neutralDm, flexedDms[idx])

	# Upper bound
	highKey = keys[idx + 1]
	highDm = flexedDms[idx + 1]

	# Calculate t [0, 1]
	t = (activation - lowKey) / (highKey - lowKey + 1e-6)
	t = jnp.clip(t, 0.0, 1.0)

	return (1.0 - t) * lowDm + t * highDm


def computeHolisticParams(simParams:state.SimStaticParams,
						  frameData:state.FrameStaticData
						  ):
	"""
	Calculates the substep-static parameters for the entire continuum.
	"""
	# 1. Calculate Skinning for Pinning
	targetPos = computeLbs(simParams, frameData)

	# 2. Derive Muscle Activations from NURBS Strands
	# strandActivations: (numStrands,)
	# tetActivations: (nM,)
	# strandIndices: (nM, nMaxStrands), strandWeights: (nM, nMaxStrands)
	strandActivations = maths.uint8ToFloat16(frameData.strandActivations)
	strandWeights = maths.uint8ToFloat16(simParams.strandWeights)
	muscleActivations = jnp.sum(
		strandActivations[
			simParams.strandIndices] * strandWeights,
		axis=1
	)

	# 3. Interpolate Muscle DmInv (Only for the first nM tets)
	flexedActivationKeys = maths.uint8ToFloat16(simParams.flexedActivationKeys)
	muscleDmInv = interpolateMuscleDm(
		muscleActivations,
		simParams.dmInvNeutral[:simParams.nMuscleTets],
		simParams.dmInvFlexed,
		flexedActivationKeys
	)

	# 4. Concatenate with Passive Tissue DmInv
	# Non-muscle tets use their neutral matrix throughout
	passiveDmInv = simParams.dmInvNeutral[simParams.nMuscleTets:]
	fullCurrentDmInv = jnp.concatenate([muscleDmInv, passiveDmInv], axis=0)

	return muscleActivations, fullCurrentDmInv, targetPos

@jax.vmap
def deriveDmInv(restPos, indices):
	"""
	restPos: (V, 3) vertex positions in the rest pose.
	indices: (T, 4) vertex indices for each tetrahedron.
	"""
	# Gather the four vertices of the tet
	v = restPos[indices]  # Shape: (4, 3)

	# Calculate edge vectors from v0
	# e1 = v1 - v0, e2 = v2 - v0, e3 = v3 - v0
	dm = jnp.stack([
		v[1] - v[0],
		v[2] - v[0],
		v[3] - v[0]
	], axis=1)  # Shape: (3, 3)

	# Compute the inverse
	# We add a tiny epsilon to the diagonal to ensure invertibility
	# if the authoring tool accidentally produced a degenerate (flat) tet.
	dmInv = jnp.linalg.inv(dm + jnp.eye(3) * 1e-10)

	# Calculate rest volume: V = 1/6 * |det(Dm)|
	volume = jnp.abs(jnp.linalg.det(dm)) / 6.0

	return dmInv, volume


def precomputeTetData(restPos, tetIndices):
	"""
	Main helper to prepare the 'data' dict for the simulation.
	"""
	dmInvs, volumes = deriveDmInv(restPos, tetIndices)

	return {
		"dmInv": dmInvs,
		"volWeights": volumes,
		"indices": tetIndices
	}

@jax.vmap
def computeF(pos, dm_inv, indices):
	"""
	overall deformation gradient of a tet
	pos: (V, 3) current vertex positions
	dm_inv: (T, 3, 3) precomputed inverse rest-shape matrix
	indices: (T, 4) tet indices
	"""
	v = pos[indices] # (4, 3)
	# Edge vectors: [v1-v0, v2-v0, v3-v0]
	Ds = jnp.stack([v[1] - v[0], v[2] - v[0], v[3] - v[0]], axis=1)
	return Ds @ dm_inv # (T, 3, 3) @ (T, 3, 3)


@jax.vmap
def polar_decomp_3x3(F):
	"""
	Extracts Rotation (R) from Deformation Gradient (F) using SVD.
	F = U * diag(S) * Vt  =>  R = U * Vt
	"""
	U, S, Vt = jnp.linalg.svd(F)
	R = U @ Vt
	# Reflection check to ensure a valid rotation matrix (det=1)
	det = jnp.linalg.det(R)
	U_modified = U.at[:, 2].multiply(jnp.where(det < 0, -1.0, 1.0))
	R = U_modified @ Vt
	return R, S

def neoHookeanEnergy(F, mu, kappa):
	J = jnp.linalg.det(F)
	Ic = jnp.sum(jnp.square(F)) # Trace of C
	# Stable Neo-Hookean formulation
	energy = (mu / 2) * (Ic - 3) - mu * (J - 1) + (kappa / 2) * jnp.square(J - 1)
	return energy


def muscleActiveEnergy(F, a0, alpha, sigma_max):
	"""
	a0: (3,) rest fiber direction
	alpha: (1,) activation [0, 1]
	sigma_max: Peak isometric stress
	"""
	# Current fiber vector in deformed space
	a_curr = F @ a0
	lam = jnp.linalg.norm(a_curr)  # Stretch ratio lambda

	# Simple Active Tension model: Energy = 0.5 * k * (lam - 1)^2 * alpha
	# In a production Blemker model, this would be a spline lookup
	# representing the Force-Length curve.
	active_k = sigma_max * alpha
	energy = 0.5 * active_k * jnp.square(lam - 1.0)
	return energy


@jax.vmap
def computeBendingEnergy(
		fA: jnp.ndarray[3, 3],
		fB: jnp.ndarray[3, 3],
		rA: jnp.ndarray[3, 3],
		rB: jnp.ndarray[3, 3],
		kBend: float
) -> jnp.ndarray:
	"""
	Penalizes the difference in rotation between two adjacent elements.
	rA, rB: Rotational components from PolarDecomp3x3.
	simpler and cheaper general form of bending energy
	"""
	# Relative rotation between adjacent elements
	# In a rod/cloth, this captures bending and twisting
	relRot = rA.T @ rB

	# Logarithmic rotation error (approximate)
	# trace(relRot) is 3 for identical rotations
	error = 3.0 - jnp.trace(relRot)

	return 0.5 * kBend * error ** 2


@jax.vmap
def discreteShellBending(
		pos: jnp.ndarray[nV, 3],
		edgeIndices: jnp.ndarray[nClothEdges, 4],
		kBend: float,
		restAngle: float
) -> float:
	"""
	Calculates bending energy for two triangles sharing an edge.
	If you've seen Vellum's cloth constraint primitives, with the zigzag
	across each edge, this is the same thing.
	edgeIndices: [v0, v1, v2, v3] where v0-v1 is the shared edge.
	v2 and v3 are the 'wing' vertices.
	"""
	x0, x1, x2, x3 = pos[edgeIndices]

	e0 = x1 - x0  # Shared edge
	e1 = x2 - x0
	e2 = x3 - x0

	# Compute normals
	n1 = jnp.cross(e0, e1)
	n2 = jnp.cross(e2, e0)  # Winding to ensure normals point 'out'

	n1Len = jnp.linalg.norm(n1)
	n2Len = jnp.linalg.norm(n2)

	# Normalization
	n1 = n1 / (n1Len + 1e-10)
	n2 = n2 / (n2Len + 1e-10)

	# Dihedral Angle components: cos(theta) and sin(theta)
	cosTheta = jnp.clip(jnp.dot(n1, n2), -1.0, 1.0)

	# Energy: Quadratic penalty on the angle difference
	# Using (1 - cos(theta - rest)) is more stable than acos()
	# For simplicity: penalizing the dot product difference
	error = cosTheta - jnp.cos(restAngle)
	return 0.5 * kBend * (error ** 2)


@jax.vmap
def torsionalTwistEnergy(
		rA: jnp.ndarray[3, 3],
		rB: jnp.ndarray[3, 3],
		kTorsion: float,
		restRelRot: jnp.ndarray[3, 3]
) -> float:
	"""
	Penalizes relative rotation (twist) between two adjacent tet frames.
	We attempt to emulate a cosserat rod by propagating torsion
	rA, rB: Rotational components from PolarDecomp3x3.
	restRelRot: Precomputed rest relative rotation.
	"""
	# Compute the current relative rotation
	# currentRelRot = inv(rA) * rB
	currentRelRot = rA.T @ rB

	# Difference between current and rest relative rotations
	# diffRot: jnp.ndarray[3, 3]
	diffRot = currentRelRot @ restRelRot.T

	# The 'Angle' of the difference rotation
	# trace(R) = 1 + 2*cos(theta)
	# So, (trace(R) - 1) / 2 is the cosine of the error angle
	cosError = (jnp.trace(diffRot) - 1.0) / 2.0
	cosError = jnp.clip(cosError, -1.0, 1.0)

	# Energy = 0.5 * k * theta^2
	# Approximate theta^2 with (1 - cosError) for speed/stability
	return kTorsion * (1.0 - cosError)


def computeSuctionPotential(pos, slidingVertexIndices, gridData, surfaceTris,
							data):
	"""
	Calculates the total potential energy generated by suction/sliding interfaces.
	slidingVertexIndices: (numSlidingVertices,) indices of skin/fascia vertices.
	"""
	skinPoints = pos[slidingVertexIndices]

	# Find nearest points on the muscle/bone surfaces
	# This uses our vmapped spatial query
	results = spatial.querySpatialNearest(skinPoints, gridData, surfaceTris,
										 data)

	# Potential Energy = 0.5 * k * distanceSq
	# The distanceSq is returned directly from our projection kernel
	suctionEnergy = 0.5 * data["suctionStiffness"] * jnp.sum(
		results["bestDistSq"])

	return suctionEnergy


def total_system_potential(pos, data):
	"""
	returns total energy of the system - this is the objective function to minimize
	pos: (V, 3) current positions
	data: dict containing:
		- indices: (T, 4)
		- dm_inv: (T, 3, 3)
		- fiber_dirs: (M, 3) only for muscle tets
		- activations: (M,)
		- mu, kappa: (T,)
		- vol_weights: (T,) precomputed tet volumes
	"""
	# 1. Compute all Deformation Gradients
	Fs = computeF(pos, data['dm_inv'], data['indices'])

	# 2. Passive Energy (Apply to ALL tets)
	passive_e = jax.vmap(neoHookeanEnergy)(Fs, data['mu'], data['kappa'])

	# 3. Active Muscle Energy (Apply only to first M tets)
	# We slice Fs to only include the muscle tets
	muscle_Fs = Fs[:data['num_muscles']]
	active_e = jax.vmap(muscleActiveEnergy)(
		muscle_Fs,
		data['fiber_dirs'],
		data['activations'],
		data['sigma_max']
	)

	# 4. Sum up (weighted by rest volume)
	# Note: active_e is padded with zeros for non-muscle tets to keep shapes static
	total_passive = jnp.sum(passive_e * data['vol_weights'])
	total_active = jnp.sum(active_e * data['vol_weights'][:data['num_muscles']])

	# 5. External Potential (Gravity / Skeleton Pinning)
	# Pinning: 0.5 * k * ||pos - anim_pos||^2
	pinning_e = 0.5 * data['pin_k'] * jnp.sum(
		jnp.square(pos - data['target_pos']) * data['pin_mask'])

	return total_passive + total_active + pinning_e


def ComputeHvp(Pos, Data, Vector):
	"""
	Computes Hessian-Vector Product: H(Pos) * Vector
	"""

	def GradientFunc(P):
		return jax.grad(total_system_potential)(P, Data)

	_, Hvp = jax.jvp(GradientFunc, (Pos,), (Vector,))
	return Hvp


def SolveNewtonStep(Pos, Data):
	"""
	Performs one Newton iteration using Matrix-Free Conjugate Gradient.
	"""
	# 1. Compute current forces (Negative Gradient)
	Forces = -jax.grad(total_system_potential)(Pos, Data)

	# 2. Define the Linear Operator for CG: A(v) = H * v
	def LinearOperator(V):
		# Apply HVP
		H_v = ComputeHvp(Pos, Data, V)
		# Add a small Tikhonov regularization (Damping) for stability
		return H_v + Data["Damping"] * V

	# 3. Solve the system H * DeltaX = Forces
	# We use a padded CG or jax.scipy.sparse.linalg.cg
	DeltaX, _ = jax.scipy.sparse.linalg.cg(
		LinearOperator,
		Forces,
		tol=Data["SolverTol"],
		maxiter=Data["MaxCgIter"]
	)

	# 4. Update positions
	NewPos = Pos + Data["StepSize"] * DeltaX
	return NewPos


@partial(jax.jit, static_argnames=("newtonIter", "cgIter", "maxBucketSearch",
                                   "runCloth", "runRope"))
def simulationStep(
		simParams:state.SimStaticParams,
		frameState:state.FrameStaticData,
		substepState:state.SubstepStaticData,
		dynamicState:state.DynamicState,
		data,
		newtonIter=2,
		cgIter=20,
		maxBucketSearch=16,
		runCloth=True,
		runRope=True,
)->state.DynamicState:
	"""
	Main simulation entry point for a single frame/substep.
	state: dict {"pos", "vel"}
	data: dict containing topology, material params, and buffers.
	"""
	dt = data["dt"]
	prevPos = dynamicState.pos
	velocity = dynamicState.vel

	# --- 1. PREDICTION STEP (Inertia) ---
	# predictedPos serves as the initial guess for the Newton solver
	predictedPos = prevPos + velocity * dt

	# --- 2. SPATIAL GRID BUILD ---
	# We build the grid once at the start of the step.
	# For ultra-high precision sliding, move this inside the Newton loop.
	gridData = spatial.buildGlobalSpatialGrid(data["surfaceTris"], data)

	# --- 3. TOTAL POTENTIAL FUNCTION ---
	def computeTotalEnergy(currentPos):
		# 1. Standard Volumetric (All Tets)
		# ... (neoHookean logic) ...

		# 2. Muscle Active Tension
		# ... (Blemker logic) ...

		# Passive Elasticity (Neo-Hookean)
		fs = computeF(currentPos, data["dmInv"], data["indices"])
		passiveE = jnp.sum(
			jax.vmap(neoHookeanEnergy)(fs, data["mu"], data["kappa"]) * data[
				"volWeights"])

		# Active Muscle (Blemker)
		# Only process the first M tets which are muscles
		mFs = fs[:data["numMuscles"]]
		activeE = jnp.sum(jax.vmap(muscleActiveEnergy)(
			mFs, data["fiberDirs"], data["activations"], data["sigmaMax"]
		) * data["volWeights"][:data["numMuscles"]])

		# Sliding Suction
		suctionE = computeSuctionPotential(
			currentPos, data["slidingIndices"], gridData, data["surfaceTris"],
			data, maxBucketSearch
		)

		# Kinematic Pinning (Bone attachment)
		pinningE = 0.5 * data["pinK"] * jnp.sum(
			jnp.square(currentPos - data["targetPos"]) * data["pinMask"]
		)

		potential = passiveE + activeE + suctionE + pinningE

		if runCloth:
			# 3. Cloth Bending (Surface-based)
			# Only runs if clothEdgeIndices is not empty
			clothE = discreteShellBending(
				currentPos, data.clothEdgeIndices, data.kBendCloth,
				data.clothRestAngles
			)
			potential += clothE

		if runRope:
			# 4. Rope Torsion (Element-based)
			# Requires R matrices, computed only for muscle/rope tets
			# subsetRs: jnp.ndarray[nActive, 3, 3]
			subsetRs = libgeo.computeSubsetRotations(
				currentPos, data.dmInvNeutral, data.indices, data.activeRotIndices
			)

			# 2. Rope Torsion (using subsetRs)
			# ropePairA/B are indices into subsetRs
			rA = subsetRs[data.ropeSubsetPairs[:, 0]]
			rB = subsetRs[data.ropeSubsetPairs[:, 1]]
			ropeE = torsionalTwistEnergy(rA, rB, data.kTorsionRope,
			                             data.ropeRestRelRots)
			potential += ropeE

		return potential

	# --- 4. NEWTON-CG SOLVER ---
	def newtonIteration(currentPos, _):
		# Calculate Forces (Negative Gradient)
		forces = -jax.grad(computeTotalEnergy)(currentPos)

		# Linear Operator for CG: A(v) = H * v
		def linearOperator(v):
			# Hessian-Vector Product (HVP)
			_, hvp = jax.jvp(jax.grad(computeTotalEnergy), (currentPos,), (v,))
			# Add damping for numerical stability
			return hvp + data["damping"] * v

		# Conjugate Gradient Solver (solve H * deltaX = forces)
		# We use our own fixed-iteration CG to ensure JIT-compatibility
		def cgBody(cgState, _):
			r, p, x, r_sq = cgState
			ap = linearOperator(p)
			alpha = r_sq / (jnp.dot(p.flatten(), ap.flatten()) + 1e-10)
			x = x + alpha * p
			r = r - alpha * ap
			new_r_sq = jnp.dot(r.flatten(), r.flatten())
			beta = new_r_sq / r_sq
			p = r + beta * p
			return (r, p, x, new_r_sq), None

		r0 = forces
		p0 = r0
		x0 = jnp.zeros_like(currentPos)
		r_sq0 = jnp.dot(r0.flatten(), r0.flatten())

		(final_r, final_p, deltaX, _), _ = jax.lax.scan(
			cgBody, (r0, p0, x0, r_sq0), None, length=cgIter
		)

		# Update position
		updatedPos = currentPos + deltaX
		return updatedPos, None

	# Run fixed number of Newton steps
	finalPos, _ = jax.lax.scan(newtonIteration, predictedPos, None,
							   length=newtonIter)

	# --- 5. VELOCITY UPDATE ---
	newVel = (finalPos - prevPos) / dt
	# Apply global damping to stop oscillations
	newVel = newVel * data["velocityDamping"]

	return state.DynamicState(
		pos=finalPos, vel=newVel
	)


def calculateSculptLoss(
		params,
		dynamicState:state.DynamicState,
		targetPos,
		targetVel,
		simData,
		gamma=0.1
):
	"""
	Minimizes exact vertex correspondence
	params: activations/stiffness we are optimizing.
	targetPos, targetVel: The artistic goal.
	"""
	# 1. Run the forward simulation (Step)
	# This is differentiable!
	nextState = simulationStep(dynamicState, {**simData, "activations": params})

	# 2. Position Error
	posError = jnp.sum(jnp.square(nextState.pos - targetPos))

	# 3. Velocity Error (Penalize snapping/unnatural speed)
	velError = jnp.sum(jnp.square(nextState.vel - targetVel))

	return posError + gamma * velError


def calculateSilhouetteLoss(pos, targetGrid, targetSurfaceTris, simData):
	"""
	Minimizes distance from simulated skin vertices to the target sculpt
	manifold, without pinning exact vertices, still allowing freedom in tangent
	plane/skin sliding

	TODO: still include vel penalties here
	"""
	skinPos = pos[simData.outerTriIndices]

	# Query the grid built from the TARGET sculpt mesh
	# We stop_gradient on the grid result to ensure we only differentiate
	# with respect to the simulated vertex positions, not the search logic.
	queryResults = spatial.querySpatialNearest(skinPos, targetGrid,
										  targetSurfaceTris,
									   simData)

	# The 'targetPoints' are treated as static destinations for this iteration
	targetPoints = jax.lax.stop_gradient(queryResults["bestPoint"])

	# Point-to-Surface loss
	distSq = jnp.sum(jnp.square(skinPos - targetPoints))
	return distSq

def fitSculptParameters(
		initialParams,
		initialState,
		targetPos,
		targetVel,
		simData,
		iterations=10):
	"""
	Uses JAX gradients to find the best muscle activations to hit a sculpt.
	"""

	def step(p, _):
		grads = jax.grad(calculateSculptLoss)(p, initialState, targetPos,
											  targetVel, simData)
		# Simple gradient descent for demonstration;
		# Production would use L-BFGS or a custom Newton step.
		return p - 0.01 * grads, None

	finalParams, _ = jax.lax.scan(step,
								  initialParams,
								  None,
								  length=iterations)
	return finalParams


def registerSculptToLocal(sculptWorldPos, simEquilibriumPos, simStatic,
						  smoothing=1):
	"""
	Bakes a world-space sculpt into local frames.
	Run this once during authoring/fitting.
	"""
	# 1. Get the frames at the equilibrium/neutral state
	frames = libgeo.computeStableVertexFrames(simEquilibriumPos, simStatic,
										 smoothing)

	# 2. Calculate the world-space delta
	worldDelta = sculptWorldPos - simEquilibriumPos

	# 3. Project delta into the local frame: Delta_local = R^T * Delta_world
	# frames is (nV, 3, 3), worldDelta is (nV, 3)
	localDelta = jax.vmap(lambda r, d: r.T @ d)(frames, worldDelta)

	return localDelta


@jax.vmap
def applyResidualDeltas(currentSimPos, localDelta, currentFrames, weight):
	"""
	Converts local deltas back to world-space based on the current sim state.
	Pos_final = Pos_sim + (Weight * (R_curr * Delta_local))
	"""
	# Rotate local delta back to world space using current frame
	worldDelta = currentFrames @ localDelta
	return currentSimPos + (weight * worldDelta)


def applySculptSubset(pos,
					  sculpt:state.SculptTarget, simIndices, currentWeight):
	"""
	Only computes and applies deltas for a subset of the mesh.
	"""
	# 1. Gather current frames for ONLY the guide tets
	subsetFrames = libgeo.computeSubsetFrames(
		pos,
		sculpt.guideTetIndices,
		simIndices
	)

	# 2. Rotate local deltas to world space
	# subsetFrames: (nImpacted, 3, 3), localDeltas: (nImpacted, 3)
	worldDeltas = jax.vmap(lambda r, d: r @ d)(subsetFrames, sculpt.localDeltas)

	# 3. Apply to the global position buffer
	# Use 'at().add()' for JAX-friendly indexed updates
	weightedDeltas = worldDeltas * sculpt.weights[:, None] * currentWeight
	updatedPos = pos.at[sculpt.affectedIndices].add(weightedDeltas)

	return updatedPos




