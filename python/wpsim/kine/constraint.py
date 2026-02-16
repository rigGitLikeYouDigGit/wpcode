from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from wpsim import maths
from wpsim.kine import state


@jdc.pytree_dataclass
class ConstraintState:
	settings: ConstraintSolveSettings
	point: PointConstraintBucket
	hingeAxis: HingeJointConstraintBucket
	#orient: OrientationDriveConstraintBucket

	# Optional: measurement-based constraints with internal DOFs
	measurements: state.MeasurementState | None = None
	measurementConstraints: MeasurementConstraintBucket | None = None

	# Optional: manifold constraints
	manifold: ManifoldConstraintBucket | None = None


constraintAccumFn = Callable[
	[state.SubstepBoundData, state.DynamicData, ConstraintState, float],
	Tuple[state.DynamicData, ConstraintState]
]

"""attempt more customisation here - 
is there ever a case where a constraint weight map function wouldn't
just output a single DOF? don't think there's anywhere we would need
the DOFs themselves coupled, duplicating the same code for a ramp 3 times 
over is fine since it gets vmapped anyway

don't forget nurbs constraints, surface constraints etc
"""

constraintAlphaMapFn = Callable[
	[state.SubstepBoundData,
	 ConstraintState,
	 int,  # constraint type index
	 int,  # constraint index
	 int  # DOF index
	 ],
	float
]

@jdc.pytree_dataclass(frozen=True)
class ConstraintPlan:
	"""frozen dataclass to pass constraint pipelines"""
	accumulators: tuple[constraintAccumFn, ...]


"""constraint buckets each store Alpha arrays, equivalent in each entry to
compliance / dt^2
inheritance used here to help organisation and enforce common signatures - 
we don't do run-time polymorphism in the loop, accumulate functions
are explicitly passed as static args at jit-time

"""
@jdc.pytree_dataclass
class ConstraintSolveSettings:
	"""
	Static-ish settings that affect solver behavior.
	Keep shapes static; scalars are fine.
	"""
	iterationCount: int
	substepCount: int
	dt: float


@jdc.pytree_dataclass
class XpbdLambda1:
	"""
	Lambda storage for constraints with a single scalar residual.
	full dataclasses here may not be necessary, I think they get
	compiled out by jax anyway
	"""
	Lambda: jnp.ndarray  # (M,)


@jdc.pytree_dataclass
class XpbdLambdaN:
	"""
	Lambda storage for constraints with small vector residuals (e.g. 2-3 DOF).
	MaxDof is fixed per bucket.
	"""
	Lambda: jnp.ndarray  # (M, MaxDof)

@jdc.pytree_dataclass
class XpbdLambda2:
	lambdaValue: jnp.ndarray	# (m, 2)

@jdc.pytree_dataclass
class XpbdLambda3:
	lambdaValue: jnp.ndarray	# (m, 3)

@jdc.pytree_dataclass
class ConstraintBucket:
	pass

	@classmethod
	def accumulate(
			cls,
			bs: state.SubstepBoundData, bufs: state.DynamicData,
			#cs: ConstraintState,
			bucket: PointConstraintBucket, dt: float) -> tuple[
		state.DynamicData, ConstraintState]:
		"""accumulate also has to replace the right bucket in
		new ConstraintState
		"""
		raise NotImplementedError

@jdc.pytree_dataclass
class PointConstraintBucket(ConstraintBucket):
	"""
	3-DOF point-to-point constraint between two bodies:
		worldPoint(bodyA, localAnchorA) == worldPoint(bodyB, localAnchorB)
	"""
	isActive: jnp.ndarray		# (m,) bool/int mask

	bodyA: jnp.ndarray			# (m,) int32
	bodyB: jnp.ndarray			# (m,) int32

	localPointA: jnp.ndarray	# (m, 3) float32
	localPointB: jnp.ndarray	# (m, 3) float32

	alpha: jnp.ndarray		# (m,) float32
	weight: jnp.ndarray			# (m,) float32

	lambdaState: XpbdLambda3	# lambdaState.lambdaValue: (m, 3)

	@classmethod
	def accumulate(
			cls,
			bs: state.SubstepBoundData, bufs: state.DynamicData,
			#cs: ConstraintState,
			bucket:PointConstraintBucket, dt: float) -> tuple[
		state.DynamicData, #ConstraintState
		PointConstraintBucket
	]:
		"""
		Accumulates anchor constraint corrections into shared buffers and updates XPBD lambdas.

		bs: snapshot body state (not modified here)
		bufs: shared per-body buffers to add into
		bucket: anchor constraints + lambda
		dt: unused here because bucket.alpha is already computed for the actual substep dt
		"""
		del dt

		active = bucket.isActive.astype(jnp.float32)	# (m,)
		m = active.shape[0]

		aIdx = bucket.bodyA.astype(jnp.int32)			# (m,)
		bIdx = bucket.bodyB.astype(jnp.int32)			# (m,)

		# Gather body transforms
		xA = bs.position[aIdx]							# (m,3)
		xB = bs.position[bIdx]							# (m,3)
		qA = bs.orientation[aIdx]						# (m,4)
		qB = bs.orientation[bIdx]						# (m,4)

		invMassA = bs.invMass[aIdx]						# (m,)
		invMassB = bs.invMass[bIdx]						# (m,)

		invInertiaA = bs.invInertiaBody[aIdx]			# (m,3)
		invInertiaB = bs.invInertiaBody[bIdx]			# (m,3)

		# World anchor offsets from COM
		rA = maths.quatRotate(qA, bucket.localPointA)			# (m,3)
		rB = maths.quatRotate(qB, bucket.localPointB)			# (m,3)

		pA = xA + rA										# (m,3)
		pB = xB + rB										# (m,3)

		# Constraint residual (3 DOF)
		c = pB - pA										# (m,3)

		# Apply strength on residual
		w = bucket.weight.astype(bs.position.dtype)		# (m,)
		cWeighted = c * w[:, None]						# (m,3)

		# XPBD alpha
		alpha = bucket.alpha.astype(bs.position.dtype)	# (m,)

		# Current lambda (m,3)
		lambdaPrev = bucket.lambdaState.lambdaValue		# (m,3)

		# For a vector constraint, we solve 3 independent rows in world XYZ basis:
		# each row has direction n = e_x/e_y/e_z
		# effective mass per row: wEff = invMassA + invMassB + (rAxn)^T I^-1 (rAxn) + (rBxn)^T I^-1 (rBxn)
		#
		# We compute all 3 rows at once by taking n as the identity basis.

		n = jnp.eye(3, dtype=bs.position.dtype)[None, :, :]	# (1,3,3) rows are basis vectors
		n = jnp.repeat(n, m, axis=0)							# (m,3,3)

		# Cross terms: r x nRow -> (m,3rows,3vec)
		rAxn = jnp.cross(rA[:, None, :], n, axis=-1)		# (m,3,3)
		rBxn = jnp.cross(rB[:, None, :], n, axis=-1)		# (m,3,3)

		# Apply world inverse inertia to each row vector
		# reshape to (m*3,3) for vectorized application
		rAxnFlat = rAxn.reshape((m * 3, 3))
		rBxnFlat = rBxn.reshape((m * 3, 3))

		qAFlat = jnp.repeat(qA, 3, axis=0)					# (m*3,4)
		qBFlat = jnp.repeat(qB, 3, axis=0)					# (m*3,4)
		invInertiaAFlat = jnp.repeat(invInertiaA, 3, axis=0)	# (m*3,3)
		invInertiaBFlat = jnp.repeat(invInertiaB, 3, axis=0)	# (m*3,3)

		iInv_rAxn = maths.applyInvInertiaWorld(qAFlat, invInertiaAFlat,
		                                  rAxnFlat).reshape((m, 3, 3))	# (m,3,3)
		iInv_rBxn = maths.applyInvInertiaWorld(qBFlat, invInertiaBFlat,
		                                  rBxnFlat).reshape((m, 3, 3))	# (m,3,3)

		# Rotational scalar terms per row: dot(rxn, I^-1 * rxn)
		rotA = jnp.sum(rAxn * iInv_rAxn, axis=-1)			# (m,3)
		rotB = jnp.sum(rBxn * iInv_rBxn, axis=-1)			# (m,3)

		# Effective inverse mass per row (m,3)
		wEff = invMassA[:, None] + invMassB[:, None] + rotA + rotB

		# Prevent divide by zero for fully-kinematic pairs
		wEff = jnp.maximum(wEff, 1e-12)

		# XPBD update per component (row)
		# deltaLambda = -(C + alpha*lambda) / (wEff + alpha)
		#
		# alpha is scalar per 3-DOF constraint; broadcast to 3 components
		denom = wEff + alpha[:, None]						# (m,3)
		deltaLambda = -(cWeighted + alpha[:, None] * lambdaPrev) / denom	# (m,3)

		# Mask inactive constraints
		deltaLambda = deltaLambda * active[:, None]

		lambdaNew = lambdaPrev + deltaLambda				# (m,3)

		# Position corrections:
		# dPosA = -invMassA * sum_j (deltaLambda_j * n_j)
		# but n_j are basis vectors, so sum_j (deltaLambda_j * n_j) == deltaLambda (as a 3-vector)
		dPosA = -invMassA[:, None] * deltaLambda			# (m,3)
		dPosB =  invMassB[:, None] * deltaLambda			# (m,3)

		# Angular corrections (small-angle):
		# dThetaA = - I^-1 * sum_j (deltaLambda_j * (rA x n_j))
		# where (rA x n_j) are the rxn rows; same for B with opposite sign.
		#
		# We already have rxn per row; form tau = sum_j deltaLambda_j * rxn_j

		tauA = jnp.sum(rAxn * deltaLambda[:, :, None], axis=1)	# (m,3)
		tauB = jnp.sum(rBxn * deltaLambda[:, :, None], axis=1)	# (m,3)

		dThetaA = -maths.applyInvInertiaWorld(qA, invInertiaA, tauA)	# (m,3)
		dThetaB =  maths.applyInvInertiaWorld(qB, invInertiaB, tauB)	# (m,3)

		# Scatter-add into shared buffers
		posDelta = bufs.posDelta
		angDelta = bufs.angDelta

		posDelta = posDelta.at[aIdx].add(dPosA)
		posDelta = posDelta.at[bIdx].add(dPosB)

		angDelta = angDelta.at[aIdx].add(dThetaA)
		angDelta = angDelta.at[bIdx].add(dThetaB)

		newBufs = state.DynamicData(posDelta=posDelta, angDelta=angDelta)

		newBucket = PointConstraintBucket(
			isActive=bucket.isActive,
			bodyA=bucket.bodyA,
			bodyB=bucket.bodyB,
			localPointA=bucket.localPointA,
			localPointB=bucket.localPointB,
			alpha=bucket.alpha,
			weight=bucket.weight,
			lambdaState=XpbdLambda3(lambdaValue=lambdaNew),
		)
		# newState = ConstraintState(
		#
		# )

		return newBufs, newBucket


@jdc.pytree_dataclass
class HingeJointConstraintBucket(ConstraintBucket):
	"""
	2-DOF constraint that aligns two hinge axes (one on each body).
	This removes swing between the bodies while leaving twist about the axis unconstrained.

	Implementation detail (later): the residual can be represented in a 2D tangent basis
	orthogonal to the hinge axis, hence lambda is (m, 2).

	NB: we lift rotation into SO3 so that we track total accumulated
	rotations to bodies, not just +- pi.
	Initial rotations are inherited once at start time
	"""
	isActive: jnp.ndarray		# (m,)
	bodyA: jnp.ndarray			# (m,)
	bodyB: jnp.ndarray			# (m,)

	localHingeAxisA: jnp.ndarray	# (m,3)
	localHingeAxisB: jnp.ndarray	# (m,3)

	# Stateful joint coordinate
	unwrappedTwist: jnp.ndarray		# (m,)

	# Twist limits (in unwrapped space)
	twistMin: jnp.ndarray			# (m,)
	twistMax: jnp.ndarray			# (m,)

	alphaSwing: jnp.ndarray			# (m,)
	alphaTwist: jnp.ndarray			# (m,)
	weight: jnp.ndarray				# (m,)

	lambdaState: XpbdLambda3

	@classmethod
	def accumulate(
			cls,
			bs: state.SubstepBoundData, bufs: state.DynamicData,
			bucket: HingeJointConstraintBucket, dt: float) -> tuple[
		state.DynamicData, HingeJointConstraintBucket]:
		"""
			XPBD 2-DOF hinge-axis alignment accumulation.

			Assumptions:
			- localHingeAxisA/B are normalized (tooling ensures this).
			- invInertiaBody is diagonal in body frame.
			"""
		del dt

		active = bucket.isActive.astype(jnp.float32)
		m = active.shape[0]

		aIdx = bucket.bodyA.astype(jnp.int32)
		bIdx = bucket.bodyB.astype(jnp.int32)

		qA = bs.orientation[aIdx]
		qB = bs.orientation[bIdx]

		invInertiaA = bs.invInertiaBody[aIdx]
		invInertiaB = bs.invInertiaBody[bIdx]

		# ------------------------------------------------------------
		# World hinge axes
		# Assumption: local axes are normalized
		# ------------------------------------------------------------
		aW = maths.quatRotate(qA, bucket.localHingeAxisA)  # (m,3)
		bW = maths.quatRotate(qB, bucket.localHingeAxisB)  # (m,3)

		# Build tangent basis around aW (swing directions)
		t1, t2 = maths.buildTangentBasis(aW)  # (m,3), (m,3)

		# ------------------------------------------------------------
		# 1) Swing residual (2 DOF)
		# ------------------------------------------------------------
		cSwing = jnp.stack(
			[
				jnp.sum(t1 * bW, axis=-1),
				jnp.sum(t2 * bW, axis=-1),
			],
			axis=-1,
		)  # (m,2)

		cSwing = cSwing * bucket.weight[:, None]

		lambdaSwingPrev = bucket.lambdaState.lambdaValue[:, 0:2]
		alphaSwing = bucket.alphaSwing

		jwA1 = -jnp.cross(aW, t1)
		jwA2 = -jnp.cross(aW, t2)
		jwB1 = jnp.cross(bW, t1)
		jwB2 = jnp.cross(bW, t2)

		iInv_jwA1 = maths.applyInvInertiaWorld(qA, invInertiaA, jwA1)
		iInv_jwA2 = maths.applyInvInertiaWorld(qA, invInertiaA, jwA2)
		iInv_jwB1 = maths.applyInvInertiaWorld(qB, invInertiaB, jwB1)
		iInv_jwB2 = maths.applyInvInertiaWorld(qB, invInertiaB, jwB2)

		wEffSwing = jnp.stack(
			[
				jnp.sum(jwA1 * iInv_jwA1, axis=-1) + jnp.sum(jwB1 * iInv_jwB1,
				                                             axis=-1),
				jnp.sum(jwA2 * iInv_jwA2, axis=-1) + jnp.sum(jwB2 * iInv_jwB2,
				                                             axis=-1),
			],
			axis=-1,
		)
		wEffSwing = jnp.maximum(wEffSwing, 1e-12)

		deltaLambdaSwing = -(cSwing + alphaSwing[:, None] * lambdaSwingPrev) / (
					wEffSwing + alphaSwing[:, None])
		deltaLambdaSwing = deltaLambdaSwing * active[:, None]

		lambdaSwingNew = lambdaSwingPrev + deltaLambdaSwing

		# Angular swing corrections
		tauSwingA = deltaLambdaSwing[:, 0:1] * jwA1 + deltaLambdaSwing[
			:, 1:2] * jwA2
		tauSwingB = deltaLambdaSwing[:, 0:1] * jwB1 + deltaLambdaSwing[
			:, 1:2] * jwB2

		dThetaSwingA = -maths.applyInvInertiaWorld(qA, invInertiaA, tauSwingA)
		dThetaSwingB = maths.applyInvInertiaWorld(qB, invInertiaB, tauSwingB)

		# ------------------------------------------------------------
		# 2) Twist measurement + unwrap
		# ------------------------------------------------------------
		qRel = maths.quatMul(maths.quatConj(qA), qB)

		v = qRel[:, 0:3]
		vPar = jnp.sum(v * aW, axis=-1, keepdims=True) * aW
		qTwist = maths.quatNormalize(jnp.concatenate([vPar, qRel[:, 3:4]], axis=-1))

		rawTwist = 2.0 * jnp.arctan2(
			jnp.sum(qTwist[:, 0:3] * aW, axis=-1),
			qTwist[:, 3]
		)

		prevTwist = bucket.unwrappedTwist
		dTwist = maths.shortestAngleDelta(rawTwist - prevTwist)
		newTwist = prevTwist + dTwist

		# ------------------------------------------------------------
		# 3) Twist limits (1 DOF)
		# ------------------------------------------------------------
		violMin = newTwist - bucket.twistMin
		violMax = newTwist - bucket.twistMax

		cTwist = jnp.where(
			violMin < 0.0, violMin,
			jnp.where(violMax > 0.0, violMax, 0.0)
		)

		cTwist = cTwist * bucket.weight * active

		jwA = -aW
		jwB = aW

		iInv_jwA = maths.applyInvInertiaWorld(qA, invInertiaA, jwA)
		iInv_jwB = maths.applyInvInertiaWorld(qB, invInertiaB, jwB)

		wEffTwist = (
				jnp.sum(jwA * iInv_jwA, axis=-1) +
				jnp.sum(jwB * iInv_jwB, axis=-1)
		)
		wEffTwist = jnp.maximum(wEffTwist, 1e-12)

		lambdaTwistPrev = bucket.lambdaState.lambdaValue[:, 2]
		alphaTwist = bucket.alphaTwist

		deltaLambdaTwist = -(cTwist + alphaTwist * lambdaTwistPrev) / (
					wEffTwist + alphaTwist)
		deltaLambdaTwist = deltaLambdaTwist * (
					(violMin < 0.0) | (violMax > 0.0)) * active

		lambdaTwistNew = lambdaTwistPrev + deltaLambdaTwist

		dThetaTwistA = -(iInv_jwA * deltaLambdaTwist[:, None])
		dThetaTwistB = (iInv_jwB * deltaLambdaTwist[:, None])

		# ------------------------------------------------------------
		# Accumulate angular corrections
		# ------------------------------------------------------------
		angDelta = bufs.angDelta

		angDelta = angDelta.at[aIdx].add(dThetaSwingA + dThetaTwistA)
		angDelta = angDelta.at[bIdx].add(dThetaSwingB + dThetaTwistB)

		newBufs = state.DynamicData(
			posDelta=bufs.posDelta,
			angDelta=angDelta
		)

		newLambda = jnp.concatenate(
			[lambdaSwingNew, lambdaTwistNew[:, None]],
			axis=-1
		)

		newBucket = HingeJointConstraintBucket(
			isActive=bucket.isActive,
			bodyA=bucket.bodyA,
			bodyB=bucket.bodyB,
			localHingeAxisA=bucket.localHingeAxisA,
			localHingeAxisB=bucket.localHingeAxisB,
			unwrappedTwist=newTwist,
			twistMin=bucket.twistMin,
			twistMax=bucket.twistMax,
			alphaSwing=bucket.alphaSwing,
			alphaTwist=bucket.alphaTwist,
			weight=bucket.weight,
			lambdaState=XpbdLambda3(lambdaValue=newLambda),
		)

		return newBufs, newBucket

@jdc.pytree_dataclass
class OrientationDriveConstraintBucket:
	"""
	3-DOF orientation drive:
	Drive the relative orientation between two local frames (one on each body)
	toward a target quaternion.

	Common rig interpretation:
		worldRot(bodyA) * localFrameA  should match  worldRot(bodyB) * localFrameB * targetRel

	Exact residual definition (log-map, swing-twist, etc.) can be chosen later;
	this bucket just defines the fixed data layout.
	"""
	isActive: jnp.ndarray  # (m,)
	bodyA: jnp.ndarray  # (m,)
	bodyB: jnp.ndarray  # (m,)

	# Local frames in each body (unit quaternions)
	localFrameA: jnp.ndarray  # (m,4)
	localFrameB: jnp.ndarray  # (m,4)

	# Stateful joint coordinate (ℝ³)
	unwrappedRotation: jnp.ndarray  # (m,3)

	# Target in joint space (ℝ³), often zero
	targetRotation: jnp.ndarray  # (m,3)

	alpha: jnp.ndarray  # (m,)
	weight: jnp.ndarray  # (m,)

	lambdaState: XpbdLambda3

	@classmethod
	def accumulate(
			cls,
			bs: state.SubstepBoundData, bufs: state.DynamicData,
			bucket: OrientationDriveConstraintBucket, dt: float) -> tuple[
		state.DynamicData, OrientationDriveConstraintBucket]:
		"""
		XPBD orientation drive using unwrapped SO(3) joint coordinates.
		"""
		del dt

		active = bucket.isActive.astype(jnp.float32)
		m = active.shape[0]

		aIdx = bucket.bodyA.astype(jnp.int32)
		bIdx = bucket.bodyB.astype(jnp.int32)

		qA = bs.orientation[aIdx]
		qB = bs.orientation[bIdx]

		invInertiaA = bs.invInertiaBody[aIdx]
		invInertiaB = bs.invInertiaBody[bIdx]

		# ------------------------------------------------------------
		# Compute relative orientation in joint frame
		# ------------------------------------------------------------
		qA_frame = maths.quatMul(qA, bucket.localFrameA)
		qB_frame = maths.quatMul(qB, bucket.localFrameB)

		qRel = maths.quatMul(maths.quatConj(qA_frame), qB_frame)

		# Incremental rotation (Lie algebra)
		deltaRot = maths.quatLog(qRel)  # (m,3)

		# Unwrap / accumulate
		newUnwrapped = bucket.unwrappedRotation + deltaRot

		# ------------------------------------------------------------
		# Constraint residual in joint space ℝ³
		# ------------------------------------------------------------
		c = newUnwrapped - bucket.targetRotation
		cWeighted = c * bucket.weight[:, None]

		alpha = bucket.alpha
		lambdaPrev = bucket.lambdaState.lambdaValue

		# Jacobians (angular only)
		# dθA = -I, dθB = +I in joint space
		jwA = -jnp.eye(3, dtype=bs.position.dtype)[None, :, :]
		jwB = jnp.eye(3, dtype=bs.position.dtype)[None, :, :]

		# Apply inverse inertia row-wise
		# Flatten for reuse of applyInvInertiaWorld
		jwA_flat = jwA.reshape((m * 3, 3))
		jwB_flat = jwB.reshape((m * 3, 3))

		qA_rep = jnp.repeat(qA, 3, axis=0)
		qB_rep = jnp.repeat(qB, 3, axis=0)
		invIA_rep = jnp.repeat(invInertiaA, 3, axis=0)
		invIB_rep = jnp.repeat(invInertiaB, 3, axis=0)

		iInv_jwA = maths.applyInvInertiaWorld(qA_rep, invIA_rep, jwA_flat).reshape(
			(m, 3, 3))
		iInv_jwB = maths.applyInvInertiaWorld(qB_rep, invIB_rep, jwB_flat).reshape(
			(m, 3, 3))

		wEff = (
				jnp.einsum("mij,mij->mi", jwA, iInv_jwA) +
				jnp.einsum("mij,mij->mi", jwB, iInv_jwB)
		)
		wEff = jnp.maximum(wEff, 1e-12)

		deltaLambda = -(cWeighted + alpha[:, None] * lambdaPrev) / (
					wEff + alpha[:, None])
		deltaLambda = deltaLambda * active[:, None]

		lambdaNew = lambdaPrev + deltaLambda

		# Angular corrections
		tauA = -jnp.sum(iInv_jwA * deltaLambda[:, :, None], axis=1)
		tauB = jnp.sum(iInv_jwB * deltaLambda[:, :, None], axis=1)

		angDelta = bufs.angDelta
		angDelta = angDelta.at[aIdx].add(tauA)
		angDelta = angDelta.at[bIdx].add(tauB)

		newBufs = state.DynamicData(
			posDelta=bufs.posDelta,
			angDelta=angDelta
		)

		newBucket = OrientationDriveConstraintBucket(
			isActive=bucket.isActive,
			bodyA=bucket.bodyA,
			bodyB=bucket.bodyB,
			localFrameA=bucket.localFrameA,
			localFrameB=bucket.localFrameB,
			unwrappedRotation=newUnwrapped,
			targetRotation=bucket.targetRotation,
			alpha=bucket.alpha,
			weight=bucket.weight,
			lambdaState=XpbdLambda3(lambdaValue=lambdaNew),
		)

		return newBufs, newBucket


@jdc.pytree_dataclass
class MeasurementConstraintBucket(ConstraintBucket):
	"""
	Constraints between arbitrary user-defined measurements with internal geometric DOFs.

	This enables holonomic constraints like:
	  - "as distance(A,B) decreases, angle(C,D) increases"
	  - "point on surface S at (u,v) equals point on curve C at t"
	  - "normal of surface patch aligns with body orientation"

	Key features:
	  1. Constraints arbitrary measurement functions (not just body coordinates)
	  2. Supports internal parameters (u, v, t) that solver can adjust
	  3. Uses autodiff-computed Jacobians for both body and parameter corrections
	  4. Optional anisotropic weighting (e.g., normal vs tangent directions)

	Workflow:
	  - User defines measurement functions via tracing system
	  - Tracing system computes J_body and J_param via autodiff
	  - Solver updates both body state AND internal parameters each iteration
	"""
	isActive: jnp.ndarray            # (m,) bool/int mask

	# Which measurements to constrain (indices into MeasurementState)
	measurementA: jnp.ndarray        # (m,) int32
	measurementB: jnp.ndarray        # (m,) int32

	# Constraint type: equality (targetDelta=0) or delta relationship
	# Shape: (m, maxDim) where maxDim matches MeasurementState.values
	targetDelta: jnp.ndarray         # (m, maxDim) - usually zero for equality

	# XPBD compliance and weighting
	alpha: jnp.ndarray               # (m,) - compliance / dt^2
	weight: jnp.ndarray              # (m,) - constraint strength multiplier

	# Optional: anisotropic weighting per constraint
	# Example: weight normal direction 10x more than tangent
	# If None, uniform weighting used
	metricWeights: jnp.ndarray | None  # (m, maxDim) - diagonal metric

	# XPBD lambda accumulator (dimension matches maxDim)
	lambdaState: XpbdLambdaN         # lambdaState.Lambda: (m, maxDim)

	@classmethod
	def accumulate(
			cls,
			bs: state.SubstepBoundData,
			bufs: state.DynamicData,
			ms: state.MeasurementState,
			bucket: MeasurementConstraintBucket,
			dt: float) -> tuple[
		state.DynamicData,
		state.MeasurementState,
		MeasurementConstraintBucket
	]:
		"""
		Accumulate measurement constraint corrections.

		Strategy:
		  1. Compute residual: C = valueA - valueB - targetDelta
		  2. Update internal parameters using J_param (slide along geometry)
		  3. Apply remaining error to bodies using J_body
		  4. Update XPBD lambda

		Note: This is a sketch - full implementation requires:
		  - Gathering J_body and J_param from MeasurementState
		  - Mapping measurement body dependencies to constraint body pairs
		  - Careful handling of parameter bounds (u,v in [0,1], etc.)
		"""
		del dt  # alpha is pre-divided by dt^2

		active = bucket.isActive.astype(jnp.float32)
		m = active.shape[0]

		# Gather measurement values
		idxA = bucket.measurementA.astype(jnp.int32)
		idxB = bucket.measurementB.astype(jnp.int32)

		valA = ms.values[idxA]  # (m, maxDim)
		valB = ms.values[idxB]  # (m, maxDim)
		dimA = ms.valueDims[idxA]  # (m,)
		dimB = ms.valueDims[idxB]  # (m,)

		# Create dimension masks (only valid dims contribute to residual)
		maxDim = valA.shape[1]
		mask = (jnp.arange(maxDim)[None, :] < jnp.minimum(dimA, dimB)[:, None]).astype(jnp.float32)

		# Compute residual (masked)
		c = (valA - valB - bucket.targetDelta) * mask  # (m, maxDim)

		# Apply metric weighting if present
		if bucket.metricWeights is not None:
			c = c * bucket.metricWeights

		# Apply constraint weight
		c = c * bucket.weight[:, None]

		# XPBD update (simplified - full version needs J_body and J_param)
		alpha = bucket.alpha.astype(bs.position.dtype)[:, None]  # (m, 1)
		lambdaPrev = bucket.lambdaState.Lambda  # (m, maxDim)

		# TODO: Full implementation requires:
		# 1. Extract J_body from MeasurementState for measurements A and B
		# 2. Compute effective mass per DOF: wEff = J^T M^-1 J
		# 3. Solve for deltaLambda per DOF
		# 4. Apply corrections to bodies via J_body
		# 5. Apply corrections to parameters via J_param
		# 6. Update ms.params with new parameter values

		# For now, simplified update assuming diagonal effective mass
		# (This is a placeholder - real implementation needs proper Jacobian handling)
		wEffDummy = jnp.ones_like(c) * 2.0  # placeholder: assumes 2 bodies contribute
		wEffDummy = jnp.maximum(wEffDummy, 1e-12)

		deltaLambda = -(c + alpha * lambdaPrev) / (wEffDummy + alpha)
		deltaLambda = deltaLambda * active[:, None] * mask

		lambdaNew = lambdaPrev + deltaLambda

		# Return updated state (placeholder - needs actual body/param corrections)
		newBucket = MeasurementConstraintBucket(
			isActive=bucket.isActive,
			measurementA=bucket.measurementA,
			measurementB=bucket.measurementB,
			targetDelta=bucket.targetDelta,
			alpha=bucket.alpha,
			weight=bucket.weight,
			metricWeights=bucket.metricWeights,
			lambdaState=XpbdLambdaN(Lambda=lambdaNew),
		)

		return bufs, ms, newBucket


@jdc.pytree_dataclass
class ManifoldConstraintData:
	"""
	Batched manifold constraints with padding support.

	Represents a learned manifold of valid poses from K sample configurations.
	Bodies constrained to this manifold will be pulled toward the nearest valid
	pose on the manifold surface.

	Padding strategy:
	- All constraints padded to (kMax, nMax, mMax) for uniform GPU batching
	- sampleWeights mask ensures padding samples don't affect interpolation
	- Real samples have weight=1.0, padding samples have weight=0.0
	"""
	# Padded sample data
	validPoses: jnp.ndarray        # (kMax, nMax, 7) - sample poses (quat + pos per body)
	paramCenters: jnp.ndarray      # (kMax, mMax) - parameter values at samples

	# Sample mask (1=real sample, 0=padding)
	sampleWeights: jnp.ndarray     # (kMax,) - masks out padding in RBF interpolation

	# Pre-computed bidirectional RBF mappings
	forwardWeights: jnp.ndarray    # (kMax, nMax*7) - params → pose
	reverseWeights: jnp.ndarray    # (kMax, mMax) - pose → params
	poseCenters: jnp.ndarray       # (kMax, nMax*7) - flattened poses for reverse RBF

	# Shape metadata (actual vs padded)
	nSamples: int                  # actual K before padding
	nBodies: int                   # actual N before padding
	nParams: int                   # actual M before padding

	# RBF configuration
	epsilon: float = 1.0
	kernelType: str = 'gaussian'

	# Body mapping (padded with -1 for invalid entries)
	bodyIds: jnp.ndarray = None    # (nMax,) body indices, -1 = padding


def createManifoldConstraint(
	validPoses: jnp.ndarray, # (K, N, 7)
	paramValues: jnp.ndarray, # (K, M)
	bodyIds: list[int],
	kMax: int,  # max samples across all constraints
	nMax: int,  # max bodies across all constraints
	mMax: int,  # max params across all constraints
	epsilon: float = 1.0,
	kernelType: str = 'gaussian'
) -> ManifoldConstraintData:
	"""
	Create padded manifold constraint with bidirectional RBF interpolation.

	Args:
		validPoses: (K, N, 7) authored sample poses (K samples, N bodies, quat+pos)
		paramValues: (K, M) parameter values at each sample
		bodyIds: list of body indices this constraint applies to
		kMax: maximum samples across all constraints (for padding)
		nMax: maximum bodies across all constraints
		mMax: maximum parameters across all constraints
		epsilon: RBF shape parameter
		kernelType: 'gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate'

	Returns:
		ManifoldConstraintData with padded arrays and pre-computed RBF weights

	Example:
		# 3 samples, 4 bodies (femur, tibia, patella, fibula), 2 params (flexion, rotation)
		validPoses = jnp.array([...])  # (3, 4, 7)
		paramValues = jnp.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.2]])  # (3, 2)

		manifold = createManifoldConstraint(
			validPoses, paramValues, [femurId, tibiaId, patellaId, fibulaId],
			kMax=100, nMax=4, mMax=2
		)
	"""
	K, N, _ = validPoses.shape
	M = paramValues.shape[1]

	# Create sample weights mask (1 for real samples, 0 for padding)
	sampleWeights = jnp.concatenate([
		jnp.ones(K, dtype=jnp.float32),
		jnp.zeros(kMax - K, dtype=jnp.float32)
	])

	# Pad poses (K → kMax, N → nMax)
	identityPose = jnp.array([0, 0, 0, 1, 0, 0, 0], dtype=jnp.float32)  # identity quat + zero pos
	paddedPoses = jnp.pad(
		validPoses,
		((0, kMax - K), (0, nMax - N), (0, 0)),
		mode='constant',
		constant_values=0
	)
	# Set padding poses to identity (safer than zeros for quaternions)
	if kMax > K or nMax > N:
		for i in range(K, kMax):
			for j in range(N, nMax):
				paddedPoses = paddedPoses.at[i, j].set(identityPose)

	# Pad parameters (K → kMax, M → mMax)
	paddedParams = jnp.pad(
		paramValues,
		((0, kMax - K), (0, mMax - M)),
		mode='edge'  # duplicate last sample's params for padding
	)

	# Pad body IDs
	paddedBodyIds = jnp.pad(
		jnp.array(bodyIds, dtype=jnp.int32),
		(0, nMax - N),
		constant_values=-1  # -1 indicates padding
	)

	# Flatten poses for RBF (K_max, N_max*7)
	poseVectors = paddedPoses.reshape(kMax, -1)

	# Solve forward RBF: params → pose (WITH sample weights to mask padding!)
	forwardWeights = maths.solveRbfWeights(
		paddedParams,
		poseVectors,
		epsilon,
		kernelType,
		sampleWeights=sampleWeights
	)

	# Solve reverse RBF: pose → params (WITH sample weights to mask padding!)
	reverseWeights = maths.solveRbfWeights(
		poseVectors,
		paddedParams,
		epsilon,
		kernelType,
		sampleWeights=sampleWeights
	)

	return ManifoldConstraintData(
		validPoses=paddedPoses,
		paramCenters=paddedParams,
		sampleWeights=sampleWeights,
		forwardWeights=forwardWeights,
		reverseWeights=reverseWeights,
		poseCenters=poseVectors,
		nSamples=K,
		nBodies=N,
		nParams=M,
		epsilon=epsilon,
		kernelType=kernelType,
		bodyIds=paddedBodyIds
	)


def measureManifoldTarget(
	manifoldData: ManifoldConstraintData,
	currentParams: jnp.ndarray # (mMax,)
) -> jnp.ndarray: # (nMax*7,)
	"""
	Evaluate forward RBF: parameters → target pose.

	This is the "measurement function" used in constraints to compute
	the target pose from current parameter values.

	Args:
		manifoldData: pre-computed manifold constraint
		currentParams: (mMax,) current parameter values (padded)

	Returns:
		targetPose: (nMax*7,) interpolated target pose (flattened)
	"""
	targetPose = maths.interpolateRbf(
		currentParams[None, :],               # (1, mMax)
		manifoldData.paramCenters,            # (kMax, mMax)
		manifoldData.forwardWeights,          # (kMax, nMax*7)
		manifoldData.epsilon,
		manifoldData.kernelType,
		sampleWeights=manifoldData.sampleWeights  # mask padding samples
	)

	# Normalize quaternions in result
	targetPose = targetPose[0]  # (nMax*7,)
	nBodies = manifoldData.nBodies
	for i in range(nBodies):
		quatStart = i * 7
		quat = targetPose[quatStart:quatStart+4]
		quatNorm = maths.quatNormalize(quat)
		targetPose = targetPose.at[quatStart:quatStart+4].set(quatNorm)

	return targetPose


def findNearestParams(
	manifoldData: ManifoldConstraintData,
	currentPose: jnp.ndarray # (nMax*7,)
) -> jnp.ndarray: # (mMax,)
	"""
	Evaluate reverse RBF: pose → nearest parameters.

	Used for initialization/warmstart to find which parameters
	best represent the current body configuration.

	Args:
		manifoldData: pre-computed manifold constraint
		currentPose: (nMax*7,) current body poses (flattened)

	Returns:
		nearestParams: (mMax,) parameters that best reconstruct current pose
	"""
	nearestParams = maths.interpolateRbf(
		currentPose[None, :],                 # (1, nMax*7)
		manifoldData.poseCenters,             # (kMax, nMax*7)
		manifoldData.reverseWeights,          # (kMax, mMax)
		manifoldData.epsilon,
		manifoldData.kernelType,
		sampleWeights=manifoldData.sampleWeights  # mask padding samples
	)

	return nearestParams[0]


def measureActualPoses(
	bodies: state.SubstepBoundData,
	bodyIds: jnp.ndarray # (nMax,)
) -> jnp.ndarray: # (nMax*7,)
	"""
	Extract current poses for all bodies in constraint.

	Args:
		bodies: full body state
		bodyIds: (nMax,) body indices, -1 = padding (returns identity pose)

	Returns:
		poses: (nMax*7,) flattened poses [quat0, pos0, quat1, pos1, ...]
	"""
	def extractPose(bodyId):
		# Handle padding (bodyId = -1)
		validBody = bodyId >= 0

		# Get pose (or identity for padding)
		quat = jnp.where(
			validBody,
			bodies.orientation[jnp.maximum(bodyId, 0)],
			jnp.array([0, 0, 0, 1], dtype=bodies.orientation.dtype)
		)
		pos = jnp.where(
			validBody,
			bodies.position[jnp.maximum(bodyId, 0)],
			jnp.zeros(3, dtype=bodies.position.dtype)
		)

		return jnp.concatenate([quat, pos])

	poses = jax.vmap(extractPose)(bodyIds)  # (nMax, 7)
	return poses.reshape(-1)  # (nMax*7,)


@jdc.pytree_dataclass
class ManifoldConstraintBucket(ConstraintBucket):
	"""
	Batched manifold constraints that pull bodies toward learned valid pose manifolds.

	Each manifold constraint in the batch defines a learned surface of valid poses
	for a subset of bodies, parameterized by internal DOFs. The solver adjusts these
	internal parameters to minimize the distance from actual body poses to the manifold.

	Shape notation:
		M = number of manifold constraints in batch
		kMax = max samples across all manifolds (padded)
		nMax = max bodies per manifold (padded)
		mMax = max internal parameters per manifold (padded)
	"""
	isActive: jnp.ndarray                     # (M,) bool/int mask

	# Per-constraint manifold data (batched)
	manifoldData: list[ManifoldConstraintData]  # length M

	# Internal parameters (solver adjusts these)
	params: jnp.ndarray                       # (M, mMax) current parameter values

	# Constraint strength
	alpha: jnp.ndarray                        # (M,) compliance / dt^2
	weight: jnp.ndarray                       # (M,) overall strength multiplier

	# XPBD lambda storage (vector constraint, one per body DOF)
	lambdaState: XpbdLambdaN                  # (M, nMax*6) - 6 DOF per body (pos+ang)

	@classmethod
	def accumulate(
			cls,
			bs: state.SubstepBoundData,
			bufs: state.DynamicData,
			bucket: ManifoldConstraintBucket,
			dt: float
	) -> tuple[state.DynamicData, ManifoldConstraintBucket]:
		"""
		Accumulate manifold constraint corrections.

		For each constraint:
		1. Evaluate target pose from current parameters via RBF
		2. Compute residual (target - actual)
		3. Compute Jacobian w.r.t. body poses
		4. Apply XPBD correction to body positions/orientations

		Note: Parameters are currently fixed (not adjusted by solver).
		Future enhancement: add parameter adjustment via reverse RBF gradient.
		"""
		del dt

		active = bucket.isActive.astype(jnp.float32)  # (M,)
		M = active.shape[0]

		# Process each manifold constraint
		def processConstraint(i):
			manifold = bucket.manifoldData[i]
			currentParams = bucket.params[i]  # (mMax,)

			# Get target pose from manifold
			targetPoseFlat = measureManifoldTarget(manifold, currentParams)  # (nMax*7,)

			# Get actual poses
			actualPoseFlat = measureActualPoses(bs, manifold.bodyIds)  # (nMax*7,)

			# Compute residual (target - actual) for real bodies only
			nBodies = manifold.nBodies
			residual = jnp.zeros(nBodies * 6, dtype=bs.position.dtype)

			# Separate into position and orientation residuals
			for j in range(nBodies):
				bodyId = manifold.bodyIds[j]

				# Skip padding
				isValid = bodyId >= 0

				# Position residual (3 DOF)
				targetPos = targetPoseFlat[j*7 + 4:j*7 + 7]
				actualPos = actualPoseFlat[j*7 + 4:j*7 + 7]
				posResidual = jnp.where(isValid, targetPos - actualPos, 0.0)

				# Orientation residual (3 DOF) - convert quat difference to small angle
				targetQuat = targetPoseFlat[j*7:j*7 + 4]
				actualQuat = actualPoseFlat[j*7:j*7 + 4]

				# Quaternion error: q_error = q_target * conj(q_actual)
				qError = maths.quatMul(targetQuat, maths.quatConj(actualQuat))

				# Convert to small angle (axis * angle/2 stored in vector part)
				angResidual = jnp.where(isValid, 2.0 * qError[0:3], 0.0)

				residual = residual.at[j*6:j*6+3].set(posResidual)
				residual = residual.at[j*6+3:j*6+6].set(angResidual)

			return residual, manifold.bodyIds[:nBodies]

		# Vectorized processing would be complex due to variable nBodies
		# Use scan for efficient sequential processing
		def scanFn(carry, i):
			residuals, bodyIdsList = carry
			residual, bodyIds = processConstraint(i)
			return (residuals.at[i].set(residual), bodyIdsList.at[i].set(bodyIds)), None

		# Get maximum body count for padding
		maxBodies = max(m.nBodies for m in bucket.manifoldData)

		# Initialize storage
		initialResiduals = jnp.zeros((M, maxBodies * 6), dtype=bs.position.dtype)
		initialBodyIds = jnp.full((M, maxBodies), -1, dtype=jnp.int32)

		(residuals, bodyIdsList), _ = jax.lax.scan(
			scanFn,
			(initialResiduals, initialBodyIds),
			jnp.arange(M)
		)

		# Apply XPBD correction for each constraint
		newPosDelta = bufs.posDelta
		newAngDelta = bufs.angDelta
		newLambdas = bucket.lambdaState.Lambda

		for i in range(M):
			if not active[i]:
				continue

			manifold = bucket.manifoldData[i]
			nBodies = manifold.nBodies
			residual = residuals[i, :nBodies*6] * bucket.weight[i]

			# XPBD update per body
			for j in range(nBodies):
				bodyId = bodyIdsList[i, j]
				if bodyId < 0:
					continue

				# Get body properties
				invMass = bs.invMass[bodyId]
				invInertia = bs.invInertiaBody[bodyId]
				quat = bs.orientation[bodyId]

				# Position constraint (3 DOF)
				posRes = residual[j*6:j*6+3]
				lambdaPrevPos = newLambdas[i, j*6:j*6+3]

				wEffPos = invMass + 1e-12
				deltaLambdaPos = -(posRes + bucket.alpha[i] * lambdaPrevPos) / (wEffPos + bucket.alpha[i])
				deltaLambdaPos = deltaLambdaPos * active[i]

				newLambdas = newLambdas.at[i, j*6:j*6+3].set(lambdaPrevPos + deltaLambdaPos)
				newPosDelta = newPosDelta.at[bodyId].add(invMass * deltaLambdaPos)

				# Orientation constraint (3 DOF)
				angRes = residual[j*6+3:j*6+6]
				lambdaPrevAng = newLambdas[i, j*6+3:j*6+6]

				# Effective mass for rotation (simplified - assumes small residual)
				iInvAng = maths.applyInvInertiaWorld(quat, invInertia, angRes)
				wEffAng = jnp.dot(angRes, iInvAng) + 1e-12

				deltaLambdaAng = -(jnp.sum(angRes * angRes) + bucket.alpha[i] * jnp.sum(lambdaPrevAng * angRes)) / (wEffAng + bucket.alpha[i])
				deltaLambdaAngVec = deltaLambdaAng * angRes / (jnp.linalg.norm(angRes) + 1e-12)
				deltaLambdaAngVec = deltaLambdaAngVec * active[i]

				newLambdas = newLambdas.at[i, j*6+3:j*6+6].set(lambdaPrevAng + deltaLambdaAngVec)
				dTheta = maths.applyInvInertiaWorld(quat, invInertia, deltaLambdaAngVec)
				newAngDelta = newAngDelta.at[bodyId].add(dTheta)

		newBufs = state.DynamicData(
			posDelta=newPosDelta,
			angDelta=newAngDelta
		)

		newBucket = ManifoldConstraintBucket(
			isActive=bucket.isActive,
			manifoldData=bucket.manifoldData,
			params=bucket.params,
			alpha=bucket.alpha,
			weight=bucket.weight,
			lambdaState=XpbdLambdaN(Lambda=newLambdas)
		)

		return newBufs, newBucket
