from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from . import state, maths


@jdc.pytree_dataclass
class ConstraintState:
	settings: ConstraintSolveSettings
	point: PointConstraintBucket
	hingeAxis: HingeJointConstraintBucket
	#orient: OrientationDriveConstraintBucket


constraintAccumFn = Callable[
	[state.BodyState, state.BodyDeltaBuffers, ConstraintState, float],
	Tuple[state.BodyDeltaBuffers, ConstraintState]
]

"""attempt more customisation here - 
is there ever a case where a constraint weight map function wouldn't
just output a single DOF? don't think there's anywhere we would need
the DOFs themselves coupled, duplicating the same code for a ramp 3 times 
over is fine since it gets vmapped anyway

don't forget nurbs constraints, surface constraints etc
"""

constraintAlphaMapFn = Callable[
	[state.BodyState,
	 ConstraintState,
	 int, # constraint type index
	 int, # constraint index
	 int # DOF index
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
			bs: state.BodyState, bufs: state.BodyDeltaBuffers,
			#cs: ConstraintState,
			bucket: PointConstraintBucket, dt: float) -> tuple[
		state.BodyDeltaBuffers, ConstraintState]:
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
			bs: state.BodyState, bufs: state.BodyDeltaBuffers,
			#cs: ConstraintState,
			bucket:PointConstraintBucket, dt: float) -> tuple[
		state.BodyDeltaBuffers, #ConstraintState
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

		newBufs = state.BodyDeltaBuffers(posDelta=posDelta, angDelta=angDelta)

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
			bs: state.BodyState, bufs: state.BodyDeltaBuffers,
			bucket: HingeJointConstraintBucket, dt: float) -> tuple[
		state.BodyDeltaBuffers, HingeJointConstraintBucket]:
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

		newBufs = state.BodyDeltaBuffers(
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
			bs: state.BodyState, bufs: state.BodyDeltaBuffers,
			bucket: OrientationDriveConstraintBucket, dt: float) -> tuple[
		state.BodyDeltaBuffers, OrientationDriveConstraintBucket]:
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

		newBufs = state.bodyDeltaBuffers(
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
