from __future__ import annotations
import typing as T
import jax
from jax import numpy as jnp
from wpsim import maths, spatial
from wpsim.kine import state, constraint, collision
from wpsim.maths import quatMul, quatNormalize

""" main meat
to run at kinematics level - mainly rigid bodies of the skeleton, 
fewer hero cloth bodies / ropes etc

returning new copies of dataclasses preferred here - apparently
more efficient and easily traced than array.at[].set()

since we accumulate buffers of deltas before combining, maybe there's
a point to diffusing deltas out through connected bodies?

we store alpha as pre-divided by timestep - updating alpha live is still
possible, just need to re-apply the timestep divide when we do.

we assume rigid body centre of mass is at ORIGIN, with inertia vectors 
aligned to PRINCIPAL AXES. this will mean live-recomputing of rest frames 
when assigning or editing meshes
"""

def applyBuffers(bs: state.SubstepBoundData, bufs: state.DynamicData) -> (
		state.SubstepBoundData):
	newPos = bs.position + bufs.posDelta
	newOri = maths.applySmallAngleToQuat(bs.orientation, bufs.angDelta)

	return state.SubstepBoundData(
		position=newPos,
		orientation=newOri,
		linearVelocity=bs.linearVelocity,
		angularVelocity=bs.angularVelocity,
		invMass=bs.invMass,
		invInertiaBody=bs.invInertiaBody,
		force=bs.force,
		torque=bs.torque,
	)

def integrateBodies(bs: state.SubstepBoundData, dt: float) -> state.SubstepBoundData:
	"""basic euler - position + vel*dt
	"""
	newPosition = bs.position + dt * bs.linearVelocity

	# quaternion derivative: qdot = 0.5 * [omega, 0] ⊗ q (body-space vs world-space chosen later)
	omegaQuat = jnp.concatenate([bs.angularVelocity, jnp.zeros((bs.angularVelocity.shape[0], 1), bs.angularVelocity.dtype)], axis=-1)
	qdot = 0.5 * quatMul(omegaQuat, bs.orientation)
	newOrientation = quatNormalize(bs.orientation + dt * qdot)

	return state.SubstepBoundData(
		position=newPosition,
		orientation=newOrientation,
		linearVelocity=bs.linearVelocity,
		angularVelocity=bs.angularVelocity,
		invMass=bs.invMass,
		invInertiaBody=bs.invInertiaBody,
		force=bs.force,
		torque=bs.torque,
	)

def applyExternalForces(
		bs: state.SubstepBoundData,
		dt: float
) -> state.SubstepBoundData:
	linearVelocity = bs.linearVelocity + dt * bs.force * bs.invMass[:, None]
	angularAccel = maths.applyInvInertiaWorld(
		bs.orientation, bs.invInertiaBody, bs.torque
	)
	angularVelocity = bs.angularVelocity + dt * angularAccel

	return state.SubstepBoundData(
		position=bs.position,
		orientation=bs.orientation,
		linearVelocity=linearVelocity,
		angularVelocity=angularVelocity,
		invMass=bs.invMass,
		invInertiaBody=bs.invInertiaBody,
		force=bs.force,
		torque=bs.torque,
	)

def applyCollisionForces(
		bs: state.SubstepBoundData,
		collisionData: collision.CollisionQueryData | None,
		collisionSettings: collision.CollisionSettings | None,
		collisionConfig: spatial.SpatialConfig | None,
		collisionStrategy: spatial.SpatialStrategy | None,
		collisionFn: T.Callable,
		dt: float
) -> state.SubstepBoundData:
	if (
			collisionData is None
			or collisionSettings is None
			or collisionConfig is None
			or collisionStrategy is None
	):
		return bs
	if collisionData.queryLocalPoints.shape[0] == 0:
		return bs
	if collisionData.surfaceTris.shape[0] == 0:
		return bs

	contacts = collision.buildContactsFromSpatial(
		bs,
		collisionData,
		collisionConfig,
		collisionStrategy,
	)
	force, torque = collisionFn(bs, contacts, collisionSettings, dt)

	linearVelocity = bs.linearVelocity + dt * force * bs.invMass[:, None]
	angularAccel = maths.applyInvInertiaWorld(
		bs.orientation, bs.invInertiaBody, torque
	)
	angularVelocity = bs.angularVelocity + dt * angularAccel

	return state.SubstepBoundData(
		position=bs.position,
		orientation=bs.orientation,
		linearVelocity=linearVelocity,
		angularVelocity=angularVelocity,
		invMass=bs.invMass,
		invInertiaBody=bs.invInertiaBody,
		force=bs.force,
		torque=bs.torque,
	)


def solverIterationPlan(
		bs: state.SubstepBoundData,
		cs: constraint.ConstraintState,
		dtSub: float,
		plan: constraint.ConstraintPlan) -> tuple[
	state.SubstepBoundData, constraint.ConstraintState]:
	"""solve sequence of constraint types given in plan
	"""
	n = bs.position.shape[0]
	bufs = state.DynamicData.makeZero(n, bs.position.dtype)

	# Apply all accumulators in the plan (static tuple => unrolled at trace time)
	# for accumulateFn in plan.accumulators:
	# 	accumulateFn : constraint.constraintAccumFn
	# 	bufs, cs = accumulateFn(bs, bufs, cs, dtSub)
	### doing proper general dispatch got a bit messy here, hardcode for now
	bufs, newPointBucket = cs.point.accumulate(
		bs, bufs, cs.point, dtSub
	)
	bufs, newHingeAxisBucket = cs.hingeAxis.accumulate(
		bs, bufs, cs.hingeAxis, dtSub
	)



	# Optional manifold constraints
	newManifoldBucket = cs.manifold
	if cs.manifold is not None:
		bufs, newManifoldBucket = cs.manifold.accumulate(
			bs, bufs, cs.manifold, dtSub
		)

	cs = constraint.ConstraintState(
		settings=cs.settings,
		point=newPointBucket,
		hingeAxis=newHingeAxisBucket,
		manifold=newManifoldBucket,
	)

	bsOut = applyBuffers(bs, bufs)

	# normalize once per iteration
	bsOut = state.SubstepBoundData(
		position=bsOut.position,
		orientation=quatNormalize(bsOut.orientation),
		linearVelocity=bsOut.linearVelocity,
		angularVelocity=bsOut.angularVelocity,
		invMass=bsOut.invMass,
		invInertiaBody=bsOut.invInertiaBody,
		force=bsOut.force,
		torque=bsOut.torque,
	)

	return bsOut, cs

def solveSubstep(
		bs: state.SubstepBoundData,
		cs: constraint.ConstraintState,
		dt: float,
		plan: constraint.ConstraintPlan
) -> tuple[state.SubstepBoundData, constraint.ConstraintState]:
	def iterBody(i, carry):
		curBs, curCs = carry
		return solverIterationPlan(curBs, curCs, dt, plan)

	bsOut, csOut = jax.lax.fori_loop(0, cs.settings.iterationCount, iterBody, (bs, cs))

	# keep quaternions normalized to prevent drift
	bsOut = state.SubstepBoundData(
		position=bsOut.position,
		orientation=quatNormalize(bsOut.orientation),
		linearVelocity=bsOut.linearVelocity,
		angularVelocity=bsOut.angularVelocity,
		invMass=bsOut.invMass,
		invInertiaBody=bsOut.invInertiaBody,
		force=bsOut.force,
		torque=bsOut.torque,
	)
	return bsOut, csOut


def solveFrame(
		bs: state.SubstepBoundData,
		cs: constraint.ConstraintState,
		plan: constraint.ConstraintPlan,
		collisionData: collision.CollisionQueryData | None = None,
		collisionSettings: collision.CollisionSettings | None = None,
		collisionConfig: spatial.SpatialConfig | None = None,
		collisionStrategy: spatial.SpatialStrategy | None = None,
		collisionFn: T.Callable = collision.smoothCollisionForce,
	) -> tuple[state.SubstepBoundData, constraint.ConstraintState]:
	substepDt = cs.settings.dt / cs.settings.substepCount

	def substepBody(s, carry):
		curBs, curCs = carry

		curBs = applyExternalForces(curBs, substepDt)

		curBs = applyCollisionForces(
			curBs,
			collisionData,
			collisionSettings,
			collisionConfig,
			collisionStrategy,
			collisionFn,
			substepDt,
		)

		# dynamic integration stub (optional; you can switch off for quasi-static)
		curBs = integrateBodies(curBs, substepDt)

		# constraint projection
		curBs, curCs = solveSubstep(curBs, curCs, substepDt, plan)

		return curBs, curCs

	return jax.lax.fori_loop(0, cs.settings.substepCount, substepBody, (bs, cs))
