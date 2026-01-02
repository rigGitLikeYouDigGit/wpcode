from __future__ import annotations
#from wplib import log

from wpsim.kine.simrigid import solveFrame
from wpsim.kine.state import *
from wpsim.kine.constraint import *
from wpsim.maths import *


def makeIdentityQuat(n: int, dtype=jnp.float32) -> jnp.ndarray:
	q = jnp.zeros((n, 4), dtype)
	return q.at[:, 3].set(1.0)


def makeTestStatePointConstraint() -> tuple[BodyState, ConstraintState,
ConstraintPlan]:
	# Two bodies
	n = 2
	dtype = jnp.float32

	# Body 0 at origin, Body 1 offset in x
	bs = BodyState(
		position=jnp.array([[0.0, 0.0, 0.0],
							[1.0, 0.0, 0.0]], dtype),
		orientation=makeIdentityQuat(n, dtype),
		linearVelocity=jnp.zeros((n, 3), dtype),
		angularVelocity=jnp.zeros((n, 3), dtype),
		invMass=jnp.array([0.0, 1.0], dtype),			# body 0 kinematic, body 1 dynamic
		invInertiaBody=jnp.array([[0.0, 0.0, 0.0],		# kinematic
								  [1.0, 1.0, 1.0]], dtype),
	)

	# One point constraint between body 0 and body 1:
	# local points are both at their COM, so this enforces x1 == x0
	mPoint = 1
	point = PointConstraintBucket(
		isActive=jnp.array([1], jnp.int32),
		bodyA=jnp.array([0], jnp.int32),
		bodyB=jnp.array([1], jnp.int32),
		localPointA=jnp.array([[0.0, 0.0, 0.0]], dtype),
		localPointB=jnp.array([[0.0, 0.0, 0.0]], dtype),
		alpha=jnp.array([0.0], dtype),					# hard constraint for test
		weight=jnp.array([1.0], dtype),
		lambdaState=XpbdLambda3(lambdaValue=jnp.zeros((mPoint, 3), dtype)),
	)

	# No hinge constraints in this test: allocate an empty bucket with m=0
	mHinge = 0
	hinge = HingeJointConstraintBucket(
		isActive=jnp.zeros((mHinge,), jnp.int32),
		bodyA=jnp.zeros((mHinge,), jnp.int32),
		bodyB=jnp.zeros((mHinge,), jnp.int32),
		localHingeAxisA=jnp.zeros((mHinge, 3), dtype),
		localHingeAxisB=jnp.zeros((mHinge, 3), dtype),
		alpha=jnp.zeros((mHinge,), dtype),
		weight=jnp.zeros((mHinge,), dtype),
		lambdaState=XpbdLambda2(lambdaValue=jnp.zeros((mHinge, 2), dtype)),
	)

	settings = ConstraintSolveSettings(
		iterationCount=10,
		substepCount=2,
		dt=1.0 / 60.0,
	)

	cs = ConstraintState(
		settings=settings,
		point=point,
		hingeAxis=hinge,
	)

	plan = ConstraintPlan(
		(point.accumulate, hinge.accumulate),
	)
	return bs, cs, plan


def pointError(bs: BodyState, cs: ConstraintState) -> jnp.ndarray:
	# Computes ||pB - pA|| for the first point constraint
	iA = cs.point.bodyA[0]
	iB = cs.point.bodyB[0]
	qA = bs.orientation[iA]
	qB = bs.orientation[iB]
	xA = bs.position[iA]
	xB = bs.position[iB]
	rA = quatRotate(qA[None, :], cs.point.localPointA)[0]
	rB = quatRotate(qB[None, :], cs.point.localPointB)[0]
	pA = xA + rA
	pB = xB + rB
	return jnp.linalg.norm(pB - pA)


def runPointConstraintSmokeTest():
	bs0, cs0, plan = makeTestStatePointConstraint()

	err0 = pointError(bs0, cs0)

	solveJit = jax.jit(solveFrame, static_argnames=("plan",))
	bs1, cs1 = solveJit(bs0, cs0, plan=plan)

	err1 = pointError(bs1, cs1)

	return err0, err1, bs0, bs1, cs1



def makeTestStateHingeAxisOnly() -> tuple[bodyState, constraintState, constraintPlan]:
	n = 2
	dtype = jnp.float32

	q0 = makeIdentityQuat(n, dtype)
	# Rotate body 1 around Z by 90 degrees
	qRot = makeQuatFromAxisAngle(jnp.array([0.0, 0.0, 1.0], dtype), jnp.pi * 0.5)
	q0 = q0.at[1].set(qRot)

	bs = BodyState(
		position=jnp.zeros((n, 3), dtype),
		orientation=q0,
		linearVelocity=jnp.zeros((n, 3), dtype),
		angularVelocity=jnp.zeros((n, 3), dtype),
		invMass=jnp.array([0.0, 1.0], dtype),
		invInertiaBody=jnp.array([[0.0, 0.0, 0.0],
								  [1.0, 1.0, 1.0]], dtype),
	)

	# No point constraints
	mPoint = 0
	point = PointConstraintBucket(
		isActive=jnp.zeros((mPoint,), jnp.int32),
		bodyA=jnp.zeros((mPoint,), jnp.int32),
		bodyB=jnp.zeros((mPoint,), jnp.int32),
		localPointA=jnp.zeros((mPoint, 3), dtype),
		localPointB=jnp.zeros((mPoint, 3), dtype),
		alpha=jnp.zeros((mPoint,), dtype),
		weight=jnp.zeros((mPoint,), dtype),
		lambdaState=XpbdLambda3(lambdaValue=jnp.zeros((mPoint, 3), dtype)),
	)

	# One hinge-axis constraint aligning local X axes of both bodies
	mHinge = 1
	hinge = HingeJointConstraintBucket(
		isActive=jnp.array([1], jnp.int32),
		bodyA=jnp.array([0], jnp.int32),
		bodyB=jnp.array([1], jnp.int32),
		localHingeAxisA=jnp.array([[1.0, 0.0, 0.0]], dtype),	# normalized
		localHingeAxisB=jnp.array([[1.0, 0.0, 0.0]], dtype),	# normalized
		alpha=jnp.array([0.0], dtype),
		weight=jnp.array([1.0], dtype),
		lambdaState=XpbdLambda2(lambdaValue=jnp.zeros((mHinge, 2), dtype)),
	)

	settings = ConstraintSolveSettings(
		iterationCount=20,
		substepCount=2,
		dt=1.0 / 60.0,
	)

	cs = ConstraintState(settings=settings, point=point, hingeAxis=hinge)
	plan = ConstraintPlan(
		(point.accumulate, hinge.accumulate),
	)
	return bs, cs, plan

def hingeAxisDot(bs: bodyState, cs: constraintState) -> jnp.ndarray:
	iA = cs.hingeAxis.bodyA[0]
	iB = cs.hingeAxis.bodyB[0]
	aW = quatRotate(bs.orientation[iA][None, :], cs.hingeAxis.localHingeAxisA)[0]
	bW = quatRotate(bs.orientation[iB][None, :], cs.hingeAxis.localHingeAxisB)[0]
	return jnp.dot(aW, bW)

def runHingeAxisSmokeTest():
	bs0, cs0, plan = makeTestStateHingeAxisOnly()
	dot0 = hingeAxisDot(bs0, cs0)

	solveJit = jax.jit(solveFrame, static_argnames=("plan",))
	bs1, cs1 = solveJit(bs0, cs0, plan=plan)
	dot1 = hingeAxisDot(bs1, cs1)

	return dot0, dot1, bs0, bs1, cs1

if __name__ == '__main__':
# Example usage:
	err0, err1, bs0, bs1, cs1 = runPointConstraintSmokeTest()
	print("Point error before:", err0)
	print("Point error after :", err1)
	print("Body1 position before:", bs0.position[1])
	print("Body1 position after :", bs1.position[1])

	err0, err1, bs0, bs1, cs1 = runHingeAxisSmokeTest()
	print("hinge error before:", err0)
	print("hinge error after :", err1)
