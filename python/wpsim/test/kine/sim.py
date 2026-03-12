from __future__ import annotations

import unittest

import jax.numpy as jnp

from wpsim.kine import constraint, sim, state


def makeIdentityQuat(n: int, dtype=jnp.float32) -> jnp.ndarray:
	q = jnp.zeros((n, 4), dtype)
	return q.at[:, 3].set(1.0)


def makeEmptyConstraintState(
		dt: float,
		substepCount: int,
		iterationCount: int,
		dtype=jnp.float32
) -> tuple[constraint.ConstraintState, constraint.ConstraintPlan]:
	point = constraint.PointConstraintBucket(
		isActive=jnp.zeros((0,), jnp.int32),
		bodyA=jnp.zeros((0,), jnp.int32),
		bodyB=jnp.zeros((0,), jnp.int32),
		localPointA=jnp.zeros((0, 3), dtype),
		localPointB=jnp.zeros((0, 3), dtype),
		alpha=jnp.zeros((0,), dtype),
		weight=jnp.zeros((0,), dtype),
		lambdaState=constraint.XpbdLambda3(
			lambdaValue=jnp.zeros((0, 3), dtype)
		),
	)

	hinge = constraint.HingeJointConstraintBucket(
		isActive=jnp.zeros((0,), jnp.int32),
		bodyA=jnp.zeros((0,), jnp.int32),
		bodyB=jnp.zeros((0,), jnp.int32),
		localHingeAxisA=jnp.zeros((0, 3), dtype),
		localHingeAxisB=jnp.zeros((0, 3), dtype),
		unwrappedTwist=jnp.zeros((0,), dtype),
		twistMin=jnp.zeros((0,), dtype),
		twistMax=jnp.zeros((0,), dtype),
		alphaSwing=jnp.zeros((0,), dtype),
		alphaTwist=jnp.zeros((0,), dtype),
		weight=jnp.zeros((0,), dtype),
		lambdaState=constraint.XpbdLambda3(
			lambdaValue=jnp.zeros((0, 3), dtype)
		),
	)

	settings = constraint.ConstraintSolveSettings(
		iterationCount=iterationCount,
		substepCount=substepCount,
		dt=dt,
	)
	stateOut = constraint.ConstraintState(
		settings=settings,
		point=point,
		hingeAxis=hinge,
	)
	plan = constraint.ConstraintPlan(
		(point.accumulate, hinge.accumulate),
	)
	return stateOut, plan


class KinePipelineTests(unittest.TestCase):
	def testSingleBodyFallsUnderGravity(self):
		# Single rigid body under constant force should follow symplectic Euler.
		dtype = jnp.float32
		dt = 1.0 / 60.0
		substepCount = 2
		iterationCount = 1

		gravity = jnp.array([0.0, -9.81, 0.0], dtype)
		startPosition = jnp.array([[0.0, 1.0, 0.0]], dtype)

		bs = state.BodyState(
			position=startPosition,
			orientation=makeIdentityQuat(1, dtype),
			linearVelocity=jnp.zeros((1, 3), dtype),
			angularVelocity=jnp.zeros((1, 3), dtype),
			invMass=jnp.array([1.0], dtype),
			invInertiaBody=jnp.array([[1.0, 1.0, 1.0]], dtype),
			force=gravity[None, :],
			torque=jnp.zeros((1, 3), dtype),
		)

		cs, plan = makeEmptyConstraintState(
			dt=dt,
			substepCount=substepCount,
			iterationCount=iterationCount,
			dtype=dtype,
		)

		bsOut, _ = sim.solveFrame(bs, cs, plan=plan)

		substepDt = dt / substepCount
		stepSum = substepCount * (substepCount + 1) / 2.0
		expectedVelocity = gravity * dt
		expectedPosition = startPosition[0] + gravity * (substepDt * substepDt * stepSum)

		self.assertTrue(bool(jnp.allclose(
			bsOut.linearVelocity[0], expectedVelocity, atol=1e-6
		)))
		self.assertTrue(bool(jnp.allclose(
			bsOut.position[0], expectedPosition, atol=1e-6
		)))
		self.assertTrue(bool(jnp.allclose(
			bsOut.orientation[0], makeIdentityQuat(1, dtype)[0]
		)))


if __name__ == "__main__":
	unittest.main()
