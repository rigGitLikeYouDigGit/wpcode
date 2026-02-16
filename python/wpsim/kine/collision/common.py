from __future__ import annotations

import jax.numpy as jnp
import jax_dataclasses as jdc

from wpsim import spatial
from wpsim.kine import state


@jdc.pytree_dataclass
class CollisionContacts:
	"""Contact batch for collision response."""
	bodyA: jnp.ndarray
	bodyB: jnp.ndarray
	pointA: jnp.ndarray
	pointB: jnp.ndarray
	normal: jnp.ndarray
	distance: jnp.ndarray
	weight: jnp.ndarray


@jdc.pytree_dataclass
class CollisionSettings:
	"""Tunable parameters for collision response."""
	stiffness: float = 0.0
	damping: float = 0.0
	friction: float = 0.0
	thickness: float = 0.0
	smoothing: float = 1e-4


@jdc.pytree_dataclass
class CollisionQueryData:
	"""Inputs required to build contacts from spatial queries."""
	queryLocalPoints: jnp.ndarray
	queryBody: jnp.ndarray
	queryWeight: jnp.ndarray
	surfaceTris: jnp.ndarray
	surfaceBody: jnp.ndarray
	gridData: spatial.SpatialGridData | spatial.BVHData


def smoothRelu(x: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
	return 0.5 * (x + jnp.sqrt(x * x + eps * eps))


def contactKinematics(
		bs: state.SubstepBoundData,
		contacts: CollisionContacts
) -> tuple[
	jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
	jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
	bodyA = contacts.bodyA.astype(jnp.int32)
	bodyB = contacts.bodyB.astype(jnp.int32)

	normal = contacts.normal.astype(bs.position.dtype)
	pointA = contacts.pointA.astype(bs.position.dtype)
	pointB = contacts.pointB.astype(bs.position.dtype)

	posA = bs.position[bodyA]
	posB = bs.position[bodyB]
	rA = pointA - posA
	rB = pointB - posB

	vA = bs.linearVelocity[bodyA] + jnp.cross(bs.angularVelocity[bodyA], rA)
	vB = bs.linearVelocity[bodyB] + jnp.cross(bs.angularVelocity[bodyB], rB)
	vRel = vB - vA
	vn = jnp.sum(vRel * normal, axis=-1)
	vt = vRel - vn[:, None] * normal

	return bodyA, bodyB, rA, rB, vRel, vn, vt, normal


def accumulateContactForces(
		bodyCount: int,
		bodyA: jnp.ndarray,
		bodyB: jnp.ndarray,
		rA: jnp.ndarray,
		rB: jnp.ndarray,
		forceWorld: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
	force = jnp.zeros((bodyCount, 3), forceWorld.dtype)
	torque = jnp.zeros((bodyCount, 3), forceWorld.dtype)

	forceA = -forceWorld
	forceB = forceWorld

	force = force.at[bodyA].add(forceA)
	force = force.at[bodyB].add(forceB)

	torque = torque.at[bodyA].add(jnp.cross(rA, forceA))
	torque = torque.at[bodyB].add(jnp.cross(rB, forceB))

	return force, torque


def frictionForce(
		vt: jnp.ndarray,
		normalForce: jnp.ndarray,
		friction: jnp.ndarray,
		eps: jnp.ndarray
) -> jnp.ndarray:
	vtMag = jnp.sqrt(jnp.sum(vt * vt, axis=-1) + eps * eps)
	return -friction[:, None] * normalForce[:, None] * vt / vtMag[:, None]
