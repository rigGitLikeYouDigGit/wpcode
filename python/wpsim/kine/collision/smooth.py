from __future__ import annotations

import jax.numpy as jnp

from wpsim.kine import state
from wpsim.kine.collision.common import (
	CollisionContacts,
	CollisionSettings,
	accumulateContactForces,
	contactKinematics,
	frictionForce,
	smoothRelu,
)


def collisionForce(
		bs: state.SubstepBoundData,
		contacts: CollisionContacts,
		settings: CollisionSettings,
		dt: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
	"""Smooth penalty-style collision response."""
	del dt

	dtype = bs.position.dtype
	eps = jnp.asarray(settings.smoothing, dtype)
	stiffness = jnp.asarray(settings.stiffness, dtype)
	damping = jnp.asarray(settings.damping, dtype)
	friction = jnp.asarray(settings.friction, dtype)
	thickness = jnp.asarray(settings.thickness, dtype)

	bodyCount = bs.position.shape[0]
	bodyA, bodyB, rA, rB, vRel, vn, vt, normal = contactKinematics(bs, contacts)

	weight = contacts.weight.astype(dtype)
	distance = contacts.distance.astype(dtype)

	stiffness = jnp.broadcast_to(stiffness, distance.shape)
	damping = jnp.broadcast_to(damping, distance.shape)
	friction = jnp.broadcast_to(friction, distance.shape)
	thickness = jnp.broadcast_to(thickness, distance.shape)

	gap = distance - thickness
	penetration = -gap
	phi = smoothRelu(penetration, eps)
	activation = phi / (phi + eps)
	approach = smoothRelu(-vn, eps)

	fn = stiffness * phi + damping * approach * activation
	fn = fn * weight

	normalForce = normal * fn[:, None]
	tangentForce = frictionForce(vt, fn, friction, eps)
	forceWorld = normalForce + tangentForce

	return accumulateContactForces(bodyCount, bodyA, bodyB, rA, rB, forceWorld)
