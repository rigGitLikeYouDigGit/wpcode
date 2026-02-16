from __future__ import annotations

import jax.numpy as jnp

from wpsim import maths, spatial
from wpsim.kine import state
from wpsim.kine.collision.common import CollisionContacts, CollisionQueryData


def buildContactsFromSpatial(
		bs: state.SubstepBoundData,
		queryData: CollisionQueryData,
		config: spatial.SpatialConfig,
		strategy: spatial.SpatialStrategy,
		allowSelf: bool = False
) -> CollisionContacts:
	dtype = bs.position.dtype
	bodyA = queryData.queryBody.astype(jnp.int32)

	localPoints = queryData.queryLocalPoints.astype(dtype)
	queryWeight = queryData.queryWeight.astype(dtype)

	posA = bs.position[bodyA]
	oriA = bs.orientation[bodyA]
	queryPoints = posA + maths.quatRotate(oriA, localPoints)

	results = spatial.querySpatial(
		queryPoints,
		queryData.gridData,
		queryData.surfaceTris,
		strategy,
		config,
	)

	distSq = jnp.maximum(results.bestDistSq, 0.0)
	distance = jnp.sqrt(distSq)

	valid = results.bestTriIdx >= 0
	safeIdx = jnp.where(valid, results.bestTriIdx, 0).astype(jnp.int32)
	bodyB = queryData.surfaceBody[safeIdx].astype(jnp.int32)

	offset = queryPoints - results.bestPoint
	norm = jnp.linalg.norm(offset, axis=-1, keepdims=True)
	normal = offset / (norm + jnp.asarray(1e-8, dtype))

	weight = queryWeight * valid.astype(dtype)
	if not allowSelf:
		weight = weight * (bodyA != bodyB).astype(dtype)

	return CollisionContacts(
		bodyA=bodyA,
		bodyB=bodyB,
		pointA=queryPoints,
		pointB=results.bestPoint,
		normal=normal,
		distance=distance,
		weight=weight,
	)
