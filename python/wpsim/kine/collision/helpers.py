from __future__ import annotations

import jax.numpy as jnp

from wpsim import maths, spatial
from wpsim.kine import state
from wpsim.kine.collision.common import CollisionQueryData


def buildCollisionSurfaceTrisFromMeshes(
		bs: state.SubstepBoundData,
		metadata: state.BodyMetadata,
		meshes: state.MeshBuffers,
		useDynamicOnly: bool = True
) -> tuple[jnp.ndarray, jnp.ndarray]:
	bodyCount = bs.position.shape[0]
	surfaceTris = []
	surfaceBody = []

	for bodyId in range(bodyCount):
		if useDynamicOnly and metadata.isDynamic is not None:
			if int(metadata.isDynamic[bodyId]) == 0:
				continue

		meshStart, meshEnd = metadata.getMeshRange(bodyId)
		if meshStart < 0:
			continue

		for meshId in range(meshStart, meshEnd):
			triStart = meshes.triStart(int(meshId))
			triEnd = meshes.triEnd(int(meshId))
			triIdx = meshes.triIndices[triStart:triEnd]
			if triIdx.shape[0] == 0:
				continue

			pointStart = meshes.pointStart(int(meshId))
			triIdx = triIdx + pointStart
			localTris = meshes.points[triIdx]
			pos = bs.position[bodyId]
			ori = bs.orientation[bodyId]

			flat = localTris.reshape((-1, 3))
			oriRep = jnp.repeat(ori[None, :], flat.shape[0], axis=0)
			world = maths.quatRotate(oriRep, flat).reshape(localTris.shape)
			world = world + pos[None, None, :]

			surfaceTris.append(world)
			surfaceBody.append(
				jnp.full((world.shape[0],), bodyId, dtype=jnp.int32)
			)

	if not surfaceTris:
		return (
			jnp.zeros((0, 3, 3), dtype=bs.position.dtype),
			jnp.zeros((0,), dtype=jnp.int32),
		)

	return (
		jnp.concatenate(surfaceTris, axis=0),
		jnp.concatenate(surfaceBody, axis=0),
	)


def buildCollisionQueryPointsFromMeshes(
		metadata: state.BodyMetadata,
		meshes: state.MeshBuffers,
		queryWeight: jnp.ndarray | None = None,
		useDynamicOnly: bool = True
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	bodyCount = metadata.geometryRefs.shape[0]
	points = []
	pointBody = []
	pointWeight = []
	dtype = meshes.points.dtype

	for bodyId in range(bodyCount):
		if useDynamicOnly and metadata.isDynamic is not None:
			if int(metadata.isDynamic[bodyId]) == 0:
				continue

		meshStart, meshEnd = metadata.getMeshRange(bodyId)
		if meshStart < 0:
			continue

		baseWeight = 1.0
		if metadata.isDynamic is not None and useDynamicOnly:
			baseWeight = float(metadata.isDynamic[bodyId])
		if queryWeight is not None:
			baseWeight *= float(queryWeight[bodyId])

		for meshId in range(meshStart, meshEnd):
			pointStart = meshes.pointStart(int(meshId))
			pointEnd = meshes.pointEnd(int(meshId))
			localPoints = meshes.points[pointStart:pointEnd]
			if localPoints.shape[0] == 0:
				continue

			points.append(localPoints)
			pointBody.append(
				jnp.full((localPoints.shape[0],), bodyId, dtype=jnp.int32)
			)
			pointWeight.append(
				jnp.full((localPoints.shape[0],), baseWeight, dtype=dtype)
			)

	if not points:
		return (
			jnp.zeros((0, 3), dtype=dtype),
			jnp.zeros((0,), dtype=jnp.int32),
			jnp.zeros((0,), dtype=dtype),
		)

	return (
		jnp.concatenate(points, axis=0),
		jnp.concatenate(pointBody, axis=0),
		jnp.concatenate(pointWeight, axis=0),
	)


def buildCollisionSpatialData(
		surfaceTris: jnp.ndarray,
		cellSize: float,
		tableSize: int = 65536,
		maxTotalEntries: int = 100000,
		maxBucketSearch: int = 16,
		useNeighborhood: bool = True,
		strategy: spatial.SpatialStrategy = spatial.SpatialStrategy.AUTO
) -> tuple[spatial.SpatialGridData | spatial.BVHData, spatial.SpatialStrategy, spatial.SpatialConfig]:
	return spatial.buildSpatialAcceleration(
		surfaceTris,
		cellSize,
		tableSize=tableSize,
		maxTotalEntries=maxTotalEntries,
		maxBucketSearch=maxBucketSearch,
		useNeighborhood=useNeighborhood,
		strategy=strategy,
	)


def buildCollisionQueryDataFromMeshes(
		bs: state.SubstepBoundData,
		metadata: state.BodyMetadata,
		meshes: state.MeshBuffers,
		gridData: spatial.SpatialGridData | spatial.BVHData,
		queryWeight: jnp.ndarray | None = None,
		useDynamicOnly: bool = True,
		surfaceTris: jnp.ndarray | None = None,
		surfaceBody: jnp.ndarray | None = None
) -> CollisionQueryData:
	if surfaceTris is None or surfaceBody is None:
		surfaceTris, surfaceBody = buildCollisionSurfaceTrisFromMeshes(
			bs, metadata, meshes, useDynamicOnly=useDynamicOnly
		)

	queryLocalPoints, queryBody, queryWeight = buildCollisionQueryPointsFromMeshes(
		metadata,
		meshes,
		queryWeight=queryWeight,
		useDynamicOnly=useDynamicOnly,
	)

	return CollisionQueryData(
		queryLocalPoints=queryLocalPoints,
		queryBody=queryBody,
		queryWeight=queryWeight,
		surfaceTris=surfaceTris,
		surfaceBody=surfaceBody,
		gridData=gridData,
	)

"""
usage:

surfaceTris, surfaceBody = collision.buildCollisionSurfaceTrisFromMeshes(bs, meta, meshes)
gridData, strategy, config = collision.buildCollisionSpatialData(surfaceTris, cellSize=0.05)
collisionData = collision.buildCollisionQueryDataFromMeshes(
	bs, meta, meshes, gridData, surfaceTris=surfaceTris, surfaceBody=surfaceBody
)


"""

