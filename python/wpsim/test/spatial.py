from __future__ import annotations

import unittest

import jax.numpy as jnp

from wpsim import spatial


def makeUnitSquareTris(dtype=jnp.float32) -> jnp.ndarray:
	points = jnp.array(
		[
			[0.0, 0.0, 0.0],
			[1.0, 0.0, 0.0],
			[1.0, 1.0, 0.0],
			[0.0, 1.0, 0.0],
		],
		dtype,
	)
	triIndices = jnp.array(
		[
			[0, 1, 2],
			[0, 2, 3],
		],
		jnp.int32,
	)
	return points[triIndices]


def makeCoincidentVertexTris(dtype=jnp.float32) -> jnp.ndarray:
	points = jnp.array(
		[
			[0.0, 0.0, 0.0],
			[1.0, 0.0, 0.0],
			[1.0, 1.0, 0.0],
			[0.0, 1.0, 0.0],
			[1.0, 0.0, 0.0],
		],
		dtype,
	)
	triIndices = jnp.array(
		[
			[0, 1, 2],
			[0, 4, 3],
		],
		jnp.int32,
	)
	return points[triIndices]


def makeDegenerateTriMesh(dtype=jnp.float32) -> jnp.ndarray:
	points = jnp.array(
		[
			[0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0],
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
		],
		dtype,
	)
	triIndices = jnp.array(
		[
			[0, 1, 2],
			[0, 3, 4],
		],
		jnp.int32,
	)
	return points[triIndices]


def makeSingleTri(dtype=jnp.float32) -> jnp.ndarray:
	points = jnp.array(
		[
			[0.1, 0.1, 0.0],
			[0.6, 0.1, 0.0],
			[0.1, 0.6, 0.0],
		],
		dtype,
	)
	triIndices = jnp.array([[0, 1, 2]], jnp.int32)
	return points[triIndices]


def makeTwoTrisSameCell(dtype=jnp.float32) -> jnp.ndarray:
	points = jnp.array(
		[
			[0.1, 0.1, 0.0],
			[0.4, 0.1, 0.0],
			[0.1, 0.4, 0.0],
			[0.6, 0.6, 0.0],
			[0.9, 0.6, 0.0],
			[0.6, 0.9, 0.0],
		],
		dtype,
	)
	triIndices = jnp.array(
		[
			[0, 1, 2],
			[3, 4, 5],
		],
		jnp.int32,
	)
	return points[triIndices]


def makeTwoTrisSeparated(dtype=jnp.float32) -> jnp.ndarray:
	points = jnp.array(
		[
			[0.0, 0.0, 0.0],
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[10.0, 0.0, 0.0],
			[11.0, 0.0, 0.0],
			[10.0, 1.0, 0.0],
		],
		dtype,
	)
	triIndices = jnp.array(
		[
			[0, 1, 2],
			[3, 4, 5],
		],
		jnp.int32,
	)
	return points[triIndices]


def buildSpatial(surfaceTris: jnp.ndarray, strategy: spatial.SpatialStrategy):
	return spatial.buildSpatialAcceleration(
		surfaceTris,
		cellSize=0.5,
		tableSize=64,
		maxTotalEntries=256,
		maxBucketSearch=16,
		useNeighborhood=True,
		strategy=strategy,
	)


class SpatialTests(unittest.TestCase):
	def testSpatialNearestSingle(self):
		# Basic single-point query above a unit square mesh.
		surfaceTris = makeUnitSquareTris()
		gridData, strategy, config = buildSpatial(
			surfaceTris, spatial.SpatialStrategy.AUTO
		)

		queryPoint = jnp.array([0.25, 0.25, 0.5], surfaceTris.dtype)
		result = spatial.querySpatial(
			queryPoint[None, :], gridData, surfaceTris, strategy, config
		)

		bestPoint = result.bestPoint[0]
		bestDistSq = result.bestDistSq[0]

		self.assertGreaterEqual(int(result.bestTriIdx[0]), 0)
		self.assertTrue(bool(jnp.allclose(
			bestPoint,
			jnp.array([0.25, 0.25, 0.0], surfaceTris.dtype),
		)))
		self.assertAlmostEqual(float(bestDistSq), 0.25, delta=1e-5)

	def testSpatialNearestBatch(self):
		# Batch query across two points with known squared distances.
		surfaceTris = makeUnitSquareTris()
		gridData, strategy, config = buildSpatial(
			surfaceTris, spatial.SpatialStrategy.AUTO
		)

		queryPoints = jnp.array(
			[
				[-0.2, 0.5, 0.0],
				[0.75, 0.75, -0.2],
			],
			surfaceTris.dtype,
		)

		result = spatial.querySpatial(
			queryPoints, gridData, surfaceTris, strategy, config
		)
		expected = jnp.array([0.04, 0.04], surfaceTris.dtype)

		self.assertEqual(int(jnp.sum(result.bestTriIdx >= 0)), 2)
		self.assertTrue(bool(jnp.allclose(result.bestDistSq, expected)))

	def testSpatialBVHStrategy(self):
		# BVH strategy should still return correct nearest distances.
		surfaceTris = makeUnitSquareTris()
		gridData, strategy, config = buildSpatial(
			surfaceTris, spatial.SpatialStrategy.BVH
		)

		self.assertEqual(strategy, spatial.SpatialStrategy.BVH)

		queryPoint = jnp.array([0.25, 0.25, -0.5], surfaceTris.dtype)
		result = spatial.querySpatial(
			queryPoint[None, :], gridData, surfaceTris, strategy, config
		)

		self.assertGreaterEqual(int(result.bestTriIdx[0]), 0)
		self.assertAlmostEqual(float(result.bestDistSq[0]), 0.25, delta=1e-5)

	def testSpatialCoincidentVertexMesh(self):
		# Mesh with coincident vertices should still query cleanly.
		surfaceTris = makeCoincidentVertexTris()
		gridData, strategy, config = buildSpatial(
			surfaceTris, spatial.SpatialStrategy.AUTO
		)

		queryPoint = jnp.array([0.25, 0.25, 0.2], surfaceTris.dtype)
		result = spatial.querySpatial(
			queryPoint[None, :], gridData, surfaceTris, strategy, config
		)

		bestPoint = result.bestPoint[0]
		bestDistSq = result.bestDistSq[0]

		self.assertGreaterEqual(int(result.bestTriIdx[0]), 0)
		self.assertTrue(bool(jnp.allclose(
			bestPoint,
			jnp.array([0.25, 0.25, 0.0], surfaceTris.dtype),
		)))
		self.assertAlmostEqual(float(bestDistSq), 0.04, delta=1e-5)

	def testSpatialDegenerateTriangleMesh(self):
		# Zero-area triangle should not break nearest query.
		surfaceTris = makeDegenerateTriMesh()
		gridData, strategy, config = buildSpatial(
			surfaceTris, spatial.SpatialStrategy.AUTO
		)

		queryPoint = jnp.array([0.25, 0.25, 0.3], surfaceTris.dtype)
		result = spatial.querySpatial(
			queryPoint[None, :], gridData, surfaceTris, strategy, config
		)

		bestPoint = result.bestPoint[0]
		bestDistSq = result.bestDistSq[0]

		self.assertGreaterEqual(int(result.bestTriIdx[0]), 0)
		self.assertTrue(bool(jnp.allclose(
			bestPoint,
			jnp.array([0.25, 0.25, 0.0], surfaceTris.dtype),
		)))
		self.assertAlmostEqual(float(bestDistSq), 0.09, delta=1e-5)

	def testProjectPointToTriangleRegions(self):
		# Projection should resolve vertex, edge, interior, and degenerate cases.
		triA = jnp.array([0.0, 0.0, 0.0], jnp.float32)
		triB = jnp.array([1.0, 0.0, 0.0], jnp.float32)
		triC = jnp.array([0.0, 1.0, 0.0], jnp.float32)

		# Vertex region near A.
		point = jnp.array([-0.1, -0.1, 0.0], jnp.float32)
		closest, distSq = spatial.projectPointToTriangle(point, triA, triB, triC)
		self.assertTrue(bool(jnp.allclose(closest, triA)))
		self.assertAlmostEqual(float(distSq), 0.02, delta=1e-6)

		# Edge region on AB.
		point = jnp.array([0.5, -0.2, 0.0], jnp.float32)
		closest, distSq = spatial.projectPointToTriangle(point, triA, triB, triC)
		self.assertTrue(bool(jnp.allclose(closest, jnp.array([0.5, 0.0, 0.0]))))
		self.assertAlmostEqual(float(distSq), 0.04, delta=1e-6)

		# Interior projection with positive normal offset.
		point = jnp.array([0.25, 0.25, 0.5], jnp.float32)
		closest, distSq = spatial.projectPointToTriangle(point, triA, triB, triC)
		self.assertTrue(bool(jnp.allclose(closest, jnp.array([0.25, 0.25, 0.0]))))
		self.assertAlmostEqual(float(distSq), 0.25, delta=1e-6)

		# Degenerate triangle collapses to a single point.
		degenerate = jnp.array([0.0, 0.0, 0.0], jnp.float32)
		point = jnp.array([1.0, 0.0, 0.0], jnp.float32)
		closest, distSq = spatial.projectPointToTriangle(
			point, degenerate, degenerate, degenerate
		)
		self.assertTrue(bool(jnp.allclose(closest, degenerate)))
		self.assertAlmostEqual(float(distSq), 1.0, delta=1e-6)

	def testSpatialHashAndMortonCodes(self):
		# Hashes should stay within range for negative and positive coords.
		coords = jnp.array(
			[
				[0, 0, 0],
				[1, 0, 0],
				[-1, 2, -3],
			],
			jnp.int32,
		)
		hashes = spatial.getSpatialHash(coords, 16)
		self.assertTrue(bool(jnp.all((hashes >= 0) & (hashes < 16))))

		# Bit expansion should match known interleave results.
		values = jnp.array([0, 1, 2, 3, 4], dtype=jnp.uint32)
		expected = jnp.array([0, 1, 8, 9, 64], dtype=jnp.uint32)
		self.assertTrue(bool(jnp.array_equal(spatial.expandBits(values), expected)))

		# Morton codes should be monotonic across min/max bounds.
		points = jnp.array(
			[
				[0.0, 0.0, 0.0],
				[1.0, 1.0, 1.0],
			],
			jnp.float32,
		)
		codes = spatial.computeMortonCodes(points, jnp.zeros(3), jnp.ones(3))
		self.assertEqual(int(codes[0]), 0)
		self.assertGreater(int(codes[1]), int(codes[0]))

		# Morton codes should be stable when bounds collapse.
		codes = spatial.computeMortonCodes(
			jnp.array([[1.0, 1.0, 1.0]], jnp.float32),
			jnp.ones(3),
			jnp.ones(3),
		)
		self.assertEqual(int(codes[0]), 0)

		# Sort indices should match Morton code ordering.
		points = jnp.array(
			[
				[0.0, 0.0, 0.0],
				[2.0, 2.0, 2.0],
				[1.0, 1.0, 1.0],
			],
			jnp.float32,
		)
		codes = spatial.computeMortonCodes(
			points, jnp.min(points, axis=0), jnp.max(points, axis=0)
		)
		sortIdx = spatial.getSpatialSortIndices(points)
		self.assertTrue(bool(jnp.array_equal(sortIdx, jnp.argsort(codes))))

	def testTriangleAabbMergeAndIntersection(self):
		# AABB computation and merging should be consistent.
		surfaceTris = jnp.array(
			[
				[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
				[[2.0, 2.0, 2.0], [3.0, 2.0, 2.0], [2.0, 3.0, 2.0]],
			],
			jnp.float32,
		)
		aabbs = spatial.computeTriangleAABBs(surfaceTris)
		self.assertTrue(bool(jnp.allclose(aabbs[0, 0], jnp.array([0.0, 0.0, 0.0]))))
		self.assertTrue(bool(jnp.allclose(aabbs[0, 1], jnp.array([1.0, 1.0, 0.0]))))

		merged = spatial.mergeAABBs(aabbs[0], aabbs[1])
		self.assertTrue(bool(jnp.allclose(merged[0], jnp.array([0.0, 0.0, 0.0]))))
		self.assertTrue(bool(jnp.allclose(merged[1], jnp.array([3.0, 3.0, 2.0]))))

		# AABB distance test should detect inside and outside points.
		self.assertTrue(bool(spatial.testPointAABBIntersection(
			jnp.array([1.5, 1.5, 1.5], jnp.float32), merged, 0.0
		)))
		self.assertFalse(bool(spatial.testPointAABBIntersection(
			jnp.array([10.0, 0.0, 0.0], jnp.float32), merged, 1.0
		)))

	def testBuildGlobalSpatialGridOverflow(self):
		# Overflow detection should trigger when a bucket exceeds limits.
		surfaceTris = makeTwoTrisSameCell()
		config = spatial.SpatialConfig(
			cellSize=1.0,
			tableSize=8,
			maxTotalEntries=2,
			maxBucketSearch=1,
			useNeighborhood=True,
		)
		gridData = spatial.buildGlobalSpatialGrid(surfaceTris, config)

		self.assertEqual(gridData.maxBucketSize, 2)
		self.assertEqual(gridData.overflowCount, 1)
		self.assertEqual(int(jnp.sum(gridData.sortedHashes == 0)), 2)

	def testQueryNearestInBucket(self):
		# Bucket query should return the closest triangle in the cell.
		surfaceTris = makeTwoTrisSameCell()
		config = spatial.SpatialConfig(
			cellSize=1.0,
			tableSize=8,
			maxTotalEntries=2,
			maxBucketSearch=2,
			useNeighborhood=True,
		)
		gridData = spatial.buildGlobalSpatialGrid(surfaceTris, config)

		queryPoint = jnp.array([0.15, 0.15, 0.0], jnp.float32)
		targetHash = spatial.getSpatialHash(
			jnp.array([[0, 0, 0]], jnp.int32), config.tableSize
		)[0]
		result = spatial.queryNearestInBucket(
			queryPoint, int(targetHash), gridData, surfaceTris, config.maxBucketSearch
		)

		self.assertEqual(int(result.bestTriIdx), 0)
		self.assertTrue(bool(jnp.allclose(result.bestPoint, jnp.array([0.15, 0.15, 0.0]))))

	def testSpatialNeighborhoodVsSingleCell(self):
		# Neighborhood search should find triangles in adjacent cells.
		surfaceTris = makeSingleTri()
		config = spatial.SpatialConfig(
			cellSize=1.0,
			tableSize=16,
			maxTotalEntries=1,
			maxBucketSearch=1,
			useNeighborhood=True,
		)
		gridData = spatial.buildGlobalSpatialGrid(surfaceTris, config)

		queryPoint = jnp.array([1.1, 0.2, 0.0], jnp.float32)
		resultNeighborhood = spatial.querySpatialNearestWithNeighborhood(
			queryPoint, gridData, surfaceTris, config, config.maxBucketSearch
		)
		self.assertEqual(int(resultNeighborhood.bestTriIdx), 0)

		# Single-cell query should miss when the triangle is in a neighbor.
		configNoNeighborhood = spatial.SpatialConfig(
			cellSize=1.0,
			tableSize=16,
			maxTotalEntries=1,
			maxBucketSearch=1,
			useNeighborhood=False,
		)
		resultSingle = spatial.querySpatialNearest(
			queryPoint[None, :], gridData, surfaceTris, configNoNeighborhood
		)
		self.assertEqual(int(resultSingle.bestTriIdx[0]), -1)
		self.assertGreater(float(resultSingle.bestDistSq[0]), 1e9)

	def testBuildLbvhEdgeCases(self):
		# BVH build should handle empty, single, and two-triangle inputs.
		with self.assertRaises(ValueError):
			spatial.buildLBVH(jnp.zeros((0, 3, 3), jnp.float32))

		singleTri = makeSingleTri()
		bvh = spatial.buildLBVH(singleTri)
		self.assertEqual(bvh.numLeaves, 1)
		self.assertTrue(bool(jnp.array_equal(bvh.leafTriIndices, jnp.array([0]))))

		twoTris = makeTwoTrisSeparated()
		bvh = spatial.buildLBVH(twoTris)
		self.assertEqual(bvh.numLeaves, 2)
		self.assertEqual(int(bvh.leftChild[0]), -1)
		self.assertEqual(int(bvh.rightChild[0]), -2)

	def testQueryBvhSingleAndVectorized(self):
		# BVH query should return nearest triangle for single and batched inputs.
		surfaceTris = makeTwoTrisSeparated()
		bvh = spatial.buildLBVH(surfaceTris)

		queryPoint = jnp.array([0.1, 0.1, 0.0], jnp.float32)
		resultSingle = spatial.queryBVHSingle(queryPoint, bvh, surfaceTris, maxDepth=8)
		self.assertEqual(int(resultSingle.bestTriIdx), 0)

		queryPoints = jnp.array(
			[
				[0.1, 0.1, 0.0],
				[10.1, 0.1, 0.0],
			],
			jnp.float32,
		)
		resultBatch = spatial.queryBVH(queryPoints, bvh, surfaceTris, maxDepth=8)
		self.assertTrue(bool(jnp.array_equal(
			resultBatch.bestTriIdx,
			jnp.array([0, 1], jnp.int32),
		)))

	def testAnalyzeDeformationAndStrategySelection(self):
		# Deformation analysis should drive auto strategy selection.
		surfaceTris = jnp.array(
			[
				[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
				[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
			],
			jnp.float32,
		)
		analysis = spatial.analyzeMeshDeformation(surfaceTris, cellSize=1.0)
		self.assertGreater(analysis.extremeFraction, 0.1)
		self.assertEqual(
			spatial.selectSpatialStrategy(analysis),
			spatial.SpatialStrategy.BVH,
		)

		# Explicit strategy should override analysis.
		self.assertEqual(
			spatial.selectSpatialStrategy(analysis, spatial.SpatialStrategy.MULTI_CELL_HASH),
			spatial.SpatialStrategy.MULTI_CELL_HASH,
		)

		# Non-extreme meshes should default to hash strategy.
		surfaceTris = makeSingleTri()
		analysis = spatial.analyzeMeshDeformation(surfaceTris, cellSize=1.0)
		self.assertEqual(
			spatial.selectSpatialStrategy(analysis),
			spatial.SpatialStrategy.MULTI_CELL_HASH,
		)

	def testBuildSpatialAccelerationConfig(self):
		# Build helper should preserve explicit strategy and config fields.
		surfaceTris = makeSingleTri()
		gridData, strategy, config = spatial.buildSpatialAcceleration(
			surfaceTris,
			cellSize=0.25,
			tableSize=32,
			maxTotalEntries=4,
			maxBucketSearch=2,
			useNeighborhood=False,
			strategy=spatial.SpatialStrategy.MULTI_CELL_HASH,
		)

		self.assertEqual(strategy, spatial.SpatialStrategy.MULTI_CELL_HASH)
		self.assertEqual(config.cellSize, 0.25)
		self.assertEqual(config.tableSize, 32)
		self.assertFalse(config.useNeighborhood)
		self.assertTrue(hasattr(gridData, "sortedHashes"))


if __name__ == "__main__":
	unittest.main()
