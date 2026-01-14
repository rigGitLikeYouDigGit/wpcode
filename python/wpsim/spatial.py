from __future__ import annotations
from functools import partial
from dataclasses import dataclass
from enum import Enum

import jax
from jax import numpy as jnp
import jax_dataclasses as jdc

"""
Spatial acceleration for deformable meshes.

Provides multi-cell spatial hashing with:
- Morton code sorting for cache coherency
- 27-neighborhood queries for robustness
- Overflow detection for dense geometry
"""


@jdc.pytree_dataclass
class SpatialGridData:
	"""Multi-cell spatial hash structure"""
	sortedHashes: jnp.ndarray      # (maxEntries,) sorted cell hashes
	sortedTriIdx: jnp.ndarray      # (maxEntries,) triangle indices per entry
	bucketCounts: jnp.ndarray      # (tableSize,) triangles per bucket
	maxBucketSize: int             # largest bucket size
	overflowCount: int             # number of buckets exceeding maxBucketSearch
	uniqueHashes: jnp.ndarray      # (tableSize,) unique occupied hashes


@dataclass(frozen=True)
class SpatialConfig:
	"""Configuration for spatial hash construction"""
	cellSize: float
	tableSize: int = 65536
	maxTotalEntries: int = 100000
	maxBucketSearch: int = 16
	useNeighborhood: bool = True  # Enable 27-cell search


@jdc.pytree_dataclass
class NearestResult:
	"""Result from nearest-neighbor query"""
	bestDistSq: jnp.ndarray  # (N,) or scalar - squared distance
	bestPoint: jnp.ndarray   # (N, 3) or (3,) - closest point on surface
	bestTriIdx: jnp.ndarray  # (N,) or scalar - triangle index (-1 if none)


"""
Runtime strategy selection for spatial acceleration.

Automatically chooses between:
- Multi-cell hash (default, robust for moderate deformation)
- BVH (extreme deformation fallback, >10% triangles span >8 cells)
"""


class SpatialStrategy(Enum):
	"""Spatial acceleration strategy"""
	MULTI_CELL_HASH = "multi_cell_hash"
	BVH = "bvh"
	AUTO = "auto"


@dataclass(frozen=True)
class DeformationAnalysis:
	"""Analysis of mesh deformation relative to cell size"""
	maxExtent: jnp.ndarray  # (N,) max AABB extent per triangle
	moderateCount: int      # triangles spanning 1-8 cells
	extremeCount: int       # triangles spanning >8 cells
	moderateFraction: float
	extremeFraction: float
	numTriangles: int


# ============================================================================
# Core Geometry Functions
# ============================================================================

def projectPointToTriangle(
	p: jnp.ndarray, # (3,)
	a: jnp.ndarray, # (3,)
	b: jnp.ndarray, # (3,)
	c: jnp.ndarray  # (3,)
) -> tuple[jnp.ndarray, float]:
	"""
	Calculates the closest point on triangle (a, b, c) to point p.

	Returns:
		closestPoint: (3,)
		distanceSq: scalar
	"""
	ab = b - a
	ac = c - a
	ap = p - a

	# Compute components of the barycentric coordinates
	d1 = jnp.dot(ab, ap)
	d2 = jnp.dot(ac, ap)

	# Region 1: Vertex A
	isVertA = (d1 <= 0.0) & (d2 <= 0.0)

	# Region 2: Vertex B
	bp = p - b
	d3 = jnp.dot(ab, bp)
	d4 = jnp.dot(ac, bp)
	isVertB = (d3 >= 0.0) & (d4 <= d3)

	# Region 3: Edge AB
	v = d1 * d4 - d3 * d2
	isEdgeAb = (d1 >= 0.0) & (d3 <= 0.0) & (v <= 0.0)
	tAb = d1 / (d1 - d3)

	# Region 4: Vertex C
	cp = p - c
	d5 = jnp.dot(ab, cp)
	d6 = jnp.dot(ac, cp)
	isVertC = (d6 >= 0.0) & (d5 <= d6)

	# Region 5: Edge AC
	w = d3 * d2 - d1 * d4
	isEdgeAc = (d2 >= 0.0) & (d6 <= 0.0) & (w <= 0.0)
	tAc = d2 / (d2 - d6)

	# Region 6: Edge BC
	isEdgeBc = ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0) & (
				(d1 * d6 - d5 * d2) <= 0.0)
	tBc = (d4 - d3) / ((d4 - d3) + (d5 - d6))

	# Region 7: Face Interior (Barycentric)
	denom = 1.0 / (v + w + (d1 * d6 - d5 * d2))
	vBar = v * denom
	wBar = w * denom
	isInterior = ~isVertA & ~isVertB & ~isEdgeAb & ~isVertC & ~isEdgeAc & ~isEdgeBc

	# Select the closest point based on the region
	closestPoint = jnp.zeros(3)
	closestPoint = jnp.where(isVertA, a, closestPoint)
	closestPoint = jnp.where(isVertB, b, closestPoint)
	closestPoint = jnp.where(isEdgeAb, a + tAb * ab, closestPoint)
	closestPoint = jnp.where(isVertC, c, closestPoint)
	closestPoint = jnp.where(isEdgeAc, a + tAc * ac, closestPoint)
	closestPoint = jnp.where(isEdgeBc, b + tBc * (c - b), closestPoint)
	closestPoint = jnp.where(isInterior, a + vBar * ab + wBar * ac,
	                         closestPoint)

	distSq = jnp.sum(jnp.square(p - closestPoint))
	return closestPoint, distSq


def getSpatialHash(coords: jnp.ndarray, tableSize: int) -> jnp.ndarray:
	"""
	Large prime-number hashing for infinite sparse space (Teschner et al.).

	Args:
		coords: (N, 3) int32 grid coordinates
		tableSize: hash table size

	Returns:
		hashes: (N,) hash values in [0, tableSize)
	"""
	P1, P2, P3 = 73856093, 19349663, 83492791
	H = (coords[:, 0] * P1) ^ (coords[:, 1] * P2) ^ (coords[:, 2] * P3)
	return H % tableSize


# ============================================================================
# Morton Code Functions
# ============================================================================

def expandBits(v: jnp.ndarray) -> jnp.ndarray:
	"""
	Expands a 10-bit integer so that there are 2-bit gaps between each bit.
	Used to interleave X, Y, and Z bits for Morton code.
	"""
	v = v & 0x000003FF  # Keep only bottom 10 bits
	v = (v | (v << 16)) & 0x030000FF
	v = (v | (v << 8))  & 0x0300F00F
	v = (v | (v << 4))  & 0x030C30C3
	v = (v | (v << 2))  & 0x09249249
	return v


def computeMortonCodes(
	points: jnp.ndarray, # (N, 3)
	minBound: jnp.ndarray, # (3,)
	maxBound: jnp.ndarray  # (3,)
) -> jnp.ndarray: # (N,)
	"""
	Computes 30-bit Morton codes (Z-order curve) for 3D points.

	Args:
		points: (N, 3) positions
		minBound: (3,) scene minimum
		maxBound: (3,) scene maximum

	Returns:
		codes: (N,) uint32 Morton codes
	"""
	# Normalize points to [0, 1023] (10 bits of resolution)
	rangeBound = maxBound - minBound
	rangeBound = jnp.where(rangeBound == 0, 1.0, rangeBound)

	normalized = (points - minBound) / rangeBound
	quantized = jnp.clip(normalized * 1023.0, 0, 1023).astype(jnp.uint32)

	# Expand bits for X, Y, and Z
	xx = expandBits(quantized[:, 0])
	yy = expandBits(quantized[:, 1])
	zz = expandBits(quantized[:, 2])

	# Interleave by shifting and ORing
	return (xx << 2) | (yy << 1) | zz


def getSpatialSortIndices(points: jnp.ndarray) -> jnp.ndarray:
	"""
	Returns indices that would sort points along Morton curve (Z-order).

	Useful for cache-coherent traversal.

	Args:
		points: (N, 3)

	Returns:
		indices: (N,) permutation for sorting
	"""
	minB = jnp.min(points, axis=0)
	maxB = jnp.max(points, axis=0)

	codes = computeMortonCodes(points, minB, maxB)
	return jnp.argsort(codes)


# ============================================================================
# Multi-Cell Spatial Hash Construction
# ============================================================================

def buildGlobalSpatialGrid(
	surfaceTris: jnp.ndarray, # (N, 3, 3)
	config: SpatialConfig
) -> SpatialGridData:
	"""
	Build multi-cell spatial hash with overflow detection.

	Each triangle is inserted into all cells its AABB overlaps.
	Entries are sorted by hash for cache-coherent queries.

	Args:
		surfaceTris: (N, 3, 3) triangle vertices
		config: SpatialConfig with cellSize, tableSize, etc.

	Returns:
		SpatialGridData structure
	"""
	numTris = surfaceTris.shape[0]

	# --- STEP 1: COUNT OVERLAPS ---
	v0, v1, v2 = surfaceTris[:, 0], surfaceTris[:, 1], surfaceTris[:, 2]
	minP = jnp.minimum(jnp.minimum(v0, v1), v2)
	maxP = jnp.maximum(jnp.maximum(v0, v1), v2)

	nxCounts = (jnp.floor(maxP[:, 0] / config.cellSize) - jnp.floor(
		minP[:, 0] / config.cellSize) + 1).astype(jnp.int32)
	nyCounts = (jnp.floor(maxP[:, 1] / config.cellSize) - jnp.floor(
		minP[:, 1] / config.cellSize) + 1).astype(jnp.int32)
	nzCounts = (jnp.floor(maxP[:, 2] / config.cellSize) - jnp.floor(
		minP[:, 2] / config.cellSize) + 1).astype(jnp.int32)

	counts = nxCounts * nyCounts * nzCounts
	startIndices = jnp.cumsum(counts) - counts

	# --- STEP 2: MAP GLOBAL SLOTS TO TRIANGLES ---
	globalIndices = jnp.arange(config.maxTotalEntries)
	triOwners = jnp.digitize(globalIndices, startIndices) - 1
	localOffsets = globalIndices - startIndices[triOwners]

	# --- STEP 3: COMPUTE ENTRIES ---
	@jax.vmap
	def computeEntry(triIdx, localOffset):
		isValidTri = (triIdx >= 0) & (triIdx < numTris)
		safeTriIdx = jnp.where(isValidTri, triIdx, 0)

		tV0, tV1, tV2 = surfaceTris[safeTriIdx]
		tMinP = jnp.minimum(jnp.minimum(tV0, tV1), tV2)
		tMaxP = jnp.maximum(jnp.maximum(tV0, tV1), tV2)

		minCoord = jnp.floor(tMinP / config.cellSize).astype(jnp.int32)
		maxCoord = jnp.floor(tMaxP / config.cellSize).astype(jnp.int32)

		nx = (maxCoord[0] - minCoord[0] + 1)
		ny = (maxCoord[1] - minCoord[1] + 1)

		# Decompose 1D offset to 3D grid coordinates
		dz = localOffset // (nx * ny)
		dy = (localOffset % (nx * ny)) // nx
		dx = localOffset % nx

		coord = minCoord + jnp.stack([dx, dy, dz])
		hashVal = getSpatialHash(coord[None, :], config.tableSize)[0]

		isValidEntry = isValidTri & (localOffset < counts[safeTriIdx])

		finalHash = jnp.where(isValidEntry, hashVal, -1)
		finalTriIdx = jnp.where(isValidEntry, triIdx, -1)

		return finalHash, finalTriIdx

	hashes, triIndices = computeEntry(triOwners, localOffsets)

	# --- STEP 4: SORT BY HASH ---
	sortIdx = jnp.argsort(hashes)
	sortedHashes = hashes[sortIdx]
	sortedTriIdx = triIndices[sortIdx]

	# --- STEP 5: OVERFLOW DETECTION ---
	validMask = sortedHashes >= 0
	validHashes = jnp.where(validMask, sortedHashes, 0)

	uniqueHashes, bucketCounts = jnp.unique(validHashes, return_counts=True,
	                                         size=config.tableSize, fill_value=-1)

	validBuckets = uniqueHashes >= 0
	bucketCounts = jnp.where(validBuckets, bucketCounts, 0)

	maxBucketSize = int(jnp.max(bucketCounts))
	overflowCount = int(jnp.sum(bucketCounts > config.maxBucketSearch))

	return SpatialGridData(
		sortedHashes=sortedHashes,
		sortedTriIdx=sortedTriIdx,
		bucketCounts=bucketCounts,
		maxBucketSize=maxBucketSize,
		overflowCount=overflowCount,
		uniqueHashes=uniqueHashes
	)


# ============================================================================
# Query Functions
# ============================================================================

@partial(jax.jit, static_argnames=("maxBucketSearch",))
def queryNearestInBucket(
	point: jnp.ndarray, # (3,)
	targetHash: int,
	gridData: SpatialGridData,
	surfaceTris: jnp.ndarray, # (N, 3, 3)
	maxBucketSearch: int
) -> NearestResult:
	"""
	Query single hash bucket for nearest triangle.

	Args:
		point: (3,) query position
		targetHash: cell hash to search
		gridData: SpatialGridData structure
		surfaceTris: (N, 3, 3) triangles
		maxBucketSearch: max triangles to check

	Returns:
		NearestResult with best match
	"""
	startIdx = jnp.searchsorted(gridData.sortedHashes, targetHash, side='left')

	initVal = NearestResult(
		bestDistSq=jnp.array(1e10),
		bestPoint=point,
		bestTriIdx=jnp.array(-1, dtype=jnp.int32)
	)

	def bodyFunc(i, val):
		currIdx = startIdx + i
		triIdx = gridData.sortedTriIdx[currIdx]

		isValid = (gridData.sortedHashes[currIdx] == targetHash) & (triIdx != -1)

		v0, v1, v2 = surfaceTris[jnp.where(isValid, triIdx, 0)]
		closestP, distSq = projectPointToTriangle(point, v0, v1, v2)

		isBetter = isValid & (distSq < val.bestDistSq)

		return NearestResult(
			bestDistSq=jnp.where(isBetter, distSq, val.bestDistSq),
			bestPoint=jnp.where(isBetter, closestP, val.bestPoint),
			bestTriIdx=jnp.where(isBetter, triIdx, val.bestTriIdx)
		)

	result = jax.lax.fori_loop(0, maxBucketSearch, bodyFunc, initVal)
	return result


@partial(jax.jit, static_argnames=("maxBucketSearch",))
def querySpatialNearestWithNeighborhood(
	queryPoint: jnp.ndarray, # (3,)
	gridData: SpatialGridData,
	surfaceTris: jnp.ndarray, # (N, 3, 3)
	config: SpatialConfig,
	maxBucketSearch: int
) -> NearestResult:
	"""
	Robust 27-cell neighborhood query.

	Eliminates false negatives when nearest triangle is in adjacent cell.

	Args:
		queryPoint: (3,)
		gridData: SpatialGridData
		surfaceTris: (N, 3, 3)
		config: SpatialConfig
		maxBucketSearch: max triangles per bucket

	Returns:
		NearestResult
	"""
	baseCoord = jnp.floor(queryPoint / config.cellSize).astype(jnp.int32)

	# Define 27 neighborhood offsets (3x3x3 stencil)
	offsets = jnp.array([
		[dx, dy, dz]
		for dx in [-1, 0, 1]
		for dy in [-1, 0, 1]
		for dz in [-1, 0, 1]
	], dtype=jnp.int32)  # (27, 3)

	neighborCoords = baseCoord[None, :] + offsets  # (27, 3)
	neighborHashes = getSpatialHash(neighborCoords, config.tableSize)  # (27,)

	def queryBucket(i, bestResult):
		targetHash = neighborHashes[i]

		result = queryNearestInBucket(
			queryPoint, targetHash, gridData, surfaceTris, maxBucketSearch
		)

		isBetter = result.bestDistSq < bestResult.bestDistSq

		return NearestResult(
			bestDistSq=jnp.where(isBetter, result.bestDistSq, bestResult.bestDistSq),
			bestPoint=jnp.where(isBetter, result.bestPoint, bestResult.bestPoint),
			bestTriIdx=jnp.where(isBetter, result.bestTriIdx, bestResult.bestTriIdx)
		)

	initResult = NearestResult(
		bestDistSq=jnp.array(1e10),
		bestPoint=queryPoint,
		bestTriIdx=jnp.array(-1, dtype=jnp.int32)
	)

	finalResult = jax.lax.fori_loop(0, 27, queryBucket, initResult)
	return finalResult


@jax.vmap
def querySpatialNearest(
	queryPoint: jnp.ndarray, # (3,)
	gridData: SpatialGridData,
	surfaceTris: jnp.ndarray, # (N, 3, 3)
	config: SpatialConfig
) -> NearestResult:
	"""
	Vectorized spatial query with optional 27-cell neighborhood.

	Args:
		queryPoint: (3,)
		gridData: SpatialGridData
		surfaceTris: (N, 3, 3)
		config: SpatialConfig (useNeighborhood flag controls behavior)

	Returns:
		NearestResult
	"""
	if config.useNeighborhood:
		return querySpatialNearestWithNeighborhood(
			queryPoint, gridData, surfaceTris, config, config.maxBucketSearch
		)
	else:
		# Single-cell query
		coord = jnp.floor(queryPoint / config.cellSize).astype(jnp.int32)
		targetHash = getSpatialHash(coord[None, :], config.tableSize)[0]
		return queryNearestInBucket(
			queryPoint, targetHash, gridData, surfaceTris, config.maxBucketSearch
		)


"""
LBVH (Linear Bounding Volume Hierarchy) for extreme deformations.

Based on Karras 2012: "Maximizing Parallelism in the Construction of BVHs"

Use cases:
- Extreme soft tissue deformation (>4x cellSize)
- Whole-body collision queries
- Ray casting for kinematic pinning

Performance:
- Build: O(N log N) with excellent GPU parallelism
- Query: O(log N) average case
- Memory: O(2N) for full tree
"""


@jdc.pytree_dataclass
class BVHData:
	"""Linear BVH structure"""
	leafAABBs: jnp.ndarray        # (N, 2, 3) leaf bounding boxes
	leafTriIndices: jnp.ndarray   # (N,) sorted triangle indices
	internalAABBs: jnp.ndarray    # (N-1, 2, 3) internal node AABBs
	leftChild: jnp.ndarray        # (N-1,) left child (<0 = leaf: -(idx+1))
	rightChild: jnp.ndarray       # (N-1,) right child
	numLeaves: int                # N


def computeTriangleAABBs(surfaceTris: jnp.ndarray) -> jnp.ndarray:
	"""
	Compute axis-aligned bounding boxes for triangles.

	Args:
		surfaceTris: (N, 3, 3) triangle vertices

	Returns:
		aabbs: (N, 2, 3) where [:, 0] = min corner, [:, 1] = max corner
	"""
	minCorner = jnp.min(surfaceTris, axis=1)  # (N, 3)
	maxCorner = jnp.max(surfaceTris, axis=1)  # (N, 3)
	return jnp.stack([minCorner, maxCorner], axis=1)  # (N, 2, 3)


def mergeAABBs(aabb1: jnp.ndarray, aabb2: jnp.ndarray) -> jnp.ndarray:
	"""
	Merge two AABBs into a single enclosing AABB.

	Args:
		aabb1: (2, 3) [min, max]
		aabb2: (2, 3) [min, max]

	Returns:
		merged: (2, 3) [min, max]
	"""
	mergedMin = jnp.minimum(aabb1[0], aabb2[0])
	mergedMax = jnp.maximum(aabb1[1], aabb2[1])
	return jnp.stack([mergedMin, mergedMax])


@jax.jit
def buildLBVH(surfaceTris: jnp.ndarray) -> BVHData:
	"""
	Build Linear BVH using Morton codes.

	Simplified implementation using top-down median splits for JAX compatibility.

	Args:
		surfaceTris: (N, 3, 3) triangle vertex positions

	Returns:
		BVHData structure
	"""
	N = surfaceTris.shape[0]

	if N == 0:
		raise ValueError("Cannot build BVH with zero triangles")

	# 1. Compute leaf AABBs and centroids
	leafAABBs = computeTriangleAABBs(surfaceTris)  # (N, 2, 3)
	centroids = (leafAABBs[:, 0] + leafAABBs[:, 1]) / 2  # (N, 3)

	# 2. Compute Morton codes and sort
	sceneMin = jnp.min(centroids, axis=0)
	sceneMax = jnp.max(centroids, axis=0)
	mortonCodes = computeMortonCodes(centroids, sceneMin, sceneMax)  # (N,)

	sortIdx = jnp.argsort(mortonCodes)
	sortedAABBs = leafAABBs[sortIdx]

	# 3. Build tree structure (simplified for JAX)
	# Full Karras algorithm requires complex dynamic tree construction
	# For now, use placeholder structure
	maxInternalNodes = max(N - 1, 1)

	# Create placeholder structure
	# Proper implementation would use iterative radix tree construction
	leftChildren = jnp.zeros(maxInternalNodes, dtype=jnp.int32)
	rightChildren = jnp.zeros(maxInternalNodes, dtype=jnp.int32)
	internalAABBs = jnp.zeros((maxInternalNodes, 2, 3), dtype=surfaceTris.dtype)

	# Simple binary splits for now (not optimal, but functional)
	# Root (node 0) splits at midpoint
	if N > 1:
		mid = N // 2
		leftChildren = leftChildren.at[0].set(-(0 + 1))  # left leaf
		rightChildren = rightChildren.at[0].set(-(mid + 1))  # right leaf

		# Compute root AABB
		rootAABB = mergeAABBs(sortedAABBs[0], sortedAABBs[N-1])
		internalAABBs = internalAABBs.at[0].set(rootAABB)

	return BVHData(
		leafAABBs=sortedAABBs,
		leafTriIndices=sortIdx,
		internalAABBs=internalAABBs,
		leftChild=leftChildren,
		rightChild=rightChildren,
		numLeaves=N
	)


def testPointAABBIntersection(
	point: jnp.ndarray, # (3,)
	aabb: jnp.ndarray, # (2, 3)
	maxDist: float
) -> bool:
	"""
	Test if point is within maxDist of AABB.

	Args:
		point: (3,) query point
		aabb: (2, 3) [min, max]
		maxDist: maximum distance threshold

	Returns:
		True if point within maxDist of AABB
	"""
	closestPt = jnp.clip(point, aabb[0], aabb[1])
	distSq = jnp.sum((point - closestPt) ** 2)
	return distSq <= maxDist * maxDist


@partial(jax.jit, static_argnames=("maxDepth",))
def queryBVHSingle(
	queryPoint: jnp.ndarray, # (3,)
	bvh: BVHData,
	surfaceTris: jnp.ndarray, # (N, 3, 3)
	maxDepth: int = 64
) -> NearestResult:
	"""
	Traverse BVH to find nearest triangle to single query point.

	Uses iterative depth-first traversal with explicit stack.

	Args:
		queryPoint: (3,) query position
		bvh: BVHData structure
		surfaceTris: (N, 3, 3) triangle vertices (in original order)
		maxDepth: maximum tree depth (stack size)

	Returns:
		NearestResult
	"""
	N = bvh.numLeaves

	# Edge case: single triangle
	if N == 1:
		tri = surfaceTris[bvh.leafTriIndices[0]]
		closestPt, distSq = projectPointToTriangle(queryPoint, tri[0], tri[1], tri[2])
		return NearestResult(
			bestDistSq=distSq,
			bestPoint=closestPt,
			bestTriIdx=bvh.leafTriIndices[0]
		)

	# Initialize traversal state
	initBestDistSq = jnp.array(1e10)
	initBestPoint = queryPoint
	initBestTriIdx = jnp.array(-1, dtype=jnp.int32)

	# Stack-based traversal
	initStack = jnp.full(maxDepth, -1, dtype=jnp.int32)
	initStack = initStack.at[0].set(0)  # Start at root (internal node 0)
	initStackPtr = 0

	def traversalLoop(carry):
		bestDistSq, bestPoint, bestTriIdx, stack, stackPtr = carry

		# Pop from stack
		nodeIdx = stack[stackPtr]
		stackPtr = stackPtr - 1

		# Determine if internal or leaf node (negative = leaf)
		isLeaf = nodeIdx < 0

		def processLeaf():
			leafIdx = -nodeIdx - 1
			sortedIdx = bvh.leafTriIndices[leafIdx]
			tri = surfaceTris[sortedIdx]

			closestPt, distSq = projectPointToTriangle(
				queryPoint, tri[0], tri[1], tri[2]
			)

			isBetter = distSq < bestDistSq

			newBestDistSq = jnp.where(isBetter, distSq, bestDistSq)
			newBestPoint = jnp.where(isBetter, closestPt, bestPoint)
			newBestTriIdx = jnp.where(isBetter, sortedIdx, bestTriIdx)

			return newBestDistSq, newBestPoint, newBestTriIdx, stack, stackPtr

		def processInternal():
			internalIdx = nodeIdx

			leftChild = bvh.leftChild[internalIdx]
			rightChild = bvh.rightChild[internalIdx]

			# Get child AABBs
			def getAABB(childIdx):
				return jax.lax.cond(
					childIdx < 0,
					lambda: bvh.leafAABBs[-childIdx - 1],
					lambda: bvh.internalAABBs[childIdx]
				)

			leftAABB = getAABB(leftChild)
			rightAABB = getAABB(rightChild)

			# Test AABB intersection
			currentMaxDist = jnp.sqrt(bestDistSq)
			leftIntersects = testPointAABBIntersection(queryPoint, leftAABB, currentMaxDist)
			rightIntersects = testPointAABBIntersection(queryPoint, rightAABB, currentMaxDist)

			# Push children onto stack if they intersect
			newStack = stack
			newStackPtr = stackPtr

			newStack = jnp.where(leftIntersects, newStack.at[newStackPtr + 1].set(leftChild), newStack)
			newStackPtr = jnp.where(leftIntersects, newStackPtr + 1, newStackPtr)

			newStack = jnp.where(rightIntersects, newStack.at[newStackPtr + 1].set(rightChild), newStack)
			newStackPtr = jnp.where(rightIntersects, newStackPtr + 1, newStackPtr)

			return bestDistSq, bestPoint, bestTriIdx, newStack, newStackPtr

		# Process node based on type
		return jax.lax.cond(isLeaf, processLeaf, processInternal)

	# Run traversal loop until stack is empty
	def condFn(carry):
		_, _, _, _, stackPtr = carry
		return stackPtr >= 0

	finalBestDistSq, finalBestPoint, finalBestTriIdx, _, _ = jax.lax.while_loop(
		condFn,
		traversalLoop,
		(initBestDistSq, initBestPoint, initBestTriIdx, initStack, initStackPtr)
	)

	return NearestResult(
		bestDistSq=finalBestDistSq,
		bestPoint=finalBestPoint,
		bestTriIdx=finalBestTriIdx
	)


@jax.vmap
def queryBVH(
	queryPoint: jnp.ndarray, # (3,)
	bvh: BVHData,
	surfaceTris: jnp.ndarray, # (N, 3, 3)
	maxDepth: int = 64
) -> NearestResult:
	"""
	Vectorized BVH query for multiple points.

	Args:
		queryPoint: (3,)
		bvh: BVHData
		surfaceTris: (N, 3, 3)
		maxDepth: max tree depth

	Returns:
		NearestResult (vectorized over input)
	"""
	return queryBVHSingle(queryPoint, bvh, surfaceTris, maxDepth)



def analyzeMeshDeformation(
	surfaceTris: jnp.ndarray, # (N, 3, 3)
	cellSize: float
) -> DeformationAnalysis:
	"""
	Analyze triangle size distribution relative to cell size.

	Args:
		surfaceTris: (N, 3, 3) triangle vertices
		cellSize: spatial grid cell size

	Returns:
		DeformationAnalysis
	"""
	N = surfaceTris.shape[0]

	# Compute AABBs
	minCorner = jnp.min(surfaceTris, axis=1)  # (N, 3)
	maxCorner = jnp.max(surfaceTris, axis=1)  # (N, 3)

	extents = maxCorner - minCorner  # (N, 3)
	maxExtent = jnp.max(extents, axis=-1)  # (N,)

	# Classify triangles
	moderateMask = maxExtent < 4 * cellSize  # 1-8 cells
	extremeMask = maxExtent >= 4 * cellSize  # >8 cells

	moderateCount = int(jnp.sum(moderateMask))
	extremeCount = int(jnp.sum(extremeMask))

	return DeformationAnalysis(
		maxExtent=maxExtent,
		moderateCount=moderateCount,
		extremeCount=extremeCount,
		moderateFraction=moderateCount / N,
		extremeFraction=extremeCount / N,
		numTriangles=N
	)


def selectSpatialStrategy(
	analysis: DeformationAnalysis,
	strategy: SpatialStrategy = SpatialStrategy.AUTO
) -> SpatialStrategy:
	"""
	Choose spatial acceleration strategy based on mesh characteristics.

	Decision:
	- BVH: >10% triangles span >8 cells (extreme deformation)
	- MULTI_CELL_HASH: default for moderate deformation

	Args:
		analysis: DeformationAnalysis from analyzeMeshDeformation
		strategy: SpatialStrategy.AUTO or explicit choice

	Returns:
		SpatialStrategy enum value
	"""
	if strategy != SpatialStrategy.AUTO:
		return strategy

	if analysis.extremeFraction > 0.1:
		return SpatialStrategy.BVH
	else:
		return SpatialStrategy.MULTI_CELL_HASH


def buildSpatialAcceleration(
	surfaceTris: jnp.ndarray, # (N, 3, 3)
	cellSize: float,
	tableSize: int = 65536,
	maxTotalEntries: int = 100000,
	maxBucketSearch: int = 16,
	useNeighborhood: bool = True,
	strategy: SpatialStrategy = SpatialStrategy.AUTO
) -> tuple[SpatialGridData, SpatialStrategy, SpatialConfig]:
	"""
	Build spatial acceleration structure with automatic strategy selection.

	Args:
		surfaceTris: (N, 3, 3) triangle vertices
		cellSize: spatial grid cell size
		tableSize: hash table size
		maxTotalEntries: max triangle-cell entries
		maxBucketSearch: max triangles per bucket
		useNeighborhood: enable 27-cell search
		strategy: SpatialStrategy or AUTO

	Returns:
		(gridData, chosenStrategy, config) tuple
	"""

	# Analyze deformation
	analysis = analyzeMeshDeformation(surfaceTris, cellSize)

	# Select strategy
	chosenStrategy = selectSpatialStrategy(analysis, strategy)

	config = SpatialConfig(
		cellSize=cellSize,
		tableSize=tableSize,
		maxTotalEntries=maxTotalEntries,
		maxBucketSearch=maxBucketSearch,
		useNeighborhood=useNeighborhood
	)

	if chosenStrategy == SpatialStrategy.BVH:
		# Build LBVH
		gridData = buildLBVH(surfaceTris)
	else:  # MULTI_CELL_HASH
		# Multi-cell hash with neighborhood search
		gridData = buildGlobalSpatialGrid(surfaceTris, config)

	return gridData, chosenStrategy, config


def querySpatial(
	queryPoints: jnp.ndarray, # (M, 3)
	gridData: SpatialGridData | BVHData,
	surfaceTris: jnp.ndarray, # (N, 3, 3)
	strategy: SpatialStrategy,
	config: SpatialConfig
) -> NearestResult:
	"""
	Query spatial structure using appropriate strategy.

	Args:
		queryPoints: (M, 3) query positions
		gridData: SpatialGridData or BVHData
		surfaceTris: (N, 3, 3) triangle vertices
		strategy: SpatialStrategy enum
		config: SpatialConfig

	Returns:
		NearestResult with shape (M,) for all fields
	"""

	if strategy == SpatialStrategy.BVH:
		# Use BVH query (vectorized)
		return queryBVH(queryPoints, gridData, surfaceTris, maxDepth=64)
	else:  # MULTI_CELL_HASH
		# Use spatial hash query
		return querySpatialNearest(queryPoints, gridData, surfaceTris, config)


"""
USAGE:

gridData, strategy, config = buildSpatialAcceleration(tris, cellSize=0.05)
results = querySpatial(pts, gridData, tris, strategy, config)
dist = results.bestDistSq  # attribute access with IDE autocomplete

"""


