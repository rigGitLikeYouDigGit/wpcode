from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from functools import partial

import jax
from jax import numpy as jnp


def projectPointToTriangle(p, a, b, c):
	"""
	Calculates the closest point on triangle (a, b, c) to point p.
	Returns: closestPoint, distanceSq
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

def countTriangleOverlaps(V0, V1, V2, CellSize):
	"""
	Calculates the number of grid cells each triangle overlaps based on AABB.
	"""
	MinP = jnp.minimum(jnp.minimum(V0, V1), V2)
	MaxP = jnp.maximum(jnp.maximum(V0, V1), V2)

	# Number of cells in each dimension
	Diff = jnp.floor(MaxP / CellSize) - jnp.floor(MinP / CellSize) + 1
	Counts = Diff[:, 0] * Diff[:, 1] * Diff[:, 2]
	return Counts.astype(jnp.int32)

def getSpatialHash(Coords, tableSize):
	"""
	Large prime-number hashing for infinite sparse space.
	Table size
	"""
	# Large primes for spatial hashing (Teschner et al.)
	P1, P2, P3 = 73856093, 19349663, 83492791
	H = (Coords[:, 0] * P1) ^ (Coords[:, 1] * P2) ^ (Coords[:, 2] * P3)
	return H % tableSize

def expandBits(v):
	"""
	Expands a 10-bit integer so that there are 2-bit gaps between each bit.
	Used to interleave X, Y, and Z bits.
	"""
	v = v & 0x000003FF # Keep only bottom 10 bits
	v = (v | (v << 16)) & 0x030000FF
	v = (v | (v << 8))  & 0x0300F00F
	v = (v | (v << 4))  & 0x030C30C3
	v = (v | (v << 2))  & 0x09249249
	return v


def computeMortonCodes(points, minBound, maxBound):
	"""
	Computes 30-bit Morton codes for a set of 3D points.
	points: (N, 3) array
	"""
	# 1. Normalize points to [0, 1023] (10 bits of resolution)
	rangeBound = maxBound - minBound
	# Avoid division by zero for flat meshes
	rangeBound = jnp.where(rangeBound == 0, 1.0, rangeBound)

	normalized = (points - minBound) / rangeBound
	quantized = jnp.clip(normalized * 1023.0, 0, 1023).astype(jnp.uint32)

	# 2. Expand bits for X, Y, and Z
	xx = expandBits(quantized[:, 0])
	yy = expandBits(quantized[:, 1])
	zz = expandBits(quantized[:, 2])

	# 3. Interleave by shifting and ORing
	return (xx << 2) | (yy << 1) | zz


def getSpatialSortIndices(points):
	"""
	Returns the indices that would sort the points along a Morton curve.
	"""
	minB = jnp.min(points, axis=0)
	maxB = jnp.max(points, axis=0)

	codes = computeMortonCodes(points, minB, maxB)
	return jnp.argsort(codes)

def buildGlobalSpatialGrid(surfaceTris, data):
	"""
	surfaceTris: (numTris, 3, 3) vertex positions.
	data: dict containing 'cellSize', 'tableSize', 'maxTotalEntries'.
	"""
	numTris = surfaceTris.shape[0]
	maxTotalEntries = data["maxTotalEntries"]
	cellSize = data["cellSize"]
	tableSize = data["tableSize"]

	# --- STEP 1: COUNT OVERLAPS ---
	# Calculate how many cells each triangle's AABB touches
	v0, v1, v2 = surfaceTris[:, 0], surfaceTris[:, 1], surfaceTris[:, 2]
	minP = jnp.minimum(jnp.minimum(v0, v1), v2)
	maxP = jnp.maximum(jnp.maximum(v0, v1), v2)

	nxCounts = (jnp.floor(maxP[:, 0] / cellSize) - jnp.floor(
		minP[:, 0] / cellSize) + 1).astype(jnp.int32)
	nyCounts = (jnp.floor(maxP[:, 1] / cellSize) - jnp.floor(
		minP[:, 1] / cellSize) + 1).astype(jnp.int32)
	nzCounts = (jnp.floor(maxP[:, 2] / cellSize) - jnp.floor(
		minP[:, 2] / cellSize) + 1).astype(jnp.int32)

	counts = nxCounts * nyCounts * nzCounts
	startIndices = jnp.cumsum(counts) - counts

	# --- STEP 2: MAP GLOBAL SLOTS TO TRIANGLES ---
	# Create a linear buffer of all possible entry slots
	globalIndices = jnp.arange(maxTotalEntries)

	# Use binary search to find which triangle owns each slot in the global buffer
	# triOwners[i] tells us which triangle index is responsible for globalIndices[i]
	triOwners = jnp.digitize(globalIndices, startIndices) - 1
	localOffsets = globalIndices - startIndices[triOwners]

	# --- STEP 3: COMPUTE ENTRIES ---
	@jax.vmap
	def computeEntry(triIdx, localOffset):
		# Guard against indices out of the valid triangle range (for padding slots)
		isValidTri = (triIdx >= 0) & (triIdx < numTris)

		# Use jnp.where for safe indexing to prevent OOB errors during the JAX trace
		safeTriIdx = jnp.where(isValidTri, triIdx, 0)

		# Get triangle bounds in grid space
		# We recalculate these to avoid passing massive arrays into the vmap closure
		tV0, tV1, tV2 = surfaceTris[safeTriIdx]
		tMinP = jnp.minimum(jnp.minimum(tV0, tV1), tV2)
		tMaxP = jnp.maximum(jnp.maximum(tV0, tV1), tV2)

		minCoord = jnp.floor(tMinP / cellSize).astype(jnp.int32)
		maxCoord = jnp.floor(tMaxP / cellSize).astype(jnp.int32)

		nx = (maxCoord[0] - minCoord[0] + 1)
		ny = (maxCoord[1] - minCoord[1] + 1)

		# Decompose the 1D localOffset back into 3D local grid coordinates (dx, dy, dz)
		# localOffset = dx + dy*nx + dz*nx*ny
		dz = localOffset // (nx * ny)
		dy = (localOffset % (nx * ny)) // nx
		dx = localOffset % nx

		# Final 3D grid coordinate
		coord = minCoord + jnp.stack([dx, dy, dz])

		# Compute the hash key for this cell
		# getSpatialHash is assumed to be the prime-number xor function defined previously
		hashVal = getSpatialHash(coord[None, :], tableSize)[0]

		# An entry is only valid if the triangle is valid and the offset
		# is within that triangle's specific cell count
		isValidEntry = isValidTri & (localOffset < counts[safeTriIdx])

		# We return -1 for invalid entries; argsort will push these to the end
		finalHash = jnp.where(isValidEntry, hashVal, -1)
		finalTriIdx = jnp.where(isValidEntry, triIdx, -1)

		return finalHash, finalTriIdx

	# Run the computeEntry vmap across the entire pre-allocated global buffer
	hashes, triIndices = computeEntry(triOwners, localOffsets)

	# --- STEP 4: SORT BY HASH ---
	# This groups all triangles belonging to the same spatial cell together
	sortIdx = jnp.argsort(hashes)

	return {
		"sortedHashes": hashes[sortIdx],
		"sortedTriIdx": triIndices[sortIdx]
	}

# advised to reduce cell size if needed, rather than increase bucket count -
# keep bucket at 16
@partial(jax.jit, static_argnames=("maxBucketSearch",))
def queryNearestInBucket(point, targetHash, gridData, surfaceTris,
                         maxBucketSearch):
	"""
	Iterates through a specific hash bucket to find the nearest triangle.
	"""
	sortedHashes = gridData["sortedHashes"]
	sortedTriIdx = gridData["sortedTriIdx"]

	# Find the start and end of the block of entries with this hash
	startIdx = jnp.searchsorted(sortedHashes, targetHash, side='left')

	# Initialize search state
	initVal = {
		"bestDistSq": 1e10,
		"bestPoint": point,
		"bestTriIdx": -1
	}

	def bodyFunc(i, val):
		# Calculate the actual index in the sorted list
		currIdx = startIdx + i
		triIdx = sortedTriIdx[currIdx]

		# Only process if the hash matches (we are still in the bucket)
		# and the index is valid
		isValid = (sortedHashes[currIdx] == targetHash) & (triIdx != -1)

		# Retrieve triangle vertices
		v0, v1, v2 = surfaceTris[jnp.where(isValid, triIdx, 0)]

		# Compute distance
		closestP, distSq = projectPointToTriangle(point, v0, v1, v2)

		# Update if this triangle is closer and valid
		isBetter = isValid & (distSq < val["bestDistSq"])
		return {
			"bestDistSq": jnp.where(isBetter, distSq, val["bestDistSq"]),
			"bestPoint": jnp.where(isBetter, closestP, val["bestPoint"]),
			"bestTriIdx": jnp.where(isBetter, triIdx, val["bestTriIdx"])
		}

	# Search up to maxBucketSearch triangles in the cell
	result = jax.lax.fori_loop(0, maxBucketSearch, bodyFunc, initVal)
	return result

# Wrap the query in a checkpoint to save memory during backprop
checkpointedQuery = jax.checkpoint(queryNearestInBucket, static_argnums=(4,))


@jax.vmap
def querySpatialNearest(queryPoint, gridData, surfaceTris, data):
	"""
	General interface for finding the nearest surface point via spatial hash.
	queryPoint: (3,)
	data: dict containing 'cellSize', 'tableSize', 'maxBucketSearch'
	"""
	# Get hash for the query point's cell
	coord = jnp.floor(queryPoint / data["cellSize"]).astype(jnp.int32)
	targetHash = getSpatialHash(coord[None, :], data["tableSize"])[0]

	# Perform the bucket search
	# For higher accuracy, you could iterate through the 27 neighboring
	# cells here and take the minimum of all results.
	result = queryNearestInBucket(
		queryPoint,
		targetHash,
		gridData,
		surfaceTris,
		data["maxBucketSearch"]
	)

	return result  # Contains 'bestDistSq', 'bestPoint', 'bestTriIdx'
