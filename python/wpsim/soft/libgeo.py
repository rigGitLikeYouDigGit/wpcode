from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import jax
from jax import numpy as jnp

from wpsim import maths

@jax.vmap
def computeRawTetFrame(
		pos:jnp.ndarray[..., 3],
		indices:jnp.ndarray[..., 4]
)->jnp.ndarray[3,3]:
	"""
	Computes an orthonormal 3x3 basis for a single tetrahedron.
	"""
	v0, v1, v2, v3 = pos[indices]

	# Primary axis (e.g., along one edge)
	e1 = v1 - v0
	xVal = e1 / (jnp.linalg.norm(e1) + 1e-10)

	# Secondary axis (to define the plane)
	e2 = v2 - v0
	zVal = jnp.cross(xVal, e2)
	zVal = zVal / (jnp.linalg.norm(zVal) + 1e-10)

	# Tertiary axis (orthonormal completion)
	yVal = jnp.cross(zVal, xVal)

	# Returns [x, y, z] as columns
	return jnp.stack([xVal, yVal, zVal], axis=1)


@jax.vmap
def computeSubsetFramesGramSchmidt(
		pos:jnp.ndarray[V, 3],
		tetIndices:jnp.ndarray[T, 4],
		activeTetIndices:jnp.ndarray,
		)->jnp.ndarray[
	...,
3, 3]:
	"""
	Computes frames only for a specific list of tets.
	"""
	# Gather only the 4 vertices of the specific tets we need
	tetIndices = tetIndices[activeTetIndices]
	v0, v1, v2, _ = pos[tetIndices]

	# Raw frame calculation (Gram-Schmidt)
	e1 = v1 - v0
	x = e1 / (jnp.linalg.norm(e1) + 1e-10)
	e2 = v2 - v0
	z = jnp.cross(x, e2)
	z = z / (jnp.linalg.norm(z) + 1e-10)
	y = jnp.cross(z, x)

	return jnp.stack([x, y, z], axis=1)


def computeSubsetRotations(
		pos: jnp.ndarray[nV, 3],
		dmInv: jnp.ndarray[nT, 3, 3],
		indices: jnp.ndarray[nT, 4],
		activeRotIndices: jnp.ndarray[nActive]
) -> jnp.ndarray[nActive, 3, 3]:
	"""
	Computes Polar Decomposition (R) only for a specific subset of tets.
	"""
	# 1. Gather only the required vertices and precomputed dmInv
	# subsetIndices: jnp.ndarray[nActive, 4]
	subsetIndices = indices[activeRotIndices]
	subsetDmInv = dmInv[activeRotIndices]

	# 2. Compute F for the subset
	# v: jnp.ndarray[nActive, 4, 3]
	v = pos[subsetIndices]
	ds = jnp.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]],
	               axis=2)
	fs = ds @ subsetDmInv

	# 3. Vmap the SVD-based Polar Decomposition over the subset
	# rs: jnp.ndarray[nActive, 3, 3]
	rs = jax.vmap(maths.polarDecomp3x3)(fs)

	return rs

@jax.vmap
def computeSubsetFramesSVD(
		pos:jnp.ndarray[V, 3],
		tetIndices:jnp.ndarray[T, 4],
		activeTetIndices:jnp.ndarray,
		)->jnp.ndarray[
	...,
3, 3]:
	"""
	Computes frames only for a specific list of tets.
	"""
	# Gather only the 4 vertices of the specific tets we need
	tetIndices = tetIndices[activeTetIndices]
	v0, v1, v2, _ = pos[tetIndices]

	# Raw frame calculation (Gram-Schmidt)
	e1 = v1 - v0
	x = e1 / (jnp.linalg.norm(e1) + 1e-10)
	e2 = v2 - v0
	z = jnp.cross(x, e2)
	z = z / (jnp.linalg.norm(z) + 1e-10)
	y = jnp.cross(z, x)

	return jnp.stack([x, y, z], axis=1)



def computeStableVertexFrames(
		pos:jnp.ndarray[..., 3],
		indices:jnp.ndarray[..., 4],
		smoothingPasses=2
)->jnp.ndarray[..., 3, 3]:
	"""
	Computes vertex-level frames by averaging neighboring tet frames.
	Provides a more 'stable' basis for broad sculpts.
	"""
	rawFrames = computeRawTetFrame(pos, indices)
	numVertices = pos.shape[0]

	# 1. Accumulate tet frames to vertices
	# We use a 3x3 matrix sum for the average
	vFrameSum = jnp.zeros((numVertices, 3, 3))
	for i in range(4):
		vFrameSum = vFrameSum.at[indices[:, i]].add(rawFrames)

	# 2. Re-orthonormalize using SVD (The most robust way to average rotations)
	@jax.vmap
	def orthonormalize(m):
		u, _, vt = jnp.linalg.svd(m)
		return u @ vt

	vFrames = orthonormalize(vFrameSum)

	# 3. Optional: Iterative Laplacian Smoothing of the frames
	def smoothStep(currentFrames, _):
		# Average frames of vertices connected by edges
		# (Simplified: using tet-connectivity to find neighbors)
		neighborSum = jnp.zeros_like(currentFrames)
		for i in range(4):
			# This diffuses the 3x3 basis across the tet topology
			neighborSum = neighborSum.at[indices].add(
				currentFrames[indices].sum(axis=1)[:, None, :, :])
		return orthonormalize(neighborSum), None

	if smoothingPasses > 0:
		vFrames, _ = jax.lax.scan(smoothStep, vFrames, None,
		                          length=smoothingPasses)

	return vFrames