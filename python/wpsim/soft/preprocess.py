from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import jax
from jax import numpy as jnp

from wpsim.soft import state


""" here we put any preprocessing functions to get
rest values, 
and any functions to take place outside the main jax sim
"""

def deriveRestAngles(
		restPos: jnp.ndarray[nV, 3],
		clothEdgeIndices: jnp.ndarray[nClothEdges, 4]
) -> jnp.ndarray[nClothEdges]:
	"""
	Calculates the dihedral rest angle for all cloth hinges.
	edgeIndices: [v0, v1, v2, v3] where v0-v1 is the shared edge.
	"""
	x0, x1, x2, x3 = restPos[clothEdgeIndices].transpose(1, 0, 2)

	e0 = x1 - x0
	e1 = x2 - x0
	e2 = x3 - x0

	n1 = jnp.cross(e0, e1)
	n2 = jnp.cross(e2, e0)

	# Normalize normals
	n1 = n1 / (jnp.linalg.norm(n1, axis=1, keepdims=True) + 1e-10)
	n2 = n2 / (jnp.linalg.norm(n2, axis=1, keepdims=True) + 1e-10)

	# Compute cos(theta)
	cosTheta = jnp.sum(n1 * n2, axis=1)
	cosTheta = jnp.clip(cosTheta, -1.0, 1.0)

	# We store the angle itself or the cosTheta for the potential
	return jnp.arccos(cosTheta)


def deriveRopeRestRots(
		restPos: jnp.ndarray[nV, 3],
		indices: jnp.ndarray[nT, 4],
		dmInv: jnp.ndarray[nT, 3, 3],
		ropeAdjPairs: jnp.ndarray[nRopePairs, 2]
) -> jnp.ndarray[nRopePairs, 3, 3]:
	"""
	Bakes the relative rest rotation between adjacent rope segments.
	"""
	# 1. Compute all rotations for the whole mesh at rest
	# (Note: at rest, F = Identity, so R is just the rotation of the authored mesh)
	# However, for robustness, we compute it properly.
	allRs = computeSubsetRotations(
		restPos, dmInv, indices, jnp.arange(indices.shape[0])
	)

	# 2. Extract pairs
	rA = allRs[ropeAdjPairs[:, 0]]
	rB = allRs[ropeAdjPairs[:, 1]]

	# 3. Relative rotation: rRel = rA^T * rB
	# This represents how tetB is oriented relative to tetA at rest.
	restRelRots = jax.vmap(lambda a, b: a.T @ b)(rA, rB)

	return restRelRots

"""
consider separate values for how active any given sculpt is, vs how strongly 
we should match to it?
why is vfx so complicated
starting to come round to the idea that options are evil - constrain the 
number of modes in a tool, rely on a person growing totally at home with it

"""

def resolveMixedSculptTarget(
		sculpts:list[state.SculptTarget],
		globalWeights
):
	"""
	Combines multiple sculpts outside the sim. Meant to be done in offline
	python
	sculpts: list of SculptTarget
	globalWeights: (numSculpts,) current rig sliders

	TODO: better support here of per-target modes - additive, weighted
		average, max length, etc
	"""
	# Simple weighted average of deltas
	totalDelta = jnp.zeros_like(sculpts[0].deltas)
	totalWeight = 1e-6

	for i, s in enumerate(sculpts):
		w = globalWeights[i] * s.weights
		totalDelta += s.deltas * w[:, None]
		totalWeight += w

	# Normalized Additive approach
	return totalDelta / totalWeight[:, None]


