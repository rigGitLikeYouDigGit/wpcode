from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from typing import NamedTuple, Optional


"""
where:
nT is number of tets,
nV is number of vertices,
nM is number of muscle tetrahedra

nOuterTris is number of outer surface triangles

float16s used here but for weighting could probably just use 8 bits each - 
see if we can quickly convert to uint8 for this


TODO: work out a decent way of embedding/wrapping arbitrary geo with the 
output tet positions, probably exists outside the sim.
"""


@jdc.pytree_dataclass
class SimStaticParams:
	"""Anatomical 'DNA' that never changes during a session.

	for sculpted activation keys, FOR NOW we don't support keys on physical
	extension of muscle, only activation during contraction.
	"""
	indices: jnp.ndarray  # (nT, 4)
	volWeights: jnp.ndarray  # (nT,) float16

	# mapping to outer skin surface
	nOuterTris: int
	outerTriTets: jnp.ndarray # (nOuterTris,) # indices into tets,
	# outer tri guaranteed to be first 3 vertices of each tet

	# neutral dmInv matrices for each tet's rest state
	dmInvNeutral: jnp.ndarray  # (nT, 3, 3) float16

	# Material constants
	mu: jnp.ndarray  # (nT,)
	kappa: jnp.ndarray  # (nT,)
	mass: jnp.ndarray  # (nT,)
	parentObj: jnp.ndarray # (nT,) uint16, pointing back to a parent mesh id

	# linear skin weights - for now, used as a fallback for offering
	# anim control over parts of the sim
	nSkinWeights: int # nSW
	skinWeights: jnp.ndarray # (nV, nSW) uint8
	skinIndices: jnp.ndarray # (nV, nSW) uint16
	restPositions: jnp.ndarray # (nV, 3)

	# Material Type Mask
	# 0 = Passive, 1 = Muscle, 2 = Cloth, 3 = Rope
	materialType: jnp.ndarray[nT]  # int8

	# indices where different types of tet begin - for now muscles, non-muscle,
	nMuscleTets: int # nM
	muscleTetIndices: jnp.ndarray
	clothTetIndices: jnp.ndarray
	ropeTetIndices: jnp.ndarray

	# muscle data
	#fiberDirs: jnp.ndarray  # (nM, 3) float16 rest fiber vectors
	# muscle shape keys:
	# how many sculpted activation shape keys each muscle may have
	nMuscleKeys : int
	# flexed base dmInv states:
	# (where no keys defined, we just copy existing neutral here as padding)
	dmInvFlexed: jnp.ndarray  # (nM, nMuscleKeys, 3, 3)
	# parametres of shape keys
	flexedActivationKeys: jnp.ndarray # (nM, nMuscleKeys) uint8

	# Strand Mapping: Every tet links to nMaxStrands
	nMaxStrands: int
	strandIndices: jnp.ndarray # (nM, nMaxStrands) uint16
	strandWeights: jnp.ndarray # (nM, nMaxStrands) uint8

	# CLOTH and ROPE
	# Anisotropy Vectors
	# For Muscle: fiberA is fiberDir
	# For Cloth: fiberA is Warp, fiberB is Weft
	# For Rope: fiberA is Tangent, fiberB is Normal (Twist reference)
	fiberA: jnp.ndarray[nT, 3]
	fiberB: jnp.ndarray[nT, 3]

	# Cloth Bending Data
	nClothEdges: int
	clothEdgeIndices: jnp.ndarray[nClothEdges, 4]  # [v0, v1, v2, v3]
	clothRestAngles: jnp.ndarray[nClothEdges]  # Precomputed acos(dot(n1, n2))
	kBendCloth: jnp.ndarray[nClothEdges]

	# Rope Torsion Data
	nRopePairs: int
	ropeAdjPairs: jnp.ndarray[nRopePairs, 2]  # [tetA, tetB]
	ropeRestRelRots: jnp.ndarray[nRopePairs, 3, 3]  # Precomputed rA.T @ rB
	kTorsionRope: jnp.ndarray[nRopePairs]



@jdc.pytree_dataclass
class FrameStaticData:
	"""Inputs from the kinematic rig/animation for the current frame."""
	pinMask: jnp.ndarray  # (nV,) float16, 1.0 = follow anim, 0.0 = simulate

	# Strand Activations derived from NURBS curves in Maya
	# e.g., extension values or manual rig attributes
	strandActivations: jnp.ndarray  # (numStrands,) uint8

	# kinematic joints as translation and quaternion
	nJoints : int
	jointTranslates : jnp.ndarray # (nJoints, 3)
	jointOrients : jnp.ndarray # (nJoints, 4)

	# computed base skinned positions
	targetKinematicPos: jnp.ndarray  # (V, 3) for pinning

	# optional output attributes, sketch for now
	# compression : jnp.ndarray # (nOuterTris,) float16


@jdc.pytree_dataclass
class SubstepStaticData:
	"""Data derived at the start of each substep (e.g. spatial hash)."""
	gridHashes: jnp.ndarray
	gridTriIdx: jnp.ndarray
	# Interpolated activations for every muscle tet
	tetActivations: jnp.ndarray  # (nM,) uint8
	# The 'Holistic' rest matrices for this specific activation level
	currentDmInv: jnp.ndarray  # (nT, 3, 3)


@jdc.pytree_dataclass
class DynamicState:
	"""Data that evolves per substep via the solver."""
	pos: jnp.ndarray  # (nV, 3)
	vel: jnp.ndarray  # (nV, 3)


@jdc.pytree_dataclass
class SculptTarget:
	# Only store data for the vertices that actually move
	affectedIndices: jnp.ndarray # (nImpacted,) int32
	localDeltas: jnp.ndarray     # (nImpacted, 3) float16
	weights: jnp.ndarray          # (nImpacted,) float16
	# The tets used to derive the local frames for these specific vertices
	# Usually one tet per impacted vertex
	guideTetIndices: jnp.ndarray  # (nImpacted,) int32

	targetActivation: float # The alpha level this sculpt represents
	# or the active weight of this target

@jdc.pytree_dataclass
class SculptState:
	"""The result of the inverse solve."""
	# The 'Final Delta' used to guarantee the exact hit at weight 1.0
	residualDeltas: jnp.ndarray # (nV, 3)
	# Optimized material tweaks found during the solve
	optimizedActivations: jnp.ndarray # (nM,) muscle activation params
	optimizedMu: jnp.ndarray # (nT,) # mus for all tets


@jdc.pytree_dataclass
class InverseSolveData:
	# The target spatial grid (built from the artistic sculpt)
	targetGrid: dict
	targetSurfaceTris: jnp.ndarray

	# Weights for the composite loss function
	weightSilhouette: float  # How much we care about the shape
	weightConformal: float  # How much we care about skin stretching
	weightVelocity: float  # How much we care about smooth motion

	# The level of frame smoothing to use [0.0 = sharp, 1.0 = smooth]
	frameSmoothing: float
