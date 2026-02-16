from __future__ import annotations
import types, typing as T
from typing import Optional, Any
import pprint
from dataclasses import dataclass

from jax import numpy as jnp
import jax_dataclasses as jdc



@jdc.pytree_dataclass
class SimStaticData:
	"""all data that will never change across entire sim"""
	dtFrame: float = 1.0 / 60.0
	substepCount: int = 2
	iterationCount: int = 8

@jdc.pytree_dataclass
class FrameBoundData:
	"""data static over the course of a frame"""
	simParams : jnp.ndarray | None # freely specified aux param inputs
	simParamLengths : jnp.ndarray | None # length of each simParam
	paramSizes : int
	simParamNames : tuple[str, ...] | None
	#TEMP, later do proper force field primitives
	gravity: jnp.ndarray      # (3,)


@jdc.pytree_dataclass
class SubstepBoundData:
	"""
	Data static over the course of a substep.
	Dynamic per-frame state for all rigid bodies.
	Shape conventions:
	  N = number of bodies
	"""
	## BODY STATE
	# Configuration
	position: jnp.ndarray        # (N, 3)
	orientation: jnp.ndarray     # (N, 4)  unit quaternions
	# Velocities
	linearVelocity: jnp.ndarray     # (N, 3)
	angularVelocity: jnp.ndarray    # (N, 3)
	# Mass properties (world-constant or body-constant)
	invMass: jnp.ndarray            # (N,)
	invInertiaBody: jnp.ndarray    # (N, 3)  diagonal inertia in body frame

	# forces
	force: jnp.ndarray             # (N, 3)
	torque: jnp.ndarray            # (N, 3)

@jdc.pytree_dataclass
class DynamicData:
	"""dynamic state scatter-add built over the course of a substep
	"""

	## BODY DELTA BUFFERS
	posDelta: jnp.ndarray		# (n, 3)
	angDelta: jnp.ndarray		# (n, 3)
	@classmethod
	def makeZero(cls, bodyCount:int, dtype=jnp.float32) -> DynamicData:
		return DynamicData(
			posDelta=jnp.zeros((bodyCount, 3), dtype),
			angDelta=jnp.zeros((bodyCount, 3), dtype),
		)

@jdc.pytree_dataclass
class MeasurementState:
	"""
	Storage for user-defined measurement functions and their internal parameters.
	Measurements are arbitrary functions of sim state (body positions, geometry, etc.)
	that produce scalar/vector outputs to be constrained.

	Examples:
	  - distance_between_bodies(A, B) -> scalar
	  - sample_surface(bodyA, surfaceS, u, v) -> (pos, normal) = 6D vector
	  - relative_rotation(bodyC, bodyD) -> 3D axis-angle
	"""
	# Measured values (padded to uniform dimension per bucket)
	# Hot path: accessed every solver iteration
	values: jnp.ndarray              # (nMeasurements, maxDim) - padded with zeros
	valueDims: jnp.ndarray           # (nMeasurements,) int32 - actual used dimensions

	# Internal geometric parameters that measurements depend on
	# These are solver variables that can be adjusted (e.g., surface u,v or curve t)
	# Cold path: accessed less frequently, truly variable length -> use indirection
	params: jnp.ndarray              # (totalParams,) - flattened, all parameters concatenated
	paramIndices: jnp.ndarray        # (nMeasurements + 1,) int32 - cumulative offsets, start at 0

	# Jacobians computed by autodiff through measurement functions
	# J_body: how measurement changes with body state (position, orientation)
	# Shape: (nMeasurements, maxDim, maxBodyDOFs) where maxBodyDOFs depends on dependencies
	J_body: jnp.ndarray | None

	# J_param: how measurement changes with internal parameters (u, v, t, ...)
	# Shape: (nMeasurements, maxDim, maxParams)
	J_param: jnp.ndarray | None

	# Optional: body dependency tracking for efficient updates
	# Each measurement may depend on 0-N bodies
	bodyDependencies: jnp.ndarray | None    # (totalDeps,) - flattened body indices
	bodyDepIndices: jnp.ndarray | None      # (nMeasurements + 1,) - indirection

	def paramStart(self, measurementId: int) -> int:
		return self.paramIndices[measurementId]

	def paramEnd(self, measurementId: int) -> int:
		return self.paramIndices[measurementId + 1]

	def bodyDepStart(self, measurementId: int) -> int:
		return self.bodyDepIndices[measurementId] if self.bodyDepIndices is not None else 0

	def bodyDepEnd(self, measurementId: int) -> int:
		return self.bodyDepIndices[measurementId + 1] if self.bodyDepIndices is not None else 0


@jdc.pytree_dataclass
class ResidualDescriptor:
	"""
	Rich residual information for a single constraint evaluation.
	Used when constraining measurements that have internal geometric DOFs.

	This allows the solver to:
	1. Apply corrections to body state (via J_body)
	2. Adjust internal parameters like surface u,v (via J_param)
	3. Weight components differently (via metric, e.g. tangent vs normal)
	4. Respect geometric structure (via local_frame)

	Example: Constraining point on surface to point on curve
	  - value: 3D position error
	  - J_body: ∂error/∂(bodyA_pos, bodyA_ori, bodyB_pos, bodyB_ori) = (3, 12)
	  - J_param: ∂error/∂(u, v, t) = (3, 3) - can slide along surface/curve
	  - metric: weight normal direction more than tangent directions
	  - local_frame: surface frame (normal, tangent_u, tangent_v)
	"""
	# Mandatory: residual value and body Jacobian
	value: jnp.ndarray              # (k,) - residual in k dimensions
	J_body: jnp.ndarray             # (k, nBodyDOFs) - Jacobian wrt body state

	# Optional: Jacobian wrt internal geometric parameters
	J_param: jnp.ndarray | None = None  # (k, nParams) - allows sliding along geometry

	# Optional: anisotropic weighting (diagonal preferred for efficiency)
	# Example: weight normal component 10x more than tangent components
	metric: jnp.ndarray | None = None   # (k,) diagonal or (k, k) full matrix

	# Optional: local coordinate frame for residual interpretation
	# Example: surface frame with columns [normal, tangent_u, tangent_v]
	local_frame: jnp.ndarray | None = None  # (k, k) or (3, 3)

	# Optional: limits on residual components (for inequality constraints)
	limit_min: jnp.ndarray | None = None  # (k,)
	limit_max: jnp.ndarray | None = None  # (k,)

	# Optional: whether residual lives in wrapped space (angles, SO(3))
	is_unwrapped: bool = True




@jdc.pytree_dataclass
class DerivedState:
	"""
	Optional caches computed once per frame.
	"""
	world_inertia: jnp.ndarray | None   # (N, 3, 3)
	rotation_matrix: jnp.ndarray | None # (N, 3, 3)


@jdc.pytree_dataclass
class RampBuffers:
	"""uniformly sampled remap ramps for JAX-friendly lookup"""
	samples: jnp.ndarray # (nRamps, sampleCount)
	sampleCount: int # normally 32
	nameIndexMap: dict[str, int]

	def sample(self, rampId: int, u: jnp.ndarray) -> jnp.ndarray:
		uClamped = jnp.clip(u, 0.0, 1.0)
		maxIndex = jnp.array(self.sampleCount - 1, dtype=jnp.int32)
		scaled = uClamped * maxIndex.astype(uClamped.dtype)
		index0 = jnp.floor(scaled).astype(jnp.int32)
		index1 = jnp.minimum(index0 + 1, maxIndex)
		t = scaled - index0.astype(scaled.dtype)
		value0 = self.samples[rampId, index0]
		value1 = self.samples[rampId, index1]
		return (1.0 - t) * value0 + t * value1



@jdc.pytree_dataclass
class SimFrame:
	"""
	Complete state of sim at single frame
	"""
	frameData : FrameBoundData
	metadata: BodyMetadata

	# Optional, depending on pipeline
	derived: DerivedState | None

	# Global time info
	startT: float
	dt: float
	index: int





@jdc.pytree_dataclass
class MeshBuffers:
	"""Struct-of-arrays storage for triangle meshes.
	All meshes packed into single flat arrays with indirection.
	Note: Multiple bodies can reference the same mesh (instancing),
	or a single body can own multiple meshes.
	"""
	# Flattened point data for all meshes
	points: jnp.ndarray  # (totalPoints, 3)

	# Flattened triangle vertex indices for all meshes
	# Indices are relative to each mesh's point range
	triIndices: jnp.ndarray  # (totalTris, 3) int32

	# Indirection arrays
	pointIndices: jnp.ndarray  # (nMeshes + 1,) - cumulative point offsets, start at 0
	triOffsets: jnp.ndarray  # (nMeshes + 1,) - cumulative triangle offsets

	# Optional per-point attributes
	normals: jnp.ndarray | None  # (totalPoints, 3)
	tangents: jnp.ndarray | None  # (totalPoints, 3)
	uvs: jnp.ndarray | None  # (totalPoints, 2)
	colors: jnp.ndarray | None  # (totalPoints, 3) or (totalPoints, 4)

	# Optional per-triangle attributes
	triNormals: jnp.ndarray | None  # (totalTris, 3)

	def pointStart(self, meshId: int) -> int:
		return self.pointIndices[meshId]

	def pointEnd(self, meshId: int) -> int:
		return self.pointIndices[meshId + 1]

	def triStart(self, meshId: int) -> int:
		return self.triOffsets[meshId]

	def triEnd(self, meshId: int) -> int:
		return self.triOffsets[meshId + 1]


@jdc.pytree_dataclass
class NurbsCurveBuffers:
	"""Struct-of-arrays storage for NURBS curves.
    All curves packed into single flat arrays with indirection.
    """
	# Flattened CV data for all curves
	cvs: jnp.ndarray  # (totalCVs, 3)

	# Flattened knot vectors for all curves
	knots: jnp.ndarray  # (totalKnots,)

	# Indirection arrays
	cvIndices: jnp.ndarray  # (nCurves + 1,) - cumulative CV offsets, start at 0
	knotIndices: jnp.ndarray  # (nCurves + 1,) - cumulative knot offsets

	# Per-curve scalar attributes
	degrees: jnp.ndarray  # (nCurves,) - curve degree (typically 3)

	# Optional per-CV attributes
	weights: jnp.ndarray | None  # (totalCVs,) - NURBS weights
	upvectors: jnp.ndarray | None  # (totalCVs, 3) - for oriented curves

	# Optional per-curve attributes
	isPeriodic: jnp.ndarray | None  # (nCurves,) - bool/int mask

	def cvStart(self, curveId: int) -> int:
		return self.cvIndices[curveId]

	def cvEnd(self, curveId: int) -> int:
		return self.cvIndices[curveId + 1]

	def knotStart(self, curveId: int) -> int:
		return self.knotIndices[curveId]

	def knotEnd(self, curveId: int) -> int:
		return self.knotIndices[curveId + 1]

	def getPos(self, curveId:int, u:float)->jnp.ndarray:
		"""assume U is not normalized
		TODO:consider if we should put helpers here or in a separate lib
		"""
		return


@jdc.pytree_dataclass
class NurbsSurfaceBuffers:
	"""Struct-of-arrays storage for NURBS surfaces.
    All surfaces packed into single flat arrays with indirection.
    """
	# Flattened CV data for all surfaces
	# CVs stored in row-major order: [u0v0, u0v1, ..., u0vN, u1v0, ...]
	cvs: jnp.ndarray  # (totalCVs, 3)

	# Flattened knot vectors (U and V separate)
	knotsU: jnp.ndarray  # (totalKnotsU,)
	knotsV: jnp.ndarray  # (totalKnotsV,)

	# Indirection arrays
	cvIndices: jnp.ndarray  # (nSurfaces + 1,) - cumulative CV offsets
	knotUIndices: jnp.ndarray  # (nSurfaces + 1,) - cumulative U knot offsets
	knotVIndices: jnp.ndarray  # (nSurfaces + 1,) - cumulative V knot offsets

	# Per-surface topology
	uCount: jnp.ndarray  # (nSurfaces,) - number of CVs in U direction
	vCount: jnp.ndarray  # (nSurfaces,) - number of CVs in V direction
	degreeU: jnp.ndarray  # (nSurfaces,) - degree in U
	degreeV: jnp.ndarray  # (nSurfaces,) - degree in V

	# Optional per-CV attributes
	weights: jnp.ndarray | None  # (totalCVs,)

	# Optional per-surface attributes
	isPeriodicU: jnp.ndarray | None  # (nSurfaces,)
	isPeriodicV: jnp.ndarray | None  # (nSurfaces,)

	def cvStart(self, surfaceId: int) -> int:
		return self.cvIndices[surfaceId]

	def cvEnd(self, surfaceId: int) -> int:
		return self.cvIndices[surfaceId + 1]

	def knotUStart(self, surfaceId: int) -> int:
		return self.knotUIndices[surfaceId]

	def knotUEnd(self, surfaceId: int) -> int:
		return self.knotUIndices[surfaceId + 1]

	def knotVStart(self, surfaceId: int) -> int:
		return self.knotVIndices[surfaceId]

	def knotVEnd(self, surfaceId: int) -> int:
		return self.knotVIndices[surfaceId + 1]


@jdc.pytree_dataclass
class GeometryBuffers:
	"""Top-level geometry storage with typed buffers.
    Bodies reference geometry via (type, index) pairs in BodyMetadata.
    """
	meshes: MeshBuffers | None
	curves: NurbsCurveBuffers | None
	surfaces: NurbsSurfaceBuffers | None

	# Future extensions
	sdfGrids: jnp.ndarray | None  # Could follow similar indirection pattern


@jdc.pytree_dataclass
class BodyMetadata:
	"""
	Indirect references and flags.
	Not expected to be touched every solver iteration.
	"""
	# Indirection into geometry buffers
	shapeIndex: jnp.ndarray     # (N,) index into shape/geometry tables - legacy, might deprecate

	# Optional flags
	isDynamic: jnp.ndarray      # (N,) bool or int mask

	# Optional rest/bind information (for rigs)
	restPosition: jnp.ndarray | None     # (N, 3) or None
	restOrientation: jnp.ndarray | None  # (N, 4) or None

	# Compact geometry indirection array
	# Layout per body: [nMeshes, meshStartIdx, nCurves, curveStartIdx, nSurfaces, surfaceStartIdx, ...]
	# If body has no geometry of a type: nType=0, startIdx=-1
	# Shape: (N, 2 * nGeometryTypes) where nGeometryTypes is currently 3
	geometryRefs: jnp.ndarray    # (N, 6) int16

	# Index constants for geometryRefs array
	GEOM_MESH_COUNT = 0
	GEOM_MESH_START = 1
	GEOM_CURVE_COUNT = 2
	GEOM_CURVE_START = 3
	GEOM_SURFACE_COUNT = 4
	GEOM_SURFACE_START = 5

	def getMeshRange(self, bodyId: int) -> tuple[int, int]:
		"""Returns (start, end) indices into MeshBuffers, or (-1, -1) if no meshes"""
		count = self.geometryRefs[bodyId, self.GEOM_MESH_COUNT]
		start = self.geometryRefs[bodyId, self.GEOM_MESH_START]
		return (start, start + count) if count > 0 else (-1, -1)

	def getCurveRange(self, bodyId: int) -> tuple[int, int]:
		"""Returns (start, end) indices into NurbsCurveBuffers, or (-1, -1) if no curves"""
		count = self.geometryRefs[bodyId, self.GEOM_CURVE_COUNT]
		start = self.geometryRefs[bodyId, self.GEOM_CURVE_START]
		return (start, start + count) if count > 0 else (-1, -1)

	def getSurfaceRange(self, bodyId: int) -> tuple[int, int]:
		"""Returns (start, end) indices into NurbsSurfaceBuffers, or (-1, -1) if no surfaces"""
		count = self.geometryRefs[bodyId, self.GEOM_SURFACE_COUNT]
		start = self.geometryRefs[bodyId, self.GEOM_SURFACE_START]
		return (start, start + count) if count > 0 else (-1, -1)
