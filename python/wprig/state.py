from __future__ import annotations
import types, typing as T
import pprint
from dataclasses import dataclass

from jax import numpy as jnp
import jax_dataclasses as jdc

@jdc.pytree_dataclass
class BodyState:
	"""
	Dynamic per-frame state for all rigid bodies.
	Shape conventions:
	  N = number of bodies
	"""
	# Configuration
	position: jnp.ndarray        # (N, 3)
	orientation: jnp.ndarray     # (N, 4)  unit quaternions

	# Velocities
	linearVelocity: jnp.ndarray     # (N, 3)
	angularVelocity: jnp.ndarray    # (N, 3)

	# Mass properties (world-constant or body-constant)
	invMass: jnp.ndarray            # (N,)
	invInertiaBody: jnp.ndarray    # (N, 3)  diagonal inertia in body frame

@jdc.pytree_dataclass
class BodyDeltaBuffers:
	posDelta: jnp.ndarray		# (n, 3)
	angDelta: jnp.ndarray		# (n, 3)

	@classmethod
	def makeZero(cls, bodyCount:int, dtype=jnp.float32) -> BodyDeltaBuffers:
		return BodyDeltaBuffers(
			posDelta=jnp.zeros((bodyCount, 3), dtype),
			angDelta=jnp.zeros((bodyCount, 3), dtype),
		)


@jdc.pytree_dataclass
class BodyMetadata:
	"""
	Indirect references and flags.
	Not expected to be touched every solver iteration.
	"""
	# Indirection into geometry buffers
	shapeIndex: jnp.ndarray     # (N,) index into shape/geometry tables

	# Optional flags
	isDynamic: jnp.ndarray      # (N,) bool or int mask

	# Optional rest/bind information (for rigs)
	restPosition: jnp.ndarray | None     # (N, 3) or None
	restOrientation: jnp.ndarray | None  # (N, 4) or None

@jdc.pytree_dataclass
class GeometryBuffers:
	"""we have mapping from body->mesh from the metadata array -
	we may here need mapping from mesh->body, for example as
	 in a skincluster"""
	vertices: jnp.ndarray        # (V, 3)
	indices: jnp.ndarray         # (T, 3) or flattened
	sdfGrids: jnp.ndarray | None  # e.g. (G, nx, ny, nz)
	primvars: dict[str, jnp.ndarray]  # optional

@jdc.pytree_dataclass
class RampBuffers:
	"""single flattened storage for all ramp mappings used in sim?
	dubious but we'll try it
	TODO: find a general way of retrieving named values from sim state
		statically - consider each constraint having a default kwarg
		of 'name' when compiled  """
	points : jnp.ndarray # all points in all ramps, (N, 2)
	indices: jnp.ndarray # (nRamps + 1) , start at 0
	pointModes: jnp.ndarray #
	indicesModes: jnp.ndarray
	nameIndexMap : dict[str, int]

	def start(self, rampId:int)->int:
		return self.indices[rampId]
	def end(self, rampId:int)->int:
		return self.indices[rampId + 1]


@jdc.pytree_dataclass
class DerivedState:
	"""
	Optional caches computed once per frame.
	"""
	world_inertia: jnp.ndarray | None   # (N, 3, 3)
	rotation_matrix: jnp.ndarray | None # (N, 3, 3)



@jdc.pytree_dataclass
class SimFrame:
	"""
	Complete state of sim at single frame
	"""
	bodies: BodyState
	metadata: BodyMetadata

	# Optional, depending on pipeline
	derived: DerivedState | None

	# Global time info
	startT: float
	dt: float
	index: int


@jdc.pytree_dataclass(frozen=True)
class SimSettings:
	dtFrame: float = 1.0 / 60.0
	substepCount: int = 2
	iterationCount: int = 8

	simParams : jnp.ndarray | None # freely specified aux param inputs
	simParamIndices: jnp.ndarray | None  # ( N+1, start at 0)
	simParamNames : tuple[str, ...] | None



