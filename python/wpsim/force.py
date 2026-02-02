from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

"""Force field buffer implementation using struct-of-arrays pattern.

This module provides a ForceFieldBuffer dataclass for managing force field
primitives in a JAX-compatible format using the struct-of-arrays pattern.
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jax import Array


@dataclass
class ForceFieldBuffer:
	"""Force field buffer using struct-of-arrays pattern.

	Stores force field primitive data in separate arrays for efficient
	vectorized operations with JAX. Supports sphere, capsule, and cube
	force field shapes.

	Shape type indices:
		0 = Sphere (uses 1 float: radius)
		1 = Capsule (uses 2 floats: radius, height)
		2 = Cube (uses 3 floats: halfExtents x, y, z)

	Attributes:
		typeIndex: Integer array of shape (n,) indicating the primitive type
			for each force field (0=sphere, 1=capsule, 2=cube).
		transform: Float array of shape (n, 4, 4) containing world transform
			matrices for each force field primitive.
		forceParams: Float array of shape (n, 3) containing force parameters:
			- forceParams[:, 0]: magnitude of the force
			- forceParams[:, 1]: falloff exponent (e.g., 1.0 for linear, 2.0 for quadratic)
			- forceParams[:, 2]: direction mode (0=radial, 1=directional, 2=vortex, etc.)
		shapeParams: Float array of shape (n, 4) containing shape-specific parameters.
			Padded to 4 floats per primitive to accommodate the largest shape definition:
			- Sphere: [radius, 0, 0, 0]
			- Capsule: [radius, height, 0, 0]
			- Cube: [halfExtentX, halfExtentY, halfExtentZ, 0]

	Example:
		>>> import jax.numpy as jnp
		>>> # Create buffer for 3 force fields
		>>> buffer = ForceFieldBuffer(
		...     typeIndex=jnp.array([0, 1, 2], dtype=jnp.int32),
		...     transform=jnp.eye(4)[None, :, :].repeat(3, axis=0),
		...     forceParams=jnp.array([
		...         [10.0, 2.0, 0.0],  # sphere: magnitude 10, quadratic falloff, radial
		...         [5.0, 1.0, 1.0],   # capsule: magnitude 5, linear falloff, directional
		...         [15.0, 2.0, 0.0],  # cube: magnitude 15, quadratic falloff, radial
		...     ]),
		...     shapeParams=jnp.array([
		...         [2.0, 0.0, 0.0, 0.0],        # sphere: radius=2.0
		...         [0.5, 3.0, 0.0, 0.0],        # capsule: radius=0.5, height=3.0
		...         [1.0, 1.5, 2.0, 0.0],        # cube: halfExtents=(1.0, 1.5, 2.0)
		...     ]),
		... )
		>>> buffer.count
		3
	"""

	typeIndex: Array  # shape: (n,), dtype: int32
	transform: Array  # shape: (n, 4, 4), dtype: float32
	forceParams: Array  # shape: (n, 3), dtype: float32
	shapeParams: Array  # shape: (n, 4), dtype: float32

	@property
	def count(self) -> int:
		"""Return the number of force field primitives in the buffer."""
		return self.typeIndex.shape[0]

	@staticmethod
	def empty() -> "ForceFieldBuffer":
		"""Create an empty force field buffer with no primitives.

		Returns:
			ForceFieldBuffer: An empty buffer with zero-sized arrays.

		Example:
			>>> buffer = ForceFieldBuffer.empty()
			>>> buffer.count
			0
		"""
		return ForceFieldBuffer(
			typeIndex=jnp.zeros(0, dtype=jnp.int32),
			transform=jnp.zeros((0, 4, 4), dtype=jnp.float32),
			forceParams=jnp.zeros((0, 3), dtype=jnp.float32),
			shapeParams=jnp.zeros((0, 4), dtype=jnp.float32),
		)

	@staticmethod
	def create(
		typeIndex: Array,
		transform: Array,
		forceParams: Array,
		shapeParams: Array,
	) -> "ForceFieldBuffer":
		"""Create a force field buffer with explicit arrays.

		Args:
			typeIndex: Integer array of shape (n,) with primitive types.
			transform: Float array of shape (n, 4, 4) with transforms.
			forceParams: Float array of shape (n, 3) with force parameters.
			shapeParams: Float array of shape (n, 4) with shape parameters.

		Returns:
			ForceFieldBuffer: A new buffer with the provided data.

		Raises:
			ValueError: If array shapes are inconsistent.

		Example:
			>>> n = 10
			>>> buffer = ForceFieldBuffer.create(
			...     typeIndex=jnp.zeros(n, dtype=jnp.int32),
			...     transform=jnp.eye(4)[None, :, :].repeat(n, axis=0),
			...     forceParams=jnp.ones((n, 3)),
			...     shapeParams=jnp.ones((n, 4)),
			... )
		"""
		n = typeIndex.shape[0]

		if transform.shape != (n, 4, 4):
			raise ValueError(
				f"transform shape {transform.shape} does not match "
				f"expected ({n}, 4, 4)"
			)
		if forceParams.shape != (n, 3):
			raise ValueError(
				f"forceParams shape {forceParams.shape} does not match "
				f"expected ({n}, 3)"
			)
		if shapeParams.shape != (n, 4):
			raise ValueError(
				f"shapeParams shape {shapeParams.shape} does not match "
				f"expected ({n}, 4)"
			)

		return ForceFieldBuffer(
			typeIndex=typeIndex,
			transform=transform,
			forceParams=forceParams,
			shapeParams=shapeParams,
		)

	@staticmethod
	def allocate(n: int) -> "ForceFieldBuffer":
		"""Allocate a force field buffer for n primitives with zero-initialized data.

		Args:
			n: Number of force field primitives to allocate.

		Returns:
			ForceFieldBuffer: A buffer with n zero-initialized primitives.

		Example:
			>>> buffer = ForceFieldBuffer.allocate(100)
			>>> buffer.count
			100
		"""
		return ForceFieldBuffer(
			typeIndex=jnp.zeros(n, dtype=jnp.int32),
			transform=jnp.zeros((n, 4, 4), dtype=jnp.float32),
			forceParams=jnp.zeros((n, 3), dtype=jnp.float32),
			shapeParams=jnp.zeros((n, 4), dtype=jnp.float32),
		)

	def sample(self, pos: Array) -> Array:
		"""Sample the combined force at a single world-space position.

		Args:
			pos: World-space position, shape (3,).

		Returns:
			Array: World-space force vector, shape (3,).
		"""
		def safeNormalize(v: Array, eps: float = 1e-8) -> Array:
			n = jnp.linalg.norm(v, axis=-1, keepdims=True)
			return v / (n + eps)

		pos = jnp.asarray(pos, dtype=self.transform.dtype)
		posHom = jnp.concatenate([pos, jnp.ones((1,), pos.dtype)])
		invTransform = jnp.linalg.inv(self.transform)
		localHom = invTransform @ posHom
		localPos = localHom[:, :3]

		center = self.transform[:, :3, 3]
		offsetWorld = pos - center

		radius = self.shapeParams[:, 0]
		height = self.shapeParams[:, 1]
		halfExtents = self.shapeParams[:, :3]

		sphereDist = jnp.linalg.norm(localPos, axis=-1) - radius

		halfHeight = 0.5 * height
		clampedY = jnp.clip(localPos[:, 1], -halfHeight, halfHeight)
		capsuleDelta = localPos - jnp.stack(
			[jnp.zeros_like(clampedY), clampedY, jnp.zeros_like(clampedY)], axis=-1
		)
		capsuleDist = jnp.linalg.norm(capsuleDelta, axis=-1) - radius

		boxQ = jnp.abs(localPos) - halfExtents
		boxOutside = jnp.linalg.norm(jnp.maximum(boxQ, 0.0), axis=-1)
		boxInside = jnp.minimum(jnp.max(boxQ, axis=-1), 0.0)
		boxDist = boxOutside + boxInside

		isSphere = self.typeIndex == shapeSphere
		isCapsule = self.typeIndex == shapeCapsule
		dist = jnp.where(isSphere, sphereDist, jnp.where(isCapsule, capsuleDist, boxDist))
		distOutside = jnp.maximum(dist, 0.0)

		magnitude = self.forceParams[:, 0]
		falloffExp = self.forceParams[:, 1]
		atten = magnitude / jnp.power(1.0 + distOutside, falloffExp)

		axisWorld = safeNormalize(self.transform[:, :3, 2])
		radialDir = safeNormalize(offsetWorld)
		vortexDir = safeNormalize(jnp.cross(axisWorld, offsetWorld))

		dirMode = self.forceParams[:, 2].astype(jnp.int32)
		dirWorld = radialDir
		dirWorld = jnp.where(dirMode[:, None] == forceDirectional, axisWorld, dirWorld)
		dirWorld = jnp.where(dirMode[:, None] == forceVortex, vortexDir, dirWorld)

		return jnp.sum(dirWorld * atten[:, None], axis=0)

	def getSphereRadius(self, index: int) -> Array:
		"""Get the radius of a sphere force field at the given index.

		Args:
			index: Index of the sphere force field.

		Returns:
			Array: The sphere radius.
		"""
		return self.shapeParams[index, 0]

	def getCapsuleParams(self, index: int) -> tuple[Array, Array]:
		"""Get the radius and height of a capsule force field at the given index.

		Args:
			index: Index of the capsule force field.

		Returns:
			tuple[Array, Array]: A tuple of (radius, height).
		"""
		return self.shapeParams[index, 0], self.shapeParams[index, 1]

	def getCubeHalfExtents(self, index: int) -> Array:
		"""Get the half extents of a cube force field at the given index.

		Args:
			index: Index of the cube force field.

		Returns:
			Array: The half extents as a 3D vector [x, y, z].
		"""
		return self.shapeParams[index, :3]

	def slice(self, start: int, end: int) -> "ForceFieldBuffer":
		"""Create a new buffer containing a slice of this buffer.

		Args:
			start: Start index (inclusive).
			end: End index (exclusive).

		Returns:
			ForceFieldBuffer: A new buffer containing the sliced data.

		Example:
			>>> buffer = ForceFieldBuffer.allocate(100)
			>>> subBuffer = buffer.slice(10, 20)
			>>> subBuffer.count
			10
		"""
		return ForceFieldBuffer(
			typeIndex=self.typeIndex[start:end],
			transform=self.transform[start:end],
			forceParams=self.forceParams[start:end],
			shapeParams=self.shapeParams[start:end],
		)

	def concatenate(self, other: "ForceFieldBuffer") -> "ForceFieldBuffer":
		"""Concatenate this buffer with another buffer.

		Args:
			other: Another ForceFieldBuffer to concatenate.

		Returns:
			ForceFieldBuffer: A new buffer containing data from both buffers.

		Example:
			>>> buffer1 = ForceFieldBuffer.allocate(10)
			>>> buffer2 = ForceFieldBuffer.allocate(5)
			>>> combined = buffer1.concatenate(buffer2)
			>>> combined.count
			15
		"""
		return ForceFieldBuffer(
			typeIndex=jnp.concatenate([self.typeIndex, other.typeIndex]),
			transform=jnp.concatenate([self.transform, other.transform]),
			forceParams=jnp.concatenate(
				[self.forceParams, other.forceParams]),
			shapeParams=jnp.concatenate(
				[self.shapeParams, other.shapeParams]),
		)


# Shape type constants for convenience
shapeSphere = 0
shapeCapsule = 1
shapeCube = 2

# Force direction mode constants
forceRadial = 0  # Force points outward/inward from center
forceDirectional = 1  # Force points in a fixed direction
forceVortex = 2  # Force creates a vortex/curl pattern

__all__ = [
	"ForceFieldBuffer",
	"shapeSphere",
	"shapeCapsule",
	"shapeCube",
	"forceRadial",
	"forceDirectional",
	"forceVortex",
]
