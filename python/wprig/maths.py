from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from jax import numpy as jnp

def safeNormalize(v: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
	return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)

def scatterAddVec3(dst: jnp.ndarray, idx: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
	"""
	dst: (n, 3)
	idx: (k,)
	values: (k, 3)
	"""
	return dst.at[idx].add(values)


def scatterAddVec2(dst: jnp.ndarray, idx: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
	return dst.at[idx].add(values)

def applySmallAngleToQuat(q: jnp.ndarray, dTheta: jnp.ndarray) -> jnp.ndarray:
	"""
	Applies a small rotation vector dTheta (axis*angle, radians) to quaternion q.
	Small-angle approximation: dq ≈ [0.5*dTheta, 1]
	"""
	dq = jnp.concatenate([0.5 * dTheta, jnp.ones((dTheta.shape[0], 1), dTheta.dtype)], axis=-1)
	# quatMul(dq, q) assumes dq is a delta rotation in world frame; conventions to be finalized later
	qNew = quatMul(dq, q)
	return quatNormalize(qNew)

def quatNormalize(q: jnp.ndarray) -> jnp.ndarray:
	return q / jnp.linalg.norm(q, axis=-1, keepdims=True)

def quatMul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
	aw, ax, ay, az = a[..., 3], a[..., 0], a[..., 1], a[..., 2]
	bw, bx, by, bz = b[..., 3], b[..., 0], b[..., 1], b[..., 2]
	x = aw * bx + ax * bw + ay * bz - az * by
	y = aw * by - ax * bz + ay * bw + az * bx
	z = aw * bz + ax * by - ay * bx + az * bw
	w = aw * bw - ax * bx - ay * by - az * bz
	return jnp.stack([x, y, z, w], axis=-1)


def quatConj(q: jnp.ndarray) -> jnp.ndarray:
	return jnp.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], axis=-1)


def quatRotate(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
	vq = jnp.concatenate([v, jnp.zeros(v[..., :1].shape, v.dtype)], axis=-1)
	return quatMul(quatMul(q, vq), quatConj(q))[..., :3]

def quatLog(q: jnp.ndarray) -> jnp.ndarray:
	"""
	Log map from unit quaternion to rotation vector (axis * angle).

	Assumes q is unit length.
	Output is in (-pi, pi] but *increments* are safe to accumulate.
	"""
	v = q[..., :3]
	w = q[..., 3]

	norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
	angle = 2.0 * jnp.arctan2(norm_v, w)

	# Avoid division by zero
	scale = jnp.where(norm_v > 1e-8, angle / norm_v, 2.0 * jnp.ones_like(norm_v))
	return v * scale

def applyInvInertiaWorld(q: jnp.ndarray, invInertiaBodyDiag: jnp.ndarray, vWorld: jnp.ndarray) -> jnp.ndarray:
	"""
	Applies world-space inverse inertia to a world-space vector vWorld, given diagonal inv inertia in body frame.

	q: (k,4) body orientation
	invInertiaBodyDiag: (k,3) diagonal entries in body frame
	vWorld: (k,3)

	return: (k,3) = I_world^{-1} * vWorld
	"""
	# Rotate into body frame
	vBody = quatRotate(quatConj(q), vWorld)			# (k,3)

	# Apply diagonal inverse inertia in body frame
	vBodyOut = vBody * invInertiaBodyDiag			# (k,3)

	# Rotate back to world frame
	return quatRotate(q, vBodyOut)

def buildTangentBasis(n: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
	"""
	Builds (t1,t2) orthonormal basis for the plane perpendicular to n.
	Assumption: n is normalized.
	"""
	# Choose a helper vector not parallel to n
	helper = jnp.where(jnp.abs(n[..., 0:1]) < 0.9,
	                   jnp.array([1.0, 0.0, 0.0], n.dtype),
	                   jnp.array([0.0, 1.0, 0.0], n.dtype))
	t1 = safeNormalize(jnp.cross(n, helper), eps=1e-8)
	t2 = jnp.cross(n, t1)
	return t1, t2

def makeQuatFromAxisAngle(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
	axis = axis / (jnp.linalg.norm(axis) + 1e-8)
	s = jnp.sin(0.5 * angle)
	c = jnp.cos(0.5 * angle)
	return jnp.array([axis[0] * s, axis[1] * s, axis[2] * s, c], jnp.float32)

def shortestAngleDelta(delta: jnp.ndarray) -> jnp.ndarray:
	"""
	Map angle delta to (-pi, pi], elementwise.
	"""
	return (delta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


