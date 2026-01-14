from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import jax
from jax import numpy as jnp

def uint8ToFloat16(n:jnp.ndarray):
	return n.astype(jnp.float16) / 255.0
def uint8ToFloat32(n:jnp.ndarray):
	return n.astype(jnp.float32) / 2 ^ 32
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

def pose_distance_se3(pose1: jnp.ndarray, pose2: jnp.ndarray,
                      rot_weight: float = 1.0,
                      trans_weight: float = 1.0) -> float:
	"""
    Distance between poses in SE(3).

    Args:
        pose1, pose2: (7,) arrays [quat(4), pos(3)]
        rot_weight: relative weight of rotation error
        trans_weight: relative weight of translation error
    """
	q1, p1 = pose1[:4], pose1[4:]
	q2, p2 = pose2[:4], pose2[4:]

	# Rotation distance: geodesic on SO(3)
	q_rel = quatMul(quatConj(q1), q2)
	angle = 2 * jnp.arccos(jnp.clip(jnp.abs(q_rel[3]), 0, 1))
	rot_dist = angle * rot_weight

	# Translation distance: Euclidean
	trans_dist = jnp.linalg.norm(p2 - p1) * trans_weight

	return rot_dist ** 2 + trans_dist ** 2


def rbf_interpolate_poses(params: jnp.ndarray, centers: jnp.ndarray,
                          weights: jnp.ndarray,
                          poses: jnp.ndarray) -> jnp.ndarray:
	"""
    RBF interpolation from M-dimensional parameter space to 6N-dimensional pose space.

    Args:
        params: (M,) current parameter values [flexion, rotation, ...]
        centers: (K, M) parameter values at each sample
        weights: (K, 6N) RBF weights (pre-solved from poses)
        poses: (K, N, 7) authored poses (N bodies × [quat(4), pos(3)])

    Returns:
        target_pose: (N, 7) interpolated target pose for all bodies
    """
	M = params.shape[0]
	K = poses.shape[0]
	N = poses.shape[1]
	# Compute RBF basis functions
	# Distance in parameter space
	diffs = params[None, :] - centers  # (K, M)
	distances = jnp.linalg.norm(diffs, axis=-1)  # (K,)

	# Gaussian RBF kernel
	epsilon = 1.0  # shape parameter
	phi = jnp.exp(-(epsilon * distances) ** 2)  # (K,)

	# Weighted sum of poses
	# Problem: poses are in SE(3), can't just linearly interpolate!
	# Need to work in tangent space

	# Flatten poses to vectors (for now, treat as Euclidean - we'll fix this)
	pose_vectors = poses.reshape(K, -1)  # (K, 6N) flattened

	# Interpolate
	target_vector = jnp.sum(phi[:, None] * pose_vectors, axis=0)  # (6N,)
	target_pose = target_vector.reshape(N, 7)  # (N, 7)

	# Normalize quaternions
	target_pose = target_pose.at[:, :4].set(
		quatNormalize(target_pose[:, :4])
	)

	return target_pose


# Pre-compute inverse map during setup
def train_inverse_rbf(centers, weights, poses):
	"""
    Train a separate RBF that maps poses → params.

    Use the same K samples but swap input/output:
        Forward RBF: params[i] → poses[i]
        Inverse RBF: poses[i] → params[i]
    """
	K = poses.shape[0]
	# RBF centers in pose space (high-dimensional)
	pose_centers = poses.reshape(K, -1)  # (K, 6N)

	# Target outputs are the parameter values
	param_targets = centers  # (K, M)

	# Solve for weights: Φ @ W = param_targets
	# where Φ[i,j] = phi(||pose[i] - pose[j]||)
	Phi = compute_rbf_matrix(pose_centers, pose_centers)  # (K, K)
	W = jnp.linalg.solve(Phi, param_targets)  # (K, M)

	return pose_centers, W


def inverse_rbf(current_pose, pose_centers, inv_weights,
                epsilon=1.0):
	"""Fast approximate inverse: pose → params"""
	pose_vector = current_pose.reshape(-1)  # (6N,)
	diffs = pose_vector[None, :] - pose_centers  # (K, 6N)
	distances = jnp.linalg.norm(diffs, axis=-1)  # (K,)
	phi = jnp.exp(-(epsilon * distances) ** 2)  # (K,)
	params = jnp.sum(phi[:, None] * inv_weights, axis=0)  # (M,)
	return params


@jax.vmap
def polarDecomp3x3(mat:jnp.ndarray[3,3]):
	"""
	Extracts Rotation (R) and Stretch (S) from F.
	Uses SVD to handle inverted tets gracefully.
	"""
	U, S, Vt = jnp.linalg.svd(mat)
	R = U @ Vt
	# Correct for reflection to maintain det(R) = 1
	Det = jnp.linalg.det(R)
	U = U.at[:, 2].multiply(jnp.where(Det < 0, -1.0, 1.0))
	R = U @ Vt
	return R


def fastOrthonormalize(m):
	"""
	A cheaper alternative to SVD for 'almost-orthonormal' matrices.
	"""
	x = m[:, 0]
	x = x / (jnp.linalg.norm(x) + 1e-10)

	z = jnp.cross(x, m[:, 1])
	z = z / (jnp.linalg.norm(z) + 1e-10)

	y = jnp.cross(z, x)
	return jnp.stack([x, y, z], axis=1)

@jax.jit(static_argnames=('kernelType',))
def computeRbfMatrix(centers: jnp.ndarray, # (K, M)
                     queryPoints: jnp.ndarray, # (N, M)
                     epsilon: float = 1.0,
                     kernelType: str = 'gaussian') -> jnp.ndarray: # (N, K)
	"""
	Computes RBF kernel matrix between query points and centers.

	Args:
		centers: (K, M) RBF center points in M-dimensional space
		queryPoints: (N, M) query points to evaluate
		epsilon: shape parameter for RBF kernel
		kernelType: 'gaussian', 'multiquadric', 'inverse_multiquadric', or 'thin_plate'

	Returns:
		Phi: (N, K) kernel evaluations where Phi[i,j] = kernel(||query[i] - center[j]||)

	Common kernels:
		gaussian: exp(-(epsilon * r)^2)
		multiquadric: sqrt(1 + (epsilon * r)^2)
		inverse_multiquadric: 1 / sqrt(1 + (epsilon * r)^2)
		thin_plate: r^2 * log(r) if r > 0 else 0
	"""
	# Compute pairwise distances
	# centers: (K, M), queryPoints: (N, M)
	# Want distances[i, j] = ||queryPoints[i] - centers[j]||
	diff = queryPoints[:, None, :] - centers[None, :, :]  # (N, K, M)
	distances = jnp.linalg.norm(diff, axis=-1)  # (N, K)

	# Apply kernel function
	if kernelType == 'gaussian':
		phi = jnp.exp(-(epsilon * distances)**2)
	elif kernelType == 'multiquadric':
		phi = jnp.sqrt(1.0 + (epsilon * distances)**2)
	elif kernelType == 'inverse_multiquadric':
		phi = 1.0 / jnp.sqrt(1.0 + (epsilon * distances)**2)
	elif kernelType == 'thin_plate':
		# r^2 * log(r), with special handling for r=0
		r2 = distances**2
		phi = jnp.where(distances > 1e-10, r2 * jnp.log(distances + 1e-10), 0.0)
	else:
		raise ValueError(f"Unknown kernel type: {kernelType}")

	return phi


def solveRbfWeights(centers: jnp.ndarray, # (K, M)
                    targetValues: jnp.ndarray, # (K, D)
                    epsilon: float = 1.0,
                    kernelType: str = 'gaussian',
                    regularization: float = 1e-8,
                    sampleWeights: jnp.ndarray = None) -> jnp.ndarray: # (K, D)
	"""
	Solves for RBF weights given centers and target values.

	Sets up and solves the linear system:
		Φ @ W = targetValues
	where Φ[i,j] = kernel(||centers[i] - centers[j]||)

	Args:
		centers: (K, M) RBF center points in parameter space
		targetValues: (K, D) target function values at each center
		epsilon: RBF shape parameter
		kernelType: kernel function to use
		regularization: small value added to diagonal for numerical stability
		sampleWeights: (K,) optional per-sample weights (0=ignore, 1=full weight)
		               Use to mask out padding samples: [1,1,1,0,0,0,...]

	Returns:
		weights: (K, D) RBF weights such that RBF(centers[i]) ≈ targetValues[i]

	Usage:
		# Training: compute weights from samples
		weights = solveRbfWeights(paramSamples, poseSamples)

		# With padding mask (padded samples have weight 0)
		sampleWeights = jnp.array([1, 1, 1, 0, 0, 0])  # 3 real, 3 padding
		weights = solveRbfWeights(paramSamples, poseSamples, sampleWeights=sampleWeights)

		# Inference: evaluate at new point
		phi = computeRbfMatrix(paramSamples, newParam[None, :])  # (1, K)
		interpolated = phi @ weights  # (1, D)
	"""
	K = centers.shape[0]

	# Build kernel matrix Φ (centers vs centers)
	phi = computeRbfMatrix(centers, centers, epsilon, kernelType)  # (K, K)

	# Apply sample weights if provided
	if sampleWeights is not None:
		# Weight both rows and columns to maintain symmetry
		# This effectively removes padding samples from the interpolation
		W = jnp.sqrt(sampleWeights)  # (K,)
		phi = phi * W[:, None] * W[None, :]  # (K, K) weighted
		targetValues = targetValues * sampleWeights[:, None]  # (K, D) weighted

	# Add regularization to diagonal for numerical stability
	phi_reg = phi + regularization * jnp.eye(K)

	# Solve linear system: Φ @ W = targetValues
	# weights: (K, D)
	weights = jnp.linalg.solve(phi_reg, targetValues)

	return weights


def interpolateRbf(queryPoints: jnp.ndarray, # (N, M)
                   centers: jnp.ndarray, # (K, M)
                   weights: jnp.ndarray, # (K, D)
                   epsilon: float = 1.0,
                   kernelType: str = 'gaussian',
                   sampleWeights: jnp.ndarray = None) -> jnp.ndarray: # (N, D)
	"""
	Evaluates trained RBF at query points.

	Args:
		queryPoints: (N, M) points to evaluate RBF at
		centers: (K, M) RBF centers (from training)
		weights: (K, D) RBF weights (from solveRbfWeights)
		epsilon: RBF shape parameter (must match training)
		kernelType: kernel type (must match training)
		sampleWeights: (K,) optional per-sample weights (must match training)
		               If used during training, must be provided here too

	Returns:
		values: (N, D) interpolated values at query points
	"""
	# Compute kernel evaluations
	phi = computeRbfMatrix(centers, queryPoints, epsilon, kernelType)  # (N, K)

	# Apply sample weights if provided (mask out padding)
	if sampleWeights is not None:
		phi = phi * sampleWeights[None, :]  # (N, K) - zero out padding columns

	# Weighted sum: phi[i, :] @ weights = sum_j phi[i,j] * weights[j, :]
	values = phi @ weights  # (N, D)

	return values


def computeMeshInertiaTensor(points: jnp.ndarray, faces: jnp.ndarray, density: float = 1.0) -> tuple[
	jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
	"""
	Computes inertia tensor and principal axes for a triangle mesh.

	Uses the standard method of integrating over tetrahedra formed by each triangle
	and the origin. Assumes mesh is closed and consistently wound (CCW from outside).

	Args:
		points: (V, 3) vertex positions
		faces: (F, 3) triangle vertex indices (int32)
		density: uniform material density (mass per unit volume)

	Returns:
		centerOfMass: (3,) center of mass position
		principalInertia: (3,) diagonal inertia tensor in principal axes frame (sorted)
		principalAxes: (3, 3) rotation matrix, columns are principal axes (largest inertia last)
		totalMass: scalar total mass

	Algorithm:
		1. Compute signed volume and center of mass via divergence theorem
		2. Shift to CoM frame
		3. Compute inertia tensor via tetrahedral decomposition
		4. Diagonalize via eigendecomposition to get principal axes

	Reference:
		"Computing the Moment of Inertia of a Solid Defined by a Triangle Mesh"
		Kallay, 2006
		https://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
	"""

	# Gather triangle vertices
	v0 = points[faces[:, 0]]  # (F, 3)
	v1 = points[faces[:, 1]]  # (F, 3)
	v2 = points[faces[:, 2]]  # (F, 3)

	# Compute signed volume of each tetrahedron (triangle + origin)
	# Volume = (1/6) * det([v0, v1, v2])
	cross_prod = jnp.cross(v1, v2)  # (F, 3)
	tet_volumes = jnp.sum(v0 * cross_prod, axis=-1) / 6.0  # (F,)

	# Total signed volume
	total_volume = jnp.sum(tet_volumes)
	total_mass = density * total_volume

	# Compute center of mass
	# CoM contribution from each tet: (v0 + v1 + v2) / 4 * volume
	tet_centroids = (v0 + v1 + v2) / 4.0  # (F, 3)
	com = jnp.sum(tet_centroids * tet_volumes[:, None], axis=0) / total_volume  # (3,)

	# Shift vertices to CoM frame
	v0_com = v0 - com[None, :]
	v1_com = v1 - com[None, :]
	v2_com = v2 - com[None, :]

	# Compute inertia tensor contributions from each tetrahedron
	# Using canonical tetrahedral inertia formulas

	# For a tetrahedron with vertices at origin and v0, v1, v2:
	# I_xx = (volume/20) * (y0^2 + y1^2 + y2^2 + y0*y1 + y0*y2 + y1*y2 + z0^2 + ...)
	# We use the full formula from the reference paper

	def compute_tet_inertia(v0, v1, v2, vol):
		"""Compute inertia tensor for single tet (origin, v0, v1, v2)"""
		# Extract coordinates
		x = jnp.stack([v0[:, 0], v1[:, 0], v2[:, 0]], axis=-1)  # (F, 3)
		y = jnp.stack([v0[:, 1], v1[:, 1], v2[:, 1]], axis=-1)  # (F, 3)
		z = jnp.stack([v0[:, 2], v1[:, 2], v2[:, 2]], axis=-1)  # (F, 3)

		# Compute products (vectorized over faces)
		x2 = x * x  # (F, 3)
		y2 = y * y
		z2 = z * z

		# Sum of squares and products
		x2_sum = jnp.sum(x2, axis=-1)  # (F,)
		y2_sum = jnp.sum(y2, axis=-1)
		z2_sum = jnp.sum(z2, axis=-1)

		xy_sum = jnp.sum(x * y, axis=-1)
		xz_sum = jnp.sum(x * z, axis=-1)
		yz_sum = jnp.sum(y * z, axis=-1)

		# Pairwise products
		xy_prod = x[:, 0]*y[:, 1] + x[:, 0]*y[:, 2] + x[:, 1]*y[:, 0] + x[:, 1]*y[:, 2] + x[:, 2]*y[:, 0] + x[:, 2]*y[:, 1]
		xz_prod = x[:, 0]*z[:, 1] + x[:, 0]*z[:, 2] + x[:, 1]*z[:, 0] + x[:, 1]*z[:, 2] + x[:, 2]*z[:, 0] + x[:, 2]*z[:, 1]
		yz_prod = y[:, 0]*z[:, 1] + y[:, 0]*z[:, 2] + y[:, 1]*z[:, 0] + y[:, 1]*z[:, 2] + y[:, 2]*z[:, 0] + y[:, 2]*z[:, 1]

		# Inertia tensor diagonal elements (per tet)
		# I_xx = integral(y^2 + z^2) dV
		I_xx = (y2_sum + z2_sum) / 20.0
		I_yy = (x2_sum + z2_sum) / 20.0
		I_zz = (x2_sum + y2_sum) / 20.0

		# Off-diagonal elements (per tet)
		# I_xy = -integral(x*y) dV
		I_xy = -(xy_sum + xy_prod) / 20.0
		I_xz = -(xz_sum + xz_prod) / 20.0
		I_yz = -(yz_sum + yz_prod) / 20.0

		# Weight by volume and density
		weight = vol * density  # (F,)

		return I_xx * weight, I_yy * weight, I_zz * weight, I_xy * weight, I_xz * weight, I_yz * weight

	# Compute inertia contributions from all tets
	I_xx, I_yy, I_zz, I_xy, I_xz, I_yz = compute_tet_inertia(v0_com, v1_com, v2_com, tet_volumes)

	# Sum over all tetrahedra
	I_xx_total = jnp.sum(I_xx)
	I_yy_total = jnp.sum(I_yy)
	I_zz_total = jnp.sum(I_zz)
	I_xy_total = jnp.sum(I_xy)
	I_xz_total = jnp.sum(I_xz)
	I_yz_total = jnp.sum(I_yz)

	# Build symmetric inertia tensor
	I_tensor = jnp.array([
		[I_xx_total, I_xy_total, I_xz_total],
		[I_xy_total, I_yy_total, I_yz_total],
		[I_xz_total, I_yz_total, I_zz_total]
	])

	# Eigendecomposition to get principal axes
	# Eigenvalues are principal moments, eigenvectors are principal axes
	eigenvalues, eigenvectors = jnp.linalg.eigh(I_tensor)

	# Sort by eigenvalue (smallest to largest)
	# Convention: principal axes ordered by increasing moment of inertia
	sort_idx = jnp.argsort(eigenvalues)
	principal_inertia = eigenvalues[sort_idx]
	principal_axes = eigenvectors[:, sort_idx]

	# Ensure right-handed coordinate system
	# If det(principal_axes) < 0, flip the last axis
	det = jnp.linalg.det(principal_axes)
	principal_axes = jnp.where(
		det < 0,
		principal_axes.at[:, 2].multiply(-1),
		principal_axes
	)

	return com, principal_inertia, principal_axes, total_mass
