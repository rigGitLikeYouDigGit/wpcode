from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
from functools import partial
import jax
from jax import jit, numpy as jnp, value_and_grad
from typing import NamedTuple, Tuple


@jit
def lbs_sparse(rest_vertices, weight_values, weight_joint_indices, joint_transforms):
    """Blend matrices first, then transform — useful when you also need normals."""
    nV = rest_vertices.shape[0]

    # (nV, K, 4, 4)
    gathered = joint_transforms[weight_joint_indices]

    # Blend matrices: M[v] = sum_k w[v,k] * T[v,k]
    # → (nV, 4, 4)
    blended_matrices = jnp.einsum('vk,vkip->vip', weight_values, gathered)

    # Homogeneous vertex transform
    ones = jnp.ones((nV, 1), dtype=rest_vertices.dtype)
    rest_homo = jnp.concatenate([rest_vertices, ones], axis=-1)

    # (nV, 4, 4) @ (nV, 4) → (nV, 4)
    deformed_homo = jnp.einsum('vip,vp->vi', blended_matrices, rest_homo)

    return deformed_homo[:, :3] / deformed_homo[:, 3:4], blended_matrices


def lbs_loss(rest_vertices, weight_values, weight_joint_indices,
             joint_transforms, target_vertices):
    """L2 loss between deformed mesh and target pose."""
    deformed = lbs_sparse(rest_vertices, weight_values,
                          weight_joint_indices, joint_transforms)
    residual = deformed - target_vertices                                # (nV, 3)
    return 0.5 * jnp.sum(residual ** 2)

# ── Autodiff backward — gradients w.r.t. weights & transforms ──
#    argnums=(1, 3) → d_loss/d_weight_values, d_loss/d_joint_transforms
@jit
def backward_autodiff(rest_vertices, weight_values, weight_joint_indices,
                      joint_transforms, target_vertices):
    loss, (grad_weights, grad_transforms) = value_and_grad(
        lbs_loss, argnums=(1, 3)
    )(rest_vertices, weight_values, weight_joint_indices,
      joint_transforms, target_vertices)
    return loss, grad_weights, grad_transforms


@jit
def jacobian_vertices_wrt_transforms(rest_vertices, weight_values,
                                     weight_joint_indices, joint_transforms):
    """d(deformed) / d(joint_transforms) — full (nV,3,nJ,4,4) Jacobian."""
    return jax.jacobian(
        lambda T: lbs_sparse(rest_vertices, weight_values,
                             weight_joint_indices, T)[0]
    )(joint_transforms)

@jit
def project_onto_simplex(y: jnp.ndarray) -> jnp.ndarray:
    """
    Project each row of y onto the probability simplex Δ^K.

    Parameters
    ----------
    y : (nV, K)
        Unconstrained weight vectors (e.g. after gradient update).

    Returns
    -------
    w : (nV, K)
        Projected weights satisfying  w >= 0,  Σ_k w_k = 1  per row.

    Algorithm
    ---------
    Duchi et al., "Efficient Projections onto the ℓ1-Ball for
    Learning in High Dimensions", ICML 2008.
    """
    # ── Step 1: Sort each row descending ────────────────────────
    # (nV, K)
    mu = jnp.sort(y, axis=-1)[:, ::-1]

    # ── Step 2: Cumulative sum along K ──────────────────────────
    # cumsum[v, j] = Σ_{k=0}^{j} μ[v, k]
    cumsum = jnp.cumsum(mu, axis=-1)                              # (nV, K)

    # ── Step 3: Find ρ — number of active components ───────────
    # rho_candidates[j] = j + 1  (1-indexed count)
    K = y.shape[-1]
    rho_candidates = jnp.arange(1, K + 1, dtype=y.dtype)         # (K,)

    # Test condition:  μ[j] + (1 - cumsum[j]) / (j+1) > 0
    # Equivalent to:   μ[j] * (j+1) + 1 - cumsum[j] > 0
    condition = mu + (1.0 - cumsum) / rho_candidates              # (nV, K)
    # Boolean mask of passing candidates
    valid = (condition > 0).astype(y.dtype)                       # (nV, K)

    # ρ = max j such that condition holds (1-indexed)
    # Multiply valid mask by index, take max
    rho = jnp.sum(valid, axis=-1, keepdims=True)                  # (nV, 1)

    # ── Step 4: Compute threshold λ ─────────────────────────────
    # λ[v] = (cumsum[v, ρ-1] - 1) / ρ
    # Gather cumsum at position ρ-1 for each row
    rho_idx = (rho - 1).astype(jnp.int32)                        # (nV, 1)
    cumsum_at_rho = jnp.take_along_axis(cumsum, rho_idx, axis=-1) # (nV, 1)

    lam = (cumsum_at_rho - 1.0) / rho                            # (nV, 1)

    # ── Step 5: Shift and clamp ─────────────────────────────────
    w = jnp.maximum(y - lam, 0.0)                                # (nV, K)

    return w

@jit
def project_onto_sparse_simplex(y: jnp.ndarray, max_nnz: int = 4) -> jnp.ndarray:
    """
    Project onto simplex AND enforce at most max_nnz nonzeros per row.

    1. Zero out all but the top-max_nnz entries per row.
    2. Project the survivors onto the simplex.
    """
    K = y.shape[-1]

    # ── Keep only top max_nnz per row ───────────────────────────
    # Get indices of top-max_nnz values
    top_vals, top_idx = jax.lax.top_k(y, max_nnz)                # (nV, max_nnz)

    # Project the top-k slice onto the simplex
    top_projected = project_onto_simplex(top_vals)                # (nV, max_nnz)

    # ── Scatter back into full K-dimensional vector ─────────────
    w = jnp.zeros_like(y)                                         # (nV, K)

    # Advanced scatter: place projected values at original positions
    rows = jnp.arange(y.shape[0])[:, None]                       # (nV, 1)
    w = w.at[rows, top_idx].set(top_projected)

    return w

@jit
def projected_gradient_step(
    rest_vertices: jnp.ndarray,        # (nV, 3)
    weight_values: jnp.ndarray,        # (nV, K)
    weight_joint_indices: jnp.ndarray, # (nV, K)
    joint_transforms: jnp.ndarray,     # (nJ, 4, 4)
    target_vertices: jnp.ndarray,      # (nV, 3)
    lr: float = 1e-3,
):
    """
    Single step of projected gradient descent on skin weights.

    1. Compute loss + gradient w.r.t. weights
    2. Gradient step in unconstrained space
    3. Project back onto simplex

    Returns updated weights guaranteed to lie on Δ^K.
    """
    # ── Forward + Backward ──────────────────────────────────────
    loss, grad_w = jax.value_and_grad(lbs_loss, argnums=1)(
        rest_vertices, weight_values, weight_joint_indices,
        joint_transforms, target_vertices
    )

    # ── Unconstrained gradient step ─────────────────────────────
    w_unconstrained = weight_values - lr * grad_w                 # (nV, K)

    # ── Project onto simplex ────────────────────────────────────
    w_projected = project_onto_simplex(w_unconstrained)           # (nV, K)

    return loss, w_projected



class SkinningState(NamedTuple):
    """All static (non-optimised) skinning data."""
    rest_vertices: jnp.ndarray          # (nV, 3)
    weight_values: jnp.ndarray          # (nV, K)
    weight_joint_indices: jnp.ndarray   # (nV, K)   int32
    bind_joint_positions: jnp.ndarray   # (nJ, 3)   anchor positions
    joint_rotations: jnp.ndarray        # (nJ, 3, 3)
    inverse_bind_matrices: jnp.ndarray  # (nJ, 4, 4)
    stiffness: jnp.ndarray             # (nJ,)
    target_vertices: jnp.ndarray        # (nV, 3)


# ════════════════════════════════════════════════════════════════
#  Transform construction
# ════════════════════════════════════════════════════════════════

@jit
def build_joint_transforms(
    joint_positions: jnp.ndarray,      # (nJ, 3)  — optimisable
    joint_rotations: jnp.ndarray,      # (nJ, 3, 3)
    inverse_bind_matrices: jnp.ndarray # (nJ, 4, 4)
) -> jnp.ndarray:
    """
    Construct per-joint skinning matrices from position + rotation.

    skinning_matrix[j] = world_transform[j] @ inverse_bind_matrix[j]

    where world_transform[j] = [R[j]  t[j]]
                                [0     1   ]

    Parameters
    ----------
    joint_positions : (nJ, 3)
        Current (optimisable) joint world translations.
    joint_rotations : (nJ, 3, 3)
        Current joint rotation matrices.
    inverse_bind_matrices : (nJ, 4, 4)
        Precomputed inverse of bind-pose transforms.

    Returns
    -------
    skinning_matrices : (nJ, 4, 4)
    """
    nJ = joint_positions.shape[0]

    # Build 4×4 world transforms:  [R | t]
    #                                [0 | 1]
    world = jnp.zeros((nJ, 4, 4), dtype=joint_positions.dtype)
    world = world.at[:, :3, :3].set(joint_rotations)
    world = world.at[:, :3, 3].set(joint_positions)
    world = world.at[:, 3, 3].set(1.0)

    # Skinning matrix = world @ inverse_bind
    # (nJ, 4, 4) @ (nJ, 4, 4) → (nJ, 4, 4)
    skinning_matrices = jnp.einsum('jik,jkp->jip', world, inverse_bind_matrices)

    return skinning_matrices


# ════════════════════════════════════════════════════════════════
#  SO(3) Exponential Map — Rodrigues
# ════════════════════════════════════════════════════════════════

@jit
def skew_symmetric(v: jnp.ndarray) -> jnp.ndarray:
    """
    Batch skew-symmetric matrix from 3-vectors.

    Parameters
    ----------
    v : (..., 3)

    Returns
    -------
    K : (..., 3, 3)   where K @ u = v × u
    """
    # Extract components with full leading-dim support
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = jnp.zeros_like(x)

    # Stack into (..., 3, 3)
    K = jnp.stack([
        jnp.stack([zero, -z,    y  ], axis=-1),
        jnp.stack([z,    zero, -x  ], axis=-1),
        jnp.stack([-y,   x,    zero], axis=-1),
    ], axis=-2)

    return K


@jit
def exp_so3(omega: jnp.ndarray) -> jnp.ndarray:
    """
    Exponential map from so(3) → SO(3) via Rodrigues' formula.
    Numerically stable for small angles using Taylor expansion.

    Parameters
    ----------
    omega : (..., 3)
        Axis-angle vectors. Direction = axis, magnitude = angle (radians).

    Returns
    -------
    R : (..., 3, 3)
        Rotation matrices.
    """
    # Angle θ = ‖ω‖
    theta_sq = jnp.sum(omega ** 2, axis=-1, keepdims=True)       # (..., 1)
    theta = jnp.sqrt(jnp.maximum(theta_sq, 1e-12))               # (..., 1)

    # For numerical stability near θ=0, use Taylor:
    #   sin(θ)/θ     ≈ 1 - θ²/6
    #   (1-cos(θ))/θ² ≈ 1/2 - θ²/24
    #
    # We blend between exact and Taylor based on θ magnitude.
    small = (theta_sq < 1e-8).squeeze(-1)                         # (...,)

    # Exact coefficients
    sinc = jnp.sin(theta) / theta                                 # sin(θ)/θ
    cosc = (1.0 - jnp.cos(theta)) / theta_sq                     # (1-cos(θ))/θ²

    # Taylor coefficients
    sinc_taylor = 1.0 - theta_sq / 6.0
    cosc_taylor = 0.5 - theta_sq / 24.0

    # Blend
    sinc = jnp.where(small[..., None], sinc_taylor, sinc)        # (..., 1)
    cosc = jnp.where(small[..., None], cosc_taylor, cosc)        # (..., 1)

    # Skew-symmetric matrix K
    K = skew_symmetric(omega)                                     # (..., 3, 3)

    # K² = K @ K
    K2 = jnp.einsum('...ij,...jk->...ik', K, K)                  # (..., 3, 3)

    # R = I + sin(θ)/θ · K + (1-cos(θ))/θ² · K²
    # Reshape coefficients for broadcasting with (3,3)
    I = jnp.eye(3, dtype=omega.dtype)
    R = I + sinc[..., None] * K + cosc[..., None] * K2

    return R


@jit
def log_so3(R: jnp.ndarray) -> jnp.ndarray:
    """
    Logarithmic map from SO(3) → so(3).
    Inverse of exp_so3. Used to initialise ω from existing rotations.

    Parameters
    ----------
    R : (..., 3, 3)

    Returns
    -------
    omega : (..., 3)
    """
    # cos(θ) = (tr(R) - 1) / 2
    trace = jnp.trace(R, axis1=-2, axis2=-1)                     # (...,)
    cos_theta = jnp.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)                                 # (...,)

    # For small θ: ω ≈ (R - Rᵀ)^∨ / 2
    # General:     ω = (θ / 2sinθ) · (R - Rᵀ)^∨
    small = (theta < 1e-6)

    # Vee map: extract axis from skew part (R - Rᵀ)/2
    skew_part = 0.5 * (R - jnp.swapaxes(R, -1, -2))             # (..., 3, 3)
    vee = jnp.stack([
        skew_part[..., 2, 1],
        skew_part[..., 0, 2],
        skew_part[..., 1, 0],
    ], axis=-1)                                                   # (..., 3)

    # Scale factor
    sin_theta = jnp.sin(theta)
    safe_sin = jnp.where(small, 1.0, sin_theta)
    scale = jnp.where(small, 1.0, theta / safe_sin)              # (...,)

    omega = scale[..., None] * vee                                # (..., 3)
    return omega


# ════════════════════════════════════════════════════════════════
#  Extended state & parameter containers
# ════════════════════════════════════════════════════════════════

class JointParams(NamedTuple):
    """Optimisable per-joint parameters — all unconstrained."""
    positions: jnp.ndarray       # (nJ, 3)  translation
    log_rotations: jnp.ndarray   # (nJ, 3)  axis-angle (ω in so(3))


class StiffnessConfig(NamedTuple):
    """Separate stiffness for translation and rotation."""
    position_stiffness: jnp.ndarray    # (nJ,)  κ_t[j]
    rotation_stiffness: jnp.ndarray    # (nJ,)  κ_r[j]


class RigState(NamedTuple):
    """All static (non-optimised) rig data."""
    rest_vertices: jnp.ndarray          # (nV, 3)
    weight_values: jnp.ndarray          # (nV, K)
    weight_joint_indices: jnp.ndarray   # (nV, K) int32
    bind_joint_positions: jnp.ndarray   # (nJ, 3)
    bind_joint_rotations: jnp.ndarray   # (nJ, 3, 3)
    bind_log_rotations: jnp.ndarray     # (nJ, 3)  log of bind rotations
    inverse_bind_matrices: jnp.ndarray  # (nJ, 4, 4)
    stiffness: StiffnessConfig
    target_vertices: jnp.ndarray        # (nV, 3)


# ════════════════════════════════════════════════════════════════
#  Transform construction (rotation from so(3))
# ════════════════════════════════════════════════════════════════

@jit
def build_joint_transforms(
    params: JointParams,
    inverse_bind_matrices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Build (nJ, 4, 4) skinning matrices from unconstrained params.

    world[j] = [exp(ω[j]) | t[j]]  @  inv_bind[j]
               [    0      |  1  ]
    """
    nJ = params.positions.shape[0]

    # Axis-angle → rotation matrices
    R = exp_so3(params.log_rotations)                             # (nJ, 3, 3)

    # Assemble 4×4
    world = jnp.zeros((nJ, 4, 4), dtype=params.positions.dtype)
    world = world.at[:, :3, :3].set(R)
    world = world.at[:, :3, 3].set(params.positions)
    world = world.at[:, 3, 3].set(1.0)

    # Skinning = world @ inverse_bind
    return jnp.einsum('jik,jkp->jip', world, inverse_bind_matrices)


# ════════════════════════════════════════════════════════════════
#  Sparse LBS (unchanged)
# ════════════════════════════════════════════════════════════════

@jit
def lbs_sparse(rest_vertices, weight_values, weight_joint_indices, joint_transforms):
    nV = rest_vertices.shape[0]
    ones = jnp.ones((nV, 1), dtype=rest_vertices.dtype)
    h = jnp.concatenate([rest_vertices, ones], axis=-1)
    G = joint_transforms[weight_joint_indices]
    t = jnp.einsum('vkip,vp->vki', G, h)
    b = jnp.einsum('vk,vki->vi', weight_values, t)
    return b[:, :3] / b[:, 3:4]


# ════════════════════════════════════════════════════════════════
#  Loss function: data + position stiffness + rotation stiffness
# ════════════════════════════════════════════════════════════════

@jit
def joint_loss(
    params: JointParams,
    state: RigState,
) -> jnp.ndarray:
    """
    L = L_data + L_pos_stiffness + L_rot_stiffness

    L_data           = (1/2) Σ_v ‖LBS(v) - target[v]‖²

    L_pos_stiffness  = (1/2) Σ_j κ_t[j] · ‖t[j] - t_bind[j]‖²

    L_rot_stiffness  = (1/2) Σ_j κ_r[j] · ‖ω[j] - ω_bind[j]‖²
                     = (1/2) Σ_j κ_r[j] · ‖Δω[j]‖²

    The rotation penalty ‖Δω‖² is the squared geodesic distance on SO(3)
    (exact for small angles, excellent approximation otherwise).
    This is far superior to penalising matrix element differences.
    """
    # ── Forward LBS ─────────────────────────────────────────────
    transforms = build_joint_transforms(params, state.inverse_bind_matrices)
    deformed = lbs_sparse(
        state.rest_vertices,
        state.weight_values,
        state.weight_joint_indices,
        transforms,
    )

    # ── Data term ───────────────────────────────────────────────
    residual = deformed - state.target_vertices
    loss_data = 0.5 * jnp.sum(residual ** 2)

    # ── Position stiffness ──────────────────────────────────────
    pos_delta = params.positions - state.bind_joint_positions     # (nJ, 3)
    pos_dist_sq = jnp.sum(pos_delta ** 2, axis=-1)               # (nJ,)
    loss_pos = 0.5 * jnp.sum(
        state.stiffness.position_stiffness * pos_dist_sq
    )

    # ── Rotation stiffness ──────────────────────────────────────
    # ‖ω - ω_bind‖² in so(3) ≈ geodesic distance² on SO(3)
    rot_delta = params.log_rotations - state.bind_log_rotations   # (nJ, 3)
    rot_dist_sq = jnp.sum(rot_delta ** 2, axis=-1)               # (nJ,)
    loss_rot = 0.5 * jnp.sum(
        state.stiffness.rotation_stiffness * rot_dist_sq
    )

    return loss_data + loss_pos + loss_rot


# ════════════════════════════════════════════════════════════════
#  Gradient
# ════════════════════════════════════════════════════════════════

@jit
def joint_grad(
    params: JointParams,
    state: RigState,
) -> Tuple[jnp.ndarray, JointParams]:
    """
    Returns (loss, JointParams(∇positions, ∇log_rotations)).

    JAX autodiff through exp_so3 gives us exact Lie-algebra gradients.
    No manual Jacobian of Rodrigues needed.
    """
    return value_and_grad(joint_loss)(params, state)


# ════════════════════════════════════════════════════════════════
#  Adam optimiser — handles JointParams pytree natively
# ════════════════════════════════════════════════════════════════

class AdamState(NamedTuple):
    m: JointParams       # first moment (same tree structure)
    v: JointParams       # second moment
    step: int


def adam_init(params: JointParams) -> AdamState:
    return AdamState(
        m=jax.tree.map(jnp.zeros_like, params),
        v=jax.tree.map(jnp.zeros_like, params),
        step=0,
    )


@partial(jit, static_argnames=('beta1', 'beta2', 'eps'))
def adam_step(
    params: JointParams,
    grads: JointParams,
    adam_state: AdamState,
    per_joint_lr: JointParams,           # tree of (nJ, 1) learning rates
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[JointParams, AdamState]:
    """
    Adam update over the JointParams pytree.
    Per-joint, per-parameter-type learning rates.
    """
    t = adam_state.step + 1

    # Moment updates
    m = jax.tree.map(
        lambda m_prev, g: beta1 * m_prev + (1.0 - beta1) * g,
        adam_state.m, grads
    )
    v = jax.tree.map(
        lambda v_prev, g: beta2 * v_prev + (1.0 - beta2) * g ** 2,
        adam_state.v, grads
    )

    # Bias correction
    m_hat = jax.tree.map(lambda m_i: m_i / (1.0 - beta1 ** t), m)
    v_hat = jax.tree.map(lambda v_i: v_i / (1.0 - beta2 ** t), v)

    # Per-joint scaled update
    new_params = jax.tree.map(
        lambda p, mh, vh, lr: p - lr * mh / (jnp.sqrt(vh) + eps),
        params, m_hat, v_hat, per_joint_lr
    )

    return new_params, AdamState(m=m, v=v, step=t)


# ════════════════════════════════════════════════════════════════
#  Stiffness → per-joint, per-param-type learning rates
# ════════════════════════════════════════════════════════════════

def stiffness_to_lr(
    stiffness: StiffnessConfig,
    base_lr_pos: float = 1e-3,
    base_lr_rot: float = 1e-3,
) -> JointParams:
    """
    Map stiffness to per-joint learning rates for both
    position and rotation parameters.

    lr[j] = base_lr / (1 + κ[j])

    Returns a JointParams with (nJ, 1) shaped LR arrays
    for broadcasting against (nJ, 3) param arrays.
    """
    lr_pos = (base_lr_pos / (1.0 + stiffness.position_stiffness))[:, None]
    lr_rot = (base_lr_rot / (1.0 + stiffness.rotation_stiffness))[:, None]
    return JointParams(positions=lr_pos, log_rotations=lr_rot)


# ════════════════════════════════════════════════════════════════
#  Rotation-aware diagnostics
# ════════════════════════════════════════════════════════════════

@jit
def compute_diagnostics(
    params: JointParams,
    state: RigState,
) -> dict:
    """Per-joint displacement and rotation magnitude diagnostics."""

    # ── Position displacement ───────────────────────────────────
    pos_delta = params.positions - state.bind_joint_positions
    pos_dist = jnp.linalg.norm(pos_delta, axis=-1)               # (nJ,)

    # ── Rotation displacement (geodesic on SO(3)) ───────────────
    rot_delta = params.log_rotations - state.bind_log_rotations
    rot_angle = jnp.linalg.norm(rot_delta, axis=-1)              # (nJ,) radians
    rot_angle_deg = jnp.rad2deg(rot_angle)

    # ── Stiffness energy breakdown ──────────────────────────────
    pos_energy = 0.5 * state.stiffness.position_stiffness * pos_dist ** 2
    rot_energy = 0.5 * state.stiffness.rotation_stiffness * rot_angle ** 2

    return {
        'pos_displacement': pos_dist,           # (nJ,)
        'rot_angle_rad': rot_angle,             # (nJ,)
        'rot_angle_deg': rot_angle_deg,         # (nJ,)
        'pos_energy': pos_energy,               # (nJ,)
        'rot_energy': rot_energy,               # (nJ,)
        'total_pos_energy': jnp.sum(pos_energy),
        'total_rot_energy': jnp.sum(rot_energy),
        'max_pos_disp': jnp.max(pos_dist),
        'max_rot_deg': jnp.max(rot_angle_deg),
    }


# ════════════════════════════════════════════════════════════════
#  Full optimisation loop
# ════════════════════════════════════════════════════════════════

def optimise_joints(
    state: RigState,
    base_lr_pos: float = 1e-3,
    base_lr_rot: float = 5e-4,
    n_steps: int = 1000,
    log_every: int = 100,
) -> Tuple[JointParams, list]:
    """
    Co-optimise joint positions and rotations with:
      - Axis-angle parameterisation (unconstrained in ℝ³)
      - Per-joint, per-param-type stiffness regularisation
      - Per-joint, per-param-type adaptive learning rates
      - Adam momentum

    Parameters
    ----------
    state : RigState
        Immutable rig data including bind pose and stiffness.
    base_lr_pos : float
        Base learning rate for positions (κ=0 joints).
    base_lr_rot : float
        Base learning rate for rotations. Typically smaller than
        position LR because small angle changes produce large
        vertex displacements for vertices far from the joint.
    n_steps : int
        Optimisation iterations.
    log_every : int
        Logging interval.

    Returns
    -------
    optimised_params : JointParams
    loss_history : list[float]
    """
    # ── Initialise at bind pose ─────────────────────────────────
    params = JointParams(
        positions=state.bind_joint_positions.copy(),
        log_rotations=state.bind_log_rotations.copy(),
    )
    opt_state = adam_init(params)

    # ── Per-joint learning rates from stiffness ─────────────────
    per_joint_lr = stiffness_to_lr(
        state.stiffness, base_lr_pos, base_lr_rot
    )

    loss_history = []

    for step in range(n_steps):
        # ── Forward + backward ──────────────────────────────────
        loss, grads = joint_grad(params, state)

        # ── Adam step with per-joint LR ─────────────────────────
        params, opt_state = adam_step(
            params, grads, opt_state, per_joint_lr
        )

        loss_val = float(loss)
        loss_history.append(loss_val)

        # ── Logging ─────────────────────────────────────────────
        if step % log_every == 0:
            diag = compute_diagnostics(params, state)
            print(
                f"Step {step:5d}  │  "
                f"Loss: {loss_val:12.4f}  │  "
                f"Max Δpos: {diag['max_pos_disp']:.5f}  │  "
                f"Max Δrot: {diag['max_rot_deg']:.2f}°  │  "
                f"E_pos: {diag['total_pos_energy']:.4f}  │  "
                f"E_rot: {diag['total_rot_energy']:.4f}"
            )

    return params, loss_history


# ════════════════════════════════════════════════════════════════
#  State construction helper
# ════════════════════════════════════════════════════════════════

def build_rig_state(
    rest_vertices: jnp.ndarray,
    weight_values: jnp.ndarray,
    weight_joint_indices: jnp.ndarray,
    bind_joint_positions: jnp.ndarray,
    bind_joint_rotations: jnp.ndarray,
    inverse_bind_matrices: jnp.ndarray,
    position_stiffness: jnp.ndarray,
    rotation_stiffness: jnp.ndarray,
    target_vertices: jnp.ndarray,
) -> RigState:
    """Construct RigState, computing bind log-rotations via log_so3."""
    return RigState(
        rest_vertices=rest_vertices,
        weight_values=weight_values,
        weight_joint_indices=weight_joint_indices,
        bind_joint_positions=bind_joint_positions,
        bind_joint_rotations=bind_joint_rotations,
        bind_log_rotations=log_so3(bind_joint_rotations),
        inverse_bind_matrices=inverse_bind_matrices,
        stiffness=StiffnessConfig(
            position_stiffness=position_stiffness,
            rotation_stiffness=rotation_stiffness,
        ),
        target_vertices=target_vertices,
    )