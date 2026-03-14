from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import NamedTuple, Tuple


class SparseCOO(NamedTuple):
    """Sparse matrix in COO format."""
    row_indices: jnp.ndarray   # (nnz,) int32
    col_indices: jnp.ndarray   # (nnz,) int32
    values: jnp.ndarray        # (nnz,) float32
    shape: Tuple[int, int]     # (nrows, ncols)


@jit
def spmv(A: SparseCOO, x: jnp.ndarray) -> jnp.ndarray:
    """
    Sparse matrix-vector product:  y = A @ x

    For each non-zero entry A[r, c] = v:
        y[r] += v * x[c]

    Parameters
    ----------
    A : SparseCOO
    x : (ncols,)

    Returns
    -------
    y : (nrows,)
    """
    # Gather: fetch x values at column indices
    x_gathered = x[A.col_indices]                        # (nnz,)

    # Multiply by stored values
    contributions = A.values * x_gathered                # (nnz,)

    # Scatter-add into output at row indices
    y = jnp.zeros(A.shape[0], dtype=x.dtype)
    y = y.at[A.row_indices].add(contributions)

    return y


@jit
def spmv_batch(A: SparseCOO, X: jnp.ndarray) -> jnp.ndarray:
    """
    Sparse matrix × dense matrix:  Y = A @ X

    Parameters
    ----------
    A : SparseCOO   — (n, m) sparse
    X : (m, k)      — dense

    Returns
    -------
    Y : (n, k)
    """
    # Gather rows from X at column indices: (nnz, k)
    X_gathered = X[A.col_indices]

    # Scale by sparse values: (nnz, k)
    contributions = A.values[:, None] * X_gathered

    # Scatter-add
    Y = jnp.zeros((A.shape[0], X.shape[1]), dtype=X.dtype)
    Y = Y.at[A.row_indices].add(contributions)

    return Y


@partial(jit, static_argnames=('n_lanczos', 'n_reorth'))
def lanczos(
    A: SparseCOO,
    n_lanczos: int = 100,         # Krylov subspace dimension
    n_reorth: int = 2,            # full reorthogonalisation passes
    key: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Lanczos tridiagonalisation of a sparse symmetric matrix.

    Parameters
    ----------
    A : SparseCOO
        Symmetric (n, n) sparse matrix.
    n_lanczos : int
        Number of Lanczos iterations. Determines accuracy.
        Typically 2-5x the number of desired eigenvalues.
    n_reorth : int
        Number of full reorthogonalisation sweeps per step.
        0 = classic Lanczos (fast but numerically unstable).
        1 = partial reorth (usually sufficient).
        2 = full reorth (safest).
    key : PRNGKey
        For random initial vector.

    Returns
    -------
    alphas : (n_lanczos,)    — diagonal of T
    betas  : (n_lanczos-1,)  — off-diagonal of T
    Q      : (n, n_lanczos)  — orthonormal Lanczos basis
    """
    n = A.shape[0]

    if key is None:
        key = jax.random.PRNGKey(0)

    # ── Initial random unit vector ──────────────────────────────
    q = jax.random.normal(key, (n,))
    q = q / jnp.linalg.norm(q)

    # ── Preallocate storage ─────────────────────────────────────
    # Q[:, j] will store the j-th Lanczos vector
    Q = jnp.zeros((n, n_lanczos), dtype=q.dtype)
    Q = Q.at[:, 0].set(q)

    alphas = jnp.zeros(n_lanczos, dtype=q.dtype)
    betas = jnp.zeros(n_lanczos, dtype=q.dtype)  # betas[0] = β₁ (first off-diag)

    # ── Lanczos iteration via scan ──────────────────────────────
    def lanczos_step(carry, j):
        Q, alphas, betas, q_prev = carry

        q_j = Q[:, j]

        # 1. SpMV: z = A @ q_j
        z = spmv(A, q_j)

        # 2. α_j = q_j · z
        alpha_j = jnp.dot(q_j, z)
        alphas = alphas.at[j].set(alpha_j)

        # 3. Subtract projections onto q_j and q_{j-1}
        beta_prev = jnp.where(j > 0, betas[j - 1], 0.0)
        z = z - alpha_j * q_j - beta_prev * q_prev

        # 4. Full reorthogonalisation against all previous q's
        #    This is the key to numerical stability.
        #    Cost: O(n * j) per step — dominated by SpMV for sparse A.
        def reorth_pass(z, _):
            # Project out all components along existing basis vectors
            # coeffs[k] = Q[:, k] · z   for k = 0..j
            coeffs = Q.T @ z                             # (n_lanczos,)
            # Zero out coefficients for columns > j (not yet filled)
            mask = (jnp.arange(n_lanczos) <= j).astype(z.dtype)
            coeffs = coeffs * mask
            # Subtract: z -= Q @ coeffs
            z = z - Q @ coeffs
            return z, None

        z, _ = jax.lax.scan(reorth_pass, z, None, length=n_reorth)

        # 5. β_{j+1} = ‖z‖
        beta_j = jnp.linalg.norm(z)
        betas = betas.at[j].set(beta_j)

        # 6. q_{j+1} = z / β_j  (guard against zero)
        q_next = z / jnp.maximum(beta_j, 1e-12)

        # Store q_{j+1} (if within bounds, handled by scan length)
        Q_new = jnp.where(
            j + 1 < n_lanczos,
            Q.at[:, j + 1].set(q_next),
            Q
        )

        return (Q_new, alphas, betas, q_j), None

    init_carry = (Q, alphas, betas, jnp.zeros(n, dtype=q.dtype))
    (Q, alphas, betas, _), _ = jax.lax.scan(
        lanczos_step,
        init_carry,
        jnp.arange(n_lanczos),
    )

    return alphas, betas[:-1], Q  # betas has n_lanczos-1 off-diags


@jit
def build_tridiagonal(
    alphas: jnp.ndarray,    # (m,)
    betas: jnp.ndarray,     # (m-1,)
) -> jnp.ndarray:
    """
    Construct dense tridiagonal matrix T from Lanczos coefficients.

    T[i,i]   = α[i]
    T[i,i+1] = β[i]
    T[i+1,i] = β[i]
    """
    m = alphas.shape[0]
    T = jnp.diag(alphas)
    T = T + jnp.diag(betas, k=1)
    T = T + jnp.diag(betas, k=-1)
    return T


@partial(jit, static_argnames=('n_eigen', 'which'))
def sparse_eigsh(
    A: SparseCOO,
    n_eigen: int = 20,
    n_lanczos: int = 100,
    n_reorth: int = 2,
    which: str = 'smallest',
    key: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Find n_eigen smallest (or largest) eigenvalues/vectors
    of a sparse symmetric matrix via Lanczos.

    Equivalent to scipy.sparse.linalg.eigsh but in pure JAX.

    Parameters
    ----------
    A : SparseCOO
        Sparse symmetric (n, n) matrix.
    n_eigen : int
        Number of eigenvalues to compute.
    n_lanczos : int
        Krylov subspace size. Should be ≥ 2*n_eigen,
        typically 5-10x for good convergence.
    n_reorth : int
        Reorthogonalisation passes (0, 1, or 2).
    which : str
        'smallest' — for graph Laplacian (Fiedler vectors).
        'largest'  — for PCA-style applications.
    key : PRNGKey

    Returns
    -------
    eigenvalues : (n_eigen,)
    eigenvectors : (n, n_eigen)
    """
    # ── Run Lanczos ─────────────────────────────────────────────
    alphas, betas, Q = lanczos(A, n_lanczos, n_reorth, key)

    # ── Eigendecompose the small tridiagonal T ──────────────────
    T = build_tridiagonal(alphas, betas)                  # (m, m)
    ritz_values, ritz_vectors = jnp.linalg.eigh(T)       # (m,), (m, m)

    # eigh returns ascending order
    if which == 'smallest':
        # Take the n_eigen smallest
        idx = jnp.arange(n_eigen)
    else:
        # Take the n_eigen largest (from the end)
        idx = jnp.arange(n_lanczos - n_eigen, n_lanczos)

    selected_values = ritz_values[idx]                    # (n_eigen,)
    selected_ritz = ritz_vectors[:, idx]                  # (m, n_eigen)

    # ── Recover full eigenvectors: v = Q @ s ────────────────────
    # Q: (n, m),  selected_ritz: (m, n_eigen)
    eigenvectors = Q @ selected_ritz                      # (n, n_eigen)

    return selected_values, eigenvectors


@partial(jit, static_argnames=('n_eigen', 'n_lanczos', 'n_reorth', 'cg_maxiter'))
def sparse_eigsh_shift_invert(
    A: SparseCOO,
    sigma: float = 0.0,
    n_eigen: int = 20,
    n_lanczos: int = 60,
    n_reorth: int = 2,
    cg_maxiter: int = 200,
    cg_tol: float = 1e-8,
    key: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shift-invert Lanczos for eigenvalues near sigma.

    Uses Conjugate Gradient (CG) as the inner linear solver
    since (A - σI) is SPD for σ below the smallest eigenvalue.
    """
    n = A.shape[0]

    # ── Build shifted operator: B = A - σI ──────────────────────
    # Add -σ to the diagonal entries
    diag_mask = (A.row_indices == A.col_indices)
    shifted_values = A.values - sigma * diag_mask.astype(A.values.dtype)
    B = SparseCOO(
        row_indices=A.row_indices,
        col_indices=A.col_indices,
        values=shifted_values,
        shape=A.shape,
    )

    # ── CG solver for B @ z = rhs ──────────────────────────────
    @partial(jit, static_argnames=('maxiter',))
    def cg_solve(B: SparseCOO, rhs: jnp.ndarray,
                 maxiter: int = 200, tol: float = 1e-8) -> jnp.ndarray:
        """Conjugate Gradient for sparse SPD system B @ z = rhs."""

        def cg_step(carry, _):
            z, r, p, rsold = carry

            Ap = spmv(B, p)
            alpha = rsold / (jnp.dot(p, Ap) + 1e-30)
            z = z + alpha * p
            r = r - alpha * Ap
            rsnew = jnp.dot(r, r)
            beta = rsnew / (rsold + 1e-30)
            p = r + beta * p

            return (z, r, p, rsnew), rsnew

        z0 = jnp.zeros_like(rhs)
        r0 = rhs.copy()                    # r = rhs - B@z0 = rhs
        p0 = r0.copy()
        rs0 = jnp.dot(r0, r0)

        (z, r, p, _), residuals = jax.lax.scan(
            cg_step, (z0, r0, p0, rs0), None, length=maxiter
        )

        return z

    # ── Modified Lanczos: SpMV replaced with CG solve ──────────
    if key is None:
        key = jax.random.PRNGKey(0)

    q = jax.random.normal(key, (n,))
    q = q / jnp.linalg.norm(q)

    Q = jnp.zeros((n, n_lanczos), dtype=q.dtype)
    Q = Q.at[:, 0].set(q)
    alphas = jnp.zeros(n_lanczos, dtype=q.dtype)
    betas = jnp.zeros(n_lanczos, dtype=q.dtype)

    def lanczos_si_step(carry, j):
        Q, alphas, betas, q_prev = carry
        q_j = Q[:, j]

        # Instead of z = A @ q_j, solve B @ z = q_j
        # i.e. z = (A - σI)⁻¹ @ q_j
        z = cg_solve(B, q_j, maxiter=cg_maxiter, tol=cg_tol)

        alpha_j = jnp.dot(q_j, z)
        alphas = alphas.at[j].set(alpha_j)

        beta_prev = jnp.where(j > 0, betas[j - 1], 0.0)
        z = z - alpha_j * q_j - beta_prev * q_prev

        # Reorthogonalise
        def reorth_pass(z, _):
            coeffs = Q.T @ z
            mask = (jnp.arange(n_lanczos) <= j).astype(z.dtype)
            z = z - Q @ (coeffs * mask)
            return z, None

        z, _ = jax.lax.scan(reorth_pass, z, None, length=n_reorth)

        beta_j = jnp.linalg.norm(z)
        betas = betas.at[j].set(beta_j)
        q_next = z / jnp.maximum(beta_j, 1e-12)

        Q_new = jnp.where(j + 1 < n_lanczos, Q.at[:, j + 1].set(q_next), Q)

        return (Q_new, alphas, betas, q_j), None

    init = (Q, alphas, betas, jnp.zeros(n, dtype=q.dtype))
    (Q, alphas, betas, _), _ = jax.lax.scan(
        lanczos_si_step, init, jnp.arange(n_lanczos)
    )

    # ── Extract eigenvalues ─────────────────────────────────────
    T = build_tridiagonal(alphas, betas[:-1])
    mu_vals, mu_vecs = jnp.linalg.eigh(T)

    # μ = 1/(λ-σ) → λ = σ + 1/μ
    # Largest |μ| → λ closest to σ
    # Sort by descending |μ| to get eigenvalues nearest to σ
    sort_idx = jnp.argsort(jnp.abs(mu_vals))[::-1][:n_eigen]

    mu_selected = mu_vals[sort_idx]
    lambda_selected = sigma + 1.0 / mu_selected                  # (n_eigen,)

    # Ritz vectors → full eigenvectors
    ritz_selected = mu_vecs[:, sort_idx]                          # (m, n_eigen)
    eigenvectors = Q @ ritz_selected                              # (n, n_eigen)

    # Sort by ascending eigenvalue for consistency
    final_order = jnp.argsort(lambda_selected)
    lambda_selected = lambda_selected[final_order]
    eigenvectors = eigenvectors[:, final_order]

    return lambda_selected, eigenvectors
