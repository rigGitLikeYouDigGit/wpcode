from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
from functools import lru_cache, partial

import jax
from jax import jit, numpy as jnp, value_and_grad, vmap, lax

from .matrix import SparseCOO, spmv_batch, sparse_eigsh


@jit
def mesh_edges_from_faces(
	face_vertices: jnp.ndarray,
	face_vertex_counts: jnp.ndarray,
	max_face_size: int
) -> jnp.ndarray:
	"""Extract unique edges from polygon faces.

	Args:
		face_vertices: (n_faces, max_face_size) padded face vertex indices (-1 for padding)
		face_vertex_counts: (n_faces,) actual vertex count per face
		max_face_size: maximum vertices per face (for padding)

	Returns:
		edges: (n_edges, 2) array of unique edge vertex indices
	"""
	n_faces = face_vertices.shape[0]

	# Generate all possible edges from faces (with duplicates)
	# Each face of size N generates N edges
	def extract_face_edges(face_verts, face_size):
		"""Extract edges from a single face."""
		# For each vertex i, create edge (i, i+1) wrapping around
		edges = jnp.stack([
			face_verts,
			jnp.roll(face_verts, -1, axis=0)
		], axis=-1)  # (max_face_size, 2)

		# Mark invalid edges (where vertex index is -1 or beyond face size)
		valid = jnp.arange(max_face_size) < face_size

		# Sort edge endpoints to canonicalize (undirected edges)
		edges = jnp.sort(edges, axis=-1)

		# Invalidate by setting to [-1, -1]
		edges = jnp.where(valid[:, None], edges, -1)

		return edges

	# Extract edges from all faces: (n_faces, max_face_size, 2)
	all_edges = vmap(extract_face_edges)(face_vertices, face_vertex_counts)
	all_edges = all_edges.reshape(-1, 2)  # (n_faces * max_face_size, 2)

	# Filter out invalid edges
	valid_mask = all_edges[:, 0] >= 0
	all_edges = all_edges[valid_mask]

	# Remove duplicates by lexicographic uniqueness
	# JAX doesn't have a native unique() that preserves order and returns indices,
	# so we use a sort-based approach
	all_edges = jnp.unique(all_edges, axis=0, size=all_edges.shape[0], fill_value=-1)

	# Remove fill values
	valid_mask = all_edges[:, 0] >= 0
	edges = all_edges[valid_mask]

	return edges


@jit
def mesh_vertex_connectivity(edges: jnp.ndarray, n_vertices: int, max_valence: int) -> tuple[jnp.ndarray, jnp.ndarray]:
	"""Compute sparse vertex connectivity as padded index arrays.

	Args:
		edges: (n_edges, 2) array of edge vertex indices
		n_vertices: total number of vertices
		max_valence: maximum number of neighbors per vertex (for padding)

	Returns:
		neighbor_indices: (n_vertices, max_valence) padded neighbor indices (-1 for padding)
		neighbor_counts: (n_vertices,) actual number of neighbors per vertex
	"""
	# Initialize arrays
	neighbor_indices = jnp.full((n_vertices, max_valence), -1, dtype=jnp.int32)
	neighbor_counts = jnp.zeros(n_vertices, dtype=jnp.int32)

	def add_edge(carry, edge):
		neighbor_indices, neighbor_counts = carry
		v0, v1 = edge

		# Add v1 as neighbor of v0
		count0 = neighbor_counts[v0]
		neighbor_indices = neighbor_indices.at[v0, count0].set(v1)
		neighbor_counts = neighbor_counts.at[v0].add(1)

		# Add v0 as neighbor of v1
		count1 = neighbor_counts[v1]
		neighbor_indices = neighbor_indices.at[v1, count1].set(v0)
		neighbor_counts = neighbor_counts.at[v1].add(1)

		return (neighbor_indices, neighbor_counts), None

	(neighbor_indices, neighbor_counts), _ = lax.scan(
		add_edge,
		(neighbor_indices, neighbor_counts),
		edges
	)

	return neighbor_indices, neighbor_counts


@jit
def mesh_laplacian_uniform(
	neighbor_indices: jnp.ndarray,
	neighbor_counts: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	"""Compute uniform Laplacian matrix as sparse index arrays.

	Args:
		neighbor_indices: (n_vertices, max_valence) padded neighbor indices
		neighbor_counts: (n_vertices,) actual neighbor counts

	Returns:
		row_indices: (nnz,) row indices for sparse matrix
		col_indices: (nnz,) column indices for sparse matrix
		values: (nnz,) Laplacian weights
	"""
	n_vertices, max_valence = neighbor_indices.shape

	# Vectorized approach: build diagonal entries
	row_diag = jnp.arange(n_vertices, dtype=jnp.int32)
	col_diag = jnp.arange(n_vertices, dtype=jnp.int32)
	val_diag = jnp.maximum(neighbor_counts, 1)  # degree (avoid zero for isolated vertices)

	# Build off-diagonal entries: vectorized over all vertex-neighbor pairs
	# Create (n_vertices, max_valence) arrays for rows, cols, values
	rows_offdiag = jnp.repeat(jnp.arange(n_vertices, dtype=jnp.int32)[:, None], max_valence, axis=1)
	cols_offdiag = neighbor_indices
	vals_offdiag = jnp.where(neighbor_indices >= 0, -1.0, 0.0)

	# Flatten and filter valid entries
	rows_offdiag_flat = rows_offdiag.reshape(-1)
	cols_offdiag_flat = cols_offdiag.reshape(-1)
	vals_offdiag_flat = vals_offdiag.reshape(-1)

	valid_mask = cols_offdiag_flat >= 0

	# Concatenate diagonal + off-diagonal
	row_indices = jnp.concatenate([row_diag, rows_offdiag_flat[valid_mask]])
	col_indices = jnp.concatenate([col_diag, cols_offdiag_flat[valid_mask]])
	values = jnp.concatenate([val_diag, vals_offdiag_flat[valid_mask]])

	return row_indices, col_indices, values


@jit
def mesh_laplacian_cotangent(
	positions: jnp.ndarray,
	neighbor_indices: jnp.ndarray,
	neighbor_counts: jnp.ndarray,
	face_indices: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	"""Compute cotangent Laplacian matrix as sparse index arrays.

	Args:
		positions: (n_vertices, 3) vertex positions
		neighbor_indices: (n_vertices, max_valence) padded neighbor indices
		neighbor_counts: (n_vertices,) actual neighbor counts
		face_indices: (n_faces, 3) triangle vertex indices

	Returns:
		row_indices: (nnz,) row indices for sparse matrix
		col_indices: (nnz,) column indices for sparse matrix
		values: (nnz,) cotangent Laplacian weights
	"""
	n_vertices, max_valence = neighbor_indices.shape

	# Vectorized cotangent computation over all faces
	def compute_face_cotangents(face):
		v0, v1, v2 = face

		p0 = positions[v0]
		p1 = positions[v1]
		p2 = positions[v2]

		# Edges
		e0 = p1 - p0
		e1 = p2 - p1
		e2 = p0 - p2

		# Cotangent weights (half-cotangent formula)
		# cot(angle) = dot(u,v) / ||cross(u,v)||
		eps = 1e-8

		# At v0: angle between e0 and -e2
		cot0 = jnp.dot(e0, -e2) / (jnp.linalg.norm(jnp.cross(e0, -e2)) + eps)

		# At v1: angle between e1 and -e0
		cot1 = jnp.dot(e1, -e0) / (jnp.linalg.norm(jnp.cross(e1, -e0)) + eps)

		# At v2: angle between e2 and -e1
		cot2 = jnp.dot(e2, -e1) / (jnp.linalg.norm(jnp.cross(e2, -e1)) + eps)

		# Return edge pairs and their cotangent weights
		# Each edge gets 0.5 * cotangent from opposite angle
		return jnp.array([
			[v0, v1, 0.5 * cot2],  # edge v0-v1, weight from angle at v2
			[v1, v0, 0.5 * cot2],
			[v1, v2, 0.5 * cot0],  # edge v1-v2, weight from angle at v0
			[v2, v1, 0.5 * cot0],
			[v2, v0, 0.5 * cot1],  # edge v2-v0, weight from angle at v1
			[v0, v2, 0.5 * cot1],
		])

	# Compute all face contributions: (n_faces, 6, 3) -> (n_faces * 6, 3)
	face_contributions = vmap(compute_face_cotangents)(face_indices)
	edge_data = face_contributions.reshape(-1, 3)  # (n_faces * 6, 3)

	# Aggregate weights by edge using scatter_add
	# Build edge weight matrix (n_vertices, max_valence)
	edge_weights = jnp.zeros((n_vertices, max_valence), dtype=jnp.float32)

	def add_edge_weight(carry, data):
		edge_weights = carry
		src, dst, weight = data[0].astype(jnp.int32), data[1].astype(jnp.int32), data[2]

		# Find dst in src's neighbor list and add weight
		matches = (neighbor_indices[src] == dst)
		edge_weights = edge_weights.at[src].add(matches * weight)

		return edge_weights, None

	edge_weights, _ = lax.scan(add_edge_weight, edge_weights, edge_data)

	# Build sparse matrix from edge weights (vectorized)
	n_vertices_range = jnp.arange(n_vertices, dtype=jnp.int32)

	# Diagonal: sum of weights per vertex
	row_diag = n_vertices_range
	col_diag = n_vertices_range
	val_diag = jnp.sum(edge_weights * (neighbor_indices >= 0), axis=1)

	# Off-diagonal: -weight for each neighbor
	rows_offdiag = jnp.repeat(n_vertices_range[:, None], max_valence, axis=1)
	cols_offdiag = neighbor_indices
	vals_offdiag = jnp.where(neighbor_indices >= 0, -edge_weights, 0.0)

	# Flatten and filter
	rows_offdiag_flat = rows_offdiag.reshape(-1)
	cols_offdiag_flat = cols_offdiag.reshape(-1)
	vals_offdiag_flat = vals_offdiag.reshape(-1)

	valid_mask = cols_offdiag_flat >= 0

	# Concatenate
	row_indices = jnp.concatenate([row_diag, rows_offdiag_flat[valid_mask]])
	col_indices = jnp.concatenate([col_diag, cols_offdiag_flat[valid_mask]])
	values = jnp.concatenate([val_diag, vals_offdiag_flat[valid_mask]])

	return row_indices, col_indices, values


@jit
def build_sparse_laplacian(
    edges: jnp.ndarray,            # (nE, 2) int32 — undirected edge list
    edge_weights: jnp.ndarray,     # (nE,)   — affinity weights per edge
    nV: int,
) -> SparseCOO:
    """
    Build the symmetric normalised Laplacian in sparse COO.

    L_sym = I - D^{-1/2} W D^{-1/2}

    where W is the weighted adjacency and D is the degree matrix.

    Non-zeros:
      - Diagonal:     L[v,v] = 1
      - Off-diagonal: L[u,v] = -w(u,v) / sqrt(d(u) * d(v))

    Total nnz = nV + 2*nE  (diagonal + symmetric edges)
    """
    nE = edges.shape[0]
    src = edges[:, 0]                                    # (nE,)
    dst = edges[:, 1]                                    # (nE,)

    # ── Degree: d[v] = Σ_{u~v} w(u,v) ──────────────────────────
    # Scatter-add weights to both endpoints (undirected)
    degree = jnp.zeros(nV, dtype=edge_weights.dtype)
    degree = degree.at[src].add(edge_weights)
    degree = degree.at[dst].add(edge_weights)            # (nV,)

    # ── Inverse sqrt degree ─────────────────────────────────────
    inv_sqrt_deg = 1.0 / jnp.sqrt(jnp.maximum(degree, 1e-10))

    # ── Off-diagonal entries ────────────────────────────────────
    # L[src, dst] = -w * inv_sqrt_deg[src] * inv_sqrt_deg[dst]
    # Both directions for symmetric matrix
    off_diag_vals = -edge_weights * inv_sqrt_deg[src] * inv_sqrt_deg[dst]

    # Forward edges: (src → dst) and reverse: (dst → src)
    off_row = jnp.concatenate([src, dst])                # (2*nE,)
    off_col = jnp.concatenate([dst, src])                # (2*nE,)
    off_val = jnp.concatenate([off_diag_vals, off_diag_vals])  # (2*nE,)

    # ── Diagonal entries ────────────────────────────────────────
    # L[v,v] = 1  (from normalised Laplacian definition)
    diag_idx = jnp.arange(nV, dtype=jnp.int32)
    diag_val = jnp.ones(nV, dtype=edge_weights.dtype)

    # ── Assemble COO ────────────────────────────────────────────
    row_indices = jnp.concatenate([diag_idx, off_row])
    col_indices = jnp.concatenate([diag_idx, off_col])
    values = jnp.concatenate([diag_val, off_val])

    return SparseCOO(
        row_indices=row_indices,
        col_indices=col_indices,
        values=values,
        shape=(nV, nV),
    )


@partial(jit, static_argnames=('n_clusters', 'n_eigenvectors', 'n_lanczos',
                                'kmeans_iter'))
def sparse_spectral_clustering(
    edges: jnp.ndarray,              # (nE, 2) int32
    edge_weights: jnp.ndarray,       # (nE,)
    nV: int,
    n_clusters: int = 16,
    n_eigenvectors: int = 20,
    n_lanczos: int = 100,
    kmeans_iter: int = 100,
    key: jnp.ndarray = None,
) -> dict:
    """
    Full sparse spectral clustering pipeline.

    1. Build sparse normalised Laplacian
    2. Lanczos → smallest eigenvectors
    3. K-Means on spectral embedding

    Complexity: O(nE × n_lanczos) for Lanczos
                O(nV × n_clusters × n_eig × kmeans_iter) for K-Means
    Total:      O(nE × n_lanczos + nV × n_clusters × n_eig)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # ── 1. Sparse Laplacian ─────────────────────────────────────
    L = build_sparse_laplacian(edges, edge_weights, nV)

    # ── 2. Smallest eigenvalues via Lanczos ─────────────────────
    eigvals, eigvecs = sparse_eigsh(
        L,
        n_eigen=n_eigenvectors + 1,      # +1 to include trivial
        n_lanczos=n_lanczos,
        n_reorth=2,
        which='smallest',
        key=key,
    )

    # Skip the trivial eigenvector (constant, eigenvalue ≈ 0)
    embedding = eigvecs[:, 1:n_eigenvectors + 1]                  # (nV, n_eig)

    # ── Row-normalise (Ng-Jordan-Weiss) ─────────────────────────
    row_norms = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
    embedding_norm = embedding / jnp.maximum(row_norms, 1e-10)

    # ── 3. K-Means on embedding ─────────────────────────────────
    key, subkey = jax.random.split(key)
    init_idx = jax.random.choice(subkey, nV, (n_clusters,), replace=False)
    centers = embedding_norm[init_idx]

    def lloyd(centers, _):
        dists = jnp.sum(
            (embedding_norm[:, None, :] - centers[None, :, :]) ** 2,
            axis=-1
        )
        assignments = jnp.argmin(dists, axis=-1)
        one_hot = jax.nn.one_hot(assignments, n_clusters)
        counts = jnp.maximum(one_hot.sum(0, keepdims=True).T, 1.0)
        new_centers = (one_hot.T @ embedding_norm) / counts
        return new_centers, None

    centers, _ = jax.lax.scan(lloyd, centers, None, length=kmeans_iter)

    dists = jnp.sum(
        (embedding_norm[:, None, :] - centers[None, :, :]) ** 2,
        axis=-1
    )
    assignments = jnp.argmin(dists, axis=-1)

    return {
        'assignments': assignments,                   # (nV,)
        'embedding': embedding_norm,                  # (nV, n_eig)
        'laplacian_eigenvalues': eigvals[1:n_eigenvectors + 1],
        'eigengap': jnp.diff(eigvals[1:n_eigenvectors + 1]),
    }