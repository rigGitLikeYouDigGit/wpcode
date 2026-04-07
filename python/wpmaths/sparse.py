from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import numpy as np
from scipy.sparse import csr_array


def csr_to_skin_arrays(csr_mat: csr_array, max_influences: int):
	"""
	Converts a sparse CSR matrix of skin weights into dense arrays.

	Parameters:
		csr_mat: scipy.sparse.csr_array of shape (num_vertices, num_bones)
		max_influences: int, the maximum number of bones per vertex (e.g., 4 or 8)

	Returns:
		bone_indices: np.ndarray of shape (num_vertices, max_influences)
		bone_weights: np.ndarray of shape (num_vertices, max_influences)
	"""
	num_vertices = csr_mat.shape[0]

	# Initialize target arrays with zeros (padding)
	bone_indices = np.zeros((num_vertices, max_influences),
	                        dtype=csr_mat.indices.dtype)
	bone_weights = np.zeros((num_vertices, max_influences), dtype=csr_mat.dtype)

	# Get the number of influences per vertex
	counts = np.diff(csr_mat.indptr)

	# Create an array that maps every non-zero element back to its row (vertex) index
	row_idx = np.repeat(np.arange(num_vertices), counts)

	# Calculate the local influence index (0, 1, 2, ...) for each element within its row
	# e.g., if a row has 3 elements, they will be assigned 0, 1, 2.
	col_idx = np.arange(len(csr_mat.data)) - csr_mat.indptr[row_idx]

	# In case the sparse matrix has a vertex with MORE than max_influences,
	# we create a mask to safely truncate/ignore the overflow.
	valid_mask = col_idx < max_influences

	# Filter the indices based on the mask
	row_idx = row_idx[valid_mask]
	col_idx = col_idx[valid_mask]

	# Scatter the sparse data into the dense arrays
	bone_indices[row_idx, col_idx] = csr_mat.indices[valid_mask]
	bone_weights[row_idx, col_idx] = csr_mat.data[valid_mask]

	return bone_indices, bone_weights


def skin_arrays_to_csr(bone_indices: np.ndarray, bone_weights: np.ndarray,
                       num_bones: int) -> csr_array:
	"""
	Converts dense index and weight arrays back into a sparse CSR array.

	Parameters:
		bone_indices: np.ndarray of shape (num_vertices, max_influences)
		bone_weights: np.ndarray of shape (num_vertices, max_influences)
		num_bones: int, the total number of bones in the armature

	Returns:
		csr_mat: scipy.sparse.csr_array of shape (num_vertices, num_bones)
	"""
	num_vertices = bone_weights.shape[0]

	# Create a boolean mask of valid influences.
	# (Checking for != 0 filters out all the padding we added earlier)
	active_mask = bone_weights != 0

	# Extract the flat arrays of non-zero data
	data = bone_weights[active_mask]
	cols = bone_indices[active_mask]

	# np.nonzero on a 2D array returns a tuple of (row_indices, col_indices).
	# We only need the row (vertex) indices.
	rows = np.nonzero(active_mask)[0]

	# Construct the CSR matrix. SciPy automatically handles
	# the internal indptr construction and sorts the rows if necessary.
	csr_mat = csr_array((data, (rows, cols)), shape=(num_vertices, num_bones))

	return csr_mat



