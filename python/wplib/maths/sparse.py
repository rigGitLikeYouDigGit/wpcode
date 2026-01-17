from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import numpy as np
from scipy.sparse import dok_array, csr_array, coo_array, lil_array


def lilFromIndexArrs(
		indexArr:np.ndarray,
		valueArr:np.ndarray,
)->lil_array:
	"""builds sparse array - convert it on receipt to
	the format best for your case"""
	result = lil_array((indexArr.shape[0], indexArr.max()+1))
	for row, vs in zip(indexArr, valueArr):
		result.rows[row] = vs
	return result

def minShapeForSparse(
		sparseArr:csr_array
)->tuple[int, int]:
	"""returns minimum shape for sparse array"""
	nEntries = 0
	for i in range(sparseArr.shape[0]):
		nEntries = max(nEntries, len(sparseArr[i].nonzero()[0]))
	return sparseArr.shape[0], nEntries

def indexArrsFromSparse(
		sparseArr:csr_array,
		indexArr:np.ndarray=None,
		valueArr:np.ndarray=None,
)->tuple[np.ndarray, np.ndarray]:
	"""returns index and value arrays from sparse array"""
	if indexArr is None:
		indexArr = np.zeros(minShapeForSparse(sparseArr), dtype=np.int32)
	if valueArr is None:
		valueArr = np.zeros(minShapeForSparse(sparseArr),
		                     dtype=sparseArr.dtype)
	assert indexArr.shape == valueArr.shape

	# need to iterate in loop since nonzero returns different lengths per row
	for i in range(sparseArr.shape[0]):
		nz = sparseArr[i].nonzero()[0]
		indexArr[i, :len(nz)] = nz
		valueArr[i, :len(nz)] = sparseArr[i].data

	return indexArr, valueArr

if __name__ == '__main__':

	dense = np.eye(6)
	sparse = csr_array(dense)
	print(indexArrsFromSparse(sparse))

	sparse[3, 4] = 7.0
	print(indexArrsFromSparse(sparse))
