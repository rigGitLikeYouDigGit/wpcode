from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import numpy as np
from scipy import sparse, spatial

"""pure np array-based mesh rep
"""


def triangulatePolygonMesh(faceVertexLengths, faceVertexConnects):
	faceLengths = np.asarray(faceVertexLengths)
	faceConnects = np.asarray(faceVertexConnects)


	# Each polygon of length V yields V - 2 triangles.
	# We clamp at 0 to gracefully handle points/lines
	numTrianglesPerFace = np.maximum(0, faceLengths - 2)
	numFaces = len(faceLengths)
	nTriangles = np.sum(numTrianglesPerFace)

	# origFaceIndices maps every new triangle back to its parent polygon index
	origFaceIndices = np.repeat(np.arange(numFaces), numTrianglesPerFace)

	# faceOffsets represents the starting index of each polygon in the flat faceConnects array
	faceOffsets = np.zeros(numFaces, dtype=int)
	if numFaces > 1:
		faceOffsets[1:] = np.cumsum(faceLengths[:-1])

	# triOffsets represents the starting index of each polygon's triangles in our output
	triOffsets = np.zeros(numFaces, dtype=int)
	if numFaces > 1:
		triOffsets[1:] = np.cumsum(numTrianglesPerFace[:-1])

	# For each triangle, calculate its local index within its parent polygon (e.g., 0, 1, 2...)
	localTriIdx = np.arange(nTriangles) - triOffsets[origFaceIndices]

	# Map to the specific vertices for a simple triangle fan:
	# Triangle i in a polygon connects vertex 0, vertex i+1, and vertex i+2
	baseIdx = faceOffsets[origFaceIndices]

	v0Idx = baseIdx
	v1Idx = baseIdx + localTriIdx + 1
	v2Idx = baseIdx + localTriIdx + 2

	# Extract the actual vertex indices from the flat array
	v0 = faceConnects[v0Idx]
	v1 = faceConnects[v1Idx]
	v2 = faceConnects[v2Idx]

	# Form the final [nTriangles, 3] array
	triangles = np.column_stack((v0, v1, v2))
	origFaceIndices = origFaceIndices.ravel()
	return triangles, origFaceIndices

def getEdgePointConnects(faceVertices:np.ndarray[nfaces, 3]):
	# Extract the three edges from each triangle
	edgeOne = faceVertices[:, [0, 1]]
	edgeTwo = faceVertices[:, [1, 2]]
	edgeThree = faceVertices[:, [2, 0]]

	# Combine them into a (3N, 2) array
	allEdges = np.vstack((edgeOne, edgeTwo, edgeThree))

	# Sort each edge's vertices so that (v1, v2) and (v2, v1) are treated as identical
	sortedEdges = np.sort(allEdges, axis=1)

	# Remove duplicate edges to get an (NEdges, 2) array
	uniqueEdges = np.unique(sortedEdges, axis=0)
	return uniqueEdges

class WMesh:
	""" don't know the best syntax for attributes -
	also just storing full dense buffers for topology

	"""

	def __init__(self):

		self.a = {
			"p" : {"P" : np.zeros((0, 3))},
			"e" : (),
			"f" : {},
			"v" : {},
			"g" : {}
		}

		# group arrays still index based; if bool, need to
		# resize all of them on any submesh added
		self.g = {
			"p" : {},
			"e" : {},
			"f" : {},
			"v" : {}
		}
		self.fv = np.zeros((0, 3), dtype=int)

	@property
	def pA(self):
		return self.a["p"]
	@property
	def eA(self):
		return self.a["e"]
	@property
	def fA(self):
		return self.a["f"]
	@property
	def vA(self):
		return self.a["v"]
	@property
	def gA(self):
		return self.a["g"]
	@property
	def pG(self):
		return self.g["p"]
	@property
	def eG(self):
		return self.g["e"]
	@property
	def fG(self):
		return self.g["f"]
	@property
	def vG(self):
		return self.g["v"]

	def extendAttrs(self,
	                nFaces,
	                nPoints,
	                nEdges,
	                nVertices,
	                ):
		"""extend dense attr buffers when new meshes added
		TODO: defaults"""
		for k, v in tuple(self.fA.items()):
			self.fA[k] = np.concatenate(
				(v, np.zeros((nFaces, v.shape[1]))), axis=0)
		for k, v in tuple(self.pA.items()):
			self.pA[k] = np.concatenate(
				(v, np.zeros((nPoints, v.shape[1]))), axis=0)
		for k, v in tuple(self.eA.items()):
			self.eA[k] = np.concatenate(
				(v, np.zeros((nEdges, v.shape[1]))),
			                            axis=0)
		for k, v in tuple(self.vA.items()):
			self.vA[k] = np.concatenate(
				(v, np.zeros((nVertices, v.shape[1]))), axis=0
			)

	def getA(self, elType:str, name:str):
		"""get attr array for the given element type and attr name
		"""
		attrDict = {
			"point": self.pA,
			"edge": self.eA,
			"face": self.fA,
			"vertex": self.vA,
			"group": self.gA,
		}
		return attrDict[elType][name]

	def addSubTriMesh(self,
	               faceVertexConnects:np.ndarray[int],
	               posArr:np.ndarray[float],
	                  name="",
	               ):
		"""expect fvc in range (0, nvertices in submesh)
		"""
		nNewFaces = faceVertexConnects.shape[0]
		nNewPoints = np.max(posArr) + 1
		nNewVertices = faceVertexConnects.shape[0] * faceVertexConnects.shape[1]
		edgePointConnects = getEdgePointConnects(faceVertexConnects)
		nNewEdges = edgePointConnects.shape[0]

		self.extendAttrs(nNewFaces, nNewPoints, nNewEdges, nNewVertices)

		faceStartIndex = self.fv.shape[0]
		faceEndIndex = faceStartIndex + nNewFaces
		ptStartIndex = np.max(self.fv) + 1
		ptEndIndex = ptStartIndex + nNewPoints

		self.fv = np.concatenate(
			(self.fv, faceVertexConnects + ptStartIndex), axis=0)
		self.pA["P"][ptStartIndex : ptEndIndex] = posArr
		self.fG[name] = np.arange(faceStartIndex, faceEndIndex)
		self.pG[name] = np.arange(ptStartIndex, ptEndIndex)

	def addSubMesh(self,
	               faceVertexLengths,
	               faceVertexConnects,
		               posArr:np.ndarray[float],
	               name="",
	               ):
		triangles, origFaceIndices = triangulatePolygonMesh(
			faceVertexLengths, faceVertexConnects
		)
		self.addSubTriMesh(
			triangles,
			posArr,
			name=name
		)
