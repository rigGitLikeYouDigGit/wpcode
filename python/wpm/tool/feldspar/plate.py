
"""a single rigid piece of a linkage"""
from __future__ import annotations
import typing as T
import scipy
import numpy as np
from sklearn.preprocessing import normalize
from dataclasses import dataclass

from tree.lib.uidelement import UidElement

from edRig.maya.constant import vectorType
from edRig.maya.tool.feldspar.lib.matrix import multMatrixVectorArray, averageMatrixFromVectors, translationRotationFromPointSets, atLeast3PlatePoints, rotMatrixAligningVectors, jitterValues
from edRig.maya.tool.feldspar.vertex import Vertex

if T.TYPE_CHECKING:
	from edRig.maya.tool.feldspar.assembly import Assembly

@dataclass
class PlateParamData:
	fixed : bool
	mass : float = 1.0
	rigidPriority : int = 1
	collisionMesh = None



class Plate(UidElement):
	def __init__(self,
	             matrix=np.identity(4),
	             vertexPositions:T.List[vectorType]=None ,
	             fixed=False
	             ):
		"""for now using single matrix, might
		be beneficial to split to position and orientation

		points are single vector positions in relative space to matrix
		they will be multiplied out from matrix to constraint positions,
		while keeping matrix orientation

		if fixed, no solving forces will act
		(plate can still move by user input, acting as goal)

		"""
		super(Plate, self).__init__()
		self.graph : Assembly = None
		self.params : PlateParamData = None
		self.matrix = matrix
		self.restMatrix = np.array(matrix)
		vertexPositions = vertexPositions or []
		self.vertices = [Vertex(self, i, pos) for i, pos in enumerate(vertexPositions)]

		self._solveFn = None

	def isValid(self):
		"""check that plate has at least 2 vertices"""
		return len(self.vertices) > 1 or self.params.fixed

	def preSolve(self):
		"""run a valid check, then set the correct solve function"""
		assert self.isValid()
		if self.params.fixed:
			return

		# special-case 2 vertex plates
		if len(self.vertices) == 2:
			self._solveFn = self.solveRod
		else:
			self._solveFn = self.solveRigidPlate

	def solve(self):
		if self.params.fixed:
			return
		basePositions = np.array([v.pos for v in self.vertices])
		translators = np.array([v.translator for v in self.vertices])
		globalPositions = np.array([v.globalPos() for v in self.vertices])

		basePositions = jitterValues(basePositions)
		translators = jitterValues(translators)
		globalPositions = jitterValues(globalPositions)

		if len(self.vertices) == 2:
			self.solveRod(basePositions, globalPositions, translators)
		else:
			self.solveRigidPlate(basePositions,
			                     globalPositions,
			                     translators,
			                     )



	def solveRod(self, basePositions, globalPositions,
	             translators):
		"""get matrix that aims at v2 in v1-space, apply it,
		done"""
		#print("solve rod")
		#print(" ")

		# sumPositions = basePositions + translators
		sumPositions = globalPositions + translators
		# baseAim = basePositions[1] - basePositions[0]
		baseAim = globalPositions[1] - globalPositions[0]
		relativeAim = sumPositions[1] - sumPositions[0]

		# print(baseAim)
		# print(relativeAim)

		mat = rotMatrixAligningVectors(baseAim, relativeAim)
		# print("rot")
		# print(mat)


		#baseMeanPos = np.sum(basePositions, axis=0) / 2
		targetMeanPos = np.mean(translators, axis=0)

		#print("target mean", targetMeanPos)

		self.matrix[:3, :3] = normalize(np.matmul(
			self.matrix[:3, :3],
			np.linalg.inv(mat),

		))

		# self.matrix[:3, :3] = np.linalg.inv(mat)
		self.matrix[3] = self.matrix[3] + targetMeanPos
		self.matrix[3, 3] = 1.0



		pass

	def solveRigidPlate(self,
	                      basePositions,
	                    globalPositions,
	                      translators,

	                    ):
		"""check over all connected vertices,
		align to their deltas

		special case algorithms for 1- and 2-vertex plates
		"""
		# print("")
		# print(globalPositions)
		# print(globalPositions + translators)
		translation, rotation = translationRotationFromPointSets(
			globalPositions, globalPositions + translators
			#basePositions, basePositions + translators
		)
		# print("tR", translation)
		# print(rotation)


		newOrient = normalize(np.matmul(
			self.matrix[:3, :3],
			#np.linalg.inv(self.matrix[:3, :3]),
			#rotation,
			np.linalg.inv(rotation),
			# self.matrix[:3, :3]
		))
		translation = np.mean(translators, axis=0)

		# self.matrix[3] = [*translation, 0.0] + self.matrix[3]
		self.matrix[:3, :3] = newOrient
		self.matrix[3] = translation + self.matrix[3]
		self.matrix[3, 3] = 1.0



	def rotateByRotators(self):
		pass


	def __str__(self):
		return """<Plate {} - vtxes {}>""".format(self.uidRepr(), self.vertices)

	def __repr__(self):
		return str(self)



