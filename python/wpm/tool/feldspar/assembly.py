
from __future__ import annotations

import sklearn.preprocessing

from sklearn.preprocessing import normalize
"""main class for a mechanical assembly"""

import itertools
import typing as T
import networkx as nx
import numpy as np

np.set_printoptions(precision=3, linewidth=3000, suppress=True)

from edRig.maya.tool.feldspar import datastruct
from edRig.maya.tool.feldspar.lib import matrix

class Assembly(nx.Graph):
	"""holds collection of plates, points and constraints"""

	def __init__(self, incomingData=None, **attrs):
		super(Assembly, self).__init__(incomingData, **attrs)

		self.params : datastruct.AssemblyParams = None

		self.data : datastruct.FeldsparData = None
		self.constraintMatrix : np.ndarray = None

	def setData(self, data:datastruct.FeldsparData):
		"""update the data struct, regenerate the constraint matrix"""
		self.data = data
		self.data.recache()
		self.buildConstraintMatrix()

	# def pinMatrix(self):
	# 	"""return consistent matrix rows"""

	@property
	def nPoints(self):
		return len(self.data.positions)

	def buildConstraintMatrix(self):
		"""not worth complexity to cache vertex pin rows
		cannibalised from Harold Cooper
		test locking z axis of all points

		"""
		# build constraint matrix
		#rows = len(self.data.fixedVertexArray) * 3 + len(self.data.rigidBarDatas())
		rows = 0
		columns = self.nPoints * 3

		rigidMat = np.zeros((rows, columns))
		row = 0

		"""build rows for fixed vertices"""
		for index in range(len(self.data.positions)):
			if index in self.data.fixedVertexSet:
				fixLines = np.zeros((3, columns))
				fixLines[0, 3 * index] = 1.0
				fixLines[1, 3 * index + 1] = 1.0
				fixLines[2, 3 * index + 2] = 1.0
				rigidMat = np.append(rigidMat, fixLines, axis=0)
				#row += 3
		#for fixedVertexId in self.data.fixedVertexArray:
			else:
				# fix x axis
				fixLines = np.zeros((1, columns))
				fixLines[0, index * 3] = 1.0
				rigidMat = np.append(rigidMat, fixLines, axis=0)

		"""build rows for rigid bars"""
		for barData in self.data.rigidBarDatas():
			ids = barData.indices
			fixLines = np.zeros((1, columns))
			posA, posB = self.data.positions[ids[0]], self.data.positions[ids[1]]
			posDelta = (posA - posB)# / np.linalg.norm(posA - posB)

			fixLines[0, 3 * ids[0] : 3 * ids[0] + 3] = -posDelta
			fixLines[0, 3 * ids[1] : 3 * ids[1] + 3] = posDelta
			rigidMat = np.append(rigidMat, fixLines, axis=0)

		#row += 1

		return rigidMat


	def applySoftBars(self):
		"""add soft bars to velocity at the start of the iteration"""
		for i in self.data.rigidBarDatas(state=False):
			positionA, positionB = self.data.positions[i.indices[0]], self.data.positions[i.indices[1]]
			vel = positionB - positionA
			self.data.velocities[i.indices[0]] += vel * 0.1
			self.data.velocities[i.indices[1]] += -vel * 0.1

			# midpt = (positionB - positionA) / 2
			# self.data.velocities[:] += (midpt - self.data.positions[:]) * 0.1
			pass

	#def applyVelocities

	def gatherBarVelocities(self,
	                        livePositions:np.ndarray,
	                        barTies:np.ndarray,
	                        barLengths:np.ndarray # [ length, bindLength, targetLength ]
	                        )->np.ndarray:
		"""gather offsets in point positions necessary to stop gradual
		stretching of bars"""
		#print("barties", barTies)
		#print("barlengths", barLengths)
		newVelocities = np.zeros_like(livePositions)
		vectorTies = livePositions[barTies]
		#print("vec ties", vectorTies)
		# print(vectorTies)
		barDeltas = np.diff(vectorTies, axis=1).reshape(len(barTies), 3)
		#print("deltas", barDeltas)
		lengths = np.linalg.norm(barDeltas, axis=1).reshape(len(barTies))
		#print("lengths", lengths)
		scaleFactors = 1.0 - np.divide(barLengths[:, 1], lengths)
		#print("scaleFactors", scaleFactors)
		scaleDeltas = barDeltas * scaleFactors.reshape(len(barTies), 1)
		newVelocities[barTies[:, 0]] += scaleDeltas[:]
		newVelocities[barTies[:, 1]] -= scaleDeltas[:]

		#print("newVelocities", newVelocities)

		# average
		flatTies = barTies.reshape(len(barTies) * 2)
		vertexIndices, counts = np.unique(flatTies, return_counts=True)
		#print("vertexIndices", vertexIndices, "counts", counts)
		newVelocities[vertexIndices] = newVelocities[vertexIndices] / counts.reshape(len(counts), 1)

		#print("end velocities", newVelocities)
		return newVelocities

	def iterate(self,
	            maxIterations:int,
	            currentIteration:int):
		self.data.velocities.fill(0.0)
		self.applySoftBars()
		self.data.velocities[self.data.fixedVertexArray] = np.zeros(3)

		#self.data.positions += self.data.velocities
		barTies, barLengths = self.data.rigidBarArrays(state=True)


		# for iteration in range(10):
		# 	barVels = self.gatherBarVelocities(self.data.positions,
		# 	                                   barTies,
		# 	                                   #self.data.rigidBarDatas(True),
		# 	                                   barLengths
		# 	                                   )
		# 	barVels[self.data.fixedVertexArray] = np.zeros(3)
		# 	self.data.positions = self.data.positions + barVels
		# return



		rigidMat = self.buildConstraintMatrix()
		# print("rigid")
		# print(rigidMat)
		nullBasis = matrix.nullspace(rigidMat)
		for iteration in range( 1
				#self.params.constraintIterations
		):
			self.data.velocities[self.data.fixedVertexArray] = np.zeros(3)

			nullVels = np.array([i.reshape((self.nPoints, 3)) for i in nullBasis])

			"""
			returns a variable number of lists of vectors, depending on
			degrees of freedom of the system -
			each one of these sets is a POTENTIAL, VALID set - the vectors contained
			in it are guaranteed not to violate any of the constraints on the system.
			
			With each, any unfiltered goal velocities are dotted with these options to
			get the COMPONENTS of those goal velocities that lie inline with the valid
			constraint vectors.
			
			Final, filtered velocities are the sum of each of these component filters,
			passed through each list of vectors in the nullspace.
			"""

			# print("reshape")
			# print(nullVels)
			#
			# print("prevels", self.data.velocities)
			#print("pre pos", self.data.positions)

			#baseVels = self.data.velocities.copy()
			#newVels = np.zeros_like(self.data.velocities)

			v0 = self.data.velocities[-2]

			for i, nullVelList in enumerate(nullVels):
				newVels = np.zeros_like(self.data.velocities)

				v = nullVelList[-2]
				c = np.dot(v0, v) / np.dot(v, v)

				v0 = v0 - v * c

				for n in range(self.nPoints):
					self.data.positions[n] = self.data.positions[n] + nullVelList[n] * c# * 0.3

				# print("nullvelList")
				# print(nullVelList)

				# nullVelList = np.array([n / max(np.linalg.norm(n), 0.00001) for n in nullVelList])

				# dotVels = (nullVelList * self.data.velocities).sum(1)
				# #print("dotVels", dotVels)
				# prod = nullVelList * np.reshape(dotVels, (len(dotVels), 1))
				#
				# # print("prod", prod)
				# newVels += prod# * 0.1
				#
				# self.data.velocities -= prod
				#
				# # print("newVels", newVels)
				#
				# self.data.positions += newVels



			# reset fixed points

		# 	#self.data.velocities = newVels
		# 	newVels[self.data.fixedVertexArray] = np.zeros(3)
		# 	self.data.velocities = newVels
		# self.data.positions += self.data.velocities


			# # gather forces to restore bar lengths
			# barVels = self.gatherBarVelocities(self.data.positions,
			#                                    barTies,
			#                                    #self.data.rigidBarDatas(True),
			#                                    barLengths)
			# # barVels[self.data.fixedVertexArray] = np.zeros(3)
			# # self.data.positions = self.data.positions + barVels
			#
			# #newVels += barVels
			# newVels[self.data.fixedVertexArray] = np.zeros(3)
			# barVels[self.data.fixedVertexArray] = np.zeros(3)
			#
			#
			# self.data.velocities = newVels
			# self.data.positions = self.data.positions + barVels

		#rint("postvels", self.data.velocities)
		# self.data.positions = self.data.positions + self.data.velocities





		pass

	def runIteration(self):
		for i in range(#self.params.iterations
		               3):
			self.iterate(maxIterations=self.params.iterations,
			             currentIteration=i)

		pass


