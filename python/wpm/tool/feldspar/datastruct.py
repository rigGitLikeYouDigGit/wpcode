
from __future__ import annotations

"""test for data class holding feldspar arrays and groups
"""

from dataclasses import dataclass
from collections import namedtuple

import numpy as np

from edRig.maya.tool.feldspar.lib import matrix as libmat

@dataclass
class BarData:
	indices :tuple[int, int]
	length: float
	soft : bool = False
	bindLength : float = 1.0
	targetLength : float = 1.0


#BarData = namedtuple("BarData", ("indices", "soft", "bindLength", "targetLength", "length"))

@dataclass
class GroupData:
	vertices : np.ndarray
	fixed : False
	#matrix : np.ndarray

	def bindMatrix(self, fpData:FeldsparData)->np.ndarray:
		return libmat.averageMatrixFromVectors(fpData.basePositions[self.vertices])

	def liveMatrix(self, fpData:FeldsparData)->np.ndarray:
		return libmat.translationRotationFromPointSets(
			fpData.basePositions[self.vertices],
			fpData.positions[self.vertices]
		)



@dataclass
class FeldsparData:

	basePositions : np.ndarray
	positions : np.ndarray
	velocities : np.ndarray

	barDatas : list[BarData]
	groupDatas : list[GroupData]

	def __post_init__(self):
		self.needsRecache = True
		self.fixedVertexArray : np.ndarray = None
		self.fixedVertexSet : set[int] = set()
		self.barTies : np.ndarray = None
		self.barLengths : np.ndarray = None # nx3 array with entries [length, bindLength, targetLength]
		# self.softBarTies: np.ndarray = None
		# self.softBarLengths: np.ndarray = None  # nx3 array with entries [length, bindLength, targetLength]
		# self.rigidBarTies: np.ndarray = None
		# self.rigidBarLengths: np.ndarray = None  # nx3 array with entries [length, bindLength, targetLength]
		self.recacheIfNeeded()

	def recacheIfNeeded(self):
		if not self.needsRecache:
			return
		self.recache()

	def recache(self):
		"""regenerate any cached values from applied data"""
		fixed = set()
		print(self.groupDatas)
		for i in self.groupDatas:
			if i.fixed:
				fixed.update(i.vertices)
		self.fixedVertexSet = fixed
		self.fixedVertexArray = np.array(sorted(fixed), dtype=int)
		self.barTies = np.array([i.indices for i in self.barDatas], dtype=int)
		self.barLengths = np.array([[i.length, i.bindLength, i.targetLength] for i in self.barDatas])

		self.needsRecache = False

	def fixedVertexIndices(self, state=True):
		return np.array([i for i in range(len(self.fixedVertexArray))])

	def rigidBarDatas(self, state=True):
		return [i for i in self.barDatas if i.soft != state]

	def rigidBarArrays(self, state=True)->tuple[np.ndarray, np.ndarray]:
		return (
			np.array([i.indices for i in self.rigidBarDatas(state)], dtype=int),
			np.array([[i.length, i.bindLength, i.targetLength] for i in self.rigidBarDatas(state)])
		)


@dataclass
class AssemblyParams:
	iterations : int
	constraintIterations : int = 3
