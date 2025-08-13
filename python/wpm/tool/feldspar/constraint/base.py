
from __future__ import annotations
import typing as T
from dataclasses import dataclass
import numpy as np

from tree.lib.uidelement import UidElement
from edRig.maya.tool.feldspar.plate import Plate
from edRig.maya.tool.feldspar.vertex import Vertex

"""for ease of implementation, use a single parametre class among all 
constraint types - if a new field is needed, add it here as an optional
attribute"""

@dataclass
class ConstraintParams:
	weight : float = 1.0 # crude version of constraint "strength"
	localIterations : int = 1 # how many constraint iterations per full iteration
	soft : bool = False
	neutral : (float, np.ndarray) = None
	restoreForce : (float, np.ndarray) = None
	axis : np.ndarray = None
	limits = None
	weld = True


class ConstraintBase(UidElement):
	"""base class for linkage constraints
	each constraint links to at least one plate -
	each plate connection defines multiple vertices to consider

	plateList looks like
	[ ( Plate, ( plate vertices for this constraint ) ) ]

	during solve, return a map of { vertex : (translator, rotator) }
	for each vertex - these will be combined later

	preparation for parallelisation

	"""

	paramCls = ConstraintParams

	plateListType : list[tuple[Plate, tuple[Vertex]]]

	def __init__(self):
		super(ConstraintBase, self).__init__()
		self.plateList : ConstraintBase.plateListType = []
		self.params = self.paramCls() # set as needed

	# def activeIterations(self, maxIterations:int):
	# 	"""given number of max iterations, return number of iterations
	# 	that this constraint should be active"""
	# 	return int(self.params.weight * maxIterations)

	@property
	def plates(self):
		return [i[0] for i in self.plateList]

	def iterVertices(self):
		for plateTie in self.plateList:
			for vertex in plateTie[1]:
				yield vertex

	def vertices(self)->tuple[Vertex]:
		return tuple(self.iterVertices())


	def setupNeutral(self):
		"""run this when constraint is initialised in neutral position
		in graph - override to gather any static offsets, rest lengths etc"""


	def solve(self,
	          globalIteration:int,
	          maxGlobalIterations:int,
	          localIteration:int=1,
	          maxLocalIterations:int=1):
		"""method to solve this constraint against its connected plates
		not sure yet how to handle constraint types - for now assume all
		are simple weld constraints

		soft constraints simply deal with current vertex position
		and add force -
		hard constraints deal with current position + would-be result of force,
		and add counteracting force as needed

		"""


		allVertices = self.vertices()

		if self.params.soft:
			#iterWeight = 0.5 * (globalIteration + 1) / maxGlobalIterations
			iterWeight = 0.1
		else:
			#iterWeight = 2.0 - ((localIteration + 1) / maxLocalIterations)
			iterWeight = 1.0

		#iterWeight = 0.1
		#iterWeight = 1.0

		# check rigid priorities
		rigidSum = sum(i.params.rigidPriority for i in self.plates) or 1
		rigidMax = max(i.params.rigidPriority for i in self.plates)

		for vertex in allVertices:
			globalTargetPos = np.average([i.globalPos() for i in self.iterVertices() if i is not vertex], axis=0)
			vtxWeight = float(vertex.plate.params.rigidPriority) / rigidSum
			delta = globalTargetPos - vertex.globalPos()
			delta = delta * iterWeight# * vtxWeight

			vertex.translator = vertex.translator + delta





