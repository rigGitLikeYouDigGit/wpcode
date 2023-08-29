
from __future__ import annotations
import typing as T

from dataclasses import dataclass

import numpy as np


from wplib import Expression

from wptree import Tree

from chimaera import ChimaeraNode, PlugNode

"""
Consider a tree
branches form a spine
values form rich data

groupA : transform
	groupB : transform
		meshA : Mesh 
			topology : buffer
			positions : buffer
			
so do we actually need more than one "real" output plug?
if the real plug just represents the outgoing data stream, then we 
have various filter plugs to separate out substreams from it?

can we / how can we embed expressions in data?
not for now
"""

@dataclass
class Transform:
	"""data object for transform
	TEMP TEMP TEMP just a sketch for tree value
	"""
	matrix : np.ndarray

class SkeletonOp(PlugNode):
	"""Node for creating joint hierarchies.
	Create joints in place from saved data view
	"""

	@classmethod
	def defaultParams(cls, paramRoot:Tree) ->Tree:
		jointRoot = paramRoot("joints", create=True)
		jointRoot.desc = "Branches below this will each correspond to a joint."
		return paramRoot

	@classmethod
	def makePlugs(cls, inRoot:Tree, outRoot:Tree)->None:
		"""create plugs for this node
		Output root will be set with data object, to be filtered if needed"""
		inRoot("parent", create=True)

		outRoot("joints", create=True)

	@classmethod
	def makeDataFromJointTree(cls, jointTree:Tree):
		"""create data from joint tree -
		skip root"""
		for branch in jointTree.allBranches(includeSelf=False):
			data = Transform(np.identity(4))
			branch.value = data
		return jointTree

	#@classmethod
	def compute(self, **kwargs) ->None:
		"""get parent data,
		generate new hierarchy data for joints including transforms
		add to parent,
		return new data"""
		parentData = self.plugChildMap(self.ownInputPlugRoot())["parent"].resultData()

		# load existing data from storage,
		existData = self.resultStorage()("joints", create=True)
		# create any new data from params
		newData = self.makeDataFromJointTree(self.sourceParams()("joints"))

		# update with any existing entries
		# don't create new entries if none are found from storage,
		mergeData = combineDataTrees(newData, existData, mode="intersect")

		# then update storage again, preserve any unknown entries in case of mistype?
		combineDataTrees(self.sourceStorage()("joints", create=True),
		                 mergeData, mode="union")

		# editing in maya is then done by creating a data view on
		# the STORAGE tree

		# update output data
		self.plugChildMap(self.ownOutputPlugRoot())["joints"].setValue(mergeData)




if __name__ == '__main__':
	graph = ChimaeraNode("graph")
	skOp = SkeletonOp.create("skelOp", graph)
	print(skOp.ownInputPlugRoot())

	print(skOp.node.value())


	jointRoot = skOp.sourceParams()("joints")
	skOp.sourceParams().lookupCreate = True
	print(jointRoot)
	jointRoot("joint1")
	jointRoot("joint1", "joint2")
	jointRoot("joint1", "joint2", "joint3")

	#result = skOp.resultParams()
	#print(result)
	#print(result.serialise())

	skOp.resultParams().display()









