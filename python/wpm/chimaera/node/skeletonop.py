
from __future__ import annotations
import typing as T

from dataclasses import dataclass

import numpy as np


#from wplib import Expression

from wptree import Tree

from chimaera import ChimaeraNode
from wpm import cmds, om, oma, WN

"""
coming back to this with a proper system backing Chimaera,
don't worry about doing fancy stuff with plugs,
just match the output hierarchy exactly
"""



class SkeletonOp(ChimaeraNode):
	"""Node for creating joint hierarchies.
	Create joints in place from saved data view
	- set up template inputs
	- set up template outputs
	"""

	@classmethod
	def defaultSettings(cls, forNode:ChimaeraNode) ->Tree:
		"""TODO: set up cls-super
		"""
		t = Tree("root")
		t("joints").description = "Branches below this will be literally created as joints"
		t["joints", "a", "b", "c"]


	@classmethod
	def templateFlowOut(self) ->Tree:
		pass


	#@classmethod
	def compute(self, **kwargs) ->None:
		"""get parent data,
		generate new hierarchy data for joints including transforms
		add to parent,
		return new data"""
		settings = self.settings()

		### what in god's country was past ed cooking
		#parentData = self.plugChildMap(self.ownInputPlugRoot())["parent"].resultData() # ????




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









