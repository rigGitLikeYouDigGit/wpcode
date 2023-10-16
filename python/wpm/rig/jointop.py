
from __future__ import annotations
import typing as T

from wptree import Tree

from wpm import cmds, om, WN

"""make some joints and then worry about the rest
outputs should be reference to joints -
we can extract local and world if needed.

but should it be valid to output a plug from a rig op? probably - 


"""

class GraphPlugData:

	def __init__(self,
	             node=None,
	             main=None,
	             local=None,
	             world=None,
	             ):
		"""main is any straight float plugs -
		local and world are for matrix plugs?
		"""


class JointOp:


	def __init__(self):
		self.params = Tree("params", lookupCreate=True)
		self.params["count"] = 3

		self.outputs = Tree("out", lookupCreate=True)
		self.outputs("joints")


	def build(self):

		self.outputs.clear()

		baseJoint = None
		currentParent = None
		for i in range(self.params["count"]):
			joint = WN.create("joint", n="op_joint{}".format(i))
			if baseJoint is None:
				baseJoint = joint
			if currentParent is not None:
				joint.setParent(currentParent)
			else:
				currentParent = joint
			self.outputs[str(i)] = joint





		pass





