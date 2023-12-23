




from __future__ import annotations
import typing as T

from pathlib import Path

from wplib import log
from wplib.inheritance import clsSuper

from wptree import Tree

from wpm import WN, cmds, om, oma

from ..chimaera import MayaOp


class SkeletonOp(MayaOp):
	"""build some joints.

	- create joints based on patterns in parametres
	- optionally specify each chain's parent in incoming data
	- link transform of each joint to item in node storage
	"""

	@classmethod
	def getDefaultParams(cls, forNode:MayaOp) ->Tree:
		"""create default params.
		Value dicts control parenting -
		parent: path in incomingData
		parentIndex: if tree parent is a curve or chain
		"""
		return Tree("C_root", v={"parent" : None})


	def compute(self, inputData:Tree
	            ) ->Tree:
		"""iterate over all params branches,
		make nodes,
		link to storage"""

		for branch in self.params().allBranches(
			includeSelf=False,
			depthFirst=True,
			topDown=True
		):

			joint = WN.create("joint", n=branch.name)
			branch.scratch["joint"] = joint

			# use "parent" key if defined, else the parent branch
			if branch.v.get("parent"):
				log("looking up parent {} in inputData".format(branch.v["parent"]))
				jointParent = inputData(branch.v["parent"]).v

			else:
				jointParent = branch.parent.scratch.get("joint",
				                                        self.rigGrp()
				                                        )
			# parent joint to parent
			joint.parentTo(jointParent)

		# link top-level branches to storage
		for i in self.params().branches:
			# get joint from scratch
			joint = i.scratch["joint"]
			# link to storage
			self.storage(i)

		# link to storage
		self.storage(branch.address())







