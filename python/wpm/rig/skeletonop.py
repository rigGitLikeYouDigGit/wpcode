
from __future__ import annotations
import typing as T

from dataclasses import dataclass

import numpy as np


from wplib import Expression

from wptree import Tree

from chimaera import ChimaeraNode, PlugNode


class DataBlock:
	"""test sketch of retrieving data
	and operating on it in a separable way"""
	params : Tree
	data : Tree # consider packing different components in another object, since
	# not all nodes may define all of them
	inPlug : Tree
	outPlug : Tree


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

	@classmethod
	def compute(cls, dataBlock:DataBlock, graph:ChimaeraNode)->None:
		"""Create joint hierarchy from template hierarchy.
		"""
# get data
		jointRoot = dataBlock.params("joints")
		# get parent plug
		parentPlug = dataBlock.inPlug("parent")




