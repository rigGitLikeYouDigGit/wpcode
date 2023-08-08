
from __future__ import annotations
import typing as T

from wplib import Expression

from wptree import Tree

from chimaera import ChimaeraNode, PlugNode, ChimaeraGraph


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
	Create a live template hierarchy feeding into live constrained joints.
	"""

	def defaultParams(self)->Tree:
		"""set up default params for placing joints
		maybe pass in tree root as argument"""
		paramRoot = Tree("params")
		jointRoot = paramRoot("joints", create=True)
		jointRoot.desc = "Branches below this will each correspond to a joint."
		return paramRoot

	def makePlugs(self, inRoot:Tree, outRoot:Tree)->None:
		"""create plugs for this node
		Output root will be set with data object, to be filtered if needed"""
		inRoot("parent", create=True)
		outRoot("joint", create=True)

	def compute(self, dataBlock):
		"""build template hierarchy from data
		copy out live hierarchy
		local constrain each
		constrain output to parent
		connect output hierarchy to data object
		update output tree
		"""





