
from __future__ import annotations
import typing as T

from dataclasses import dataclass

import numpy as np


from wplib import Expression

from wptree import Tree

from chimaera import ChimaeraNode, PlugNode




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
	def compute(cls, node:ChimaeraNode, graph:ChimaeraNode) ->None:





