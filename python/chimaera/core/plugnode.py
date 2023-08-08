from __future__ import annotations
import typing as T

from wplib import Expression
from wptree import Tree

from chimaera.core.node import ChimaeraNode

"""not sure if we should have a type disconnect between base nodes
and plug nodes - we could probably combine them if needed.


Plug node is associated with separate Chimaera nodes forming input
and output plug trees, managed by main node

Unsure if these should automatically be subgraphs or live at same
hierarchy level as main node.

PlugNode classes should REALLY be function owners operating on a 
datablock, in the same model as Maya -
separates operation and data, makes it easier to serialise and merge data,
makes it more robust to missing class references, etc

"""

class PlugNode(ChimaeraNode):

	@classmethod
	def defaultParams(cls, paramRoot:Tree)->Tree:
		"""set up default params for placing joints
		maybe pass in tree root as argument"""
		return paramRoot

	@classmethod
	def makePlugs(cls, inRoot:Tree, outRoot:Tree)->None:
		"""create plugs for this node - work with trees, not plug nodes.
		"""

	@classmethod
	def syncInputPlugs(cls, fromInTree:Tree)->None:
		"""regenerate input plug nodes from tree -
		to be called if plugs change during compute
		"""
