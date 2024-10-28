
from __future__ import annotations

import pprint
import typing as T

import fnmatch

from collections import namedtuple

import wplib.sequence
from wplib import log, Sentinel, TypeNamespace, Pathable
from wplib.constant import MAP_TYPES, SEQ_TYPES, STR_TYPES, LITERAL_TYPES, IMMUTABLE_TYPES
from wplib.object.visitor import PARAMS_T, CHILD_LIST_T
from wplib.uid import getUid4
from wplib.inheritance import clsSuper
from wplib.object import UidElement, ClassMagicMethodMixin, CacheObj
from wplib.serial import Serialisable
#from wplib.pathable import Pathable
from wplib.object import VisitAdaptor, Visitable


from wptree import Tree

from wpdex import *


from chimaera.lib import tree as treelib



"""
simplest version works with node name and UI fully reactive :D
"""



class ChimaeraNode(Modelled,
                   Pathable,
                   Visitable,
                   ):
	keyT = Pathable.keyT
	pathT = Pathable.pathT

	@classmethod
	def dataT(cls):
		return Tree
	data : Tree | WpDexProxy

	@classmethod
	def newDataModel(cls, name="node", **kwargs) ->dataT():
		t = Tree(name=name)
		t.addBranch( Tree("nodes") ) #TODO: replace with the crazy incoming/defined etc
		return t

	nodeTypeRegister : dict[str, type[ChimaeraNode]] = {}

	def branchMap(self):
		return {name : ChimaeraNode(branch)
		        for name, branch in self.data("nodes").branchMap().items()}

	def getAvailableNodesToCreate(self)->list[str]:
		"""return a list of node types that this node can support as
		children - by default allow all registered types
		TODO: update this as a combined class/instance method
		"""
		return list(self.nodeTypeRegister.keys())

	def __init__(self, data:Tree):
		Modelled.__init__(self, data)

	def colour(self)->tuple[float, float, float]:
		return (0.2, 0.3, 0.7)

	@property
	def name(self) -> keyT:
		return self.data.getName()
	def setName(self, name: keyT):
		self.data.setName(name)

	@property
	def parent(self) -> (Pathable, None):
		"""parent data will be

		parentName :
			nodes :
				childName : etc
					nodes : etc

		intuitively, should the value of an attribute be
		the fully resolved version of that attribute?

		"""
		if not self.data.parent: return None

		return ChimaeraNode(self.data.parent.parent)

	def _setParent(self, parent: Pathable):
		"""private as you should use addBranch to control hierarchy
		from parent to child - addBranch will call this internally"""
		return
	def addBranch(self, branch:Pathable, name:keyT=None):
		assert isinstance(branch, (ChimaeraNode, Tree))
		if isinstance(branch, ChimaeraNode):
			branch = branch.data
		# add tree branch to this node's data
		self.data("nodes").addBranch(branch)
		return ChimaeraNode(branch)

	def branchMap(self) ->dict[keyT, Pathable]:
		return {k : ChimaeraNode(v) for k, v
		        in self.data("nodes").branchMap().items()}

	@classmethod
	def nodeForTree(cls, data:Tree):
		pass


	@classmethod
	def create(cls, name="node", **kwargs):
		return cls(cls.newDataModel(name=name, **kwargs))


if __name__ == '__main__':
	pass













