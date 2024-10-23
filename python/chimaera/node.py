
from __future__ import annotations

import pprint
import typing as T

import fnmatch

from collections import namedtuple

import wplib.sequence
from wplib import log, Sentinel, TypeNamespace, Pathable
from wplib.constant import MAP_TYPES, SEQ_TYPES, STR_TYPES, LITERAL_TYPES, IMMUTABLE_TYPES
from wplib.uid import getUid4
from wplib.inheritance import clsSuper
from wplib.object import UidElement, ClassMagicMethodMixin, CacheObj
from wplib.serial import Serialisable
#from wplib.pathable import Pathable

from wptree import Tree

from wpdex import WpDexProxy, WpDex, WX


from chimaera.lib import tree as treelib



"""
start over just for the moment,
get a full cycle of it working -
 nodes affecting data
  data driving UI
   UI modifying data
only node name for now
"""

class Modelled:
	"""test if it's useful to have a base class -
	represents a python object that refers to
	and modifies a static data model for all
	its state.

	data used in this way has to be wrapped in a proxy,
	so that modifications made by this object
	trigger events to other items pointing to the data

	can we use this to hold a tree schema for validation too?
	"""

	@classmethod
	def dataT(cls):
		return T.Any

	@classmethod
	def newDataModel(cls, **kwargs)->dataT():
		raise NotImplementedError("Define a new data structure expected "
		                          f"for Modelled {cls}")

	def __init__(self, data:dataT()):
		self.data : self.dataT() = WpDexProxy(data)

	@classmethod
	def create(cls, **kwargs):
		return cls(cls.newDataModel(**kwargs))

class ChimaeraNode(Modelled,
                   Pathable):
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

	@classmethod
	def create(cls, name="node", **kwargs):
		return cls(cls.newDataModel(**kwargs))

	def __init__(self, data:Tree):
		Modelled.__init__(self, data)

	def ref(self, path:Pathable.pathT)->WX:
		return self.data.ref(path)

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



if __name__ == '__main__':
	pass













