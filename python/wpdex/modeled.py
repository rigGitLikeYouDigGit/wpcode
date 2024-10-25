

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


#from chimaera.lib import tree as treelib



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

	def ref(self, path:Pathable.pathT)->WX:
		return self.data.ref(path)

	@classmethod
	def create(cls, **kwargs):
		return cls(cls.newDataModel(**kwargs))