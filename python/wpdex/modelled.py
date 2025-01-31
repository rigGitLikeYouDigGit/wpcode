

from __future__ import annotations
import pprint
import typing as T

import fnmatch

from collections import namedtuple

import wplib.sequence
from wplib import log, Sentinel, TypeNamespace, Pathable
from wplib.object import Visitable, VisitAdaptor
from wplib.serial import Serialisable

from wpdex import WpDexProxy, WpDex, WX


#from chimaera.lib import tree as treelib



class Modelled(#Visitable,
               Serialisable):
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
		#log("new modelled", data)
		self.data : self.dataT() = WpDexProxy(data)

	def __str__(self):
		try:
			return f"<{self.__class__.__name__}({self.rawData()})>"
		except:
			return f"<{self.__class__.__name__}(ERROR getting raw data)>"

	def rawData(self)->dataT():
		return self.data._proxyTarget()

	def ref(self, *path:Pathable.pathT)->WX:
		return self.data.ref(*path)

	# def dataChangedSignal(self, path:Pathable.pathT=()):
	# 	return self.data.ref(path).

	def setDataModel(self, data:dataT()):
		"""transplant the core WpDexProxy to the new data object -
		this should preserve and update any references
		made relative to the proxy object"""
		self.data._setProxyTarget(self.data, data)

	def encode(self, encodeParams:dict=None) ->dict:
		return self.rawData()
	@classmethod
	def decode(cls, serialData:dict, decodeParams:dict=None) ->T.Any:
		return cls(serialData)

	def childObjects(self, params:PARAMS_T) ->CHILD_LIST_T:
		"""maybe we should just bundle this in modelled
		I think for the core visit stuff we shouldn't pass in the proxy,
		might get proper crazy if we do

		"""
		return VisitAdaptor.adaptorForObject(self.rawData()).childObjects(
			self.rawData(), params)

	@classmethod
	def newObj(cls, baseObj: Visitable, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		"""this should just be a new copy of the data given"""
		return cls(childDatas[0][1])


	@classmethod
	def create(cls, **kwargs):
		return cls(cls.newDataModel(**kwargs))






