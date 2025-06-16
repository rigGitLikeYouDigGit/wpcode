from __future__ import annotations

import types
import typing as T
from collections import namedtuple
from typing import NamedTuple
import numpy as np

from wptree import TreeInterface
from wplib.sequence import flatten
from wplib.object import UnHashableDict
from wplib import log

#from setFnMap
from .cache import om
from . import bases, plug as pluglib
from . import api
from .bases import NodeBase, PlugBase
from wplib.object import Signal, Adaptor

if T.TYPE_CHECKING:
	from wpm import WN

# _WNCache : T.Type[WN] = None
# def _getWN()->_WNCache:
# 	global _WNCache
# 	if _WNCache is None:
# 		from wpm.core.node.base import WN
# 		_WNCache = WN
# 	return _WNCache

class PlugMeta(type):
	"""Metaclass to initialise plug wrapper from mplug
	or from string"""
	objMap = UnHashableDict()

	# register maps for all plugs
	plugTypeMap : dict[int, T.Type[Plug]] = {}

	@staticmethod
	def wrapperClassForMPlug(MPlug: om.MPlug):
		"""return a wrapper class for the plug -
		"""
		return WN.apiTypeClassMap().get(MPlug.attribute().apiType(), WN)
	# endregion

	def __call__(cls, plug:T.Union[str, Plug, om.MPlug])->Plug:
		"""check if existing plug object exists
		"""

		if isinstance(plug, Plug):
			return plug
		# filter input to MPlug
		mPlug = pluglib.getMPlug(plug, default=None)
		if mPlug is None:
			raise TypeError(f"Cannot retrieve MPlug from {plug}")

		# no caching for now, found it to be more trouble than it's worth
		plug = super(PlugMeta, cls).__call__(mPlug)
		return plug

		# check for (node, plug) tie in register
		tie = (mPlug.node(), mPlug.name().split(".")[1:])

		if tie not in PlugMeta.objMap:
			plug = super(PlugMeta, cls).__call__(mPlug)
			#plug = Plug(mPlug)
			PlugMeta.objMap[tie] = plug
		return PlugMeta.objMap[tie]


		# filter to MPlug - in case somehow the incorrect wrapper is used
		mplug = api.getMPlug(plug)
		cache = api.getCache()
		mobj = mplug.attribute()
		getType = mobj.apiType()
		leafMFnClass = cache.apiTypeLeafMFnMap[getType]

		print("get class for plug", mplug, type(mplug), mobj.apiTypeStr)
		print("found mfn", leafMFnClass)

		wrapCls = Plug.adaptorForType(leafMFnClass)

		print("found cls", wrapCls)
		if leafMFnClass is om.MFnTypedAttribute: # check for data type
			dataType = om.MFnTypedAttribute(mobj).attrType()
			dataTypeName = cache.classTypeIdNameMemberMap(om.MFnData)[dataType]
			print("found data type", dataType, dataTypeName)
			dataMFnCls = cache.apiTypeMFnDataMap[
				om.MFnTypedAttribute(mobj).attrType()]
			print("found data mfn", dataMFnCls)
			plugCls = Plug.adaptorForType(dataMFnCls)
			print("found plug cls", plugCls)
			wrapCls = plugCls

		print("IS ARRAY", mplug, mplug.isArray)
		#TODO: cache array types as declared, don't re-type them each time
		if mplug.isArray:
			wrapCls = type("Array_" + wrapCls.__name__,
			               (wrapCls, ArrayPlug),
			               {}
			               )

		# create instance
		ins = super(PlugMeta, wrapCls).__call__(mplug, **kwargs)
		return ins


TreeType = T.TypeVar("TreeType", bound="Plug") # type of the current class

class Plug(PlugBase,
           Adaptor,
           TreeInterface,
           metaclass=PlugMeta):

	"""wrapper for MPlugs on nodes -
	syntax matters between SINGLE, COMPOUND, ARRAY and SLICE

	single->single : good
	single->compound : good, try to connect all
	single->array : ILLEGAL, ambiguous if we connect to first, to last, or to all
	single->slice : good, try to connect to all plugs in slice

	compound->single : good, try connect first
	compound->compound : good only if structures match
	compound->array : ????? ILLEGAL for now, but might be allowed later if dims match
	compound->slice : ?????

	array->single : ILLEGAL
		LEGAL if single is element of array, try to connect to the given element onwards
		(same principle as .start(), .end() iterators from c++)
	array->compound : ILLEGAL
	array->array : good, try connect one-to-one
	array->slice : ILLEGAL for now

	slice->single : ILLEGAL
		LEGAL if single is element of array, as above
	slice->compound : good if structures match
	slice->array : good, connect from start?
	slice->slice : good, trim to the shortest length


	can we get some kind of tree dimension tuple to help expansion // broadcasting
	since expansion has to happen from leaves up

	since plug structure might change easily, this class is totally transient, just
	to aid manipulation

	adaptor set up for MFn attribute classes - we get the lowest-matching
	MFn class for the plug, and pass that into the adaptor class lookup

	NO CACHE everything transient, MPlug itself is source of truth on everything, live

	TODO: still eventually want some kind of PlugSlice, to work on a selection of plugs as a single object

	"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = (om.MFnAttribute, )

	# identification constants
	apiTypeStr : str = None # plug.attribute().apiTypeStr
	apiType : int = None # plug.attribute().apiType

	subtype = None

	VALUE_T = T.Any # type of value to retrieve from plug

	lookupCreate = False

	# if isArray:
	# 	VALUE_T = T.List[VALUE_T]

	# region core

	def __init__(self,
				 plug:T.Optional[om.MPlug, str],
				 ):
		""""""
		self.MPlug = plug
		self.hType : pluglib.HType = pluglib.plugHType(self.MPlug)
		TreeInterface.__init__(self, None)


	def __repr__(self):
		return f"<{self.__class__.__name__} ({self.MPlug.name()})>"

	def node(self)->WN:
		"""return parent node"""
		return WN(self.MPlug.node())

	def dataObject(self)->om.MObject:
		"""return data MObject from plug - used
		to pass into MFnData objects"""
		return self.MPlug.asMObject()

	def MFnData(self)->om.MFnData:
		"""return MFnData object from plug"""
		raise NotImplementedError


	def strPath(self, root=True, node=True, nodeFullPath=False) ->str:
		"""always include root on plugs, since we don't consider node to be direct parent
		maybe this is wrong"""
		baseStr = super().strPath(root=root)
		if(nodeFullPath):
			if api.isDag(self.MPlug.node()):
				nodeFn = om.MFnDagNode(self.MPlug.node())
				return nodeFn.fullPathName() + "." + baseStr
			else:
				return om.MFnDependencyNode(self.MPlug.node()).absoluteName() + "." + baseStr
		if node:
			return om.MFnDependencyNode(self.MPlug.node()).name() + "." + baseStr
		return baseStr

	#endregion


	#region tree integration
	def getName(self) ->str:
		"""test for allowing plugs to have int names, not just strings"""
		if self.MPlug.isElement:
			result = pluglib.splitPlugLastNameToken(self.MPlug)
			return int(result)
		return self.MPlug.partialName(useLongNames=1)

	def setName(self, name:str):
		# don't allow changing plug names like this
		return

	def getParent(self) ->Plug:
		if self.MPlug.isElement:
			return Plug(self.MPlug.array())
		if self.MPlug.isChild:
			return Plug(self.MPlug.parent())
		return None

	def getBranches(self) ->list[Plug]:
		return [Plug(v) for k, v in pluglib.subPlugMap(self.MPlug).items()]

	def branchMap(self) -> dict[str, TreeType]:
		return {k : Plug(v) for k, v in pluglib.subPlugMap(self.MPlug).items()}

	def _branchFromToken(self, token:keyT)->(TreeType, None):
		""" given single address token, return a known branch or none """
		if token == self.parentChar:
			return self.getParent()
		return self.branchMap().get(token)

	def __eq__(self, other:Plug):
		return self.MPlug is other.MPlug

	def __str__(self):
		return om.MFnDependencyNode(self.plug().node()).name() + "." + self.stringAddress()

	def __hash__(self):
		return hash((self.node, self.MPlug.name()))

	def stringAddress(self, includeRoot=True) -> str:
		"""reformat full path to this plug
		"""
		trunk = self.trunk(includeSelf=True,
		                   includeRoot=includeRoot,
		                   )
		s = ""
		for i in range(len(trunk)):
			s += str(trunk[i].name)
			if i != (len(trunk) - 1):
				s += trunk[i].separatorChar

		return s

	@property
	def isLeaf(self)->bool:
		return not any((self.MPlug.isCompound, self.MPlug.isArray))


	def index(self, raw=False)->(int, None):
		"""return index of plug in parent array"""
		if not self.MPlug.isElement: return None
		if not raw:	return self.MPlug.logicalIndex()
		# physical index is a bit convoluted - there's no direct way to get it,
		# so we look for the index of the logical index in the list of existing indices.
		# easy
		return tuple(self.MPlug.array().getExistingArrayAttributeIndices()).index(
			self.MPlug.logicalIndex())
		# as far as I know this should always work, indices might be sparse but they're always sorted


	#region value

	def getValue(self) ->T:
		"""retrieve MPlug value"""
		return pluglib.plugValue(self.MPlug)

	# slightly more maya-familiar versions of the above
	def get(self):
		return self.getValue()

	def valueNP(self)->np.ndarray:
		"""return value as numpy array"""
		return np.array(self.getValue())
	def valueMPoint(self)->om.MPoint:
		return om.MPoint(self.getValue())
	def valueMVector(self)->om.MVector:
		return om.MVector(self.getValue())
	def valueMMatrix(self)->om.MMatrix:
		return om.MMatrix(self.getValue())
	
	def __call__(self, *args, **kwargs):
		if not args and not kwargs:
			return self.getValue()
		return super().__call__(*args, **kwargs)

	def __getattr__(self, item:str)->Plug:
		"""check if plug has been accessed directly by name -
		always has a trailing underscore"""
		if(item[-1] == "_" and item[0] != "_"): # don't trigger on private or magic methods
			#if (foundPlug := self.plug(item[:-1])) is not None:
			if (foundPlug := self.getBranch(item[:-1])) is not None:
				return foundPlug
			raise TypeError("no maya plug found for ", item)
		return super().__getattribute__(item)

	def __setattr__(self, item:str, val):
		"""check if plug has been accessed directly by name -
		always has a trailing underscore"""
		if(item[-1] == "_" and item[0] != "_"):
			item = item[:-1]
			#if (foundPlug := self.plug(item[:-1])) is not None:
			#log("getBranch", self.getBranch(item))
			if (foundPlug := self.getBranch(item)) is not None:
				foundPlug.set(val)
				return
			raise TypeError("no maya plug found to set ", item,
			                "on ", self.strPath(root=True), self.branchMap())
		return super().__setattr__(item, val)

	def __getitem__(self, *address:(str, tuple),
	                **kwargs):
		"""override normal tree behaviour on getitem and setitem
		to return full Plug objects - just to get closer to normal Maya syntax
		"""
		first = address[0]
		if isinstance(first, int):
			if self.MPlug.isArray:
				if(len(address) == 1):
					return Plug(self.MPlug.elementByLogicalIndex(first))
				return Plug(self.MPlug.elementByLogicalIndex(first))[address[1:]]
		raise NotImplementedError

	def __setitem__(self, *address:(str, tuple), value,
	                **kwargs):
		"""override normal tree behaviour on getitem and setitem
		to return full Plug objects - just to get closer to normal Maya syntax
		"""
		first = address[0]
		nextPlug : Plug = None
		if isinstance(first, int):
			if self.MPlug.isArray:
				nextPlug = Plug(self.MPlug.elementByLogicalIndex(first))
			elif self.MPlug.isCompound:
				nextPlug = Plug(self.MPlug.child(first))
			else:
				raise TypeError("cannot use setItem on leaf plug")

			if len(address) == 1:
				nextPlug.set(value)
			else:
				nextPlug.__setitem__(*address[1:], value)
			return
		raise NotImplementedError

	#endregion



	# region connection and assignment

	def setValue(self, val):
		"""top-level method to set this plug's value,
		or connect another live plug to it"""
		if otherPlug := pluglib.getMPlug(val, default=None):
			pluglib.con(otherPlug, self.MPlug)
		else:
			pluglib.setPlugValue(self.MPlug, val)

	def set(self, *value):
		if len(value) == 1:
			value = value[0]
		self.setValue(value)
	def __lshift__(self, other:(Plug, T.Any)):
		"""self << other
		drive this plug with other
		"""
		self.set(other)
	def __rshift__(self, other):
		"""self >> other
		drive other with this plug ( if other is a plug )"""
		if(otherPlug := pluglib.getMPlug(other)):
			pluglib.con(self.MPlug, otherPlug)
			return
		raise TypeError
	def __rlshift__(self, other):
		"""other << self
		drive other with self (if other is a plug"""
		if (otherPlug := pluglib.getMPlug(other)):
			pluglib.con(self.MPlug, otherPlug)
			return
		raise TypeError
	def __rrshift__(self, other):
		"""other >> self
		drive self with other"""
		self.set(other)

	#endregion


	rootName = "plugRoot"

	trailingChar = "+" # signifies uncreated trailing plug index
	separatorChar =  "."

	def plug(self, *lookup)->om.MPlug:
		"""return child plug from string - name is a bit off,
		but lets us have the same interface as on node, which may
		be important later
		"""
		if not lookup:
			# used for uniform interfaces
			return self.MPlug
		raise NotImplementedError

	def nBranches(self):
		if self.MPlug.isArray:
			return self.MPlug.evaluateNumElements()
		if self.MPlug.isCompound:
			return self.MPlug.numChildren()
		return 0


	def arrayMPlugs(self):
		"""return MPlug objects for each existing element in index
		always returns 1 more than number of real plugs (as always
		one left open)
		"""

		if self.MPlug.evaluateNumElements() == 0:
			self.addNewElement()

		return [self.MPlug.elementByLogicalIndex(i)
				for i in range(self.MPlug.evaluateNumElements())]


	def element(self, index:int, logical=True):
		"""ensure that the given element index is present
		return that array plug"""
		return Plug( pluglib.arrayElement(self.MPlug, index) )

	def addNewElement(self):
		"""extend array plug by 1"""
		pluglib.newLastArrayElement(self.MPlug)

	def ensureOneArrayElement(self):
		"""
		"""
		pluglib.ensureOneArrayElement(self.MPlug)

	plugParamType = (str, "Plug", om.MPlug)

	def _filterPlugParam(self, plug:plugParamType)->om.MPlug:
		"""return an MPlug from variety of acceptable sources"""
		plug = getattr(plug, "plug", plug)
		if isinstance(plug, (types.FunctionType, types.MethodType)):
			plug = plug()
		if isinstance(plug, Plug):
			plug = plug.MPlug
		elif isinstance(plug, str):
			plug = pluglib.getMPlug(plug)
		return plug


	def con(self, otherPlug:(plugParamType,
	                         list[plugParamType]),
	        _dgMod=None):
		"""connect this plug to the given plug or plugs"""
		dgMod = _dgMod or om.MDGModifier()

		# allow for passing multiple trees to connect to
		if not isinstance(otherPlug, (list, tuple)):
			otherPlug = (otherPlug,)

		for otherPlug in otherPlug:
			# check if other object has a .plug attribute to use
			log("other plug start", otherPlug, type(otherPlug))
			otherPlug = self._filterPlugParam(otherPlug)
			log("other plug end", otherPlug, type(otherPlug))

			log("con:", self.MPlug, otherPlug)
			pluglib.con(self.MPlug, otherPlug, dgMod)
		dgMod.doIt()

	# region networking
	def driver(self)->("Plug", None):
		"""looks at only this specific plug, no children
		returns Plug or None"""
		driverMPlug = self.MPlug.connectedTo(True,  # as Dst
		                       False  # as Src
		                       )
		if not driverMPlug: return None
		return Plug(driverMPlug[0])

	def drivers(self, includeBranches=True)->dict[Plug, Plug]:
		"""queries either immediate subplugs, or all subplugs to find set of all
		sub plugs
		returns { driverPlug : drivenPlug }

		"""
		checkBranches = self.allBranches(includeSelf=True) if includeBranches else [self]
		plugMap = {}

		for checkBranch in checkBranches:
			driver = checkBranch.driver()
			if driver:
				plugMap[driver] = checkBranch
		return plugMap

	def _singleDestinations(self)->tuple[Plug]:
		"""return all plugs fed by this single plug"""
		drivenMPlugs = self.MPlug.connectedTo(
			False, # as Dst
		    True # as Src
		)
		return drivenMPlugs

	def destinations(self, includeBranches=False)->dict[Plug, tuple[Plug]]:
		checkBranches = self.allBranches(includeSelf=True) if includeBranches else [self]
		plugMap = {}
		for checkBranch in checkBranches:
			plugMap[checkBranch] = checkBranch._singleDestinations()
		return plugMap

	def breakConnections(self, incoming=True, outgoing=True, includeBranches=True):
		"""disconnects all incoming / outgoing edges from this plug,
		or all of its branches"""
		#checkBranches = self.allBranches(includeSelf=True) if includeBranches else [self]
		dgMod = om.MDGModifier()
		if incoming:
			driverMap = self.drivers(includeBranches=includeBranches)
			for driver, driven in driverMap.items():
				dgMod.disconnect(driver.MPlug, driven.MPlug)
		if outgoing:
			driverMap = self.drivers(includeBranches=includeBranches)
			for driver, drivens in driverMap.items():
				for driven in drivens:
					dgMod.disconnect(driver.MPlug, driven.MPlug)
		dgMod.doIt()


	# endregion

	def driverOrGet(self)->Plug:
		"""if plug is driven by live input,
		return driving plug
		else return its static value"""
		if self.driver():
			return self.driver()
		return self.getValue()

	#def driveOrSet(self, plugOrValue):


	# region convenience
	@property
	def X(self):
		"""capital for more maya-like calls - node.translate.X"""
		return self.branches[0]
	@property
	def Y(self):
		return self.branches[1]
	@property
	def Z(self):
		return self.branches[2]
	@property
	def last(self):
		"""return the last available (empty) plug for this array"""
		return self(-1)

	#end


if T.TYPE_CHECKING:
	arrayBase = Plug
else:
	arrayBase = object

class ArrayPlug(arrayBase):
	"""very uncertain about this -
	dynamically generate classes that
	work as arrays?
	we assume this will be the second parent class,
	look up other values on the first

	type hinting won't work through this, will need to
	explicitly subclass in generated node files
	"""
	@classmethod
	def checkValid(cls):
		assert Plug in cls.__bases__
	@classmethod
	def plugCls(cls)->type[Plug]:
		"""return plug class for this array"""
		return cls.__bases__[0]

	def indices(self)->np.ndarray:
		return np.array(self.MPlug.getExistingArrayAttributeIndices())

	def child(self, index:int, raw=False)->Plug:
		"""return child plug from index"""
		return Plug(self.MPlug.elementByPhysicalIndex(index) if raw else self.MPlug.elementByLogicalIndex(index))

	def value(self) ->VALUE_T:
		"""return value from plug - if compound, recurse
		into it and return list"""
		return SparseList.fromTies(
			(i, self.child(i, raw=False).value()) for i in self.indices())


class CompoundPlug(Plug):
	"""plug for compound attributes,
	return a namedtuple by default - allow returning as dict
	"""

	forTypes = (om.MFnCompoundAttribute, )
	apiType = om.MFn.kCompoundAttribute

	USE_NAMED_TUPLES = True

	if USE_NAMED_TUPLES:
		VALUE_T : type[NamedTuple]
		def _generateTypeTuple(self)->type[NamedTuple]:
			"""return a named tuple type for this plug -
			used if the plug type is not statically defined"""
			return namedtuple(
				"CompoundValue",
			    [self.child(i).name() for i in range(self.numChildren())])
		def value(self) ->NamedTuple:
			"""it might be too complex to pass through the numpy-ness of
			the values"""
			if self.VALUE_T is None:
				self.VALUE_T = self._generateTypeTuple()
			return self.VALUE_T(
				*[Plug(self.MPlug.child(i)).value()
				  for i in range(self.MPlug.numChildren())])
	else:
		VALUE_T = dict
		def value(self) ->dict:
			return {self.child(i).name() : Plug(self.MPlug.child(i)).value()
			        for i in range(self.MPlug.numChildren())}

	def nameIndexMap(self)->dict[str, int]:
		"""return child plugs as a map"""
		nameMap = {}
		for i in range(self.MPlug.numChildren()):
			mplug : om.MPlug = self.MPlug.child(i)
			nameMap[mplug.partialName().rsplit(".", 1)[-1]] = i
			nameMap[mplug.partialName(useAlias=True).rsplit(".", 1)[-1]] = i
			nameMap[mplug.partialName(useLongNames=True).rsplit(".", 1)[-1]] = i
		return nameMap

	def child(self, index:(int, str))->Plug:
		"""return child plug from index"""
		if isinstance(index, int):
			return Plug(self.MPlug.child(index))


class NumericPlug(Plug):
	"""plug for numeric attributes"""
	forTypes = (om.MFnNumericAttribute, )
	apiType = om.MFn.kNumericAttribute

	VALUE_T = list[(int, float)]

	def value(self) ->VALUE_T:
		# mfnData = om.MFnNumericData(
		# 	self.MPlug.asMDataHandle().data()		)
		# DON'T DO THIS ^ it crashes maya
		return om.MFnNumericData(self.MPlug.asMObject()).getData()

class MatrixPlug(Plug):
	"""
	still not sure what makes some plugs MatrixAttributes,
	and some TypedAttributes with Matrix data
	"""
	forTypes = (om.MFnMatrixAttribute, om.MFnMatrixData)
	VALUE_T = om.MMatrix
	apiType = om.MFn.kMatrixAttribute
	def MFnData(self) ->om.MFnMatrixData:
		return om.MFnMatrixData(self.dataObject())
	def value(self) ->VALUE_T:
		#return self.MFnData().matrix()
		return self.MPlug.constructHandle().asMatrix()
	def valueNP(self)->np.ndarray:
		return np.array(self.value()).reshape(4, 4)
	def _setValue(self, value:VALUE_T):
		"""I don't think it's safe to set the MDataHandle directly
		"""
		self.MPlug.setMObject(om.MFnMatrixData().create(
			om.MMatrix(value)))
		#self.MPlug.setMObject(om.MFnMatrixData().create(value))

class UnitPlug(Plug):
	forTypes = (om.MFnUnitAttribute, )
	apiType = om.MFn.kUnitAttribute

class EnumPlug(Plug):
	forTypes = (om.MFnEnumAttribute, )
	apiType = om.MFn.kEnumAttribute

class MessagePlug(Plug):
	forTypes = (om.MFnMessageAttribute, )
	apiType = om.MFn.kMessageAttribute

class TypedPlug(Plug):
	forTypes = (om.MFnTypedAttribute, )
	apiType = om.MFn.kTypedAttribute

class NurbsCurvePlug(TypedPlug):
	forTypes = (om.MFnNurbsCurve, om.MFnNurbsCurveData)
	subtype = om.MFnData.kNurbsCurve
class MeshPlug(TypedPlug):
	forTypes = (om.MFnMesh, om.MFnMeshData)
	subtype = om.MFnData.kMesh
class NurbsSurfacePlug(TypedPlug):
	forTypes = (om.MFnNurbsSurface, om.MFnNurbsSurfaceData)
	subtype = om.MFnData.kNurbsSurface
class FloatArrayPlug(TypedPlug):
	forTypes = (om.MFnNumericData, )
	subtype = om.MFnData.kFloatArray
class IntArrayPlug(TypedPlug):
	forTypes = (om.MFnIntArrayData, )
	subtype = om.MFnData.kIntArray
class VectorArrayPlug(TypedPlug):
	forTypes = (om.MFnVectorArrayData, )
	subtype = om.MFnData.kVectorArray
class PointArrayPlug(TypedPlug):
	forTypes = (om.MFnPointArrayData, )
	subtype = om.MFnData.kPointArray
class MatrixArrayPlug(TypedPlug):
	forTypes = (om.MFnMatrixArrayData, )
	subtype = om.MFnData.kMatrixArray
# class LatticePlug(TypedPlug):
# 	forTypes = (om.MFnLatticeData, )
# 	subtype = om.MFnData.kLattice
class StringPlug(TypedPlug):
	forTypes = (om.MFnStringData, )
	subtype = om.MFnData.kString
class StringArrayPlug(TypedPlug):
	forTypes = (om.MFnStringArrayData, )
	subtype = om.MFnData.kStringArray
# class FalloffFunctionPlug(TypedPlug):
# 	forTypes = (om.MFnFalloffFunction,)
# 	subtype = om.MFnData.kFalloffFunction


# class PlugSlice(Plug):
# 	"""object to represent a slice of a plug tree
# 	Weird inheritance but maybe it works
# 	"""

# man I don't know if any of this stuff is even a good idea
#TODO: shift the above into adaptor classes -
# this way we can also handle logic for datahandles in the same place

class PlugDescriptor:
	"""descriptor for plugs -
	declare whole attr hierarchy in one go
	TODO: consider putting valid value types here as well to warn
		if you try and set an int plug to a string or something
	"""
	def __init__(self, name:str):
		self.name = name

	# TEMP get and set
	def __get__(self,  instance:(Plug, WN), owner)->Plug:
		#log("pd _get_", instance, type(instance), owner, type(owner))
		return instance.plug(self.name)
	# # TEMP
	def __set__(self, instance:(Plug, WN),
	            value:(Plug, Plug, WN, T.Any )):
		try:
			# try and connect value if it's another plug

			instance.plug(self.name) << pluglib.getMPlug(value)
			return
		except (AttributeError, TypeError):
			pass
		instance.plug(self.name).set(value)




class MDataHandleFn:
	"""helpers for MDataHandle getting and setting
	for different types"""

