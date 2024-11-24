
from __future__ import annotations
import typing as T
import ast
from typing import TypedDict, NamedTuple
from collections import namedtuple

import numpy as np

# tree libs for core behaviour
#from wptree import Tree
from wplib.object import Signal, Adaptor
from wplib.inheritance import iterSubClasses
from wplib.wpstring import camelJoin
from wplib.object import UnHashableDict, StringLike, SparseList
#from tree.lib.treecomponent import TreeBranchLookupComponent
from wplib.sequence import toSeq, firstOrNone

# maya infrastructure
from wpm.constant import GraphTraversal, GraphDirection, GraphLevel, WPM_ROOT_PATH
from ..bases import NodeBase, PlugBase
from ..callbackowner import CallbackOwner
from ..patch import cmds, om
from ..api import getMObject, getMFn, asMFn, getMFnType
from .. import api, plug

#from ..plugtree import PlugTree
from ..namespace import NamespaceTree, getNamespaceTree


# NO TRUE IMPORTS for codegen submodules
# if T.TYPE_CHECKING:
# 	from ..api import MFMFnT
# 	from .gen import Catalogue as GenCatalogue
# 	from .author import Catalogue as ModCatalogue

"""

maya node connection syntax fully conformed to tree syntax
but not actually inheriting from tree


test more descriptive error types for things that can go wrong
when creating nodes, plugs etc

LookupError : string not found in scene - name of node, plug etc
ReferenceError : MObject is null


"""

# TODO: MOVE THIS
#  integrate properly with main plugs
class RampPlug(object):
	"""don't you love underscores and capital letters?
	called thus:
	ramp = RampPlug(myAnnoyingString)
	setAttr(ramp.point(0).value, 5.0)
	to be fair, if these are special-cased in maya, why not here"""

	class _Point(object):
		def __init__(self, root, index):
			self.root = root
			self.index = index

		@property
		def _base(self):
			return "{}[{}].{}_".format(
				self.root, self.index, self.root	)

		@property
		def value(self):
			return self._base + "FloatValue"
		@property
		def position(self):
			return self._base + "Position"
		@property
		def interpolation(self):
			return self._base + "Interp"

	def __init__(self, rootPlug):
		"""root is remapValue"""
		self.root = rootPlug
		self.points = {}

	def point(self, id):
		""":rtype : RampPlug._Point"""
		return self._Point(self.root, id)




class PlugMeta(type):
	"""
	replicate the same dynamic class lookup of the
	WN node metaclass for plugs
	"""
	# register maps for all plugs
	plugTypeMap : dict[int, T.Type[Plug]] = {}

	# def __getattr__(self, item:str):
	# 	"""return a class for a specific node type
	#
	# 	WN.Transform -> look up Transform node class
	# 	"""
	# 	#look for WN.Transform, WN.Mesh etc
	# 	if item[0].isupper():
	# 		return self.retriever.getNodeCls(item)
	# 	raise AttributeError(f"no attribute {item}")


	@staticmethod
	def wrapperClassForMPlug(MPlug: om.MPlug):
		"""return a wrapper class for the plug -
		"""
		return WN.apiTypeClassMap().get(mobj.apiType(), WN)
	# endregion


	def __call__(cls, plug:(om.MPlug, str), **kwargs)->Plug:
		""" get right plug subclass from MPlug,
		to get and set data by consistent interface
		I don't think we care about reusing one Plug object for the
		same MPlug - this should only handle the type lookup
		"""
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

class ApiType(NamedTuple):
	int : int
	str : str

	@classmethod
	def fromMObj(cls, obj:om.MObject):
		return cls(obj.apiType(), obj.apiTypeStr())
	@classmethod
	def fromConstant(cls, constant:int, holderClass:type[om.MFn]):
		pass

class MDataHandleFn:
	"""helpers for MDataHandle getting and setting
	for different types"""

#class Plug[isArray:T.Literal[False, True]]:
class Plug(PlugBase,
           Adaptor,
           metaclass=PlugMeta):
	"""base class for plugs

	for setting, take inspiration from normal variable
	assignment:
	a = 1
	b = 2
	plug.set(3)
	plug << 4

	this goes completely against the intuition of left-to-right
	data flow, but maybe it's worth it for consistent syntax

	adaptor set up for MFn attribute classes - we get the lowest-matching
	MFn class for the plug, and pass that into the adaptor class lookup

	"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = (om.MFnAttribute, )

	# identification constants
	apiTypeStr : str = None # plug.attribute().apiTypeStr
	apiType : int = None # plug.attribute().apiType

	subtype = None

	VALUE_T = T.Any # type of value to retrieve from plug
	# if isArray:
	# 	VALUE_T = T.List[VALUE_T]

	# region core
	def __init__(self, MPlug:om.MPlug):
		self.MPlug = MPlug

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

	#endregion

	#region structure
	def plug(self, s:str)->Plug:
		"""return child plug from string - name is a bit off,
		but lets us have the same interface as on node, which may
		be important later"""
		return self

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
	def value(self)->VALUE_T:
		"""return value from plug - if compound, recurse
		into it and return list"""
		raise NotImplementedError
	def _getValue(self)->VALUE_T:
		"""internal method to retrieve value from leaf plug"""
		raise NotImplementedError
	def valueNP(self)->np.ndarray:
		"""return value as numpy array"""
		return np.array(self.value())
	def valueMPoint(self)->om.MPoint:
		return om.MPoint(self.value())
	def valueMVector(self)->om.MVector:
		return om.MVector(self.value())
	def valueMMatrix(self)->om.MMatrix:
		return om.MMatrix(self.value())

	def _setValue(self, value:VALUE_T):
		"""set value on plug -
		internal method called by set() """
		raise NotImplementedError
	#endregion


	# region connection and assignment
	def set(self, arg:(Plug, str)):
		"""top-level method to set this plug's value,
		or connect another live plug to it"""

	def __lshift__(self, other:(Plug, T.Any)):
		"""connect other plug to this one"""
		self.set(other)

		pass
	# def __rlshift__(self, other):
	# 	pass

	#endregion

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

class PlugDescriptor:
	"""descriptor for plugs -
	declare whole attr hierarchy in one go"""
	def __init__(self, name:str):
		self.name = name

	# TEMP get and set
	def __get__(self,  instance:(Plug, WN), owner)->Plug:
		return instance.plug(self.name)
	# # TEMP
	def __set__(self, instance:(Plug, WN),
	            value:(Plug, WN, T.Any )):
		try:

			instance.plug(self.name) << value.plug()
			return
		except AttributeError:
			pass
		instance.plug(self.name).set(value)



# nodes

def filterToMObject(node)->om.MObject:
	"""filter input to MObject
	may not be alive"""
	node = firstOrNone(node)
	if isinstance(node, str):
		mobj = getMObject(node)
		if mobj is None:
			raise LookupError("No MObject found for string {}".format(mobj))
		return mobj
	elif isinstance(node, om.MObject):
		return node

	# filter input to MObject
	if hasattr(node, "MObject"):
		return node.MObject

	try:
		return node.object()
	except AttributeError:
		pass

	raise TypeError("Could not get MObject from {}".format((node, type(node))))

def filterToMPlug():
	pass


def nodeClassNameFromApiStr(apiTypeStr:str)->str:
	"""return node class from api type string"""
	return apiTypeStr[1:]

if T.TYPE_CHECKING:
	from . import NodeClassRetriever

class WNMeta(type):
	"""
	- get unique MObject for node / string
	- send getAttr to received to get node classes
	- support init kwargs for nodes as functions
		pass these on to node class, otherwise control gets complicated
	"""
	# region node class lookup
	retriever : NodeClassRetriever

	def __getattr__(self, item:str):
		"""return a class for a specific node type

		WN.Transform -> look up Transform node class
		"""
		#look for WN.Transform, WN.Mesh etc
		if item[0].isupper():
			return self.retriever.getNodeCls(item)
		raise AttributeError(f"no attribute {item}")

	@staticmethod
	def wrapperClassForNodeType(nodeType: str) -> T.Type[WN]:
		"""return a wrapper class for the given node's type
		if it exists"""
		return WN.nodeTypeClassMap().get(nodeType, WN)

	@staticmethod
	def wrapperClassForMObject(mobj: om.MObject):
		"""return a wrapper class for the given mobject
		bit more involved if we don't know the string type
		"""
		return WN.apiTypeClassMap().get(mobj.apiType(), WN)
	# endregion

	# region unique MObject lookup
	objMap = UnHashableDict()

	@classmethod
	def getUniqueMObject(cls, node: T.Union[str, om.MObject, WN, int])->om.MObject:
		"""filter input to MObject"""
		if isinstance(node, str):
			mobj = getMObject(node)
			if mobj is None:
				raise LookupError("No MObject found for string {}".format(mobj))
			return mobj
		elif isinstance(node, om.MObject):
			mobj = node
			if mobj.isNull():
				raise ReferenceError("MObject {} is null".format(mobj))
			return mobj

		# filter input to MObject
		if hasattr(node, "MObject"):
			return node.MObject

		try:
			return node.object()
		except AttributeError:
			pass

		raise TypeError("Could not get MObject from {}".format((node, type(node))))

	# endregion

	def __call__(cls, node:T.Union[str, om.MObject, WN], **kwargs)->WN:
		"""filter arguments to correct MObject,
		check if a node already exists for it,
		if so return that node

		create node wrapper - from a specific subclass if defined,
		else normal EdNode
		initialise that instance with MObject,
		add it to register,
		return it

		simple
		"""

		# filter input to MObject
		if isinstance(node, WN):
			return node

		mobj = filterToMObject(node)

		# check if MObject is known
		if mobj in WNMeta.objMap:
			# return node object associated with this MObject
			return WNMeta.objMap[mobj]

		# get specialised WNode subclass if it exists
		wrapCls = WNMeta.wrapperClassForMObject(mobj)

		# create instance
		ins = super(WNMeta, wrapCls).__call__(mobj, **kwargs)
		# add to MObject register
		WNMeta.objMap[mobj] = ins

		return ins


"""
WN("transform1") # this should ERROR if transform1 doesn't exist
WN.Transform("transform1") # this will create a new node
"""


class Catalogue:
	pass
if T.TYPE_CHECKING:
	from .author import Catalogue


# i've got no strings, so i have fn
class WN( # short for WePresentNode
	Catalogue,
	StringLike,
         NodeBase,
         #Composite,
         # metaclass=Singleton,
         CallbackOwner,
         metaclass=WNMeta
         ):
	# DON'T LOSE YOUR WAAAAY
	"""
	Base class for python node wrapper - we work entirely from MObjects and api
	objects, and use the tree library for data storage and manipulation.

	Node can be passed directly to the wrapped versions of cmds and OpenMaya

	Don't store any state in this python object for your sanity
	"""

	# type constant for link to api for specific subclasses
	clsApiType : int = None

	# TODO: put specific MFn types here in generated classes
	MFnCls = om.MFnDagNode


	NODE_DATA_ATTR = "_nodeAuxData"
	NODE_PROXY_ATTR = "_proxyDrivers"

	inheritStrMethods = True # might have to deactivate this if it clashes with nodeFn

	# enums
	GraphTraversal = GraphTraversal
	GraphLevel = GraphLevel
	GraphDirection = GraphDirection

	nodeArgT: (str, om.MObject, WN) # maybe move this higher

	def plug(self, lookup)->Plug:
		"""return plugtree directly from lookup
		returns None if no plug found"""
		raise NotImplementedError
		if lookup not in self._namePlugMap:
			try:
				mplug = self.MFn.findPlug(lookup, False)
			except RuntimeError: # invalid plug name
				return None
			plugTree = PlugTree(mplug)
			self._namePlugMap[lookup] = plugTree
		return self._namePlugMap[lookup]
	#endregion


	def __init__(self, node:om.MObject, **kwargs):
		"""init here is never called directly, always filtered to an MObject
		through metaclass
		we save only an MObjectHandle on the node to avoid
		holding MObject python wrappers longer than the c++ objects exist
		"""
		CallbackOwner.__init__(self)

		assert not node.isNull(), f"invalid MObject passed to WN constructor for {type(self)}"

		self.MObjectHandle = om.MObjectHandle(node)

		# maybe it's ok to cache MFn?
		self._MFn = None

		# slot to hold live data tree object
		# only used for optimisation, don't rely on it
		self._liveDataTree = None

	@property
	def MObject(self)->om.MObject:
		"""return MObject from handle"""
		return self.MObjectHandle.object()


	def object(self, checkValid=False)->om.MObject:
		"""same interface as MFnBase"""
		if checkValid:
			assert not self.MObject.isNull()
		return self.MObject

	def exists(self)->bool:
		"""check MObject is valid"""
		return not self.MObject.isNull()

	@property
	def dagPath(self)->om.MDagPath:
		return om.MDagPath.getAPathTo(self.MObject)

	@property
	def MFn(self)->MFnCls:
		"""return the best-matching function set """
		if self._MFn is not None:
			return self._MFn
		#mfnType = getMFnType(self.MObject)

		# TODO: rework this once we have specific MFn
		#  if issubclass(self.MFnCls, om.MFnDagNode):

		try:
			# dag node, initialise against dag path
			self._MFn = self.MFnCls(self.dagPath)
		except:
			self._MFn = self.MFnCls(self.MObject)
		return self._MFn

	# use create() to define order of arguments and process on creation,
	# WNMeta should delegate to this
	@classmethod
	def create(cls,
	           name:str="", dgMod_:om.MDGModifier=None, parent_:nodeArgT=None, **kwargs)->WN:
		"""suffix _ to avoid name clashes with kwargs
		if dgmod is passed, add actions to it - otherwise immediately
		execute"""
		opMod = dgMod_ or om.MDGModifier()
		if parent_ is not None:
			parent = filterToMObject(parent_)

	def setInitAttrs(self, ):
		"""subclasses should populate function signature
		with union of all their attrs and inherited attrs"""

		for attrName, val in locals().items():
			if attrName.startswith("_"):
				continue
			if attrName == "self":
				continue
			self(attrName).set(val)




	## refreshing mechanism
	def __str__(self):
		self.value = self.MFn.name()
		return self.value

	def name(self):
		return str(self).split("|")[-1]

	def setName(self, value):
		"""atomic setter, does not trigger checks for shapes etc"""
		self.MFn.setName(value)

	def address(self)->tuple[str]:
		"""return sequence of node names up to root """
		return self.dagPath.fullPathName().split("|")

	def stringAddress(self, joinChar="/"):
		"""allows more houdini-esque operations on nodes
		also lets us be a little less scared of duplicate leaf names in
		separate hierarchies"""
		return self.dagPath.fullPathName().replace("|", joinChar)

	def isTransform(self):
		return isinstance(self.MFn, om.MFnTransform)

	def isShape(self):
		return self.isDag() and not self.isTransform()

	def isDag(self):
		return isinstance(self.MFn, om.MFnDagNode)

	def isShapeTransform(self)->bool:
		"""return true if this is a transform directly over a shape"""
		if not self.isTransform():
			return False
		return len(self.children()) == 1 and self.children()[0].isShape()

	def isCurve(self):
		return isinstance(self.MFn, om.MFnNurbsCurve)
	def isMesh(self):
		return isinstance(self.MFn, om.MFnMesh)
	def isSurface(self):
		return isinstance(self.MFn, om.MFnNurbsSurface)


	#endregion

	# region creation

	@classmethod
	def create(cls, type=None, n="", parent=None, dgMod:om.MDGModifier=None, existOk=True)->WN:
		"""any subsequent wrapper class will create its own node type
		If modifier object is passed, add operation to it but do not execute yet.
		Otherwise create and execute a separate modifier each time.
		:rtype cls"""

		# initialise wrapper on existing node
		if existOk and n:
			if cmds.objExists(n):
				return cls(n)

		nodeType = type or cls.clsApiType
		name = n or nodeType

		if dgMod:
			node = cls(dgMod.createNode(nodeType))
			dgMod.renameNode(node.MObject, name)
		else:
			node = cls(om.MFnDependencyNode().create(nodeType, name)) # cheeky

		#node.setDefaults()
		return node

	# endregion

	def getChild(self, lookup)->WN:
		return self.childMap().get(lookup)


	# endregion

	def __call__(self, *args, **kwargs)-> PlugTree:
		"""may allow calling node to look up both plugs and child nodes -
		we are unlikely to ever have collisions between node and plug names"""
		if not args and not kwargs: # raw call of node()
			return self

		tokens = attr.splitPlugTokens(args)
		# try to return plug
		plug = self.plug(tokens[0])
		if plug:
			return plug(tokens[1:])

		# if no plug found, return child node
		childNode = self.getChild(tokens[0])
		return childNode(tokens[1:])

	# region convenience auxProperties
	@property
	def shapes(self)->tuple[WN]:
		if self.isShape() or not self.isDag():
			return ()
		return tuple(map(WN, cmds.listRelatives(
			self,
			s=1) or ()))

	@property
	def shape(self)->WN:
		return firstOrNone(self.shapes or ())
		return next(iter(self.shapes), None)

	@property
	def transform(self)->WN:
		if not self.isDag():
			return None
		if self.isTransform():
			return self
		return self.parent
	#endregion

	# region hierarchy
	def parent(self)->WN:
		return self.getParent()
	#endregion


	# region visibility
	def hide(self):
		#cmds.hide(self())
		self.transform("visibility").set(0)

	def show(self):
		self.transform("visibility").set(1)
	#endregion

	# region namespace stuff
	def namespace(self)->str:
		"""returns only flat namespace string -
		namespace prefixed with :
		'a:b:c:node' -> ':a:b:c'
		"""
		return om.MNamespace.getNamespaceFromName(self.name)

	def namespaceBranch(self)->NamespaceTree:
		"""return branch of namespace tree"""
		return getNamespaceTree()(list(filter(None, self.namespace().split(":"))))

	def setNamespace(self, namespace:(str, NamespaceTree)):
		# convert to tree for easier processing
		if isinstance(namespace, str):
			namespace : NamespaceTree = getNamespaceTree()(namespace, create=True)
		namespace.ensureExists()
	#endregion


	def TRS(self, *args):
		"""returns unrolled transform attrs
		args is any combination of t, r, s, x, y, z
		will return product of each side"""
		mapping = {"T" : "translate", "R" : "rotate", "S" : "scale"}
		if not args:
			args = ["T", "R", "S", "X", "Y", "Z"]
		elif isinstance(args[0], str):
			args = [i for i in args[0]]

		args = [i.upper() for i in args]
		attrs = [mapping[i] for i in "TRS" if i in args]
		dims = [i for i in "XYZ" if i in args]

		plugs = []
		for i in attrs:
			for n in dims:
				plugs.append(self+"."+i+n)
		return plugs

	# region history and future traversal
	def history(self,
	            mfnType=om.MFn.kInvalid,
	            traversal=GraphTraversal.DepthFirst,
	            #graphLevel=GraphLevel.NodeLevel
	            )->WN:
		"""create MItDependencyGraph rooted on this node
		node history gives nodes, plug history gives plugs
		"""
		it = om.MItDependencyGraph(
			self.MObject,
			mfnType,
			self.GraphDirection.History.value,
			traversal.value,
			self.GraphLevel.NodeLevel.value
		)

		return list(map(WN, [i.currentNode() for i in it]))

	def future(self,
	            mfnType=om.MFn.kInvalid,
	            traversal=GraphTraversal.DepthFirst,
	            #graphLevel=GraphLevel.NodeLevel
	            )->WN:
		"""create MItDependencyGraph rooted on this node
		node history gives nodes, plug history gives plugs"""
		it = om.MItDependencyGraph(
			self.MObject,
			mfnType,
			self.GraphDirection.History.value,
			self.GraphLevel.NodeLevel.value

		)
		return  list(map(WN, [i.currentNode() for i in it]))


	#endregion



	# region node data
	def hasAuxData(self):
		return self.plug(self.NODE_DATA_ATTR) is not None
		#return self.NODE_DATA_ATTR in self.attrs()
	def addAuxDataAttr(self):
		#self.addAttr(keyable=False, ln=self.NODE_DATA_ATTR, dt="string")
		spec = self.AttrSpec(name=self.NODE_DATA_ATTR)
		spec.data = self.AttrData(self.AttrType.String)
		self.addAttrFromSpec(spec)
		self(self.NODE_DATA_ATTR).set("{}")
		#self.set(self.NODE_DATA_ATTR, "{}")

	def auxDataPlug(self)->PlugTree:
		return self.plug(self.NODE_DATA_ATTR)
	def getAuxData(self)->dict:
		""" returns dict from node data"""
		if not self.hasAuxData():
			return {}
		data = self(self.NODE_DATA_ATTR).get()
		return ast.literal_eval(data)
	def setAuxData(self, dataDict):
		""" serialise given dictionary to string attribute ._nodeData """
		if not self.hasAuxData():
			self.addAuxDataAttr()
		self(self.NODE_DATA_ATTR).set(str(dataDict))

	@classmethod
	def templateAuxTree(cls)->Tree:
		return Tree("root")

	def getAuxTree(self)->Tree:
		""" initialise data tree object and return it.
		connect value changed signal to serialise method.
		"""
		if self._liveDataTree:
			return self._liveDataTree
		elif self.getAuxData():
			tree = Tree.deserialise(self.getAuxData())
			self._liveDataTree = tree
			return self._liveDataTree
		self._liveDataTree = self.templateAuxTree()

		# don't mess around with automatic signals for now
		return self._liveDataTree

	def saveAuxTree(self, tree=None):
		if not tree and (self._liveDataTree is None):
			raise RuntimeError("no tree passed or already created to save")
		tree = tree or self._liveDataTree
		self.setAuxData(tree.serialise())

	def setDefaults(self):
		"""called when node is created"""
		if self.defaultAttrs:
			#attr.setAttrsFromDict(self.defaultAttrs, self)
			for k, v in self.defaultAttrs.items():
				self(k).set(v)



	#endregion

