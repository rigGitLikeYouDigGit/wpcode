
from __future__ import annotations

import fnmatch
import time
import traceback
import typing as T, types
import ast
from typing import TypedDict, NamedTuple
from collections import namedtuple

import numpy as np

# tree libs for core behaviour
#from wptree import Tree
from wplib import Sentinel, log, wpstring
from wplib.object import Signal, Adaptor
from wplib.inheritance import iterSubClasses
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

from ..plugtree import Plug, PlugDescriptor
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


class DGDagModifier(om.MDagModifier):
	"""ಠ_ಠ
	this actually works quite well
	"""
	def __init__(self):
		super().__init__()
		self._dgMod = om.MDGModifier()
	def createNode(self, nodeTypeNameOrId, parent=om.MObject.kNullObj):
		baseExc = None
		try:
			return super().createNode(nodeTypeNameOrId, parent)
		except TypeError:
			pass
		except Exception as e:
			baseExc = e
		try:
			return self._dgMod.createNode(nodeTypeNameOrId)
		except:
			log("proper type error raised when trying to create with combined modifiers")
			if baseExc:
				log("base error:")
				traceback.print_exception(baseExc)
			raise
	def deleteNode(self, node:om.MObject):
		super().deleteNode(node, False) # do NOT include empty dag parents by default

	def doIt(self, *args, **kwargs):
		super().doIt()
		self._dgMod.doIt()
	def undoIt(self, *args, **kwargs):
		super().undoIt()
		self._dgMod.undoIt()





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

	@classmethod
	def wrapperClassForNodeType(cls, nodeType: (str, om.MTypeId)) -> T.Type[WN]:
		"""return a wrapper class for the given node's type
		if it exists"""
		nc = om.MNodeClass(nodeType)
		if nc.typeId.id() == 0:
			raise TypeError("No known node class matches: " + nodeType)
		if result := WN.typeIdIntWNClassMap.get(nc.typeId.id()):
			return result
		if result := WN.nodeTypeNameWNClassMap.get(nc.typeName):
			return result
		# not already cached, go looking with retriever
		try:
			return cls.retriever.getNodeCls(nc.typeName)
		except:
			return WN

	@classmethod
	def wrapperClassForMObject(cls, mobj:om.MObject):
		"""return a wrapper class for the given mobject
		bit more involved if we don't know the string type
		"""

		mfn = om.MFnDependencyNode(mobj)
		if result := WN.typeIdIntWNClassMap.get(mfn.typeId.id()):
			return result
		apiType = mobj.apiType()
		if result := WN.nodeTypeNameWNClassMap.get(apiType):
			return result

		className = mfn.typeName # still works properly for plugin nodes
		try:
			return cls.retriever.getNodeCls(className) or WN
		except:
			log("error getting node class from retriever - looked for:", className)
			raise

		try:
			return cls.retriever.getNodeCls(className)
		except:
			return WN
		#return WN.apiTypeWNClassMap.get(mobj.apiType(), WN)
		#return api.getCache().api.apiTypeClassMap().get(mobj.apiType(), WN)
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

	def __call__(cls,
	             node:(str, om.MObject, WN),
	             new=None,
	             parent:(str, om.MObject, WN)=None,
	             **kwargs)->WN:
		"""filter arguments to correct MObject,
		check if a node already exists for it,
		if so return that node

		if node is a string name:
			if new is None:
				if no node with given name found:
					create a new node under parent
				else:
					return the existing node
			if new is False:
				get existing node, error if not found
			if new is True:
				create new node under parent, increment name if necessary

		create node wrapper - from a specific subclass if defined,
		else normal EdNode
		initialise that instance with MObject,
		add it to register,
		return it

		simple
		"""
		#log("WNMeta _call_", node, new, kwargs)
		# filter input to MObject
		if isinstance(node, WN):
			return node

		if new:
			return WN.create(node, parent_=parent, **kwargs)

		mobj = filterToMObject(node)

		# check if MObject is known
		if mobj in WNMeta.objMap:
			#log("mobj known")
			# return node object associated with this MObject
			return WNMeta.objMap[mobj]

		# get specialised WNode subclass if it exists
		wrapCls = WNMeta.wrapperClassForMObject(mobj)
		#log("wrapCls", wrapCls)

		# create instance
		ins = super(WNMeta, wrapCls).__call__(mobj, new=new, **kwargs)
		# add to MObject register
		WNMeta.objMap[mobj] = ins

		return ins



class Catalogue:
	pass
if T.TYPE_CHECKING:
	from .author import Catalogue


class FilterSequence(list):
	"""return an ordered sequence allowing wildcard matching and filtering

	"""
	def __init__(self, it, getKeyFn=None):
		super().__init__(it)
		self.getKeyFn = getKeyFn

	# def _cloneFromOther(self, other:FilterSequence):
	# 	super().__init__()
	def _copy(self, it):
		return type(self)(it, getKeyFn=self.getKeyFn)

	def filter(self, pattern):
		return self._copy(
			fnmatch.filter(map(self.getKeyFn, self), pattern)
		)
	def __getitem__(self, item):
		if isinstance(item, str):
			return self.filter(item)
		return super().__getitem__(item)

	def __sub__(self, other):
		""" set subtraction """
		otherSet = set(other)
		return self._copy(
			i for i in self if not i in otherSet
		)
	def __mul__(self, other):
		""" set intersection """
		otherSet = set(other)
		return self._copy(
			i for i in self if i in otherSet
		)

	def __add__(self, other):
		""" set union """
		thisSet = set(self)
		new = self._copy(self)
		new.extend(i for i in other if not i in thisSet)
		return new


# i've got no strings, so i have fn
class WN( # short for WePresentNode
	Catalogue,
	StringLike,
         NodeBase,
         #Composite,
         CallbackOwner,
         metaclass=WNMeta
         ):
	# DON'T LOSE YOUR WAAAAY
	"""
	Base class for python node wrapper - we work entirely from MObjects and api
	objects, and use the tree library for data storage and manipulation.

	Node can be passed directly to the wrapped versions of cmds and OpenMaya

	Don't store any state in this python object for your sanity


	myNode = WN()

	myNode("child") -> return a direct child called
	myNode["plug*End"] -> return plugs of this node matching pattern
	myNode.translateX_ -> return single plug


	"""

	# attributes shared by generated classes
	"""
	typeName = "transform"
	apiTypeInt = 110
	apiTypeStr = "kTransform"
	MFnCls = om.MFnTransform """
	typeName = None
	apiTypeInt = None
	apiTypeStr = None
	typeIdInt : int = None # master, if node class has a typeId, obviously use that
	@classmethod
	def typeId(cls)->om.MTypeId:
		return om.MTypeId(cls.typeIdInt) if cls.typeIdInt is not None else None
	MFnCls = om.MFnDependencyNode

	clsIsDag = False

	nodeLeafClassAttrs : list[str] = []
	nodeLeafPlugs : list[str] = []


	apiTypeWNClassMap : dict[int, type[WN]] = {}
	nodeTypeNameWNClassMap : dict[str, type[WN]] = {} # replace with MNodeClass
	typeIdIntWNClassMap : dict[int, type[WN]] = {}
	def __init_subclass__(cls, **kwargs):
		"""happily, if you subclass a generated Node class and
		don't redefine any of the type constants here, your subclass
		will still be preferred for that type.

		And I *think* it's guaranteed that only leaf / real node classes
		will even have MTypeIds anyway, so we shouldn't need to worry
		about intermediate parent classes taking priority when they're defined.
		"""
		if cls.apiTypeInt is not None:
			cls.apiTypeWNClassMap[cls.apiTypeInt] = cls
			#cls.nodeTypeNameWNClassMap[cls.__name__] = cls
		if cls.typeIdInt is not None:
			cls.typeIdIntWNClassMap[cls.typeIdInt] = cls

		if cls.typeName is not None:
			cls.nodeTypeNameWNClassMap[cls.typeName] = cls

	NODE_DATA_ATTR = "_nodeAuxData"
	NODE_PROXY_ATTR = "_proxyDrivers"

	inheritStrMethods = True # might have to deactivate this if it clashes with nodeFn

	# enums
	GraphTraversal = GraphTraversal
	GraphLevel = GraphLevel
	GraphDirection = GraphDirection

	nodeArgT: (str, om.MObject, WN) # maybe move this higher

	#endregion

	@classmethod
	def wrapperExistsForNodeType(cls, nodeType:(int, str)):
		if(isinstance(nodeType, str)):
			return
	def __init__(self, node:om.MObject, **kwargs):
		"""init here is never called directly, always filtered to an MObject
		through metaclass
		we save only an MObjectHandle on the node to avoid
		holding MObject python wrappers longer than the c++ objects exist
		"""
		CallbackOwner.__init__(self)

		assert not node.isNull(), f"invalid MObject passed to WN constructor for {type(self)}"

		self.MObjectHandle = om.MObjectHandle(node)
		self._mobj = node

		# slot to hold live data tree object
		# only used for optimisation, don't rely on it
		self._liveDataTree = None

	# @property
	# def MObject(self)->om.MObject:
	# 	"""return MObject from handle"""
	# 	return self.object()Handle.object()
	# 	#return self._mobj


	def object(self, checkValid=False)->om.MObject:
		"""same interface as MFnBase"""
		# if checkValid:
		# 	assert not self.MObjectHandle.object().isNull()
		# return self.MObjectHandle.object()
		return self._mobj

	def exists(self)->bool:
		"""check MObject is valid"""
		return not self.object().isNull()

	def dagPath(self)->om.MDagPath:
		return om.MDagPath.getAPathTo(self.object())

	@property
	def MFn(self)->MFnCls:
		"""return the best-matching function set """
		if self.isDag():
			mfn = self.MFnCls(self.dagPath())
		else:
			mfn = self.MFnCls(self.object())
		mfn.name() # if this errors we are brusquely screwed
		# sometimes gives RuntimeError: (kFailure): Object does not exist
		return mfn
		#import maya.api.OpenMaya as om
		baseFn = om.MFnDependencyNode(self.object())
		baseFn.name()
		print("MFN for", baseFn.name(), baseFn.typeName, self.MFnCls)
		assert not self.object().isNull()
		if(self.isDag()):
			dp : om.MDagPath = om.MDagPath.getAPathTo(self.object())
			assert dp.isValid()
			print("dp", dp, dp.fullPathName(), dp.isValid())

			dpFn = om.MFnDagNode(dp)
			#dpFn.name()
			print("dpFn path", dpFn.dagPath())

			sel = om.MSelectionList()
			sel.add(dp)
			newDp = sel.getDagPath(0)
			assert newDp.isValid()

			nurbsCurveFn = om.MFnNurbsCurve(newDp)
			#nurbsCurveFn = om.MFnNurbsCurve(dp.getPath(0))
			nurbsCurveFn.name()


			nurbsCurveFn = om.MFnNurbsCurve(dp)
			#nurbsCurveFn = om.MFnNurbsCurve(dp.getPath(0))
			nurbsCurveFn.name()

			pathObj : om.MObject = dp.node()
			assert not pathObj.isNull()
			#return self.MFnCls(dp)

			objFn : om.MFnDagNode = self.MFnCls(pathObj)
			assert not pathObj.isNull()
			pathFn : om.MFnDagNode = self.MFnCls(dp)
			print("pathFn", pathFn)
			print("pathFn obj", pathFn.object())
			print("pathFn obj isNull", pathFn.object().isNull())
			print("pathFn getPath", pathFn.getPath())

			print("pathFn path", pathFn.dagPath())


			assert pathFn.dagPath().isValid()
			assert pathFn.dagPath() == dp


			pathFn.name()
			#assert pathObj == objFn.object()
			assert not objFn.object().isNull()
			objFn.name()
			objFn.setObject(dp)
			objFn.name()
			objFn.dagPath()
			# mfn : om.MFnDagNode = self.MFnCls(dp)
			# mfn.dagPath()
			# mfn.name()

			# print("mfnCls", self.MFnCls)
			# print("dp", mfn.dagPath())
			# print(mfn.dagPath().isValid())
			# assert not mfn.object().isNull()
		return self.MFnCls(self.object())

		# if self._MFn is not None:
		# 	return self._MFn
		# #mfnType = getMFnType(self.object())
		#
		# # TODO: rework this once we have specific MFn
		# #  if issubclass(self.MFnCls, om.MFnDagNode):
		#
		# try:
		# 	# dag node, initialise against dag path
		# 	self._MFn = self.MFnCls(self.dagPath)
		# except:
		# 	self._MFn = self.MFnCls(self.object())
		#
		# assert not self._MFn.object().isNull()
		# return self._MFn

	def output(self)->(None, Plug):
		"""return a plug if this node can be used in atomic operations -
		for example, ".output" from addDoubleLinear.
		For more complex nodes, for now leave null and force explicit retrieval
		"""
		return None

	# geometry node creation - basically ONLY for curve and mesh nodes
	""" something silly happens if you create these from scratch with MDagModifier - 
	the data objects never get created properly, so MFnNurbsCurve( myNewCurveShapeNode )
	fails with 'Object does not exist' , and won't let you do any modifications to it
	"""

	def getGeoDataParentObject(self)->om.MObject:
		"""return an MObject to use as parent for directly
		setting geometry data, through MFnNurbsCurve.create()
		"""
		raise TypeError("no data parent object for node type: " + self.typeName)

	def setGeoDataObject(self, dataObj:om.MObject):
		"""directly set geo data through MObject -
		use for structural changes like setting new CV number
		"""
		raise TypeError("no geo data object for node type: " + self.typeName)

	@staticmethod
	def _validateCreatedWrapper(obj:om.MObject, wrapper:WN)->WN:
		"""TRAUMA TIME:
		I was getting inconsistent and rarely reproducible problems where
		the MObject handle for newly created nodes would return a null MObject, after
		being set on the new one -
		the created MObject is evidently fine, it's just the handle that gives the issue.

		so here, we ask calmly, "did I stutter?"
		if the object handle isn't valid,
		then we just get a new object handle.

		STILL not convinced there's actually any value to the handle indirection,
		how bad can it be to just save references to MObjects? all my friends are doing it.
		"""
		limit = 10
		i = 0
		while wrapper.object().isNull():
			if i > limit:
				raise RuntimeError("invalid wrapper")
			#log("invalid wrapper found, resetting MObjectHandle")
			wrapper.MObjectHandle = om.MObjectHandle(obj)
			#time.sleep(0.1)
		return wrapper
		isValid = False
		hdl = om.MObjectHandle(obj)
		hdl.object()
		return hdl
		while not isValid:
			objValid = om.MObjectHandle(obj).isValid()
			objAlive = om.MObjectHandle(obj).isAlive()
			if not (objValid and objAlive):
				log("mobj not valid or alive - valid:", objValid, "alive:", objAlive )
				time.sleep(1)
				continue
			return om.MObjectHandle(obj)




	# use create() to define order of arguments and process on creation,
	# WNMeta should delegate to this
	@classmethod
	def create(cls,
	           name:str="", dgMod_:DGDagModifier=None, parent_:nodeArgT=None, **kwargs)->WN:
		"""
		explicitly create a new node of this type, incrementing name if necessary

		suffix _ to avoid name clashes with kwargs

		leaving this as the single method to override for specific creation behaviour,
		without a deeper "_createInner()" method to actually create the object -
		will result in some copying of the argument processing here,
		but I find copying can sometimes be comforting for work code
		"""
		# creating from raw WN class, default to Transform
		if cls is WN:
			nodeCls = WN.Transform
		else:
			nodeCls = cls
		if parent_ is not None:
			parent_ = filterToMObject(parent_)


		opMod = dgMod_ or DGDagModifier()
		name = name or nodeCls.typeName + str(1)
		typeId = nodeCls.typeId()
		newObj = opMod.createNode(
			typeId,
			parent_ or om.MObject.kNullObj)
		# returns the created transform node, if you create a shape
		# always execute the modifier for now, too complicated and not useful otherwise
		opMod.doIt()

		#hdl = cls._validateCreatedMObject(newObj)
		wrapper = nodeCls(newObj, **kwargs)
		cls._validateCreatedWrapper(newObj, wrapper) # basically sometimes an object can be valid, while its handle isn't?
		wrapper.setName(name)
		return wrapper




	@classmethod
	def createNode(cls,
	           nodeType:(str, om.MTypeId), name:str="", dgMod_:DGDagModifier=None, parent_:nodeArgT=None, **kwargs)->WN:
		"""
		explicitly create a new node of the given type type, incrementing name if necessary

		suffix _ to avoid name clashes with kwargs
		if dgmod is passed, add actions to it - otherwise immediately
		execute

		OK important technology:
			DGModifier raises TypeError if you pass it any dag node type
			DagModifier raises TypeError if you pass it a non-dag node type
		so there isn't any good way to only use one or the other as a default argument.
		we use the strange combined type DGDagModifier from above

		TODO: semantics around what node gets returned when you do (EG):
			WN.createNode("mesh", "myHappyMesh")
		we should return the WN-wrapped mesh shape node, named "myHappyMeshShape",
		under a new transform (under the world) named "myHappyMesh"

		check if the typeId returned by the modifier matches the node type created
		"""

		wrapperCls = cls.wrapperClassForNodeType(nodeType)

		opMod = dgMod_ or DGDagModifier()
		name = name or nodeType + str(1)

		if parent_ is not None:
			parent_ = filterToMObject(parent_)
		return wrapperCls.create(name, opMod, parent_, **kwargs)



	def setInitAttrs(self, ):
		"""subclasses should populate function signature
		with union of all their attrs and inherited attrs"""

		for attrName, val in locals().items():
			if attrName.startswith("_"):
				continue
			if attrName == "self":
				continue
			self(attrName).set(val)


	### WILDCARD FILTER SEQUENCES
	### MPLUGS

	## refreshing mechanism
	def __str__(self):
		if(self.object().isNull()):
			return str(type(self)) + "(NULL_OBJ)"
		#return self.MFn.absoluteName()
		return self.MFn.name()

	def name(self):
		return str(self).split("|")[-1]

	def setName(self, value:str, exact=False):
		"""if exact, only rename this exact node to exactly what you give;
		if not exact, do some checks to keep shapes in line with transforms
		"""
		# exact forces exact result
		# non-dags have no extra rules to keep in check
		# multiple shapes imply more exact naming
		if exact or (not self.isDag()) or (len(self.shapes()) != 1):
			self.MFn.setName(value)
			return
		if value.endswith("Shape"):
			self.shape().setName(value, exact=True)
			root, digits = wpstring.trailingDigits(self.shape().name())
			self.tf().setName(root.rsplit("Shape")[0] + digits, exact=True)
		else:
			self.tf().setName(value, exact=True)
			root, digits = wpstring.trailingDigits(self.tf().name())
			self.shape().setName(root + "Shape" + digits, exact=True)

	def address(self)->tuple[str]:
		"""return sequence of node names up to root """
		return self.dagPath().fullPathName().split("|")

	def stringAddress(self, joinChar="/"):
		"""allows more houdini-esque operations on nodes
		also lets us be a little less scared of duplicate leaf names in
		separate hierarchies"""
		return self.dagPath().fullPathName().replace("|", joinChar)

	def isTransform(self):
		#return isinstance(self.MFn, om.MFnTransform)
		return self.object().hasFn(om.MFn.kTransform)

	def isShape(self):
		#return self.isDag() and not self.isTransform()
		return self.object().hasFn(om.MFn.kShape)

	def isDag(self):
		#return isinstance(self.MFn, om.MFnDagNode)
		return self.object().hasFn(om.MFn.kDagNode)

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


	def plug(self, lookup)->Plug:
		"""return plugtree directly from lookup
		returns None if no plug found"""
		#if lookup not in self._namePlugMap:
		try:
			mplug = self.MFn.findPlug(lookup, False)
		except RuntimeError: # invalid plug name
			return None
		return Plug(mplug)
		# 	self._namePlugMap[lookup] = plugTree
		# return self._namePlugMap[lookup]

	# endregion


	def __call__(self, *args, **kwargs)->WN:# Plug:
		"""may allow calling node to look up both plugs and child nodes -
		we are unlikely to ever have collisions between node and plug names"""
		if not args and not kwargs: # raw call of node()
			return self
		# if no plug found, return child node
		tokens = plug.splitPlugTokens(args)
		childNode = self.getChild(tokens[0])
		return childNode(tokens[1:])

	def __getitem__(self, *item)->(Plug, str):
		"""look up plug with square brackets when using a string,
		otherwise delegate back to stringlike"""
		if isinstance(item[0], str):
			return self.plug(item[0])
			#todo: slicing, matching etc
			pass
		if len(item) == 1:
			return super().__getitem__(item[0])
		raise TypeError("invalid arg type to index into node")

	#def __getattribute__(self, item:str)->Plug:

	def _plugAttrLookup(self, plugAttrName:str):
		"""using a secondary method to avoid infinite loop when delegating lookups
		between shape/transform

		if not a plugAttr name, returns none
		if a plugAttr name and plug found, returns plug
		if plugAttr and not found, raises TypeError
		"""
		if (plugAttrName[-1] == "_" and plugAttrName[0] != "_"):
			if (foundPlug := self.plug(plugAttrName[:-1])) is not None:
				return foundPlug
			raise TypeError("no maya plug found for ", plugAttrName)
		return None

	def _shapeTfPlugAttrLookup(self, plugAttrName:str)->Plug:
		""" outer layer of plug lookup, wrapping tf->shape
		and shape->tf delegation logic"""
		try:
			if (foundPlug := self._plugAttrLookup(plugAttrName)) is not None:
				return foundPlug
		except TypeError as e: # first level of error, try delegating to transform/shape
			if self.isShape():
				try:
					return self.tf()._plugAttrLookup(plugAttrName)
				except TypeError:
					raise TypeError(self.pathStr + ": no maya plug found for", plugAttrName, "looked at shape, then at transform")
			if self.isTransform() and self.shape():
				try:
					return self.shape()._plugAttrLookup(plugAttrName)
				except TypeError:
					raise TypeError(self.pathStr + ": no maya plug found for", plugAttrName, "looked at transform, then at shape ", self.shape().pathStr)
			# errored once and no second node to check - just raise error
			raise e
		return None
	def __getattr__(self, item:str)->Plug:
		"""check if plug has been accessed directly by name -
		always has a trailing underscore

		We emulate the maya cmds behaviour for forgiving shape or transform
		plug lookups
		"""
		if (result := self._shapeTfPlugAttrLookup(item)) is not None:
			return result
		return super().__getattr__(item)

	""" we don't override __setattr__ anymore now that we have
	plug descriptors 
	"""

	# def __setattr__(self, item:str, val):
	# 	"""check if plug has been accessed directly by name -
	# 	always has a trailing underscore"""
	# 	if(item[-1] == "_" and item[0] != "_"):
	# 		if (foundPlug := self.plug(item[:-1])) is not None:
	# 			foundPlug.set(val)
	# 		raise TypeError("no maya plug found to set ", item)
	# 	return super().__setattr__(item, val)

	# region convenience auxProperties

	#endregion

	# region hierarchy
	def parent(self)->WN:
		"""all dag nodes in maya are at least children of the root 'world' object,
		which I don't know a good shorthand for -
		so here we test if their parents' parent has no parents
		"""
		if om.MFnDagNode(self.MFn.parent(0)).parentCount() == 0:
			return None
		return WN(self.MFn.parent(0))

	def _childTfObjects(self)->list[om.MObject]:
		return [self.MFn.child(i) for i in range(self.MFn.childCount())
		        if self.MFn.child(i).hasFn(om.MFn.kTransform)]
	def children(self)->list[WN]:
		"""could somehow make the WN wrapping lazy, but it doesn't seem
		too slow yet
		exclude shapes here"""
		return [WN(i) for i in self._childTfObjects()]
	def childMap(self)->dict[str, WN]:
		return {i.name() : i for i in self.children()}
	def child(self, lookup)->WN:
		"""TODO: maybe add the full Pathable syntax here,
		would rather save that for access()
		"""
		if isinstance(lookup, str):
			return self.childMap().get(lookup)
		if isinstance(lookup, int):
			return WN(self._childTfObjects()[lookup])

	def _shapeObjects(self)->list[om.MObject]:
		return [self.MFn.child(i) for i in range(self.MFn.childCount())
		        if not self.MFn.child(i).hasFn(om.MFn.kTransform)]
	def shapes(self)->list[WN]:
		return [WN(i) for i in self._shapeObjects()]

	def shape(self)->WN:
		if self.isShape():
			return self
		if shapes := self._shapeObjects():
			return WN(shapes[0])
		return None

	def tf(self)->WN.Transform:
		"""'transform()' sounds too much like a verb
		tf() isn't a great abbreviation, but more readable as an accessor
		"""
		if not self.isDag():
			return None
		if self.isTransform():
			return self
		return self.parent()

	# helpers for local and worldspace attributes -
	# reimplement in relevant shape classes

	@property
	def localIn(self)->Plug:
		raise AttributeError(f"Node type {type(self)} has no relevant local input")
	@property
	def localOut(self) -> Plug:
		raise AttributeError(f"Node type {type(self)} has no relevant local output")

	@property
	def worldIn(self) -> Plug:
		raise AttributeError(f"Node type {type(self)} has no relevant world input")
	@property
	def worldOut(self)-> Plug:
		raise AttributeError(f"Node type {type(self)} has no relevant world output")
	#endregion


	# region visibility
	def hide(self):
		#cmds.hide(self())
		self.tf().visibility_.set(0)

	def show(self):
		self.tf().visibility_.set(1)
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
			self.object(),
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
			self.object(),
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

	def auxDataPlug(self)->Plug:
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

	#region bases and RTTI inheritance
	@classmethod
	def nodeBaseClassWrappers(cls)->list[type[WN]]:
		"""return list of bases starting from this class,
		corresponding to node bases"""
		return [i for i in cls.__mro__ if issubclass(i. WN) and i != WN]

	@classmethod
	def nodeClassLeafAttrs(cls)->list[om.MObject]:
		"""return all attributes added by this exact node
		class"""


