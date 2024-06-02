
from __future__ import annotations
import typing as T
import ast

# tree libs for core behaviour
from wptree import Tree
from wplib.object import Signal
from wplib.inheritance import iterSubClasses
from wplib.string import camelJoin
from wplib.object import UnHashableDict, StringLike
#from tree.lib.treecomponent import TreeBranchLookupComponent
from wplib.sequence import toSeq, firstOrNone

# maya infrastructure
from wpm.constant import GraphTraversal, GraphDirection, GraphLevel, WPM_ROOT_PATH
from ..bases import NodeBase
from ..callbackowner import CallbackOwner
from ..patch import cmds, om
from ..api import getMObject, getMFn, asMFn, getMFnType
from .. import api, attr

from ..plug import PlugTree
from ..namespace import NamespaceTree, getNamespaceTree

# dynamic type lookups
from . import codegen

# NO TRUE IMPORTS for codegen submodules
if T.TYPE_CHECKING:
	from ..api import MFMFnT
	from .codegen.gen import Catalogue as GenCatalogue
	from .codegen.modified import Catalogue as ModCatalogue

"""

maya node connection syntax fully conformed to tree syntax
but not actually inheriting from tree


test more descriptive error types for things that can go wrong
when creating nodes, plugs etc

LookupError : string not found in scene - name of node, plug etc
ReferenceError : MObject is null


"""

class Plug:
	"""base class for plugs"""

	if T.TYPE_CHECKING:

		MFn : MFMFnT

		def __getitem__(self, item)->Plug:
			"""return child plug from string
			type-checking specialcase -

			to keep type-hinting working, we need to
			return parent plug for arrays,
			to allow string slicing, etc
			"""
			return self

	def plug(self, s:str)->Plug:
		"""return child plug from string"""

class FloatPlug(Plug):
	"""object to represent a float plug"""
	MFn : om.MFnNumericAttribute

class IntPlug(Plug):
	"""object to represent an int plug"""
	MFn : om.MFnNumericAttribute

class BoolPlug(Plug):
	"""object to represent a boolean plug"""
	MFn : om.MFnNumericAttribute

class StringPlug(Plug):
	"""object to represent a string plug"""
	MFn : om.MFnTypedAttribute

class EnumPlug(Plug):
	"""object to represent an enum plug"""
	MFn : om.MFnEnumAttribute

class MessagePlug(Plug):
	"""object to represent a message plug"""
	MFn : om.MFnMessageAttribute

# complex datatypes
class MatrixPlug(Plug):
	"""object to represent a matrix plug"""
	MFn : om.MFnMatrixAttribute
	MFnData : om.MFnMatrixData



class PlugSlice:
	"""object to represent a slice of a plug tree
	"""

class PlugDescriptor:
	"""descriptor for plugs -
	declare whole attr hierarchy in one go"""
	def __init__(self, name:str):
		self.name = name

	# TEMP get and set
	def __get__(self, instance:(Plug, WN), owner)->Plug:
		return instance.plug(self.name)
	# # TEMP
	def __set__(self, instance, value):
		instance.plug(self.name).set(value)


# nodes

def filterToMObject(node):
	"""filter input to MObject"""
	if isinstance(node, str):
		mobj = getMObject(node)
		if mobj is None:
			raise LookupError("No MObject found for {}".format(mobj))
		return mobj
	elif isinstance(node, om.MObject):
		mobj = node
		if mobj.isNull():
			raise ReferenceError("MObject {} is null".format(mobj))
		return mobj

	# filter input to MObject
	if hasattr(node, "MObject"):
		return node.MObject

	raise TypeError("Could not get MObject from {}".format((node, type(node))))
	pass


def nodeClassNameFromApiStr(apiTypeStr:str)->str:
	"""return node class from api type string"""
	#return apiTypeStr.split("k")[-1]
	return apiTypeStr[1:]

class NodeMeta(type):
	"""enforce a single node wrapper per MObject -
	we also need to support kwargs to init, to let nodes act as functions"""
	objMap = UnHashableDict()

	@staticmethod
	def wrapperClassForNodeType(nodeType:str)->T.Type[WN]:
		"""return a wrapper class for the given node's type
		if it exists"""
		return WN.nodeTypeClassMap().get(nodeType, WN)

	@staticmethod
	def wrapperClassForMObject(mobj:om.MObject):
		"""return a wrapper class for the given mobject
		bit more involved if we don't know the string type
		"""
		return WN.apiTypeClassMap().get(mobj.apiType(), WN)

	def _lookupNodeClass(cls, nodePyClassName:str)->type[WN]:
		"""return a class for a specific Python class name
		 ie Transform, Mesh, etc
		 """
		# get module
		moduleName = nodePyClassName.lower()
		module = getattr(codegen, moduleName)
		# get class
		return getattr(module, nodePyClassName)

	def __getattr__(self, item:str):
		"""return a class for a specific node type"""
		#look for WN.Transform, WN.Mesh etc
		if item[0].isupper():
			return self._lookupNodeClass(item)
		raise AttributeError(f"no attribute {item}")


	def __call__(cls, node:T.Union[str, om.MObject, WN])->WN:
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
		if mobj in NodeMeta.objMap:
			# return node object associated with this MObject
			return NodeMeta.objMap[mobj]

		# get specialised WNode subclass if it exists
		wrapCls = NodeMeta.wrapperClassForMObject(mobj)

		# create instance
		ins = super(NodeMeta, wrapCls).__call__(mobj)
		# add to MObject register
		NodeMeta.objMap[mobj] = ins

		"""
			# avoid annoying material errors
	if nodeType == "mesh" or nodeType == "nurbsSurface":
		edNode.assignMaterial("lambert1")

	if edNode.isShape() and returnTransform:
		# set naming correctly
		edNode.parent.rename(name)
		return edNode.parent
		
		"""

		return ins


"""
WN("transform1") # this should ERROR if transform1 doesn't exist
WN.Transform("transform1") # this will create a new node
"""



def WNCall(mobj:om.MObject)->WN:
	"""init function called on
	WN("transform1") # this should ERROR if transform1 doesn't exist
	handle node lookup and dispatch to retrieve the right wrapper class,
	if it hasn't been loaded yet

	handled by WN metaclass
	"""
	# turn into MObject, raise error if not found
	mobj = filterToMObject(mobj)
	testCls = WN._loadedClsForMObject(mobj)
	if testCls:
		return testCls(mobj)
	# get the right class name for this MObject type
	classTypeName = wnTypeNameForMObject(mobj)
	# load the class if not already, cache against obj type, and return it
	cls = loadWNClass(classTypeName)
	return cls(mobj)



def nodeTypeCall(cls:type[WN], n="", parent=None,
             **kwargs):
	"""init function called on
	WN.Transform("transform1", tx=1, ty="someOtherPlug) # this will create a new node

	handled by WN metaclass
	"""

	newObj : WN = cls()

	if parent is not None:
		assert newObj.isDag()
		newObj.setParent(parent)

	for plugName, val in kwargs.items(): #type: str, (Plug, Literal)
		newObj(plugName).set(val)



"""arrange classes so type hinting thinks we have attributes
for all derived classes : 
	WN.Transform, WN.Mesh, etc
in reality this is done dynamically at runtime.

If this was done for real, the inheritance chain would eat
itself, since all the derived nodes inherit from WN to start with
"""

WNBase = object
if T.TYPE_CHECKING:
	class WNBase(GenCatalogue, ModCatalogue):
		pass


# i've got no strings, so i have fn
class WN( # short for WePresentNode
	WNBase,
	StringLike,
         NodeBase,
         #Composite,
         # metaclass=Singleton,
         CallbackOwner,
         metaclass=NodeMeta
         ):
	# DON'T LOSE YOUR WAAAAY
	"""for once we don't actually use tree

	Base class for python node wrapper - we work entirely from MObjects and api
	objects, and use the tree library for data storage and manipulation.

	Node can be passed directly to the wrapped versions of cmds and OpenMaya
	"""

	# type constant for link to api for specific subclasses
	clsApiType : int = None

	NODE_DATA_ATTR = "_nodeAuxData"
	NODE_PROXY_ATTR = "_proxyDrivers"

	inheritStrMethods = True # might have to deactivate this if it clashes with nodeFn

	# enums
	GraphTraversal = GraphTraversal
	GraphLevel = GraphLevel
	GraphDirection = GraphDirection

	# region type integration
	# def __getattr__(self, item:str):
	# 	"""return plugtree directly from lookup
	# 	returns None if no plug found"""
	# 	plug = self.plug(item)
	# 	if plug:
	# 		return plug
	# 	# if no plug found, return child node
	# 	childNode = self.getChild(item)
	# 	if childNode:
	# 		return childNode
	# 	raise AttributeError(f"no plug or child node found for {item}")


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


	def __init__(self, node:T.Union[str, om.MObject] = None, **kwargs):
		"""init here is never called directly, always filtered to an MObject
		through metaclass
		node should always be filtered to an MObject by the metaclass
		"""
		#Composite.__init__(self)
		CallbackOwner.__init__(self)
		self.value = ""
		self.MObject = None
		self._MFn = None

		self.setMObject(mobj)

		# add plugs for lookup
		self._namePlugMap = {}

		# slot to hold live data tree object
		self._liveDataTree = None


	def object(self)->om.MObject:
		"""same interface as MFnBase"""
		return self.MObject

	def exists(self)->bool:
		"""check MObject is valid"""
		return not self.MObject.isNull()

	@property
	def dagPath(self)->om.MDagPath:
		return om.MDagPath.getAPathTo(self.MObject)

	# region core typing systems
	def setMObject(self, obj):
		self.MObject = obj # not weakref-able :(


	@property
	def MFn(self)->MFMFnT:
		"""return the best-matching function set """
		if self._MFn is not None:
			return self._MFn
		mfnType = getMFnType(self.MObject)
		if issubclass(mfnType, om.MFnDagNode):
			# dag node, initialise against dag path
			self._MFn = mfnType(self.dagPath)
		else:
			self._MFn = mfnType(self.MObject)
		return self._MFn

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

