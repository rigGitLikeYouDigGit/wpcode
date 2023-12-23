"""EdNode wrapper """

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
from wpm.constant import GraphTraversal, GraphDirection, GraphLevel
from ..bases import NodeBase
from ..callbackowner import CallbackOwner
from ..patch import cmds, om
from ..api import getMObject, getMFn, asMFn, getMFnType
from .. import api, attr

from ..plug import PlugTree
from ..namespace import NamespaceTree, getNamespaceTree

if T.TYPE_CHECKING:
	from ..api import MFMFnT

"""

maya node connection syntax fully conformed to tree syntax
but not actually inheriting from tree

"""


#todo: strip out like 90 percent of this stuff


class NodeMeta(type):
	"""enforce a single node wrapper per MObject"""
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

		if isinstance(node, str):
			mobj = getMObject(node)
			if mobj is None:
				raise RuntimeError("No MObject found for {}".format(mobj))
		elif isinstance(node, om.MObject):
			mobj = node
			if mobj.isNull():
				raise RuntimeError("MObject {} is null".format(mobj))

		else:
			print("mobj", node, type(node))

		# check if MObject is known
		if mobj in NodeMeta.objMap:
			# return node object associated with this MObject
			return NodeMeta.objMap[mobj]

		# get specialised EdNode subclass if it exists
		wrapCls = NodeMeta.wrapperClassForMObject(mobj)

		# create instance
		ins = super(NodeMeta, wrapCls).__call__(mobj)
		# add to MObject register
		NodeMeta.objMap[mobj] = ins
		return ins


class AttrDescriptor:
	"""small object to allow prediction on common attributes"""

	def __init__(self, attrName:str):
		self.name = attrName

	def __get__(self, instance:WN, owner)->PlugTree:
		return instance(self.name)


class WN(StringLike, # short for WePresentNode
         NodeBase,
         #Composite,
         # metaclass=Singleton,
         CallbackOwner,
         metaclass=NodeMeta
         ):
	# DON'T LOSE YOUR WAAAAY
	"""for once we don't actually use tree"""

	objMap = UnHashableDict()

	# type constant for link to api for specific subclasses
	clsApiType = None

	class Types:
		"""useful API type constants"""
		kTransform = om.MFn.kTransform
		kTransformGeometry = om.MFn.kTransformGeometry
		kSkinCluster = om.MFn.kSkinClusterFilter



	allInfo = {
		"mesh" : {
			"inShape" : "inMesh",
			"outLocal" : "outMesh",
			"outWorld" : "worldMesh[0]"
		},
		"transform" : {
			"outLocal" : "matrix",
			"outWorld" : "worldMatrix[0]"
		},
		"nurbsCurve" : {
			"outLocal" : "local",
			"outWorld" : "worldSpace[0]",
			"inShape" : "create"
		},
		"nurbsSurface": {
			"outLocal": "local",
			"outWorld": "worldSpace[0]",
			"inShape": "create"
		}
	}

	NODE_DATA_ATTR = "_nodeAuxData"
	NODE_PROXY_ATTR = "_proxyDrivers"

	# persistent dict of uid : absoluteNode, used as cache
	nodeCache = {}
	# yes I know there can be uid clashes but for now it's fine

	defaultAttrs = {}
	# override with {"nodeState" : 1} etc

	# override to explicitly set string maya nodeTypeName
	_nodeType = None

	inheritStrMethods = True

	_nameClassMap = {}
	@classmethod
	def nodeTypeClassMap(cls):
		if not cls._nameClassMap:
			cls._nameClassMap = {camelJoin(i.__name__) : i for i in iterSubClasses(cls)}
		return cls._nameClassMap

	_apiTypeClassMap = {}
	@classmethod
	def apiTypeClassMap(cls):
		if not cls._apiTypeClassMap:
			cls._nameClassMap = {i.clsApiType : i for i in iterSubClasses(cls)}
		return cls._nameClassMap

	# enums
	GraphTraversal = GraphTraversal
	GraphLevel = GraphLevel
	GraphDirection = GraphDirection


	def __init__(self, mobj:om.MObject):
		"""init here is never called directly, always filtered to an MObject
		through metaclass"""
		#Composite.__init__(self)
		CallbackOwner.__init__(self)
		self.value = ""
		self.MObject = None
		self._MFn = None

		self._isDag = False

		self.setMObject(mobj)

		# slot to hold live data tree object
		self._liveDataTree = None

		self.con = self._instanceCon
		self.conOrSet = self._instanceConOrSet
		self.nodeType = self._instanceNodeType

		# add plugs for lookup
		self._namePlugMap = {}

		# signals for callbacks
		# lazy creation for performance
		self._nodeNameChangedSignal = None
		self._nodePreRemovalSignal = None
		self._nodeAboutToDeleteSignal = None
		self._nodeDestroyedSignal = None
		# we do NOT remove signals immediately when
		# node is deleted, as it could be undone while preserving python wrapper

	def getNodeNameChangedSignal(self):
		""" creates a signal for node name changed callback and
		attaches it to the callback unit

		signature: mobject, old name, user data
		could always make this more complex, allow dynamically
		creating user data every call - not needed for now"""
		if self._nodeNameChangedSignal is None:
			self._nodeNameChangedSignal = Signal()
			self.addOwnedCallback(
				om.MNodeMessage.addNameChangedCallback(
					self.MObject, self._nodeNameChangedSignal)
			)
		return self._nodeNameChangedSignal
	def getNodePreRemovalSignal(self):
		""" fires whenever node is about to be removed, including from undos/redos
		signature : MObject, user data

		"""
		if self._nodePreRemovalSignal is None:
			self._nodePreRemovalSignal = Signal()
			self.addOwnedCallback(
				om.MNodeMessage.addNodePreRemovalCallback(
					self.MObject, self._nodePreRemovalSignal)
			)
		return self._nodePreRemovalSignal
	def getNodeAboutToDeleteSignal(self):
		""" fires whenever node is about to be delete FOR THE FIRST TIME -
		when the MDGModifier object that effects the undo operation is first created.

		While the modifier object can be undone/redone, this signal will not fire again -
		use PreRemovalSignal for that.

		Honestly if you're messing with this you should just read the Maya docs
		"""
		if self._nodeAboutToDeleteSignal is None:
			self._nodeAboutToDeleteSignal = Signal()
			self.addOwnedCallback(
				om.MNodeMessage.addNodeAboutToDeleteCallback(
					self.MObject, self._nodeAboutToDeleteSignal)
			)
		return self._nodeAboutToDeleteSignal
	def getNodeDestroyedSignal(self):
		""" called when node destructor is called -
		usually when undo queue is cleared or scene is closed"""
		if self._nodeDestroyedSignal is None:
			self._nodeDestroyedSignal = Signal()
			self.addOwnedCallback(
				om.MNodeMessage.addNodeDestroyedCallback(
					self.MObject, self._nodeDestroyedSignal)
			)
		return self._nodeDestroyedSignal



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

	# i've got no strings, so i have fn
	def isCurve(self):
		return isinstance(self.MFn, om.MFnNurbsCurve)
	def isMesh(self):
		return isinstance(self.MFn, om.MFnMesh)
	def isSurface(self):
		return isinstance(self.MFn, om.MFnNurbsSurface)


	#endregion

	# region creation
	@staticmethod
	def fromMObject(obj:om.MObject):
		"""find node associated with obj and wrap it"""
		return WN(obj)

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

	def getPlug(self, lookup)->PlugTree:
		"""return plugtree directly from lookup
		returns None if no plug found"""
		if lookup not in self._namePlugMap:
			try:
				mplug = self.MFn.findPlug(lookup, False)
			except RuntimeError: # invalid plug name
				return None
			plugTree = PlugTree(mplug)
			self._namePlugMap[lookup] = plugTree
		return self._namePlugMap[lookup]

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
		plug = self.getPlug(tokens[0])
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

	def getParent(self)->WN:
		if not self.MFn.parentCount():
			return None
		obj = self.MFn.parent(0)
		# if parent is world, return None
		if obj.isNull():
			return None
		return WN(obj)
	@property
	def parent(self)->WN:
		return self.getParent()

	def children(self)->T.List[WN]:
		"""return only transforms - use shapes for shapes"""
		if not self.isTransform():
			return []
		return list(
			map(
				WN, filter(
					lambda x: om.MFnTransform().hasObj(x),
					(self.MFn.child(i) for i in range(self.MFn.childCount())
					 )	)			)		)

	def allDescendants(self, includeSelf=False)->T.List[WN]:
		"""return all descendents, depth first
		not matching listRelatives, its order is a bit weird"""
		descendents = [self] if includeSelf else []
		for child in self.children():
			descendents.extend(child.allDescendants(includeSelf=True))
		return descendents


	def childMap(self)->T.Dict[str, WN]:
		return {c.name: c for c in self.children()}


	# convenient plug attributes - no shorthand for now
	translate = AttrDescriptor("translate")
	rotate = AttrDescriptor("rotate")
	scale = AttrDescriptor("scale")
	visibility = AttrDescriptor("visibility")
	matrix = AttrDescriptor("matrix")
	localMatrix = AttrDescriptor("matrix")
	inverseMatrix = AttrDescriptor("inverseMatrix")
	worldMatrix = AttrDescriptor("worldMatrix[0]")
	worldInverseMatrix = AttrDescriptor("worldInverseMatrix[0]")
	parentMatrix = AttrDescriptor("parentMatrix[0]")
	parentInverseMatrix = AttrDescriptor("parentInverseMatrix[0]")

	offsetParentMatrix = AttrDescriptor("offsetParentMatrix") # mi amore

	colour = AttrDescriptor("objectColorRGB")
	message = AttrDescriptor("message")
	rotateOrder = AttrDescriptor("rotateOrder")

	# specific attributes for other nodes
	jointOrient = AttrDescriptor("jointOrient")
	bindPose = AttrDescriptor("bindPose")

	inMesh = AttrDescriptor("inMesh")
	outMesh = AttrDescriptor("outMesh")
	worldMesh = AttrDescriptor("worldMesh[0]")
	inCurve = AttrDescriptor("create")
	outCurve = AttrDescriptor("local")
	worldCurve = AttrDescriptor("worldSpace[0]")

	# endregion

	#region hierarchy
	def setParent(self, targetParent=None, *args, **kwargs):
		"""reparents node under target dag
		replace with api call"""

		mfnDag : om.MFnDagNode = self.MFn
		if targetParent is None:
			mfnDag.parent(om.MObject.kNullObj)
			return


		om.MFnDagNode(getMObject(targetParent)).addChild(mfnDag.object())

	def parentTo(self, targetParent=None, *args, **kwargs):
		"""reparents node under target dag
		replace with api call"""

		mfnDag : om.MFnDagNode = self.MFn
		if targetParent is None:
			mfnDag.parent(om.MObject.kNullObj)
			return

		getMFn(targetParent).addChild(mfnDag.object())


	def addChildNodes(self, newChildren, relative=True):
		nodes = map(WN, toSeq(newChildren))
		for i in nodes:
			i.parentTo(self, relative=relative)



	#endregion

	def hide(self):
		#cmds.hide(self())
		self.transform("visibility").set(0)

	def show(self):
		self.transform("visibility").set(1)

	# def lock(self, attrs=None, locked=True):
	# 	attrs = attrs or self.attrs(keyable=True)
	# 	for i in attrs:
	# 		attr.setLocked(self + "." + i, state=locked)


	def rename(self, name, andShape=True):
		""" allows renaming transform and shape as unit,
		ensuring names are kept in sync """
		if not self.isDag():
			self.setName( name)
			return
		self.shape.name = name + "Shape"
		self.transform.name = name
		return self()


	def delete(self, full=True):
		"""deletes maya node, and by default deletes entire openmaya framework around it
		tesserae is very unstable, and i think this might be why"""
		cmds.delete(self())
		self.MObject = None

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

	#@property
	@classmethod
	def nodeTypeName(cls):
		"""returns string name of node type - "joint", "transform", etc"""
		if cls._nodeType is not None:
			return cls._nodeType
		return camelJoin(cls.__name__)

	def _instanceNodeType(self):
		return cmds.nodeTypeName(self())



	def worldPos(self):
		"""returns world position as MVector"""
		return self.MFn.translation(om.MSpace.kWorld)
	def worldPoint(self):
		return om.MPoint(self.worldPos())


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

	# ---- attribute and plug methods

	# def attrs(self, **kwargs)->T.List[PlugTree]:
	# 	"""return all the attributes of the node
	# 	rarely useful, since attributes don't show current plug structure -
	# 	prefer getPlug() to check if attribute exists"""
	# 	print("attrs")
	# 	for i in range(self.MFn.attributeCount()):
	# 		plug = self.MFn.findPlug(self.MFn.attribute(i), False)
	# 		print(plug.name())

	def parseAttrArgs(self, args=None):
		""" process args given to various attr commands
		if first token of either argument does not exist,
		append node name to it (UNLESS set, in which case string
		value is allowed)
		"""
		#print(args)
		for i in range(len(args)):
			val = args[i]
			if not isinstance(val, str):
				# print("{} is not str".format())
				continue
			if not "." in val:
				val = self() + "." + val
			elif not cmds.objExists( val.split(".")[0]):
				val = self() + ".name"
			args[i] = val
		return args


	@staticmethod
	def con(sourcePlug, destPlug):
		"""tribulations"""
		attr.con(sourcePlug, destPlug, f=True)

	def _instanceCon(self, sourcePlug, destPlug):
		""" im gonna do it """
		args = [sourcePlug, destPlug]
		conargs = self.parseAttrArgs(args)
		attr.con(conargs[0], conargs[1], f=True)

	@staticmethod
	def conOrSet(a, b, f=True):
		"""tribulations"""
		attr.conOrSet(a, b, f)

	def _instanceConOrSet(self, source, dest, **kwargs):
		args = self.parseAttrArgs([source, dest])
		attr.conOrSet(args[0], args[1])
		pass


	# region attribute methods
	AttrData = attr.AttrData
	AttrSpec = attr.AttrSpec
	AttrType = attr.AttrType
	def addAttrFromSpec(self, spec:AttrSpec):
		"""add attribute to node"""
		return attr.addAttrFromSpec(self.MFn, spec)


	# -- node data
	def hasAuxData(self):
		return self.getPlug(self.NODE_DATA_ATTR) is not None
		#return self.NODE_DATA_ATTR in self.attrs()
	def addAuxDataAttr(self):
		#self.addAttr(keyable=False, ln=self.NODE_DATA_ATTR, dt="string")
		spec = self.AttrSpec(name=self.NODE_DATA_ATTR)
		spec.data = self.AttrData(self.AttrType.String)
		self.addAttrFromSpec(spec)
		self(self.NODE_DATA_ATTR).set("{}")
		#self.set(self.NODE_DATA_ATTR, "{}")

	def auxDataPlug(self)->PlugTree:
		return self.getPlug(self.NODE_DATA_ATTR)
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

		# #saveTree = lambda : self.setAuxData(self._dataTree.serialise())
		# def saveTree(*args, **kwargs):
		# 	self.setAuxData(self._liveDataTree.serialise())
		#
		# self._liveDataTree.getSignalComponent().valueChanged.connect(saveTree)
		# self._liveDataTree.structureChanged.connect(saveTree)
		# return self._liveDataTree

		"""the alternative is to use a context handler, like
		with self.dataTree() as data:
			data[etc]
		but this is even cleaner"""

	def saveAuxTree(self, tree=None):
		if not tree and (self._liveDataTree is None):
			raise RuntimeError("no tree passed or already created to save")
		tree = tree or self._liveDataTree
		self.setAuxData(tree.serialise())


	# -- other random stuff -----
	def setColour(self, *colour):
		""" applies override RGB colour """

		self("overrideEnabled").set(1)
		self("overrideRGBColors").set(1)
		self("overrideColorRGB").set(colour)

	def showCVs(self, state=1):
		""" shows cvs of nurbs curves and surfaces """
		self.shape.set( "dispCV", state )


	def connectToShader(self, shader):
		"""takes shadingEngine and connects shape"""
		self.con(self+".instObjGroups[0]", shader+".dagSetMembers")

	def assignMaterial(self, materialName="lambert1"):
		""" very temp """
		cmds.select(cl=1)
		cmds.select(self())
		cmds.hyperShade( assign=materialName)
		cmds.select(cl=1)

	@property
	def shadingEngine(self):
		"""returns connected shadingEngine node"""
		if self.isShape():
			return attr.getImmediateFuture(self + ".instObjGroups[0]")[0]

	def setDrawingOverride(self, referenced=False, clean=False):
		""" sets object display mode override"""
		self.set("overrideEnabled", 1)
		if referenced:
			self.set("overrideDisplayType", 2)
		elif clean:
			self.set("overrideDisplayType", 0)


	def setDefaults(self):
		"""called when node is created"""
		if self.defaultAttrs:
			#attr.setAttrsFromDict(self.defaultAttrs, self)
			for k, v in self.defaultAttrs.items():
				self(k).set(v)

	def copy(self, name=None, children=False):
		"""copies node, without copying children"""
		name = name or self.name+"_copy"
		if children:
			node = WN(cmds.duplicate(self(), n=name)[0])
		else:
			node = WN(cmds.duplicate(
				self(), parentOnly=True, n=name)[0])
		if self.isShape():
			cmds.parent(node, self.parent, r=True, s=True)
		elif self.isDag() and self.parent:
			cmds.parent(node, self.parent)
		return node



	#endregion

