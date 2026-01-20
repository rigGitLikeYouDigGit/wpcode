from __future__ import annotations
import typing as T

import ast, pickle, copy, dataclasses
from collections import namedtuple
from dataclasses import dataclass

from wplib import log
from wplib.sequence import flatten

from maya.api import OpenMaya as om, OpenMayaRender as omr, OpenMayaUI as omui, OpenMayaAnim as oma




class NodeParamData:
	"""a base for classes defining a parametre struct
	to be extracted from maya nodes and then passed around as argument"""
	pass


@dataclass
class PluginNodeIdData:
	"""a struct for storing a node id and its associated data
	(struct is overkill but type registration is critical,
	fine to make it verbose)"""
	nodeName: str
	nodeType : int = om.MPxNode.kDependNode



class PluginNodeTemplate:
	"""convenience base class for hinting common functions;
	we also provide suggested methods for different stages of
	computation, to help avoid compute() becoming unreadable

	Example:
		class MyNode(om.MPxNode, PluginNodeTemplate):
			@classmethod
			def pluginNodeIdData(cls)->PluginNodeIdData:
				return PluginNodeIdData("myNode", om.MPxNode.kDependNode)

		class MyOldApiNode(omMPx.MPxNode, PluginNodeTemplate):
			pass

	"""
	# kNodeName : str
	# #kNodeId : om.MTypeId = None
	# kNodeLocalId = -1 # should range from 0 to 16 within a single plugin
	# kNodeType = om.MPxNode.kDependNode

	@classmethod
	def pluginNodeIdData(cls)->PluginNodeIdData:
		"""return a unique id for this node type"""
		raise NotImplementedError

	@classmethod
	def typeName(cls):
		return cls.pluginNodeIdData().nodeName

	@classmethod
	def kNodeName(cls):
		return cls.pluginNodeIdData().nodeName
	@classmethod
	def kNodeType(cls):
		return cls.pluginNodeIdData().nodeType

	kDrawClassification = "" # used to match up nodes with their draw overrides

	paramDataCls = NodeParamData

	# these are often useful, except in case where
	# the same plug can be input or output
	driverMObjects : list[om.MObject] = []
	drivenMObjects : list[om.MObject] = []

	def __str__(self):
		return "<{} - {}>".format(self.__class__, self.name())

	# region VITAL METHODS
	@classmethod
	def initialiseNode(cls):
		"""required method - this is where you set up
		all the attribute MObjects"""
		raise NotImplementedError

	@classmethod
	def nodeCreator(cls):
		"""required method - this is where you return
		an instance of this node"""
		return cls()

	def postConstructor(self):
		"""effectively the init of a maya node,
		called directly after node data mobject and
		user wrapper have been constructed and linked"""

	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		"""runs maya node computation on inputs"""

	#endregion

	def gatherParams(self, pPlug:om.MPlug, pData:om.MDataBlock)->paramDataCls:
		"""run on every compute even when bound
		gather node params and return param object"""

	def applyParams(self, pPlug: om.MPlug, pData: om.MDataBlock,
	                paramData: paramDataCls):
		"""do stuff with params, if applicable"""

	def setOutputs(self, pPlug:om.MPlug, pData:om.MDataBlock, paramData:paramDataCls=None):
		"""apply node data results to node output plugs"""

	def evaluate(self, pPlug:om.MPlug, pData:om.MDataBlock, paramData:paramDataCls=None):
		"""marks actual node logic, apart from maya datahandle boilerplate"""

	def syncAbstractData(self, pPlug:om.MPlug, pData:om.MDataBlock):
		"""sometimes useful if multiple nodes form an abstract structure,
		and another node controls how they rebuild it"""

	def bind(self, pPlug:om.MPlug, pData:om.MDataBlock, paramData:paramDataCls=None):
		"""a superset of the above, often a node needs to gather and cache
		information once, to then use in compute
		the normal bind states are Off, Bind, Bound and Live
		"""

	# region other node methods
	def legalConnection(self, plug: om.MPlug,
	                    otherPlug: om.MPlug,
	                    asSrc: bool) -> (bool, None):
		"""return True or False to indicate if connection is accepted
		between plugs
		return None to defer to normal Maya processing"""

	def connectionBroken(self, thisPlug:om.MPlug,
	                     otherPlug:om.MPlug, asSrc:bool):
		"""When a connection is broken between otherPlug and thisPlug,
				with thisPlug being the source if asSrc"""
		pass

	def connectionMade(self, thisPlug:om.MPlug,
	                     otherPlug:om.MPlug, asSrc:bool):
		"""When a connection is made between otherPlug and thisPlug,
		with thisPlug being the source if asSrc"""
		pass

	# region node utils
	def thisMFn(self):
		"""shortcut to return an MFnDependencyNode affecting this
		node"""
		return om.MFnDependencyNode(self.thisMObject())

	@classmethod
	def addAttributes(cls, *args:(om.MObject, list[om.MObject])):
		"""add attributes to this node"""
		for mObj in flatten(args):
			cls.addAttribute(mObj)


	@classmethod
	def setAttributesAffect(cls,
	                        drivers: T.List[om.MObject],
	                        drivens: T.List[om.MObject],
	                        ):
		log("setAttributesAffect", cls, drivers, drivens)
		for driver in drivers:
			cls.driverMObjects.append(driver)
			for driven in drivens:
				cls.drivenMObjects.append(driven)
				cls.attributeAffects(driver, driven)

	def printNodeError(self, msg:str):
		"""print given error string to script editor"""
		om.MGlobal.displayError(msg)

	def printNodeInfo(self, msg:str):
		om.MGlobal.displayInfo(msg)

	#region API-agnostic methods
	def thisNodeUniquePath(self):
		"""api version agnostic - return a unique string path to this node
		"""
		import maya.OpenMaya as om1
		import maya.api.OpenMaya as om2
		try:
			om2.MFnDependencyNode(self.thisMObject())
			try:
				return om2.MFnDagNode(self.thisMObject()).fullPathName()
			except RuntimeError: # not dag
				return om2.MFnDependencyNode(self.thisMObject()).name()

		except ValueError: # not api2
			try:
				return om1.MFnDagNode(self.thisMObject()).fullPathName()
			except RuntimeError: # not dag
				return om1.MFnDependencyNode(self.thisMObject()).name()

	def thisApi1MObject(self):
		"""explicitly return an api1 MObject for this node"""
		import maya.OpenMaya as om1
		sel = om1.MSelectionList()
		sel.add(self.thisNodeUniquePath())
		return sel.getDependNode(0)

	def thisApi2MObject(self):
		"""explicitly return an api2 MObject for this node"""
		import maya.api.OpenMaya as om2
		sel = om2.MSelectionList()
		sel.add(self.thisNodeUniquePath())
		return sel.getDependNode(0)

	#endregion

	@classmethod
	def testNode(cls):
		"""run any tests to verify this node is working properly"""
		from maya import cmds
		newNode = cmds.createNode(cls.typeName())
		print("NEW NODE", newNode, cmds.nodeType(newNode))
		assert newNode, "failed to create node of type {}".format(cls.typeName())
		assert cmds.nodeType(newNode) == cls.typeName(), "node type mismatch"

	pass


pluginNodeType = (PluginNodeTemplate, om.MPxNode)


class PluginDrawOverrideTemplate:
	@classmethod
	def kDrawRegistrantId(cls):
		"""used to register draw overrides.
		it's complicated, follow an existing example for draw overrides"""
		return cls.__name__ + "_drawRegistrant"

	# attach function here to actually draw open gl stuff
	drawCallback : T.Callable = None

	def __init__(self, obj):
		omr.MPxDrawOverride.__init__(self, obj, self.drawCallback, True)

	@classmethod
	def creator(cls, obj):
		""" create new rig object and assign it to the node"""
		newNode = cls(obj)
		return newNode


class PluginMPxData(om.MPxData):
	"""mixin for custom MPxData types, accepting
	arbitrary dataclasses.
	usage:
	>>>class MyCustomData(PluginMPxData, om.MPxData):
	>>>	dataClsT = MyDataClass
	>>>	kTypeId = om.MTypeId(0x00112233)

	"""
	dataClsT : T.Type = None
	kTypeId : om.MTypeId = None

	def __init__(self):
		om.MPxData.__init__(self)
		self._data = None

	def setData(self, data:dataClsT):
		"""set internal data from given dataclass instance"""
		self._data = data

	def data(self)->dataClsT:
		"""return internal dataclass instance"""
		return self._data

	def typeId(self)->om.MTypeId:
		"""return unique type id for this MPxData type"""
		return self.kTypeId

	def name(self)->str:
		"""return name of this MPxData type"""
		return self.__class__.__name__

	def writeASCII(self) -> str:
		"""write data to ascii file stream"""
		if self._data is None:
			return ""
		return repr(dataclasses.asdict(self._data))

	def writeBinary(self)->bytearray:
		"""write data to binary file stream"""
		if self._data is None:
			return bytearray()
		return bytearray(pickle.dumps(self._data))

	def readBinary(self, binaryIn, length) -> int:
		"""read data from binary file stream"""
		data = bytes(binaryIn[:length])
		self._data = pickle.loads(data)
		return length

	def readASCII(self, argList, endOfTheLastParsedElement)->int:
		"""read data from ascii file stream"""
		if len(argList) > endOfTheLastParsedElement:
			dataDict = ast.literal_eval(argList[endOfTheLastParsedElement])
			self._data = self.dataClsT(**dataDict)
			return endOfTheLastParsedElement + 1
		return endOfTheLastParsedElement

	def copy(self, other:PluginMPxData):
		"""copy data from another instance of this MPxData type"""
		self._data = copy.deepcopy(other._data)

	@classmethod
	def creator(cls):
		"""required creator method for maya plugin registration"""
		return cls()

