
from __future__ import annotations
import typing as T

import numpy as np
from collections import namedtuple

from wpm import cmds, om, WN

"""shelving until we have a proper character to test on - 
at first glance this works, but needs proper integration

also not sure yet how to structure constraints and iterations

"""

NamedMatrix : type[tuple[str, np.ndarray]] = namedtuple("NamedMatrix", ["name", "matrix"])

# basic IK triangle for now
jointPositions = np.array([
	(0, 0, 0),
	(2, 0, 2),
	(4, 0, 0),
	])

def addMessageAttr(node:WN, attrName:str):
	"""add named message attribute to node"""
	nodeFn = om.MFnDependencyNode(node.object())
	attrFn = om.MFnMessageAttribute()
	attrObj = attrFn.create(attrName, attrName)
	nodeFn.addAttribute(attrObj)


class EphState:
	"""holds static array of matrices - acts as input and output
	to eph operations"""

	def __init__(self, matrixArr:np.ndarray, nameList:list[str]):
		self.matrices = matrixArr
		self.nameMap = {name:idx for idx, name in enumerate(nameList)}

class EphControlScheme:
	"""holds overall graph structure for single eph evaluation -
	different rig settings generate new ControlScheme objects"""

class EphConstraint:
	"""base class for eph constraints,
	connecting 2 or more EphNodes"""

class EphCallbackData:

	def __init__(self, ephNode:EphNode):
		self.ephNode = ephNode

class EphNode:
	"""holder class for ephemeral transforms
	look up nodes by connections if not apparent
	"""

	def __init__(self):
		self.refJnt:WN = None
		self.inputGrp:WN = None
		self.inputJnt:WN = None
		self.outputJnt:WN = None

		self.callbackId = None

		self.callbackData = EphCallbackData(self)
		self.mfnMatrixData = om.MFnMatrixData()

		# TEMP for testing basic FK
		self.parentNode : EphNode = None
		self.childNodes : list[EphNode] = []

	def name(self):
		return self.inputJnt.name() + "Eph"

	def addChild(self, child:EphNode):
		"""add child node to list"""
		self.childNodes.append(child)
		child.parentNode = self

	def syncOutputJoint(self):
		"""output joint matrix should ALWAYS be
		equal to input joint matrix * input group matrix
		"""
		transform = om.MTransformationMatrix(
			self.inputJnt.matrix.get() * self.inputGrp.matrix.get()
		)
		self.outputJnt.MFn.setTransformation(transform)

	# TEMP for testing basic FK
	def matchParentTransform(self):
		"""match parent transform -
		update input group with delta between
		parent ref joint and this ref joint
		"""
		relMat = self.parentNode.refJnt.matrix.get().inverse() * self.refJnt.matrix.get()
		targetMat = relMat * self.parentNode.outputJnt.matrix.get()
		transform = om.MTransformationMatrix(targetMat)
		self.inputGrp.MFn.setTransformation(transform)

		self.syncOutputJoint()
		for i in self.childNodes:
			i.matchParentTransform()



	@staticmethod
	def _onInputJointChanged(
			attributeMessage,
			plug:om.MPlug, otherPlug:om.MPlug, clientData:EphCallbackData):
		"""callback for input joint changing"""

		# get ephNode from clientData
		ephNode:EphNode = clientData.ephNode
		ephNode.syncOutputJoint()

		for i in ephNode.childNodes:
			i.matchParentTransform()




	# @staticmethod
	# def _onManipulationFinished(
	# 		attributeMessage,

	def _setupCallbacks(self):
		"""setup callbacks for nodes"""
		self.callbackId = om.MNodeMessage.addAttributeChangedCallback(
			self.inputJnt.object(),
			self._onInputJointChanged,
			self.callbackData
		)
		#print("callbackId", self.callbackId)
		# self.dragEndCallbackId = om.MEventMessage.addEventCallback(
		#
		# pass

	def setNodes(self, refJnt:WN, inputGrp:WN, inputJnt:WN, outputJnt:WN):
		self.refJnt = refJnt
		self.inputGrp = inputGrp
		self.inputJnt = inputJnt
		self.outputJnt = outputJnt

		self._setupCallbacks()


	@classmethod
	def fromNodes(cls, refJnt:WN, inputGrp:WN, inputJnt:WN, outputJnt:WN):
		self = cls()
		self.setNodes(refJnt, inputGrp, inputJnt, outputJnt)
		return self

def makeEphNode(name:str, pos:np.ndarray)->EphNode:
	"""create reference, input group, input joint and output joint"""

	refJnt = WN("joint", name=name + "Ref_JNT")
	addMessageAttr(refJnt, "inputGrp")
	addMessageAttr(refJnt, "inputJnt")
	addMessageAttr(refJnt, "outputJnt")

	inputGrp = WN("transform", name=name+"Input_GRP")
	addMessageAttr(inputGrp, "refJnt")
	addMessageAttr(inputGrp, "inputJnt")
	addMessageAttr(inputGrp, "outputJnt")


	inputJnt = WN("joint", name=name+"Input_JNT", parent=inputGrp)
	addMessageAttr(inputJnt, "refJnt")
	addMessageAttr(inputJnt, "inputGrp")
	addMessageAttr(inputJnt, "outputJnt")

	outputJnt = WN("joint", name=name+"Output_JNT")
	addMessageAttr(outputJnt, "refJnt")
	addMessageAttr(outputJnt, "inputGrp")
	addMessageAttr(outputJnt, "inputJnt")

	# set initial positions
	for i in [inputGrp, outputJnt, refJnt]:
		i.translate.set(pos)

	# connect message attrs
	refJnt("inputGrp").use(inputGrp("refJnt"))
	refJnt("inputJnt").use(inputJnt("refJnt"))
	refJnt("outputJnt").use(outputJnt("refJnt"))

	inputGrp("inputJnt").use(inputJnt("inputGrp"))
	inputGrp("outputJnt").use(outputJnt("inputGrp"))

	inputJnt("outputJnt").use(outputJnt("inputJnt"))

	return EphNode.fromNodes(refJnt, inputGrp, inputJnt, outputJnt)


def main():
	"""
	function for testing basic eph callbacks
	load and restart maya scene
	"""

	cmds.file(new=True, force=True)

	# group holding reference transforms
	refGrp = WN("transform", name="refGrp")
	# group holding flat output transforms
	outputGrp = WN("transform", name="outputGrp")
	# group holding input transforms
	inputGrp = WN("transform", name="inputGrp")

	jointNames = "ABC"

	ephNodes = []
	for i, pos in enumerate(jointPositions):
		node = makeEphNode(jointNames[i], pos)
		node.refJnt.parentTo(refGrp)
		node.inputGrp.parentTo(inputGrp)
		node.outputJnt.parentTo(outputGrp)
		ephNodes.append(node)

	ephNodes[0].addChild(ephNodes[1])
	ephNodes[1].addChild(ephNodes[2])

	inputGrp.setColour(0.043, 0.113, 0.365)

	outputGrp.translate.set((0, 0, -6))
	outputGrp.setColour(0.7, 0.2, 0.0)

	refGrp.translate.set((0, 0, -12))
	refGrp.setColour(0.1, 0.5, 0.1)
	pass





