from __future__ import annotations

maya_useNewAPI = True

import sys, os, typing as T

from itertools import product

from maya import cmds
from maya.api import OpenMaya as om, OpenMayaUI as omui, OpenMayaRender as omr

if T.TYPE_CHECKING:
	from edRig.ephrig.node import EphNode
	from edRig.ephrig.solver import EphSolver
	from edRig.ephrig.maya.rig import MEphRig




ephNodeId = 0x00070000
ephNodeName = "ephRigMain"


class EphRigDrawData(om.MUserData):
	""" data for drawing rig"""
	def __init__(self,
	             rig=None):
		om.MUserData.__init__(self, False)
		self.rig = rig # type: MEphRig
		self.drawVecs = None # type: T.List[om.MVector]
	pass


#class EphRigControlNode(om.MPxNode):
class EphRigControlNode(omui.MPxLocatorNode):
	"""
	wanted to subclass transform directly, but there's a 4-year-old
	bug in the maya forums that kTransformNode is missing from
	the api 2.0

	central maya node for ephrig,
	gathering updates for the transforms
	and triggering EphRig evaluation
	"""

	typeId = om.MTypeId(ephNodeId)
	typeName = ephNodeName
	drawClassification = "drawdb/geometry/" + ephNodeName
	#drawClassification = "drawdb/surface/" + ephNodeName
	drawRegistrantId = "EphRigNodePlugin_py"

	# region attribute MObjects
	inputArray = None  # array of ephnode input joints
	# input child attributes
	inWorldMat = None
	inLocalMat = None
	inParentMat = None
	inChildMsg = None
	inParentMsg = None
	nodeName = None

	# output attributes
	outputArray = None
	outWorldMat = None

	# ephrig params
	paramArray = None
	nIterations = None

	dataString = None  # serialised ephrig dict

	# endregion

	if False:
		def __init__(self, *args, **kwargs):
			self.ephRig = None #type:MEphRig

	def compute(self, plug :om.MPlug, data :om.MDataBlock):
		"""reevaluate whole network on any change -
		not prohibitive for now"""

		# match output transforms to inputs
		inArrDH = data.inputArrayValue(EphRigControlNode.inputArray)

		outArrDH = data.outputArrayValue(EphRigControlNode.outputArray)

		while not inArrDH.isDone():
			inMatDH = inArrDH.inputValue().child(self.inWorldMat)
			outMatDH = outArrDH.outputValue().child(self.outWorldMat)
			inMat = om.MMatrix(inMatDH.asMatrix())
			#print("inMat", inMat)
			outMatDH.setMMatrix(inMat)
			inArrDH.next()
			outArrDH.next()

		data.setClean(plug)
		data.setClean(EphRigControlNode.outputArray)



		pass

	@classmethod
	def initialize(cls):
		nFn = om.MFnNumericAttribute()
		tFn = om.MFnTypedAttribute()
		mFn = om.MFnMessageAttribute()
		cFn = om.MFnCompoundAttribute()
		matFn = om.MFnMatrixAttribute()



		#cls.inLocalMat = tFn.create("inLocalMat", "inLocalMat", om.MFnData.kMatrix)
		baseData = om.MFnMatrixData().create(om.MMatrix())
		cls.inWorldMat = matFn.create("inWorldMat", "inWorldMat",
		                            #om.MFnData.kMatrix, defaultValue=baseData
		                              )
		cls.inParentMat = matFn.create("inParentMat", "inParentMat",
		                               #om.MFnData.kMatrix
		                               )
		cls.inChildMsg = mFn.create("inChildMsg", "inChildMsg")
		cls.inParentMsg = mFn.create("inParentMsg", "inParentMsg")
		cls.nodeName = tFn.create("nodeName", "nodeName", om.MFnData.kString)

		cls.inputArray = cFn.create("inputArray", "inputArray")
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cFn.readable = False
		for i in (cls.inWorldMat, cls.inParentMat,
			cls.inParentMsg, cls.inChildMsg,
			cls.nodeName
		          ):
			cFn.addChild(i)



		# cls.outWorldMat = tFn.create("outWorldMat", "outWorldMat", om.MFnData.kMatrix)
		cls.outWorldMat = matFn.create("outWorldMat", "outWorldMat")

		cls.outputArray = cFn.create("outputArray", "outputArray")
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cFn.writable = False
		for i in (cls.outWorldMat, ):
			cFn.addChild(i)

		cls.nIterations = nFn.create("iterations", "iterations",
		                             om.MFnNumericData.kInt)
		cls.dataString = tFn.create("dataString", "dataString", om.MFnData.kString)

		inputAttrs = [cls.inputArray, cls.inWorldMat, cls.inParentMat,
		              cls.nIterations, cls.dataString]
		outputAttrs = [cls.outputArray, cls.outWorldMat]


		for i in [cls.inputArray, cls.outputArray,
			# cls.inWorldMat, cls.outWorldMat,
		          cls.nIterations,
		          cls.dataString
		          ]:
			cls.addAttribute(i)

		#cls.attributeAffects(cls.inputArray, cls.outputArray)

		for src, dest in product(inputAttrs, outputAttrs):
			cls.attributeAffects(src, dest)


	# def postConstructor(self):
	# 	"""hook up ephrig object?"""
	# 	obj = self.thisMObject()
	#
	# 	# load rig if one is saved
	# 	mfn = om.MFnDependencyNode(obj)
	# 	data = mfn.findPlug("dataString", True).asString()
	# 	if not data.strip():
	# 		pass



	def buildRigFromConnections(self):
		"""rebuild this node's ephRig from its various connections"""

	def setRig(self, rig):
		"""run any maya-node-side cleanup needed on rig replacement
		set EphRig reference"""
		self.ephRig = rig

	@classmethod
	def creator(cls):
		""" create new rig object and assign it to the node"""
		#newRig = EphRig()
		newNode = cls(
			#rig=newRig
		)
		#obj = newNode.thisMObject()
		# object is null within creator
		return newNode

