from __future__ import annotations
import typing as T
from wplib import log


from maya.api import OpenMaya as om, OpenMayaRender as omr, OpenMayaUI as omui, OpenMayaAnim as oma
from wpm.lib.plugin import PluginNodeTemplate, PluginNodeIdData, attr


"""
leyline point, 
optionally matching manual input within a parent space
"""

def maya_useNewAPI():
	pass

class StrataPoint(PluginNodeTemplate, om.MPxNode):
	"""
	should a single maya node handle multiple points?
	"""

	# inputs
	aEditMode : om.MObject
	aLiveDeltaMatrix : om.MObject

	# root parent compound object
	aParent : om.MObject
	#aParentOffsetMatrix : om.MObject

	# outputs
	aOutMatrix : om.MObject

	#aBalanceWheel : om.MObject # bool attribute to flag that a node has been eval'd

	@classmethod
	def pluginNodeIdData(cls)->PluginNodeIdData:
		return PluginNodeIdData("strataPoint", om.MPxNode.kTransformNode)

	@classmethod
	def initialiseNode(cls):
		log("initialiseNode" )
		nFn = om.MFnNumericAttribute()
		cFn = om.MFnCompoundAttribute()
		tFn = om.MFnTypedAttribute()
		msgFn = om.MFnMessageAttribute()

		# input
		cls.aEditMode = nFn.create("editMode", "editMode",
		                           om.MFnNumericData.kBoolean, 0)
		cls.aLiveDeltaMatrix = tFn.create("liveDeltaMatrix", "liveDeltaMatrix",
		                                  om.MFnData.kMatrix)

		# only a transform node may connect its message attribute here -
		# controls direct non-graph reset action
		cls.aLiveTransformNodeMsg = msgFn.create("liveTransformNodeMsg", "liveTransformNodeMsg")
		cls.aUiData = tFn.create("uiData", "uiData",
		                         om.MFnData.kString)

		# parent array
		cls.aDriver = cFn.create("driver", "driver")
		cFn.array = True
		cFn.usesArrayDataBuilder = True

		# if parent is a point
		cls.aDriverMatrix = tFn.create(
			"driverMatrix", "driverMatrix",om.MFnData.kMatrix)
		# if parent is a curve - might need some extra data for this
		cls.aDriverCurve = tFn.create(
			"driverCurve", "driverCurve", om.MFnData.kNurbsCurve)
		# extra coord attributes
		cls.aDriverRefCurve = tFn.create(
			"driverCurveRef", "parentCurveRef", om.MFnData.kNurbsCurve)
		cls.aDriverParam = nFn.create("driverCurveParam", "driverCurveParam", om.MFnNumericData.kFloat, 0.0)
		cls.aDriverLength = nFn.create("driverCurveLength", "driverCurveLength", om.MFnNumericData.kFloat, 0.0)
		cls.aDriverNormLength = nFn.create("driverCurveNormLength", "driverCurveNormLength", om.MFnNumericData.kFloat, 0.0)
		cls.aDriverCurveReverse = nFn.create("driverCurveReverse", "driverCurveReverse", om.MFnNumericData.kFloat, 0.0)


		cls.aPointOffsetMatrix = tFn.create("pointOffsetMatrix", "pointOffsetMatrix",
		                                     om.MFnData.kMatrix)
		parentAttrs = [cls.aDriverMatrix, cls.aDriverCurve, cls.aDriverRefCurve,
		               cls.aDriverCurveReverse, cls.aDriverParam, cls.aDriverLength,
		               cls.aDriverNormLength, cls.aPointOffsetMatrix]
		for i in parentAttrs:
			cFn.addChild(i)

		# output
		cls.aOutMatrix = tFn.create("outMatrix", "outMatrix",
		                            om.MFnData.kMatrix)


		cls.driverMObjects = [
			cls.aEditMode, cls.aLiveDeltaMatrix,
			cls.aLiveTransformNodeMsg, cls.aUiData,
			cls.aDriver,
		]
		cls.drivenMObjects = [cls.aOutMatrix]

		cls.addAttributes(cls.drivenMObjects,
		                  cls.driverMObjects,
		                  #cls.aReceivedData
		                  )
		cls.setAttributesAffect(cls.driverMObjects, cls.drivenMObjects)


	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		"""check for reset - otherwise retrieve data from end node"""

		# print("solverStart compute")
		if pData.isClean(pPlug):
			print("plug clean", pPlug)
			return

		pData.setClean(pPlug)
		return


	@classmethod
	def testNode(cls):
		"""test node creation"""
		from maya import cmds
		parentLoc = cmds.spaceLocator(name="parentLoc")[0]
		cmds.setAttr(parentLoc + ".translate", 2, 3, 4)
		pt = cmds.createNode(cls.typeName())
		cmds.connectAttr(parentLoc + ".worldMatrix[0]", pt + ".parent[0].parentMatrix")

		childLoc = cmds.spaceLocator(name="childLoc")[0]
		cmds.connectAttr(pt + ".outMatrix", childLoc + ".parentOffsetMatrix")



