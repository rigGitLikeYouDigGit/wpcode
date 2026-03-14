from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpm import om, cmds
if T.TYPE_CHECKING:
	import maya.api.OpenMaya as om

from PySide6 import QtCore
from wpm.lib.plugin import (PluginNodeTemplate,
                            MayaPyPluginAid, PluginNodeIdData)
from wpm.lib.plugin import (PluginNodeTemplate, MayaPyPluginAid,
							PluginNodeIdData)
from wpm.lib.plugin.template import PluginMPxData, PluginMPxDictData
from wpsim.maya.plugin import lib
from wpsim.kine.builder import (BuilderBody, BuilderMesh,
	BuilderConstraint,
								BuilderTransform, BuilderNurbsCurve)


class WpSimConstraintMPxData(PluginMPxDictData):
	"""Constraint Data MPxData for WpSim Maya Plugin
	"""
	clsName = "wpSimConstraintMPxData"
	kTypeId = om.MTypeId(0x00112234)
def maya_useNewAPI():
	pass

class WpSimRigidConstraintNode(om.MPxNode, PluginNodeTemplate):
	"""
	general constraint node for wpsim -
	parametres and bodies linked up by string name
	"""
	@classmethod
	def pluginNodeIdData(cls) ->PluginNodeIdData:
		return PluginNodeIdData(
			"wpSimRigidConstraint",
			om.MPxNode.kDependNode
		)

	@classmethod
	def initialiseNode(cls):
		tFn = om.MFnTypedAttribute()
		mFn = om.MFnMatrixAttribute()
		nFn = om.MFnNumericAttribute()
		cFn = om.MFnCompoundAttribute()
		eFn = om.MFnEnumAttribute()
		# leave blank to just use name of node
		cls.aName = tFn.create("name", "name", om.MFnData.kString)

		#todo: make this an enum for loaded constraint types?
		cls.aType = tFn.create("type", "type", om.MFnData.kString)

		cls.aBody = cFn.create("body", "body",)
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cFn.readable = False

		cls.aBodyName = tFn.create("bodyName", "bodyName", om.MFnData.kString)
		cls.aBodyParamName = tFn.create("bodyParamName", "bodyParamName",
		                                om.MFnData.kString)
		cFn.addChild(cls.aBodyName)
		cFn.addChild(cls.aParamName)

		cls.aParam = cFn.create("param", "param")
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cFn.readable = False
		cls.aParamName = tFn.create("paramName", "paramName",
		                            om.MFnData.kString)
		cls.aParamSize = nFn.create("paramSize", "paramSize",
		                            om.MFnNumericData.kInt, 1)
		nFn.setMin(1)
		nFn.setMax(16)
		cls.aParamExp = tFn.create("paramExp", "paramExp",
		                           om.MFnData.kString)
		cls.aParamVal = nFn.create("paramVal", "paramVal", om.MFnNumericData.k4Double)
		cls.aParamMatrix = mFn.create("paramMatrix", "paramMatrix")
		cFn.addChild(cls.aParamName)
		cFn.addChild(cls.aParamSize)
		cFn.addChild(cls.aParamExp)
		cFn.addChild(cls.aParamVal)
		cFn.addChild(cls.aParamMatrix)

		cls.aData = tFn.create("data", "data",
		                       WpSimConstraintMPxData.kTypeId)
		tFn.writable = False


		cls.addAttributes(cls.aName, cls.aType,
		                  cls.aBody,
		                  cls.aParam,
		                  cls.aData)
		cls.setAttributesAffect(
			[cls.aName, cls.aType,
			 cls.aParam, cls.aData,],
			[cls.aData]
		)

	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		"""build constraint data dict as fast as possible
		"""
		if pData.isClean(pPlug):
			return

		data = {}

		data["name"] = pData.inputValue(self.aName).asString()
		data["type"] = pData.inputValue(self.aType).asString()
		bodyArrDH = pData.inputArrayValue(self.aBody)
		for i in range(len(bodyArrDH.builder())):
			bodyArrDH.jumpToPhysicalElement(i)
			bodyDH = bodyArrDH.inputValue()
			bodyName = bodyDH.child(self.aBodyName).asString()
			paramName = bodyDH.child(self.aBodyParamName).asString()
			data[paramName] = bodyName
		paramArrDH = pData.inputArrayValue(self.aParam)
		for i in range(len(bodyArrDH.builder())):
			paramArrDH.jumpToPhysicalElement(i)
			paramArrDH = paramArrDH.inputValue()
			paramName = paramArrDH.child(self.aParamName).asString()
			paramVal = paramArrDH.child(self.aParamVal).asDouble4()
			paramSize = paramArrDH.child(self.aParamSize).asInt()
			paramExp = paramArrDH.child(self.aParamExp).asString()
			paramMatrix = paramArrDH.child(self.aParamMatrix).asMatrix()
			data[paramName] = (paramSize, paramExp, paramVal, paramMatrix)

		dataObj = WpSimConstraintMPxData()
		dataObj.setData(data)
		pData.outputValue(self.aData).setMPxData(dataObj)
		pData.setClean(pPlug)
		for i in (self.aName, self.aType, self.aParam, self.aData):
			pData.setClean(i)
		return

# class WpSimRigidConstraintHingeNode(om.MPxNode, PluginNodeTemplate):
# 	"""
# 	constraint for hinge between two rigid bodies
# 	"""
# 	@classmethod
# 	def pluginNodeIdData(cls) ->PluginNodeIdData:
# 		return PluginNodeIdData(
# 			"wpSimRigidConstraintHinge",
# 			om.MPxNode.kDependNode
# 		)
#
# 	@classmethod
# 	def initialiseNode(cls):
# 		tFn = om.MFnTypedAttribute()
# 		mFn = om.MFnMatrixAttribute()
# 		nFn = om.MFnNumericAttribute()
# 		cFn = om.MFnCompoundAttribute()
# 		# leave blank to just use name of node
# 		cls.aName = tFn.create("name", "name", om.MFnData.kString)
#
# 		# connected bodies
# 		cls.aBodyAName = tFn.create("bodyNameA", "bodyNameA",
# 									om.MFnData.kString)
# 		cls.aBodyBName = tFn.create("bodyNameB", "bodyNameB",
# 									om.MFnData.kString)
# 		cls.aBodyATfName = tFn.create("bodyATfName", "bodyATfName",
# 									  om.MFnData.kString)
# 		cls.aBodyBTfName = tFn.create("bodyBTfName", "bodyBTfName",
# 									  om.MFnData.kString)
#
# 		cls.aActive = nFn.create("active", "active", om.MFnNumericData.kBoolean, True)
#
# 		cls.aWeight = nFn.create("weight", "weight", om.MFnNumericData.kDouble, 1.0)
# 		nFn.setMin(0.0)
#
# 		cls.aDamping = nFn.create("damping", "damping", om.MFnNumericData.kDouble, 0.0)
# 		nFn.setMin(0.0)
#
# 		cls.aIndex = nFn.create("index", "index", om.MFnNumericData.kInt, -1)
# 		nFn.setMin(-1)
# 		nFn.writable = False
#
# 		cls.aRestQuat = nFn.create("restQuat", "restQuat", om.MFnNumericData.k4Double)
# 		cls.aRestPos = nFn.create("restPos", "restPos", om.MFnNumericData.k3Double)
