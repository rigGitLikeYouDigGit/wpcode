from __future__ import annotations
import typing as T

from maya.api import OpenMaya as om, OpenMayaRender as omr, OpenMayaUI as omui, OpenMayaAnim as oma
from wpm.lib.plugin import PluginNodeTemplate, PluginNodeIdData, attr

from wpmplugin.solver.lib import SolverFrameData, getSolverFrameData, makeFrameCompound
from wpmplugin.solver.solverstart import WpSolverStart

"""
end node for solver - 
not much computation
"""


def maya_useNewAPI():
	pass


class WpSolverEnd(PluginNodeTemplate, om.MPxNode):
	"""end of solver - the compound attribute 'frameData'
	will be copied back to the start node for the next frame"""

	# inputs
	aFrameData : om.MObject  # array object containing data for a single frame
	aFloat : om.MObject  # float attribute
	aBalanceWheel : om.MObject # bool attribute to flag that a node has been eval'd

	aThisFrame : om.MObject

	@classmethod
	def pluginNodeIdData(cls) -> PluginNodeIdData:
		return PluginNodeIdData("wpSolverEnd", om.MPxNode.kDependNode)

	@classmethod
	def initialiseNode(cls):
		nFn = om.MFnNumericAttribute()
		tFn = om.MFnTypedAttribute()
		cFn = om.MFnCompoundAttribute()

		data = makeFrameCompound("frameData", readable=True, writable=True, array=False,
		                         floatArrName="frameFloat")
		cls.aFrameData = data["compound"]
		cls.aFloat = data["float"]

		cls.aBalanceWheel = attr.makeBalanceWheelAttr("solverStart", readable=False, writable=True)
		cls.aThisFrame = nFn.create("thisFrame", "thisFrame", om.MFnNumericData.kInt, 0)
		nFn.keyable = True

		cls.driverMObjects = [cls.aBalanceWheel, cls.aThisFrame]
		cls.drivenMObjects = [cls.aFrameData]
		cls.addAttributes(cls.drivenMObjects, cls.driverMObjects)
		cls.setAttributesAffect(cls.driverMObjects, cls.drivenMObjects)




		# cls.setExistWithoutInConnections(cls, True)
		# cls.setExistWithoutOutConnections(cls, True)

	def _getSolverStartNode(self, pData:om.MDataBlock)->om.MObject:
		"""return the linked solver start node"""
		thisMFn = self.thisMFn()
		balanceWheelPlug : om.MPlug = thisMFn.findPlug(self.aBalanceWheel, True)
		connectedPlug = balanceWheelPlug.source()
		if not connectedPlug:
			return None
		else:
			return connectedPlug.node()

	def _getSolverStartDH(self, pData:om.MDataBlock, startNode:om.MObject)->om.MArrayDataHandle:
		"""get the receivedData data handle from the start node"""
		startFn = om.MFnDependencyNode(startNode)
		frameDataPlug = startFn.findPlug("receivedData", True)
		frameDataHandle = om.MArrayDataHandle(frameDataPlug.asMDataHandle())
		return frameDataHandle

	def _setSolverStartClsData(self, pData:om.MDataBlock):
		"""set the receivedData data handle from the start node"""
		arrayDH = pData.outputValue(self.aFrameData)
		frameData = getSolverFrameData(
			arrayDH, self.aFloat,
			getOutputValues=False,
		)
		print("setting frame data", frameData.float)
		WpSolverStart.sentFrameData = frameData



	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		"""copy frame data from end node to start node"""
		if pData.isClean(pPlug):
			return
		#return
		self._setSolverStartClsData(pData)

		pData.setClean(pPlug)
		#pData.setClean(self.aFrameData)
		return

		# # get the solver start node
		# startNode = self._getSolverStartNode(pData)
		# if not startNode:
		# 	print("no linked solver start node")
		# 	return
		# startADH = self._getSolverStartDH(pData, startNode)
		# frameDataADH = pData.inputArrayValue(self.aFrameData)
		# print("copying to received data", frameDataADH.inputValue().asDouble())
		# startADH.copy(frameDataADH)
		# print("done copying to received data")
		# print(startADH.inputValue().asDouble())


