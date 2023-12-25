from __future__ import annotations
import typing as T

from maya.api import OpenMaya as om, OpenMayaRender as omr, OpenMayaUI as omui, OpenMayaAnim as oma
from wpm.lib.plugin import PluginNodeTemplate, PluginNodeIdData, attr

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
	aBalanceWheel : om.MObject # bool attribute to flag that a node has been eval'd

	aBalanceWheel : om.MObject # bool attribute to flag that a node has been eval'd

	@classmethod
	def pluginNodeIdData(cls) -> PluginNodeIdData:
		return PluginNodeIdData("wpSolverEnd", om.MPxNode.kDependNode)

	@classmethod
	def initialiseNode(cls):
		nFn = om.MFnNumericAttribute()
		tFn = om.MFnTypedAttribute()
		cFn = om.MFnCompoundAttribute()

		cls.aFrameData = tFn.create("frameData", "frameData", om.MFnData.kAny)
		tFn.array = True

		cls.aBalanceWheel = attr.makeBalanceWheelAttr("solverStart", readable=False, writable=True)

		cls.driverMObjects = [cls.aBalanceWheel]
		cls.drivenMObjects = [cls.aFrameData]
		cls.addAttributes(cls.drivenMObjects, cls.driverMObjects)
		cls.setAttributesAffect(cls.driverMObjects, cls.drivenMObjects)


		# cls.setExistWithoutInConnections(cls, True)
		# cls.setExistWithoutOutConnections(cls, True)


