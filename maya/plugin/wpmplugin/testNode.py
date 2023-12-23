

from __future__ import annotations
import typing as T

from maya.api import OpenMaya as om, OpenMayaRender as omr, OpenMayaUI as omui, OpenMayaAnim as oma
from wpm.lib.plugin import PluginNodeTemplate, PluginNodeIdData

def maya_useNewAPI():
	pass


class TestNode(PluginNodeTemplate, om.MPxNode):

	@classmethod
	def pluginNodeIdData(cls)->PluginNodeIdData:
		return PluginNodeIdData("wpTestNode", om.MPxNode.kDependNode)

	@classmethod
	def initialiseNode(cls):

		nFn = om.MFnNumericAttribute()
		cls.aTest = nFn.create("testIn", "testIn", om.MFnNumericData.kFloat, 0.0)

		cls.aTestOut = nFn.create("testOut", "testOut", om.MFnNumericData.kFloat, 0.0)

		cls.driverMObjects = [cls.aTest]
		cls.drivenMObjects = [cls.aTestOut]
		cls.addAttributes(cls.drivenMObjects, cls.driverMObjects)
		cls.setAttributesAffect(cls.driverMObjects, cls.drivenMObjects)





