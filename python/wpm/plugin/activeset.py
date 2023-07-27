from __future__ import annotations
import typing as T

import os, sys
from pathlib import Path

"""very, very dodgy - we use the old maya api for this node alone,
as it's the only place that MPxObjectSet is exposed to python.
We do no direct computation with it, only setting up callbacks
when the node is constructed.

This file should be self-contained as the entire plugin.

For the actual processing, we mix in some new api as well -
be very clear where the two interact

"""
import maya.OpenMayaMPx as ompx
import maya.OpenMaya as omOld



from wpm.lib.plugin import MayaPluginAid, NodeParamData, PluginNodeTemplate


class ActiveSetNode(ompx.MPxObjectSet, PluginNodeTemplate):
	kNodeName = "wpActiveSet"
	kNodeClassify = "utility/general"
	#kNodeTypeId = omOld.MTypeId(0x00000001)
	kNodeLocalId = 0
	kNodeType = ompx.MPxNode.kObjectSet

	aExpression = omOld.MObject()
	aSetMsgArr = omOld.MObject()

	@classmethod
	def nodeCreator(cls):
		return ompx.asMPxPtr(ActiveSetNode())

	@classmethod
	def initialiseNode(cls):
		"""add message array attr for local references to other sets
		string attr for expression to be evaluated
		"""
		aExpressionFn = omOld.MFnTypedAttribute()
		cls.aExpression = aExpressionFn.create("expression", "expression", omOld.MFnData.kString)
		aExpressionFn.setStorable(True)
		aExpressionFn.setWritable(True)
		aExpressionFn.setReadable(True)
		aExpressionFn.setKeyable(True)


		aSetMsgArrFn = omOld.MFnMessageAttribute()
		cls.aSetMsgArr = aSetMsgArrFn.create("setMsgArr", "setMsgArr")
		aSetMsgArrFn.setArray(True)
		#aSetMsgArrFn.setIndexMatters(True)
		aSetMsgArrFn.setUsesArrayDataBuilder(True)
		aSetMsgArrFn.setStorable(True)
		aSetMsgArrFn.setWritable(True)
		aSetMsgArrFn.setReadable(True)
		aSetMsgArrFn.setDisconnectBehavior(omOld.MFnAttribute.kNothing)

		drivers = [cls.aExpression, cls.aSetMsgArr]
		for i in drivers:
			cls.addAttribute(i)

	def postConstructor(self, *args: Any, **kwargs: Any) -> Any:
		"""set up callbacks"""
		print("postConstructor for", self, self.thisMObject(), omOld.MFnDependencyNode(self.thisMObject()).name())

		print("unique path", self.thisNodeUniquePath())




# set up wrappers for plugin registration
def getThisFilePath():
	"""annoying replacement for __file__, since registering
	a plugin directly executes this python file (but doesn't correctly
	add it to sys.argv)
	"""
	from wpm import plugin
	return Path(plugin.__file__).parent / "activeset.py"



pluginAid = MayaPluginAid(
	name="wpActiveSetNode",
	studioLocalPluginId=1,
	pluginPath=getThisFilePath(),
	nodeClasses=(ActiveSetNode,),
	drawOverrideClasses={},
	useOldApi=True
)

def initializePlugin(mobject):
	pluginAid.initialisePluginOldApi(mobject)

def uninitializePlugin(mobject):
	pluginAid.uninitialisePluginOldApi(mobject)



