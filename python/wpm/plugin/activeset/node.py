
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

from wpm.lib.lifespan import NodeLifespanTracker
from wpm.lib.plugin import MayaPluginAid, NodeParamData, PluginNodeTemplate
from wpm.plugin.activeset.settracker import ActiveSetLifespanTracker

class ActiveSetNode(ompx.MPxObjectSet, PluginNodeTemplate):
	kNodeName = "wpActiveSet"
	kNodeClassify = "utility/general"
	#kNodeTypeId = omOld.MTypeId(0x00000001)
	kNodeLocalId = 0
	kNodeType = ompx.MPxNode.kObjectSet

	aExpression = omOld.MObject()
	aSetMsgArr = omOld.MObject()
	aBalanceWheel = omOld.MObject()

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
		aSetMsgArrFn.setUsesArrayDataBuilder(True)
		aSetMsgArrFn.setStorable(True)
		aSetMsgArrFn.setWritable(True)
		aSetMsgArrFn.setReadable(True)
		aSetMsgArrFn.setDisconnectBehavior(omOld.MFnAttribute.kNothing)

		aBalanceWheelFn = omOld.MFnNumericAttribute()
		cls.aBalanceWheel = aBalanceWheelFn.create("balanceWheel", "balanceWheel", omOld.MFnNumericData.kNumeric, 0.0)

		drivers = [cls.aExpression, cls.aSetMsgArr]
		driven = [cls.aBalanceWheel]
		for i in drivers:
			cls.addAttribute(i)
		for i in driven:
			cls.addAttribute(i)
		for i in drivers:
			for j in driven:
				cls.attributeAffects(i, j)

	def onNodeFirstAddedToModel(self, *args, **kwargs):
		"""callback for when this node is first added to the model -
		at THIS point it should know its name, and we can cast into the om2 api
		"""
		mfnOld = omOld.MFnDependencyNode(self.thisMObject())
		print("onNodeFirstAddedToModel", self, self.thisMObject(), omOld.MFnDependencyNode(self.thisMObject()).name(), args, kwargs)

		# done now, remove the callback
		omOld.MMessage.removeCallback(self._nodeAddedCallbackId)

		#set up the node tracker
		import maya.api.OpenMaya as om2
		print("unique", mfnOld.uniqueName())

		sel = om2.MSelectionList()
		sel.add(str(mfnOld.uniqueName()))
		om2Obj = sel.getDependNode(0)
		print("om2Obj", om2Obj)
		mfn2 = om2.MFnDependencyNode(om2Obj)
		print("mfn2", mfn2, mfn2.name(), mfn2.uniqueName())

		print("adding tracker")
		self._nodeTracker = NodeLifespanTracker(om2Obj)

		#om2Obj = self.thisApi2MObject()
		#self._nodeTracker = ActiveSetLifespanTracker(om2Obj)

	def postConstructor(self, *args: Any, **kwargs: Any) -> Any:
		"""suppress the spaghett

		postConstructor runs before the node is added to the graph, before it even receives a name. That means we can't access the om2 MObject here, and we can't
		set up the node tracker.

		We CAN set ANOTHER callback in om1, to wait for this node to be added to the graph, and then set up the tracker.

		I tried a load of different ways to cast directly from the om1 MObject to the om2 MObject, but wasn't able to. It's annoying since they're both wrappers around the same c++ object
		"""
		print("postConstructor for", self, self.thisMObject(), omOld.MFnDependencyNode(self.thisMObject()).name(), args, kwargs)

		# callbackId = omOld.MModelMessage.addNodeAddedToModelCallback(self.thisMObject(),
		#                                                              self.onNodeFirstAddedToModel)

		callbackId = omOld.MDGMessage.addNodeAddedCallback(self.onNodeFirstAddedToModel, "wpActiveSet")

		self._nodeAddedCallbackId = callbackId



		# print("unique path", self.thisNodeUniquePath())
		#
		# # import tracker object, attach it to the OM2 MObject for this node
		# # so that it can be accessed from the callbacks
		# from wpm.plugin.activeset.settracker import ActiveSetLifespanTracker
		# tracker = ActiveSetLifespanTracker(self.thisApi2MObject())

