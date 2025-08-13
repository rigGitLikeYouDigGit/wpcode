
from __future__ import annotations
"""tectonic rides again"""

from collections import namedtuple
from dataclasses import dataclass

from maya.api import OpenMaya as om, OpenMayaUI as omui

from edRig.maya.lib.attr import makeEnumAttr, makeBindAttr, makeTimeAttr, makeLiveStateAttr
from edRig.constant import BindState, LiveState
from edRig.maya.lib import attr
from edRig.maya.lib.plugin import setAttributesAffect, registerClsAttrMObjects, makeTickAttr, flipTickAttr, MPxNodeDrivingPlug, MPxNodeFromMObject, jumpToElement
from edRig.maya.object.plugin import PluginNodeTemplate

from edRig.maya.tool.feldspar.assembly import Assembly
from edRig.maya.tool.feldspar.plugin import lib as fplib
from edRig.maya.tool.feldspar.plugin.setupnode import FeldsparSetupNode
from edRig.maya.tool.feldspar import datastruct

"""
the cost, obviously, is more work to add and remove attributes

tick connections are no longer necessary but for the purposes 
of keeping attribute names not crazy, we're still using them

"""

# @dataclass
# class SolverParamData:
# 	iterations:int
# 	damping:float
# 	inheritVelocity:float


def maya_useNewAPI():
	pass


class FeldsparSolverNode(
	#om.MPxNode,
	omui.MPxLocatorNode,
	PluginNodeTemplate):
	"""inputs use only tick signals - outputs
	present full graph compound attributes"""
	# define everything
	kNodeId = om.MTypeId( 0xDAA1)
	kNodeName = "feldsparSolver"

	kNodeType = om.MPxNode.kLocatorNode

	kDrawClassification = "drawdb/geometry/" + kNodeName

	paramDataCls = datastruct.AssemblyParams

	def postConstructor(self):
		self.assembly = Assembly()

	@classmethod
	def initialiseNode(cls):
		"""add attributes
		2 arrays of strings - these hold the uids of plates and constraints"""
		msgFn = om.MFnMessageAttribute()
		cFn = om.MFnCompoundAttribute()
		tFn = om.MFnTypedAttribute()
		nFn = om.MFnNumericAttribute()

		# parametres
		cls.aBind, bindFn = makeEnumAttr(name="bind", enum=BindState)

		cls.aTick = makeTickAttr()
		cls.aMaxIterations = nFn.create("maxIterations", "maxIterations",
		                             om.MFnNumericData.kInt, 10)
		nFn.keyable = True
		nFn.setMin(0)

		cls.aTime = makeTimeAttr()

		cls.aLiveState = makeLiveStateAttr()

		# inputs
		# graph network array
		cls.aGraphTick = makeTickAttr("graphTick", array=True,
		                              readable=False, writable=True)

		# trajectory for synthesis to match
		cls.aTargetCurve = tFn.create("targetCurve", "targetCurve", om.MFnData.kNurbsCurve)




		# outputs
		cls.aGraphOutput = cFn.create("graphOutput", "graphOutput")
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cFn.readable = True
		cFn.writable = False

		cls.aVertex, cls.aVertexPos, cls.aVertexIndex = fplib.makeVertexAttr(array=True)

		cls.aBar, cls.aBarVertexA, cls.aBarVertexB, cls.aSoft, cls.aBarBindLength, cls.aBarTargetLength, cls.aBarLength = fplib.makeBarAttr(array=True)
		cls.aGroup, cls.aGroupVertexIndex, cls.aGroupMatrix, cls.aGroupFixed = fplib.makeRigidGroupAttr(array=True)

		cFn.addChild(cls.aVertex)
		cFn.addChild(cls.aBar)
		cFn.addChild(cls.aGroup)

		# output trajectory of the trace vertex
		cls.aTraceCurve = tFn.create("traceCurve", "traceCurve", om.MFnData.kNurbsCurve)


		toAdd = [cls.aBind, cls.aMaxIterations, cls.aTime,
		         cls.aTick,
		         cls.aGraphTick, cls.aTargetCurve,
		         cls.aGraphOutput, cls.aTraceCurve

		         ]
		for i in toAdd:
			cls.addAttribute(i)
		cls.driverMObjects = [cls.aBind, cls.aMaxIterations, cls.aTime,
		                      cls.aGraphTick,
		                      cls.aTargetCurve,
		           ]
		cls.drivenMObjects = [cls.aGraphOutput, cls.aTraceCurve,
		                      cls.aVertexPos, cls.aVertexPos,
		                      cls.aTick]
		setAttributesAffect(cls.driverMObjects, cls.drivenMObjects, cls)




	def gatherParams(self, pPlug:om.MPlug, pData:om.MDataBlock)->paramDataCls:
		"""run on every compute even when bound
		gather numeric params and return param object"""
		data = self.paramDataCls(
			pData.inputValue(self.aMaxIterations).asInt(),
		)
		return data

	def applyParams(self, pPlug: om.MPlug, pData: om.MDataBlock,
	                paramData: paramDataCls):
		self.assembly.params = paramData


	def bind(self, pPlug:om.MPlug, pData:om.MDataBlock, paramData:paramDataCls=None):
		"""reset and rebuild python object connections in linkage assembly"""
		print("bind")

		setupNodes = self.connectedSetupNodes()
		self.assembly.setData( setupNodes[0].data )



		pass

	def evaluate(self, pPlug:om.MPlug, pData:om.MDataBlock, paramData:paramDataCls=None):
		"""main iteration of assembly"""
		paramData.iterations = 1
		self.assembly.runIteration()



	def setOutputs(self, pPlug:om.MPlug, pData:om.MDataBlock, paramData:paramDataCls=None):
		"""update output matrices and vertex positions with results of
		evaluation"""

		outArrayDH = pData.outputArrayValue(self.aGraphOutput)
		jumpToElement(outArrayDH, 0)
		parentDH = outArrayDH.outputValue()

		vertexArrayDH = om.MArrayDataHandle(parentDH.child(self.aVertex))

		#print("end positions", self.assembly.data.positions)

		for i, dh in attr.iterArrayDataHandle(vertexArrayDH):
			dh.child(self.aVertexPos).set3Float(
				*map(float, self.assembly.data.positions[i]))



	# solver compute
	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		""""""
		#print("solver compute")
		if pData.isClean(pPlug):
			return True

		# forces plates and constraints to compute properly
		plateBools = pData.inputArrayValue(self.aGraphTick)

		paramData = self.gatherParams(pPlug, pData)
		#
		# process bind value
		bindVal = pData.inputValue(self.aBind).asShort()
		bindState = BindState._value2member_map_[bindVal]
		if bindState == BindState.Off:
			pData.setClean(pPlug)
			return True
		if bindState == BindState.Bind or bindState == BindState.Live:
			self.bind(pPlug, pData, paramData)
			if bindState == BindState.Bind:
				pData.outputValue(self.aBind).setInt(BindState.Bound.value)

		# apply parametres to assembly
		self.applyParams(pPlug, pData, paramData)

		# # run actual evaluation
		self.evaluate(pPlug, pData, paramData)

		self.setOutputs(pPlug, pData, paramData)

		flipTickAttr(self.aTick, pData)

		pData.setClean(pPlug)
		return True


	def connectedComponentNodes(self, parentPlugObj,
	                            tickPlugObj,
	                            desiredType)->list[PluginNodeTemplate]:
		mfn = self.thisMFn()
		plug = mfn.findPlug(parentPlugObj, True)
		componentNodes = []
		for i in attr.plugSubPlugs(plug):
			tickPlug = i.child(tickPlugObj)
			mpx = MPxNodeDrivingPlug(tickPlug)
			if not isinstance(mpx, desiredType):
				continue
			componentNodes.append(mpx)
		return componentNodes

	# def connectedConstraintNodes(self)->list[TectonicConstraint]:
	# 	"""return MPxNode objects for all nodes connected to
	# 	constraintTick elements"""
	# 	return self.connectedComponentNodes(
	# 		self.aConstraintInput, self.aConstraintTick, TectonicConstraint)

	def connectedSetupNodes(self)->list[FeldsparSetupNode]:
		"""return MPxNode objects for all nodes connected to
		constraintTick elements"""
		tickPlug = self.thisMFn().findPlug(self.aGraphTick, True)
		setupNodes = []
		for i in attr.plugSubPlugs(tickPlug):
			mpx = MPxNodeDrivingPlug(i)
			if not isinstance(mpx, FeldsparSetupNode):
				continue
			setupNodes.append(mpx)
		return setupNodes



	def legalConnection(self, plug: om.MPlug,
	                    otherPlug: om.MPlug,
	                    asSrc: bool) -> (bool, None):
		"""check that only plate tick and constraint tick plugs
		can be connected to those positions"""
		if not plug.attribute() in (self.aGraphTick, ) or asSrc:
			return None
		otherNode = MPxNodeFromMObject(otherPlug.node())
		if plug.attribute() == self.aGraphTick:
			if not isinstance(otherNode, FeldsparSetupNode):
				om.MGlobal.displayError("Only FeldsparSetupNode.tick signal"
				                        " must be connected here")
				return False
			return True




