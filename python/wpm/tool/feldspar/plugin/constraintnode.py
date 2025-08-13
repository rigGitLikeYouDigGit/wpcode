
"""mpx node representing tectonic constraint"""

from __future__ import annotations
from maya.api import OpenMaya as om


from edRig.maya.lib import attr
from edRig.maya.lib.plugin import setAttributesAffect, registerClsAttrMObjects, makeTickAttr, flipTickAttr, MPxNodeDrivingPlug, MPxNodeFromMObject
from edRig.maya.object.plugin import PluginNodeTemplate

from edRig.maya.tool.feldspar.constant import ConstraintType
from edRig.maya.tool.feldspar.constraint import ConstraintBase
from edRig.maya.tool.feldspar.plugin.lib import makeVertexAttr

from edRig.maya.tool.feldspar.plugin.setupnode import FeldsparSetupNode


def maya_useNewAPI():
	pass


class TectonicConstraint(om.MPxNode, PluginNodeTemplate):
	# define everything
	kNodeId = om.MTypeId( 0xDAA2)
	kNodeName = "tectonicConstraint"

	paramDataCls = ConstraintBase.paramCls

	def postConstructor(self):
		self.constraint = ConstraintBase()



	@classmethod
	def initialiseNode(cls):
		"""
		constraint takes input array of plates -
		each entry has array of vertices
		array"""
		msgFn = om.MFnMessageAttribute()
		cFn = om.MFnCompoundAttribute()
		tFn = om.MFnTypedAttribute()
		nFn = om.MFnNumericAttribute()

		# parametre attributes
		cls.aWeight = nFn.create("weight", "weight", om.MFnNumericData.kFloat, 1.0)
		nFn.keyable = True
		cls.aIterations = nFn.create("iterations", "iterations",
		                             om.MFnNumericData.kInt, 3,)
		nFn.keyable = True
		nFn.setMin(0)
		nFn.setMax(10)
		cls.aSoft = nFn.create("soft", "soft", om.MFnNumericData.kBoolean, False)
		nFn.keyable = True

		# input plates
		# likely not needed to be compound, just futureproofing
		cls.aPlate = cFn.create("plate", "plate")
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cFn.readable = False
		cFn.writable = True
		cFn.disconnectBehavior = om.MFnAttribute.kDelete

		cls.aPlateTick = nFn.create("plateTick", "plateTick", om.MFnNumericData.kBoolean)

		cls.aVertex, cls.aVertexPos, cls.aLocalVertexPos, cls.aVertexIndex, cls.aVertexTrainPos, vertexCFn = makeVertexAttr(array=True)

		cFn.addChild(cls.aPlateTick)
		cFn.addChild(cls.aVertex)

		cls.aTick = makeTickAttr()

		cls.aUid = tFn.create("uid", "uid", om.MFnData.kString)

		toAdd = [
			cls.aWeight, cls.aIterations, cls.aSoft,
			cls.aPlate, cls.aTick, cls.aUid
		]
		for n, i in enumerate(toAdd):
			cls.addAttribute(i)
		cls.driverMObjects = [
			cls.aWeight, cls.aIterations, cls.aSoft,
			cls.aPlate, cls.aPlateTick,
			cls.aVertex, cls.aVertexPos, cls.aLocalVertexPos,
			cls.aVertexIndex, cls.aVertexTrainPos
		]
		cls.drivenMObjects = [cls.aTick, cls.aUid]
		setAttributesAffect(cls.driverMObjects, cls.drivenMObjects, cls)

	def gatherParams(self, pPlug:om.MPlug, pData:om.MDataBlock) ->paramDataCls:
		"""pull in any input parametres needed from node parametres"""
		data = self.paramDataCls(
			weight=pData.inputValue(self.aWeight).asFloat(),
			localIterations=pData.inputValue(self.aIterations).asInt(),
			soft=pData.inputValue(self.aSoft).asBool()
		)
		return data

	def applyParams(self, pPlug: om.MPlug, pData: om.MDataBlock,
	                paramData: paramDataCls):
		self.constraint.params = paramData


	def syncAbstractData(self, pPlug:om.MPlug, pData:om.MDataBlock):
		"""update registered plate objects"""
		connectedPlateNodes, plateVertices = self.gatherConnectedData()
		self.constraint.plateList = plateVertices

	def connectionBroken(self, thisPlug:om.MPlug,
	                     otherPlug:om.MPlug, asSrc:bool):
		self.syncAbstractData(None, None)

	def connectionMade(self, thisPlug:om.MPlug,
	                     otherPlug:om.MPlug, asSrc:bool):
		self.syncAbstractData(None, None)


	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		#print("constraint compute")
		# early filtering
		if pData.isClean(pPlug):
			return True

		# force update inputs
		plateBools = pData.inputArrayValue(self.aPlate)
		pData.outputValue(self.aUid).setString(self.constraint.uid)

		params = self.gatherParams(pPlug, pData)
		self.applyParams(pPlug, pData, params)


		# tick
		flipTickAttr(self.aTick, pData)

		for i in self.drivenMObjects:
			pData.setClean(i)
		pData.setClean(pPlug)
		return True



	def legalConnection(self, plug: om.MPlug,
	                    otherPlug: om.MPlug,
	                    asSrc: bool) -> (bool, None):
		"""check that only plate tick plugs can be connected to plateTick
		also check that the new plate node is not already connected anywhere
		could do more, but just don't try to break this system,
		you'll probably succeed
		"""
		if not plug.attribute() == self.aPlateTick or asSrc:
			return None
		otherNode = MPxNodeFromMObject(otherPlug.node())
		if not isinstance(otherNode, FeldsparSetupNode):
			om.MGlobal.displayError("Only FeldsparSetupNode.tick signal must be connected here")
			return False
		if otherNode in self.connectedPlateNodes():
			om.MGlobal.displayError("Target FeldsparSetupNode already connected to this node -"
			                        " duplicate connections illegal")
			return False
		return True

	def connectedPlateNodes(self)->list[FeldsparSetupNode]:
		return self.gatherConnectedData()[0]

	def gatherConnectedData(self)->tuple[
		list[FeldsparSetupNode], ConstraintBase.plateListType
	]:
		"""iterate over connected plates and vertices to get variety of connected data"""
		mfn = self.thisMFn()
		plug = mfn.findPlug(self.aPlate, True)
		plateVertices : ConstraintBase.plateListType = []
		plateNodes = []
		for i in attr.plugSubPlugs(plug):
			tickPlug = i.child(self.aPlateTick)
			mpx : FeldsparSetupNode = MPxNodeDrivingPlug(tickPlug)
			if mpx is None:
				continue
			plateNodes.append(mpx)
			plate = mpx.plate
			vtxPlug = i.child(self.aVertex)
			vertices = []
			for j in attr.plugSubPlugs(vtxPlug):
				index = j.child(self.aVertexIndex).asInt()
				#print("index", index, "plate", plate)
				vertices.append(plate.vertices[index])
			plateVertices.append( (plate, vertices))
		return plateNodes, plateVertices







