
"""mpx node representing tectonic plate"""

from __future__ import annotations
import numpy as np

import edRig.constant
from maya.api import OpenMaya as om

from edRig.maya.lib import attr
from edRig.maya.lib.plugin import *
from edRig.maya.tool.feldspar.plate import PlateParamData
from edRig.maya.tool.feldspar.plugin import lib as fplib

from edRig.maya.tool.feldspar.datastruct import FeldsparData


def maya_useNewAPI():
	pass


class FeldsparSetupNode(om.MPxNode, PluginNodeTemplate):
	"""creates a feldspar vertex network directly from vertices
	computation here is minimal, only setting vertices
	and measuring bar lengths
	"""
	# define everything
	kNodeId = om.MTypeId( 0xDAA3)
	kNodeName = "feldsparSetup"

	paramDataCls = PlateParamData

	def postConstructor(self):
		self.data = FeldsparData(
			np.zeros((1, 3)),
			np.zeros((1, 3)),
			np.zeros((1, 3)),
			[],
			[]
		)


	@classmethod
	def initialiseNode(cls):
		"""add attributes
		array of vertex points"""
		msgFn = om.MFnMessageAttribute()
		cFn = om.MFnCompoundAttribute()
		tFn = om.MFnTypedAttribute()
		nFn = om.MFnNumericAttribute()

		# inputs
		cls.aVertex, cls.aVertexPos, cls.aVertexIndex = fplib.makeVertexAttr(array=True)
		cls.aBar, cls.aBarVertexA, cls.aBarVertexB, cls.aSoft, cls.aBarBindLength, cls.aBarTargetLength, cls.aBarLength = fplib.makeBarAttr(
			array=True)
		cls.aGroup, cls.aGroupVertexIndex, cls.aGroupMatrix, cls.aGroupFixed = fplib.makeRigidGroupAttr(array=True)

		cls.aTick = makeTickAttr()
		cls.aBind = attr.makeBindAttr()

		toAdd = [cls.aVertex, cls.aBar, cls.aGroup, cls.aTick, cls.aBind]
		for i in toAdd:
			cls.addAttribute(i)
		drivers = [cls.aVertex, cls.aBar, cls.aGroup,
		           cls.aGroupFixed, cls.aGroupVertexIndex,
		           cls.aBind]

		cls.drivenMObjects = [cls.aTick,
		                      #cls.aGroupMatrix
		                      ]
		setAttributesAffect(drivers, cls.drivenMObjects, cls)



	def bind(self, pPlug:om.MPlug, pData:om.MDataBlock, paramData:paramDataCls=None):
		"""set vertex indices, measure bar rest lengths
		regenerate data structs
		"""

		# build arrays
		vertexArray = fplib.vertexArrayFromVertexArrayDH(
			pData.inputArrayValue(self.aVertex),
			vtxPosMObj=self.aVertexPos
		)

		for i, dh in attr.iterArrayDataHandle(pData.inputArrayValue(self.aVertex)):
			dh.child(self.aVertexIndex).setInt(i)

		barTiesArray = fplib.barTiesFromBarArrayDH(
			pData.inputArrayValue(self.aBar),
			self.aBarVertexA, self.aBarVertexB
		)

		# build data structs
		barDatas = fplib.barDatasFromBarArrayDH(
			self, pData.inputArrayValue(self.aBar),
			vertexArray
		)

		groupDatas = fplib.groupDatasFromGroupArrayDH(
			self, pData.inputArrayValue(self.aGroup)
		)
		#print("group data", groupDatas)

		self.data = FeldsparData(basePositions=vertexArray.copy(),
		                         positions=vertexArray,
		                         velocities=np.zeros_like(vertexArray),
		                         barDatas=barDatas,
		                         groupDatas=groupDatas
		                         )
		#print("setup bind done")

	def evaluate(self, pPlug:om.MPlug, pData:om.MDataBlock, paramData:paramDataCls=None):
		"""gather positions of fixed vertices and set them"""
		vertexArray = fplib.vertexArrayFromVertexArrayDH(
			pData.inputArrayValue(self.aVertex), self.aVertexPos)
		self.data.basePositions = vertexArray
		self.data.positions[self.data.fixedVertexArray] = \
			vertexArray[self.data.fixedVertexArray]


	def setOutputs(self, pPlug:om.MPlug, pData:om.MDataBlock, paramData:paramDataCls=None):
		# update rest matrices of groups
		for i, dh in attr.iterArrayDataHandle(pData.outputArrayValue(self.aGroup)):
			group = self.data.groupDatas[i]
			groupBindMat = om.MMatrix(group.bindMatrix(self.data))
			#print("set group mat", groupBindMat)
			#dh.child(self.aGroupMatrix).setMMatrix( groupBindMat )
			om.MFnMatrixData(dh.child(self.aGroupMatrix).data()).set(groupBindMat)


	# setup compute
	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		""""""
		#print("setup compute")

		# early filtering
		if pData.isClean(pPlug):
			return True

		# do bind
		#bindVal = pData.inputValue(self.aBind).asInt()
		bindVal = pData.inputValue(self.aBind).asShort()
		bindState = edRig.constant.BindState._value2member_map_[bindVal]
		if bindState == edRig.constant.BindState.Off:
			pData.setClean(pPlug)
			return True
		if bindState == edRig.constant.BindState.Bind or bindState == edRig.constant.BindState.Live:
			self.bind(pPlug, pData)
			if bindState == edRig.constant.BindState.Bind:
				pData.outputValue(self.aBind).setShort(edRig.constant.BindState.Bound.value)

		self.evaluate(pPlug, pData)
		# print("setup end eval")
		# print("positions", self.data.positions)

		self.setOutputs(pPlug, pData)

		# tick
		flipTickAttr(self.aTick, pData)

		# set clean
		for i in self.drivenMObjects:
			pData.setClean(i)
		pData.setClean(pPlug)
		#print("setup compute done")
		return True




	#endregion














