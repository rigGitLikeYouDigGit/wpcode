from __future__ import annotations
import typing as T


from maya.api import OpenMaya as om, OpenMayaRender as omr, OpenMayaUI as omui, OpenMayaAnim as oma
from wpm.lib.plugin import PluginNodeTemplate, PluginNodeIdData, attr

from wpmplugin.solver.lib import makeFrameCompound, SolverFrameData, getSolverFrameData, setSolverFrameData

"""
start node for solver - 
handles initial values, resetting,
and retrieving final values from end node,
circumventing the maya dependency graph
"""

def maya_useNewAPI():
	pass

class WpSolverStart(PluginNodeTemplate, om.MPxNode):
	"""this will be the first node in the solver chain
	currently no type / structure checking on plugs -
	literal copies"""

	dataHandleList = []
	sentFrameData = SolverFrameData(float=(0,))

	# inputs
	aResetFrame : om.MObject
	aThisFrame : om.MObject

	aFramesToStore : om.MObject # number of frames to store
	# by default only one single frame

	# root compound object
	# copy data from this compound plug into loop input
	aInputData : om.MObject
	aInputFloat : om.MObject

	# intermediate received data from solver end
	# DO NOT set attributeAffects on this
	aReceivedData : om.MObject

	# outputs
	#aFrameArray: om.MObject  # array of data copies from previous frames
	aFrameData : om.MObject  # array object containing data for a single frame
	aFrameFloat : om.MObject

	aBalanceWheel : om.MObject # bool attribute to flag that a node has been eval'd


	@classmethod
	def pluginNodeIdData(cls)->PluginNodeIdData:
		return PluginNodeIdData("wpSolverStart", om.MPxNode.kDependNode)

	# @classmethod
	# def existWithoutInConnections(self, *args, **kwargs):
	# 	return True
	#
	# @classmethod
	# def existWithoutOutConnections(self, *args, **kwargs):
	# 	return True

	@classmethod
	def initialiseNode(cls):
		nFn = om.MFnNumericAttribute()
		cFn = om.MFnCompoundAttribute()
		tFn = om.MFnTypedAttribute()

		cls.aResetFrame = nFn.create("resetFrame", "resetFrame", om.MFnNumericData.kInt, 0)
		nFn.keyable = True
		cls.aThisFrame = nFn.create("thisFrame", "thisFrame", om.MFnNumericData.kInt, 0)
		nFn.keyable = True
		cls.aFramesToStore = nFn.create("framesToStore", "framesToStore", om.MFnNumericData.kInt, 1)
		nFn.keyable = True
		nFn.setMin(1)

		inData = makeFrameCompound(
			"inputData",
			floatArrName="inputFloat",
			readable=False, writable=True,

		array=False)
		cls.aInputData = inData["compound"]
		cls.aInputFloat = inData["float"]
		# cls.aInputData = tFn.create("inputData", "inputData", om.MFnData.kAny)
		# tFn.array = True
		# tFn.usesArrayDataBuilder = True
		# tFn.readable = False
		# tFn.writable = True

		# cls.aReceivedData = tFn.create("receivedData", "receivedData", om.MFnData.kAny)
		# tFn.array = True
		# tFn.usesArrayDataBuilder = True
		# tFn.readable = True
		# tFn.writable = False

		# cls.aFrameArray = cFn.create("frameArray", "frameArray")
		# cFn.array = True
		# cFn.usesArrayDataBuilder = True
		# cFn.readable = True
		# cFn.writable = False

		frameData = makeFrameCompound(
			"frameData", readable=True, writable=False,
			array=True,
			floatArrName="frameFloat"
		)
		cls.aFrameData = frameData["compound"]
		cls.aFrameFloat = frameData["float"]


		cls.aBalanceWheel = attr.makeBalanceWheelAttr("solverEnd", writable=False)

		cls.driverMObjects = [cls.aResetFrame, cls.aThisFrame, cls.aFramesToStore, cls.aInputData]
		cls.drivenMObjects = [cls.aFrameData, cls.aBalanceWheel]

		cls.addAttributes(cls.drivenMObjects,
		                  cls.driverMObjects,
		                  #cls.aReceivedData
		                  )
		cls.setAttributesAffect(cls.driverMObjects, cls.drivenMObjects)

		# cls.setExistWithoutInConnections(cls, True)
		# cls.setExistWithoutOutConnections(cls, True)


	def _reset(self, pData:om.MDataBlock):
		"""reset the solver - copy data from input to frameArray"""
		# get input data
		inputDataADH = pData.inputValue(self.aInputData)
		# init if empty
		#attr.jumpToElement(inputDataADH, 0, elementIsArray=False)

		thisFrame = pData.inputValue(self.aThisFrame).asInt()

		inputFrameData = getSolverFrameData(
			inputDataADH,
			atFrame=thisFrame,
			floatMObject=self.aInputFloat,
			getOutputValues=False
		)

		WpSolverStart.sentFrameData = inputFrameData

		# get frames to store
		framesToStore = pData.inputValue(self.aFramesToStore).asInt()

		# get frameData array
		frameDataArrayHandle : om.MArrayDataHandle = pData.outputArrayValue(self.aFrameData)

		# grow frame array if needed
		for i in range(framesToStore - len(frameDataArrayHandle.builder())):
			attr.jumpToElement(frameDataArrayHandle, i, elementIsArray=True)

		# reset all frames
		for i in range(framesToStore):
			#print("reset frame", i)
			#print("test src numeric value", inputDataADH.inputValue().asDouble())
			attr.jumpToElement(frameDataArrayHandle, i, elementIsArray=True)
			setSolverFrameData(frameDataArrayHandle.outputValue(),
			                   inputFrameData,
			                   floatMObject=self.aFrameFloat,
			                   )



	def _getSolverEndNode(self, pData:om.MDataBlock)->om.MObject:
		"""return the solver end node"""
		thisMFn = self.thisMFn()
		balanceWheelPlug : om.MPlug = thisMFn.findPlug(self.aBalanceWheel, True)
		connectedPlugs = balanceWheelPlug.destinations()
		if not connectedPlugs:
			return None
		else:
			return connectedPlugs[0].node()


	def _shuffleFrameDatas(self, pData:om.MDataBlock):
		"""
		shuffle current entries of frameArray down one,
		then retrieve data from end node and copy into first slot of frameArray,
		"""
		framesToStore = pData.inputValue(self.aFramesToStore).asInt()
		loadFrameArrayDH : om.MArrayDataHandle = pData.outputArrayValue(self.aFrameData)
		setFrameArrayDH : om.MArrayDataHandle = pData.outputArrayValue(self.aFrameData)

		for i in range(framesToStore):
			if i == 0:
				continue
			fromIdx = framesToStore - i
			toIdx = fromIdx - 1
			loadFrameArrayDH.jumpToPhysicalElement(fromIdx)
			setFrameArrayDH.jumpToPhysicalElement(toIdx)

			frameData = getSolverFrameData(
				loadFrameArrayDH.outputValue(),
				-1,
				floatMObject=self.aFrameFloat,
				getOutputValues=True
			)

			setSolverFrameData(
				setFrameArrayDH.outputValue(),
				frameData,
				floatMObject=self.aFrameFloat,
			)

		# copy new frame data into first slot
		setFrameArrayDH.jumpToPhysicalElement(0)
		setSolverFrameData(setFrameArrayDH.outputValue(),
		                   self.sentFrameData,
		                   floatMObject=self.aFrameFloat,
		                   )

		setFrameArrayDH.setClean()



	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		"""check for reset - otherwise retrieve data from end node"""

		print("solverStart compute")
		if pData.isClean(pPlug):
			print("plug clean", pPlug)
			return

		# flip balancewheel
		pData.outputValue(self.aBalanceWheel).setBool(
			not pData.outputValue(self.aBalanceWheel).asBool()
		)
		#pData.setClean(self.aBalanceWheel)

		#return

		currentFrame = pData.inputValue(self.aThisFrame).asInt()
		resetFrame = pData.inputValue(self.aResetFrame).asInt()
		if resetFrame == currentFrame:
			print("reset solver")
			self._reset(pData)
			#pData.setClean(self.aFrameArray)
			pData.setClean(self.aBalanceWheel)
			pData.setClean(self.aFrameData)
			pData.setClean(pPlug)
			return
		# pData.setClean(pPlug)
		# return

		# # get end node
		# endNode = self._getSolverEndNode(pData)
		# if endNode is None:
		# 	print("no end node connected")
		# 	return
		# endDH = self._getSolverEndDH(pData, endNode)
		# self._shuffleFrameDatas(pData, endDH)

		# new version - copy whatever is in receivedData

		# self._setFromClsData(pData)
		# receivedDataADH = pData.outputArrayValue(self.aReceivedData)
		self._shuffleFrameDatas(pData)
		pData.setClean(self.aBalanceWheel)
		pData.setClean(self.aFrameData)
		pData.setClean(pPlug)
		return


	@classmethod
	def testNode(cls):
		"""test node creation"""
		from maya import cmds
		solverStart = cmds.createNode(cls.typeName())

		solverEnd = cmds.createNode("wpSolverEnd")
		cmds.connectAttr(solverStart + ".solverEnd", solverEnd + ".solverStart")
		#return
		# add basic solver input data
		adl = cmds.createNode("addDoubleLinear")
		cmds.setAttr(adl + ".input1", 2)
		cmds.connectAttr(adl + ".output", solverStart + ".inputData.inputFloat[0]")


		# check copying through to frame array
		checkAdl = cmds.createNode("addDoubleLinear")
		cmds.connectAttr(solverStart + ".frameData[0].frameFloat[0]", checkAdl + ".input1")
		cmds.setAttr(checkAdl + ".input2", 1)
		print("passthrough", cmds.getAttr(checkAdl + ".output"))

		# set up full solver system
		cmds.connectAttr(checkAdl + ".output", solverEnd + ".frameData.frameFloat[0]")
		print("resetFrame out", cmds.getAttr(solverEnd + ".frameData.frameFloat[0]"))
		# change the current frame
		cmds.setAttr(solverStart + ".thisFrame", 1)
		print("next frame out", cmds.getAttr(solverEnd + ".frameData.frameFloat[0]"))
		#return

		# check that new value is one more than previous
		print("next frame add", cmds.getAttr(checkAdl + ".output"))


		cube = cmds.polyCube()[0]
		cmds.connectAttr(solverEnd + ".frameData.frameFloat[0]", cube + ".ty")
		#return
		# connect time to solver
		cmds.setAttr(solverStart + ".resetFrame", 1)
		cmds.setAttr(solverStart + ".thisFrame", 1)
		cmds.connectAttr("time1.outTime", solverStart + ".thisFrame")

