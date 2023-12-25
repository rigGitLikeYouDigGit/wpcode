from __future__ import annotations
import typing as T


from maya.api import OpenMaya as om, OpenMayaRender as omr, OpenMayaUI as omui, OpenMayaAnim as oma
from wpm.lib.plugin import PluginNodeTemplate, PluginNodeIdData, attr

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

	# inputs
	aResetFrame : om.MObject
	aThisFrame : om.MObject

	aFramesToStore : om.MObject # number of frames to store
	# by default only one single frame

	# root compound object
	# copy data from this compound plug into loop input
	aInputData : om.MObject

	# outputs
	aFrameArray: om.MObject  # array of data copies from previous frames
	aFrameData : om.MObject  # array object containing data for a single frame

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

		cls.aInputData = tFn.create("inputData", "inputData", om.MFnData.kAny)
		tFn.array = True
		tFn.usesArrayDataBuilder = True
		tFn.readable = False
		tFn.writable = True

		cls.aFrameArray = cFn.create("frameArray", "frameArray")
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cFn.readable = True
		cFn.writable = False

		cls.aFrameData = tFn.create("frameData", "frameData", om.MFnData.kAny)
		tFn.array = True
		tFn.usesArrayDataBuilder = True
		tFn.readable = True
		tFn.writable = False
		cFn.addChild(cls.aFrameData)

		cls.aBalanceWheel = attr.makeBalanceWheelAttr("solverEnd", writable=False)

		cls.driverMObjects = [cls.aResetFrame, cls.aThisFrame, cls.aFramesToStore, cls.aInputData]
		cls.drivenMObjects = [cls.aFrameArray, cls.aBalanceWheel]

		cls.addAttributes(cls.drivenMObjects, cls.driverMObjects)
		cls.setAttributesAffect(cls.driverMObjects, cls.drivenMObjects)

		# cls.setExistWithoutInConnections(cls, True)
		# cls.setExistWithoutOutConnections(cls, True)


	def _reset(self, pData:om.MDataBlock):
		"""reset the solver - copy data from input to frameArray"""
		# get input data
		inputDataADH = pData.inputArrayValue(self.aInputData)
		# init if empty
		attr.jumpToElement(inputDataADH, 0, elementIsArray=False)

		# get frames to store
		framesToStore = pData.inputValue(self.aFramesToStore).asInt()
		#return
		# get frameArray
		frameArrayHandle : om.MArrayDataHandle = pData.outputArrayValue(self.aFrameArray)

		# grow frame array if needed
		for i in range(framesToStore - len(frameArrayHandle.builder())):
			attr.jumpToElement(frameArrayHandle, i, elementIsArray=True)

		# reset all frames
		# copy constructor for MArrayDataHandle doesn't seem to work here -
		# so we can't just call the .copy() method on array children of compound attrs
		for i in range(framesToStore):
			print("reset frame", i)
			print("test src numeric value", inputDataADH.inputValue().asDouble())
			attr.jumpToElement(frameArrayHandle, i, elementIsArray=True)
			frameADH = om.MArrayDataHandle(
				frameArrayHandle.outputValue().child(self.aFrameData)
			)
			for n in range(len(inputDataADH)):
				frameADH.jumpToPhysicalElement(n)
				inputDataADH.jumpToPhysicalElement(n)
				frameADH.outputValue().copy(inputDataADH.inputValue())


	def _getSolverEndNode(self, pData:om.MDataBlock)->om.MObject:
		"""return the solver end node"""
		thisMFn = self.thisMFn()
		balanceWheelPlug : om.MPlug = thisMFn.findPlug(self.aBalanceWheel, True)
		connectedPlugs = balanceWheelPlug.destinations()
		if not connectedPlugs:
			return None
		else:
			return connectedPlugs[0].node()

	def _getSolverEndDH(self, pData:om.MDataBlock, endNode:om.MObject)->om.MArrayDataHandle:
		"""get the data from the end node
		this might hard crash maya"""
		endFn = om.MFnDependencyNode(endNode)
		frameDataPlug = endFn.findPlug("frameData", True)
		frameDataHandle = om.MArrayDataHandle(frameDataPlug.asMDataHandle())
		return frameDataHandle

	def _shuffleFrameDatas(self, pData:om.MDataBlock, newFrameDH:om.MArrayDataHandle):
		"""
		shuffle current entries of frameArray down one,
		then retrieve data from end node and copy into first slot of frameArray,
		"""
		framesToStore = pData.inputValue(self.aFramesToStore).asInt()
		loadFrameArrayDH : om.MArrayDataHandle = pData.outputArrayValue(self.aFrameArray)
		setFrameArrayDH : om.MArrayDataHandle = pData.outputArrayValue(self.aFrameArray)

		for i in range(framesToStore):
			if i == 0:
				continue
			fromIdx = framesToStore - i
			loadFrameArrayDH.jumpToPhysicalElement(fromIdx)
			setFrameArrayDH.jumpToPhysicalElement(i)

			loadFrameDataADH = om.MArrayDataHandle(
				loadFrameArrayDH.child(self.aFrameData))
			setFrameDataADH = om.MArrayDataHandle(
				setFrameArrayDH.child(self.aFrameData))
			for n in range(len(loadFrameDataADH)):
				loadFrameDataADH.jumpToPhysicalElement(n)
				setFrameDataADH.jumpToPhysicalElement(n)
				setFrameDataADH.outputValue().copy(loadFrameDataADH.outputValue())

			# setFrameArrayDH.outputValue().copy(loadFrameArrayDH.outputValue())

		# copy new frame data into first slot
		setFrameArrayDH.jumpToPhysicalElement(0)
		setFrameDataADH = om.MArrayDataHandle(
			setFrameArrayDH.outputValue().child(self.aFrameData))
		# newFrameDataADH = om.MArrayDataHandle(
		# 	newFrameDH.child(self.aFrameData))
		for n in range(len(newFrameDH)):
			setFrameDataADH.jumpToPhysicalElement(n)
			newFrameDH.jumpToPhysicalElement(n)
			# newFrameDataADH.jumpToPhysicalElement(n)
			setFrameDataADH.outputValue().copy(newFrameDH.outputValue())
		# setFrameArrayDH.outputValue().copy(newFrameDH)

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
			pData.setClean(self.aFrameArray)
			pData.setClean(self.aBalanceWheel)
			pData.setClean(self.aFrameData)
			pData.setClean(pPlug)
			return
		pData.setClean(pPlug)
		#return

		# get end node
		endNode = self._getSolverEndNode(pData)
		if endNode is None:
			print("no end node connected")
			return
		endDH = self._getSolverEndDH(pData, endNode)
		self._shuffleFrameDatas(pData, endDH)

	@classmethod
	def testNode(cls):
		"""test node creation"""
		from maya import cmds
		solverStart = cmds.createNode(cls.typeName())

		solverEnd = cmds.createNode("wpSolverEnd")
		cmds.connectAttr(solverStart + ".solverEnd", solverEnd + ".solverStart")

		# add basic solver input data
		adl = cmds.createNode("addDoubleLinear")
		cmds.setAttr(adl + ".input1", 2)
		cmds.connectAttr(adl + ".output", solverStart + ".inputData[0]")


		# check copying through to frame array
		checkAdl = cmds.createNode("addDoubleLinear")
		cmds.connectAttr(solverStart + ".frameArray[0].frameData[0]", checkAdl + ".input1")
		cmds.setAttr(checkAdl + ".input2", 1)
		print("passthrough", cmds.getAttr(checkAdl + ".output"))

		# set up full solver system
		cmds.connectAttr(checkAdl + ".output", solverEnd + ".frameData[0]")
		print("resetFrame out", cmds.getAttr(solverEnd + ".frameData[0]"))

		# change the current frame
		cmds.setAttr(solverStart + ".thisFrame", 1)
		# check that new value is one more than previous
		print("next frame add", cmds.getAttr(checkAdl + ".output"))
		print("next frame out", cmds.getAttr(solverEnd + ".frameData[0]"))

		cube = cmds.polyCube()[0]
		cmds.connectAttr(solverEnd + ".frameData[0]", cube + ".ty")

		# connect time to solver
		cmds.setAttr(solverStart + ".resetFrame", 1)
		cmds.setAttr(solverStart + ".thisFrame", 1)
		cmds.connectAttr("time1.outTime", solverStart + ".thisFrame")

