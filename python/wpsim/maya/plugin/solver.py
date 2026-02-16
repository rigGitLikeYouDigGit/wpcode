from __future__ import annotations
import types, typing as T
import pprint, traceback
from wplib import log
from importlib import reload

import jax
from jax import numpy as jnp

from wpm import om, cmds
from wpm.lib.plugin import PluginNodeTemplate, MayaPyPluginAid, PluginNodeIdData


from wpsim.kine import state, builder
reload(state)
reload(builder)
from wpsim.maya.plugin.rigidbody import WpSimBodyMPxData


def maya_useNewAPI():
	pass
class WpSimRigidSolverNode(PluginNodeTemplate, om.MPxNode):
	"""
	constraint for point constraint between two rigid bodies
	"""
	@classmethod
	def pluginNodeIdData(cls) ->PluginNodeIdData:
		return PluginNodeIdData(
			"wpSimRigidSolver",
			om.MPxNode.kDependNode
		)

	simBuilder:builder.SimBuilder
	#userMeshMap : dict[str, builder.UserMesh]
	bodyMap : dict[str, builder.BuilderBody]

	# we leave it to maya's caching to store sim results
	# frameStateMap : dict[int, state.SimFrameState]

	@classmethod
	def nodeCreator(cls):
		obj = cls()
		obj.simBuilder = None
		return obj

	def postConstructor(self):

		try:
			self.simBuilder = builder.SimBuilder(self.name())
		except Exception as e:
			log("Error creating SimBuilder for node {}: {}".format(
				self.name()))
			traceback.print_exc()

	@classmethod
	def initialiseNode(cls):
		tFn = om.MFnTypedAttribute()
		mFn = om.MFnMatrixAttribute()
		nFn = om.MFnNumericAttribute()
		cFn = om.MFnCompoundAttribute()
		eFn = om.MFnEnumAttribute()

		# leave blank to just use name of node
		cls.aName = tFn.create("name", "name", om.MFnData.kString)

		cls.aBody = tFn.create("body", "body", WpSimBodyMPxData.kTypeId)
		tFn.array = True
		tFn.usesArrayDataBuilder = True

		# time attributes
		cls.aFrameBind = nFn.create("frameBind", "frameBind",
		                            om.MFnNumericData.kInt, 1)

		cls.aFrameCurrent = nFn.create("frameCurrent", "frameCurrent",
		                               om.MFnNumericData.kInt, 1)
		cls.aFramesPerSecond = nFn.create("framesPerSecond", "framesPerSecond",
		                                  om.MFnNumericData.kFloat, 24.0)
		cls.aSteps = nFn.create("steps", "steps",
		                        om.MFnNumericData.kInt, 1)
		nFn.setMin(1)
		cls.aSubSteps = nFn.create("subSteps", "subSteps",
		                           om.MFnNumericData.kInt, 1)
		nFn.setMin(1)
		# sim mode - off, active, quasistatic
		cls.aMode = eFn.create("mode", "mode", 1)
		eFn.addField("off", 0)
		eFn.addField("active", 1)
		eFn.addField("quasistatic", 2)

		cls.aSimFirstFrame = nFn.create("simFirstFrame", "simFirstFrame",
		                                om.MFnNumericData.kBoolean, 0)

		cls.aQuasistaticFrames = nFn.create("quasistaticFrames", "quasistaticFrames",
		                                   om.MFnNumericData.kInt, 10)
		nFn.setMin(1)

		cls.aGravity = nFn.create("gravity", "gravity",
		                          om.MFnNumericData.k3Float, 0.0, -9.81, 0.0)

		"""for later use as a linkage designer - pass in target paths
		for named transforms to link to bodies - 
		inputs will be values of named parametres, and target paths
		for given transforms to match
		"""
		cls.aTargetPath = cFn.create("targetPath", "targetPath")
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cls.aTargetPathName = tFn.create("targetPathName", "targetPathName", om.MFnData.kString)
		cls.aTargetPathCurve = tFn.create("targetPathCurve", "targetPathCurve", om.MFnData.kNurbsCurve)


		# output body data - use name lookup to match input bodies
		cls.aOut = cFn.create("out", "out")
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cls.aOutName = tFn.create("outName", "outName", om.MFnData.kString)
		cls.aOutBodyMatrix = mFn.create("outBodyMatrix", "outBodyMatrix")

		cFn.addChild(cls.aOutName)
		cFn.addChild(cls.aOutBodyMatrix)

		cls.addAttributes([
			cls.aName,
			cls.aBody,
			cls.aFrameBind,
			cls.aFrameCurrent,
			cls.aFramesPerSecond,
			cls.aMode,
			cls.aQuasistaticFrames,
			cls.aSimFirstFrame,
			cls.aSteps,
			cls.aSubSteps,
			cls.aGravity,
			cls.aOut
		])

	def getSimStaticData(self, pData:om.MDataBlock)->state.SimStaticData:
		return state.SimStaticData(
			dtFrame=1.0 / pData.inputValue(self.aFramesPerSecond).asFloat(),
			iterationCount=pData.inputValue(self.aSteps).asFloat(),
			substepCount=pData.inputValue(self.aSubSteps).asFloat(),
		)

	def bind(self, pPlug:om.MPlug, pData:om.MDataBlock, paramData:paramDataCls=None):
		""" bind the simulation state from input data """

		name = pData.inputValue(self.aName).asString().strip()
		if not name:
			name = self.name()
		self.simBuilder.name = name

		simStaticData = self.getSimStaticData(pData)
		bodyDatas = []

		bodyArrayHandle : om.MArrayDataHandle = pData.inputArrayValue(
			self.aBody)
		for i in range(len(bodyArrayHandle)):
			bodyArrayHandle.jumpToPhysicalElement(i)
			bodyHandle = bodyArrayHandle.inputValue()
			bodyData : WpSimBodyMPxData = om.MFnPluginData(
				bodyHandle.data()).data()
			bodyDatas.append(bodyData._data)
		self.simBuilder.builderBodyMap = {
			b.name: b for b in bodyDatas
		}
		self.simBuilder.bind()

	def simFrame(self, pPlug:om.MPlug, pData:om.MDataBlock, frame:int):
		""" simulate to given frame - assumes bound state
		"""
		log(f"Simulating frame", frame, "for node", self.name())
		self.simBuilder.simFrame()

	def dispatchQuasistatic(self, pPlug:om.MPlug, pData:om.MDataBlock):
		""" dispatch quasistatic sim over nFrames
		"""
		nFrames = pData.inputValue(self.aQuasistaticFrames).asInt()
		self.bind(pPlug, pData)
		for f in range(nFrames):
			self.simFrame(pPlug, pData, f)

	def dispatchActive(self, pPlug:om.MPlug, pData:om.MDataBlock):
		currentFrame = pData.inputValue(self.aFrameCurrent).asInt()
		bindFrame = pData.inputValue(self.aFrameBind).asInt()
		if currentFrame == bindFrame:
			# bind sim to initial buffers
			self.bind(pPlug, pData)
			simFirstFrame = pData.inputValue(self.aSimFirstFrame).asBool()
			if simFirstFrame:
				self.simFrame(pPlug, pData, currentFrame)
			return

		if pData.isClean(self.aFrameCurrent): # already simmed this frame
			return
		self.simFrame(pPlug, pData, currentFrame)

	# nothing to do

	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		""" check if it's a live time compute - if so, sim a frame
		 """
		if pData.isClean(pPlug):
			return

		mode = pData.inputValue(self.aMode).asInt()
		# off mode - do nothing
		if mode == 0:
			pData.setClean(pPlug)
			return

		# active - bind on first frame, then run per frame
		elif mode == 1:
			self.dispatchActive(pPlug, pData)

		# quasistatic - bind and then run number of frames
		elif mode == 2:
			self.dispatchQuasistatic(pPlug, pData)

		pData.setClean(self.aFrameCurrent)
		pData.setClean(self.aFrameBind)

		pData.setClean(pPlug)
		pData.setClean(self.aOut)
		pData.setClean(self.aOutBodyMatrix)


