from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from dataclasses import dataclass, field, asdict

import jax
from jax import numpy as jnp

from wpm import om, cmds
from wpm.lib.plugin import (PluginNodeTemplate, MayaPyPluginAid,
                            PluginNodeIdData)
from wpm.lib.plugin.template import PluginMPxData
from wpsim.maya.plugin import lib
from wpsim.kine.builder import (BuilderBody, BuilderMesh,
                                BuilderTransform, BuilderNurbsCurve,
                                BuilderRamp, BuilderMultiRamps)
"""supporting injecting user-controlled ramps into sim,
we allow either a maya ramp parametre, 
or a physical curve in 0-1 xy plane (local to its transform)

ramps sampled as 32 point float arrays .

maybe overkill to have separate node for ramps, but why not
"""


class WpSimRampMPxData(PluginMPxData):
	"""Ramp data
	"""
	clsName = "wpSimBodyMPxData"
	dataClsT = BuilderMultiRamps
	kTypeId = om.MTypeId(0x00112233)


class WpSimRampNode(PluginNodeTemplate, om.MPxNode):
	"""
	"""
	@classmethod
	def pluginNodeIdData(cls) ->PluginNodeIdData:
		return PluginNodeIdData(
			"wpSimRamp",
			om.MPxNode.kDependNode
		)

	@classmethod
	def initialiseNode(cls):
		tFn = om.MFnTypedAttribute()
		mFn = om.MFnMatrixAttribute()
		nFn = om.MFnNumericAttribute()
		cFn = om.MFnCompoundAttribute()
		eFn = om.MFnEnumAttribute()

		cls.aRampArr = cFn.create("ramps", "ramps")
		cFn.array = True
		cFn.usesArrayDataBuilder = True

		# leave blank to just use name of node
		cls.aName = tFn.create("name", "name",
		                    om.MFnData.kString)
		cFn.addChild(cls.aName)

		cls.aRamp = om.MRampAttribute().createCurveRamp("ramp", "ramp")
		cFn.addChild(cls.aRamp)

		cls.aCurve = tFn.create("curve", "curve", om.MFnData.kNurbsCurve)
		tFn.storable = False
		cFn.addChild(cls.aCurve)

		cls.addAttribute(cls.aRampArr)

		cls.aRampData = tFn.create(
			"data", "data", WpSimRampMPxData.kTypeId)
		tFn.writable = False
		cFn.addChild(cls.aRampData)

		cls.setAttributesAffect(
			[cls.aName, cls.aRampArr, cls.aRamp, cls.aCurve],
			[cls.aRampData]
		)


	def syncRampIndex(self, pPlug:om.MPlug, ppData:om.MData, index:int,
	                  multiDataCls:BuilderMultiRamps):
		"""sync a single ramp index from input to output
		"""
		inArrDH = ppData.inputArrayValue(self.aRampArr)
		inDH = inArrDH.jumpToPhysicalElement(index).inputValue()
		name = inDH.child(self.aName).asString()
		curveData : om.MDataHandle = inDH.child(self.aCurve)
		if curveData.data().isNull(): # sample ramp attr
			ramp = om.MRampAttribute(self.thisMObject(), self.aRamp)
			values = [ramp.getValueAtPosition(i / 31.0)
			          for i in range(32)]
		else: # sample nurbs curve
			curveFn = om.MFnNurbsCurve(curveData.asNurbsCurve())
			values = [curveFn.getPointAtParam(i / 31.0, om.MSpace.kObject).y
			          for i in range(32)]

		dataCls = BuilderRamp(
			name, jnp.array(values)
		)
		multiDataCls.ramps[index] = dataCls

	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		"""check if lengths match - if not, sync everything
		if yes, check each input for dirty

		TODO: selectively sync only dirty indices
		"""
		if pData.isClean(pPlug):
			return

		inArrDH = pData.inputArrayValue(self.aRampArr)
		outArrDH = pData.outputArrayValue(self.aRampData)

		builder = om.MArrayDataBuilder()
		builder.growArray(len(inArrDH))
		mpxData = WpSimRampMPxData()
		multiDataCls = BuilderMultiRamps([None] * len(inArrDH))
		outArrDH.set(builder)
		indicesToSync = range(len(inArrDH))

		# if len(inArrDH) != len(outArrDH):
		# 	# lengths don't match, rebuild everything
		# 	builder = om.MArrayDataBuilder()
		# 	builder.growArray(len(inArrDH))
		# 	outArrDH.set(builder)
		# 	indicesToSync = range(len(inArrDH))
		# else:
		# 	for i in range(len(inArrDH)):
		# 		inDH = inArrDH.jumpToPhysicalElement(i).inputValue()
		# 		for at in [self.aName, self.aRamp, self.aCurve]:
		# 			if not pData.isClean()
		# 		outDH = outArrDH.getElement(i)
		# 		if inDH.isDirty():
		# 			indicesToSync.append(i)

		for i in indicesToSync:
			self.syncRampIndex(pPlug, pData, i, multiDataCls)

		mpxData.setDataCls(multiDataCls)
		outDH = pData.outputValue(self.aRampData)
		outDH.setMPxData(mpxData)

		pData.setClean(self.aRampData)
		pData.setClean(pPlug)


