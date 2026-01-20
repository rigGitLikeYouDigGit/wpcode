from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
from dataclasses import dataclass
import dataclasses

from wpm import om, cmds
from wpm.lib.plugin import (PluginNodeTemplate, MayaPyPluginAid,
                            PluginNodeIdData)
from wpm.lib.plugin.template import PluginMPxData
from wpsim.maya.plugin import lib


@dataclass(frozen=True)
class BodyData:
	name:str
	matrix:om.MMatrix
	mesh:om.MObject
	active:bool
	mass:float
	damping:float
	index:int
	auxTf:list[tuple[str, om.MMatrix]]
	auxCurve:list[tuple[str, om.MObject]]

class WpSimBodyMPxData(PluginMPxData):
	"""Body Data MPxData for WpSim Maya Plugin
	"""
	dataClsT = BodyData
	kTypeId = om.MTypeId(0x00112233)


class WpSimRigidBodyNode(PluginNodeTemplate, om.MPxNode):
	"""Rigid Body Node for WpSim Maya Plugin
	todo: maybe make this a shape
	"""
	@classmethod
	def pluginNodeIdData(cls) ->PluginNodeIdData:
		return PluginNodeIdData(
			"wpSimRigidBody",
			om.MPxNode.kDependNode
		)

	@classmethod
	def initialiseNode(cls):
		tFn = om.MFnTypedAttribute()
		mFn = om.MFnMatrixAttribute()
		nFn = om.MFnNumericAttribute()
		cFn = om.MFnCompoundAttribute()
		# leave blank to just use name of node
		cls.aBodyName = tFn.create("name", "name", om.MFnData.kString)
		# worldspace initial matrix
		cls.aBodyMatrix = mFn.create("matrix", "matrix")
		# combined collision and mass mesh
		cls.aBodyMesh = tFn.create("mesh", "mesh", om.MFnData.kMesh)

		cls.aBodyActive = nFn.create("active", "active",
		                             om.MFnNumericData.kBoolean,
		                             True)

		cls.aBodyMass = nFn.create("mass", "mass", om.MFnNumericData.kDouble,
		                           1.0)
		nFn.setMin(0.0)
		cls.aBodyDamping = nFn.create("damping", "damping",
		                              om.MFnNumericData.kDouble,
		                              0.0)
		nFn.setMin(0.0)
		#
		# cls.aBodyIndex = nFn.create("index", "index", om.MFnNumericData.kInt,
		#                             -1)
		# nFn.setMin(-1)
		# nFn.writable = False

		# aux geo
		cls.auxBodyTf = cFn.create("auxTf", "auxTf")
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cls.auxBodyTfName = tFn.create("auxTfName", "auxTfName",
		                               om.MFnData.kString)
		# local space matrix
		cls.auxBodyMatrix = mFn.create("matrix", "matrix")
		cFn.addChild(cls.auxBodyTfName)
		cFn.addChild(cls.auxBodyMatrix)

		cls.auxBodyCurve = cFn.create("auxCurve", "auxCurve")
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cls.auxBodyCurveName = tFn.create("auxCurveName", "auxCurveName",
		                                  om.MFnData.kString)
		cls.auxBodyCurveData = tFn.create("auxCurveData", "auxCurveData",
		                                  om.MFnData.kNurbsCurve)
		cFn.addChild(cls.auxBodyCurveName)
		cFn.addChild(cls.auxBodyCurveData)

		cls.aBodyData = tFn.create(
			"aBodyData", "aBodyData", WpSimBodyMPxData.kTypeId)
		cls.aBodyData.writable = False

		cls.driverMObjects = [cls.aBodyName, cls.aBodyMatrix, cls.aBodyMesh,
		                      cls.aBodyActive, cls.aBodyMass,
		                      cls.aBodyDamping, cls.auxBodyTf,
		                      cls.auxBodyCurve]
		cls.setAttributesAffect(cls.drivenMObjects, [cls.aBodyData])


	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):

		return self
