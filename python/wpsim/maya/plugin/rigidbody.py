from __future__ import annotations
import types, typing as T
import pprint

import numpy as np

from wplib import log
from dataclasses import dataclass
import dataclasses

import jax
from jax import numpy as jnp

from wpm import om, cmds
from wpm.lib.plugin import (PluginNodeTemplate, MayaPyPluginAid,
                            PluginNodeIdData)
from wpm.lib.plugin.template import PluginMPxData
from wpsim.maya.plugin import lib
from wpsim.kine.builder import (BuilderBody, BuilderMesh,
                                BuilderTransform, BuilderNurbsCurve)

"""we don't unpack / process geo data until solver node 
to cut down transfer time"""
# @dataclass(frozen=True)
# class BodyData:
# 	name:str
# 	matrix:om.MMatrix
# 	mesh:om.MObject
# 	active:int # 0=disabled, 1=active, 2=static
# 	mass:float
# 	damping:float
# 	index:int
# 	auxTf:list[tuple[str, om.MMatrix]]
# 	auxCurve:list[tuple[str, om.MObject]]
# 	com : om.MMatrix # doubles as inertial tensor for now

class WpSimBodyMPxData(PluginMPxData):
	"""Body Data MPxData for WpSim Maya Plugin
	"""
	clsName = "wpSimBodyMPxData"
	dataClsT = BuilderBody
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
		#print("initialising WpSimRigidBodyNode")
		log("initialising WpSimRigidBodyNode")
		tFn = om.MFnTypedAttribute()
		mFn = om.MFnMatrixAttribute()
		nFn = om.MFnNumericAttribute()
		cFn = om.MFnCompoundAttribute()
		eFn = om.MFnEnumAttribute()
		# leave blank to just use name of node
		cls.aBodyName = tFn.create("name", "name",
		                    om.MFnData.kString)
		# worldspace initial matrix
		cls.aBodyMatrix = mFn.create("matrix", "matrix" )
		# combined collision and mass mesh
		cls.aBodyMesh = tFn.create("mesh", "mesh", om.MFnData.kMesh)
		tFn.storable = False

		cls.aBodyActive = eFn.create(
			"active", "active",
			1)
		eFn.addField("disabled", 0)
		eFn.addField("active", 1)
		eFn.addField("static", 2)
		eFn.default = 1

		cls.aBodyMass = nFn.create("mass", "mass", om.MFnNumericData.kDouble,
		                           1.0)
		nFn.setMin(0.0)
		cls.aBodyDamping = nFn.create("damping", "damping",
		                              om.MFnNumericData.kDouble,
		                              0.0)
		nFn.setMin(0.0)

		# aux geo
		cls.auxBodyTf = cFn.create("auxTf", "auxTf")
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cls.auxBodyTfName = tFn.create("auxTfName", "auxTfName",
		                               om.MFnData.kString)
		# local space matrix
		cls.auxBodyMatrix = mFn.create("auxTfMatrix", "auxTfMatrix")
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

		cls.aCom = mFn.create("com", "com")
		nFn.writable = False

		cls.aBodyData = tFn.create(
			"bodyData", "bodyData", WpSimBodyMPxData.kTypeId)
		tFn.writable = False

		cls.addAttribute(cls.aBodyName)
		cls.addAttribute(cls.aBodyMatrix) #####
		cls.addAttribute(cls.aBodyMesh)
		cls.addAttribute(cls.aBodyActive)
		cls.addAttribute(cls.aBodyMass)
		cls.addAttribute(cls.aBodyDamping)
		cls.addAttribute(cls.auxBodyTf) #####
		cls.addAttribute(cls.auxBodyCurve)
		cls.addAttribute(cls.aCom)
		cls.addAttribute(cls.aBodyData)

		# cls.driverMObjects = [cls.aBodyName, cls.aBodyMatrix, cls.aBodyMesh,
		#                       cls.aBodyActive, cls.aBodyMass,
		#                       cls.aBodyDamping, cls.auxBodyTf,
		#                       cls.auxBodyCurve]
		# cls.addAttribute(cls.aBodyData)
		# cls.addAttribute(cls.aBodyData)
		#
		# cls.addAttributes(cls.driverMObjects)
		# cls.addAttributes([cls.aBodyData])
		cls.setAttributesAffect(
			[cls.aBodyName, cls.aBodyMatrix, cls.aBodyMesh,
			 cls.aBodyActive, cls.aBodyMass, cls.aBodyDamping,
			 cls.auxBodyTf, cls.auxBodyCurve],
			[cls.aBodyData]
		)


	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		"""if mesh is given, compute centre of mass and inertia -
		otherwise just use the incoming matrix"""
		if pData.isClean(pPlug):
			return

		matrix = jnp.array(pData.inputValue(self.aBodyMatrix).asMatrix())
		name = pData.inputValue(self.aBodyName).asString().strip()
		if not name:
			name = om.MFnDependencyNode(self.thisMObject()).name()

		meshMap = {}
		curveMap = {}
		transformMap = {}

		if pData.inputValue(self.aBodyMesh).data().isNull():
			# no mesh, just use madtrix
			comMat = matrix
		else:
			builderMesh = lib.builderMeshDataFromMFnMesh(
				om.MFnMesh(pData.inputValue(self.aBodyMesh).data()),
				name=f"{name}_mesh_MAIN",
				parent=name
			)
			com = builderMesh.com
			inertia = builderMesh.inertia
			comMat = om.MMatrix(
				inertia[0], inertia[1], inertia[2], 0.0,
				inertia[3], inertia[4], inertia[5], 0.0,
				inertia[6], inertia[7], inertia[8], 0.0,
				com[0], com[1], com[2], 1.0
			)
			meshMap["MAIN"] = builderMesh

		for i in range(pData.inputValue(self.auxBodyTf).numElements()):
			elemPlug = pData.inputValue(self.auxBodyTf).elementByPhysicalIndex(i)
			tfName = elemPlug.child(self.auxBodyTfName).asString()
			mat = jnp.array(elemPlug.child(self.auxBodyMatrix).asMatrix())
			transformMap[tfName] = BuilderTransform(f"{name}_tf_{tfName}",
			                                name, mat)

		auxCurveMap = {}
		for i in range(pData.inputValue(self.auxBodyCurve).numElements()):
			elemPlug = pData.inputValue(self.auxBodyCurve).elementByPhysicalIndex(i)
			curveName = elemPlug.child(self.auxBodyCurveName).asString()
			curve = elemPlug.child(self.auxBodyCurveData).data()
			auxCurveMap[curveName] = lib.builderCurveDataFromMFnNurbsCurve(
				om.MFnNurbsCurve(curve),
				name=f"{name}_curve_{curveName}",
				parent=name
			)


		datacls = BuilderBody(
			name = name,
			restPos=matrix[3, :3],
			restQuat=jnp.array(lib.quaternionFromMatrix(om.MMatrix(matrix))),
			meshMap=meshMap,
			curveMap=curveMap,
			transformMap=transformMap,
			com=comMat[3, :3],
			inertia=comMat[3, :3],
			mass=pData.inputValue(self.aBodyMass).asDouble(),
			active=pData.inputValue(self.aBodyActive).asInt(),
			damping=pData.inputValue(self.aBodyDamping).asDouble(),
		)

		data = WpSimBodyMPxData()
		data.setData(datacls)
		pData.outputValue(self.aBodyData).setMPxData(data)
		pData.outputValue(self.aCom).setMMatrix(comMat)

		pData.setClean(self.aBodyData)
		pData.setClean(self.aCom)
		pData.setClean(pPlug)
		return
