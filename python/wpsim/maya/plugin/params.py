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
from wpsim import lib as simlib

def maya_useNewAPI():
	pass
# class WpSimBodyMPxData(PluginMPxData):
# 	"""Body Data MPxData for WpSim Maya Plugin
# 	"""
# 	clsName = "wpSimBodyMPxData"
# 	dataClsT = BuilderBody
# 	kTypeId = om.MTypeId(0x00112233)

"""
wpSimResource node loads given python module into sim space and 
catalogues all user sim functions to JIT in the sim.

parameters need to be set on final sim dataclass when numeric value
specified
"""

class WpSimParamsNode(PluginNodeTemplate, om.MPxNode):
	"""
	load python files for user parts of sim
	hold individual params for interface from maya
	"""
	@classmethod
	def pluginNodeIdData(cls) ->PluginNodeIdData:
		return PluginNodeIdData(
			"wpSimResource",
			om.MPxNode.kDependNode
		)

	def __init__(self):
		super().__init__()
		self.pyModNamespace = {}

	@classmethod
	def initialiseNode(cls):
		tFn = om.MFnTypedAttribute()
		mFn = om.MFnMatrixAttribute()
		nFn = om.MFnNumericAttribute()
		cFn = om.MFnCompoundAttribute()
		eFn = om.MFnEnumAttribute()
		gFn = om.MFnGenericAttribute()

		cls.aPyModule = cFn.create("pyModule", "pyModule")
		cFn.array = True
		cFn.usesArrayDataBuilder = True
		cFn.writable = True
		cFn.readable = False

		cls.aPyModulePath = tFn.create("path", "path",
		                               om.MFnData.kString)
		tFn.writable = True
		tFn.readable = False
		cFn.addChild(cls.aPyModulePath)

		cls.aReload = nFn.create("reload", "reload", om.kNumberData.kBoolean, 0)
		nFn.writable = True

		# for reloading
		cls.aBalanceWheel = nFn.create("balanceWheel", "balanceWheel",
		                             om.MFnNumericData.kInt, 1)
		nFn.writable = False
		nFn.readable = True

		cls.aParam = cFn.create("param", "param")
		cFn.array = True
		cFn.usesArrayDataBuilder = True

		cls.aParamName = tFn.create("paramName", "paramName",
		                               om.MFnData.kString)
		cls.aParamStr = tFn.create("paramStr", "paramStr",
		                               om.MFnData.kString)
		tFn.readable = False
		cls.aParamValue = lib.makeGenericInputAttribute(gFn, "paramValue")
		cFn.addChild(cls.aParamName)
		cFn.addChild(cls.aParamStr)
		cFn.addChild(cls.aParamValue)


		cls.addAttribute(cls.aPyModule)
		cls.addAttribute(cls.aParam)

		cls.setAttributesAffect(
			[cls.aPyModulePath,
			 cls.aReload,
			 cls.aParamName,
			 cls.aParamStr,
			 # attrs above require recompile, below just sets value
			 cls.aParamValue,],
			[cls.aBalanceWheel]
		)


