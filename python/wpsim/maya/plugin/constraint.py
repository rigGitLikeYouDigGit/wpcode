from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpm import om, cmds
from wpm.lib.plugin import PluginNodeTemplate, MayaPyPluginAid, PluginNodeIdData


class WpSimRigidConstraintPointNode(om.MPxNode, PluginNodeTemplate):
	"""
	constraint for point constraint between two rigid bodies
	"""
	@classmethod
	def pluginNodeIdData(cls) ->PluginNodeIdData:
		return PluginNodeIdData(
			"wpSimRigidConstraintPoint",
			om.MPxNode.kDependNode
		)

	@classmethod
	def initialiseNode(cls):
		tFn = om.MFnTypedAttribute()
		mFn = om.MFnMatrixAttribute()
		nFn = om.MFnNumericAttribute()
		cFn = om.MFnCompoundAttribute()
		# leave blank to just use name of node
		cls.aName = tFn.create("name", "name", om.MFnData.kString)

		# connected bodies
		cls.aBodyAName = tFn.create("bodyNameA", "bodyNameA",
		                            om.MFnData.kString)
		cls.aBodyBName = tFn.create("bodyNameB", "bodyNameB",
		                            om.MFnData.kString)
		cls.aBodyATfName = tFn.create("bodyATfName", "bodyATfName",
		                              om.MFnData.kString)
		cls.aBodyBTfName = tFn.create("bodyBTfName", "bodyBTfName",
		                              om.MFnData.kString)

		cls.aActive = nFn.create("active", "active", om.MFnNumericData.kBoolean, True)

		cls.aWeight = nFn.create("weight", "weight", om.MFnNumericData.kDouble, 1.0)
		nFn.setMin(0.0)

		cls.aDamping = nFn.create("damping", "damping", om.MFnNumericData.kDouble, 0.0)
		nFn.setMin(0.0)

		cls.aIndex = nFn.create("index", "index", om.MFnNumericData.kInt, -1)
		nFn.setMin(-1)
		nFn.writable = False

		cls.aRestLength = nFn.create("restLength", "restLength",
		                        om.MFnNumericData.kFloat)
		cls.aTargetLength = nFn.create("targetLength", "targetLength",
		                               om.MFnNumericData.kFloat)



class WpSimRigidConstraintHingeNode(om.MPxNode, PluginNodeTemplate):
	"""
	constraint for hinge between two rigid bodies
	"""
	@classmethod
	def pluginNodeIdData(cls) ->PluginNodeIdData:
		return PluginNodeIdData(
			"wpSimRigidConstraintHinge",
			om.MPxNode.kDependNode
		)

	@classmethod
	def initialiseNode(cls):
		tFn = om.MFnTypedAttribute()
		mFn = om.MFnMatrixAttribute()
		nFn = om.MFnNumericAttribute()
		cFn = om.MFnCompoundAttribute()
		# leave blank to just use name of node
		cls.aName = tFn.create("name", "name", om.MFnData.kString)

		# connected bodies
		cls.aBodyAName = tFn.create("bodyNameA", "bodyNameA",
		                            om.MFnData.kString)
		cls.aBodyBName = tFn.create("bodyNameB", "bodyNameB",
		                            om.MFnData.kString)
		cls.aBodyATfName = tFn.create("bodyATfName", "bodyATfName",
		                              om.MFnData.kString)
		cls.aBodyBTfName = tFn.create("bodyBTfName", "bodyBTfName",
		                              om.MFnData.kString)

		cls.aActive = nFn.create("active", "active", om.MFnNumericData.kBoolean, True)

		cls.aWeight = nFn.create("weight", "weight", om.MFnNumericData.kDouble, 1.0)
		nFn.setMin(0.0)

		cls.aDamping = nFn.create("damping", "damping", om.MFnNumericData.kDouble, 0.0)
		nFn.setMin(0.0)

		cls.aIndex = nFn.create("index", "index", om.MFnNumericData.kInt, -1)
		nFn.setMin(-1)
		nFn.writable = False

		cls.aRestQuat = nFn.create("restQuat", "restQuat", om.MFnNumericData.k4Double)
		cls.aRestPos = nFn.create("restPos", "restPos", om.MFnNumericData.k3Double)
