from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpm import om, cmds
from wpm.lib.plugin import PluginNodeTemplate, MayaPyPluginAid, PluginNodeIdData


class WpSimRigidSolverNode(om.MPxNode, PluginNodeTemplate):
	"""
	constraint for point constraint between two rigid bodies
	"""
	@classmethod
	def pluginNodeIdData(cls) ->PluginNodeIdData:
		return PluginNodeIdData(
			"wpSimRigidSolver",
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

		cls.aBody = cFn.create("body", "body")
		cFn.array = True
		cFn.usesArrayDataBuilder = True


