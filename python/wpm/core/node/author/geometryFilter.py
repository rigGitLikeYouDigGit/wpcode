from __future__ import annotations
import typing as T

from ..gen.geometryFilter import GeometryFilter as GenGeometryFilter
from ..gen.geometryFilter import _BASE_
#from ..gen import GeometryFilter as GenGeometryFilter
import numpy as np
from wpm import cmds, om, WN, arr, oma, Plug
from wplib.totype import to

# if T.TYPE_CHECKING:
# 	from ...node.base import Plug


class GeometryFilter(GenGeometryFilter):
	""" builtin methods for getting history, future ,
	origshapes etc
	"""
	MFn : oma.MFnGeometryFilter
	clsApiType = om.MFn.kGeometryFilt # did they just misspell it?

	def inputMeshPlug(self)->Plug:
		return self.input_(0).inputGeometry_

	def inputGeometryShape(self)->WN.Shape|None:
		dgIter = om.MItDependencyGraph(self.inputMeshPlug().MPlug,
		                               om.MItDependencyGraph.kUpstream,
		                               om.MItDependencyGraph.kNodeLevel)
		for node in dgIter:
			if node.hasFn(om.MFn.kShape):
				return WN(node)
		return None

	def outputMeshPlug(self)->Plug:
		return self.outputGeometry_(0)

	def shapesInFuture(self)->list[WN.Shape]:
		shapes = []
		dgIter = om.MItDependencyGraph(self.outputMeshPlug().MPlug,
		                               om.MItDependencyGraph.kDownstream,
		                               om.MItDependencyGraph.kNodeLevel)

		for node in dgIter:
			if node.hasFn(om.MFn.kShape):
				shapes.append(WN(node))
			if node.hasFn(om.MFn.kDagNode):
				dgIter.prune()
		return shapes


