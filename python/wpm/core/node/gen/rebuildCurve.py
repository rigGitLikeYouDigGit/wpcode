

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbstractBaseCreate = Catalogue.AbstractBaseCreate
else:
	from .. import retriever
	AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
	assert AbstractBaseCreate

# add node doc



# region plug type defs
class DegreePlug(Plug):
	node : RebuildCurve = None
	pass
class EndKnotsPlug(Plug):
	node : RebuildCurve = None
	pass
class FitRebuildPlug(Plug):
	node : RebuildCurve = None
	pass
class InputCurvePlug(Plug):
	node : RebuildCurve = None
	pass
class KeepControlPointsPlug(Plug):
	node : RebuildCurve = None
	pass
class KeepEndPointsPlug(Plug):
	node : RebuildCurve = None
	pass
class KeepRangePlug(Plug):
	node : RebuildCurve = None
	pass
class KeepTangentsPlug(Plug):
	node : RebuildCurve = None
	pass
class MatchCurvePlug(Plug):
	node : RebuildCurve = None
	pass
class OutputCurvePlug(Plug):
	node : RebuildCurve = None
	pass
class RebuildTypePlug(Plug):
	node : RebuildCurve = None
	pass
class SmartSurfaceCurveRebuildPlug(Plug):
	node : RebuildCurve = None
	pass
class SmoothPlug(Plug):
	node : RebuildCurve = None
	pass
class SpansPlug(Plug):
	node : RebuildCurve = None
	pass
class TolerancePlug(Plug):
	node : RebuildCurve = None
	pass
# endregion


# define node class
class RebuildCurve(AbstractBaseCreate):
	degree_ : DegreePlug = PlugDescriptor("degree")
	endKnots_ : EndKnotsPlug = PlugDescriptor("endKnots")
	fitRebuild_ : FitRebuildPlug = PlugDescriptor("fitRebuild")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	keepControlPoints_ : KeepControlPointsPlug = PlugDescriptor("keepControlPoints")
	keepEndPoints_ : KeepEndPointsPlug = PlugDescriptor("keepEndPoints")
	keepRange_ : KeepRangePlug = PlugDescriptor("keepRange")
	keepTangents_ : KeepTangentsPlug = PlugDescriptor("keepTangents")
	matchCurve_ : MatchCurvePlug = PlugDescriptor("matchCurve")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	rebuildType_ : RebuildTypePlug = PlugDescriptor("rebuildType")
	smartSurfaceCurveRebuild_ : SmartSurfaceCurveRebuildPlug = PlugDescriptor("smartSurfaceCurveRebuild")
	smooth_ : SmoothPlug = PlugDescriptor("smooth")
	spans_ : SpansPlug = PlugDescriptor("spans")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "rebuildCurve"
	apiTypeInt = 90
	apiTypeStr = "kRebuildCurve"
	typeIdInt = 1314013763
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["degree", "endKnots", "fitRebuild", "inputCurve", "keepControlPoints", "keepEndPoints", "keepRange", "keepTangents", "matchCurve", "outputCurve", "rebuildType", "smartSurfaceCurveRebuild", "smooth", "spans", "tolerance"]
	nodeLeafPlugs = ["degree", "endKnots", "fitRebuild", "inputCurve", "keepControlPoints", "keepEndPoints", "keepRange", "keepTangents", "matchCurve", "outputCurve", "rebuildType", "smartSurfaceCurveRebuild", "smooth", "spans", "tolerance"]
	pass

