

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
class CheckPointsPlug(Plug):
	node : OffsetCos = None
	pass
class ConnectBreaksPlug(Plug):
	node : OffsetCos = None
	pass
class CutLoopPlug(Plug):
	node : OffsetCos = None
	pass
class DistancePlug(Plug):
	node : OffsetCos = None
	pass
class InputCurvePlug(Plug):
	node : OffsetCos = None
	pass
class OutputCurvePlug(Plug):
	node : OffsetCos = None
	pass
class StitchPlug(Plug):
	node : OffsetCos = None
	pass
class SubdivisionDensityPlug(Plug):
	node : OffsetCos = None
	pass
class TolerancePlug(Plug):
	node : OffsetCos = None
	pass
# endregion


# define node class
class OffsetCos(AbstractBaseCreate):
	checkPoints_ : CheckPointsPlug = PlugDescriptor("checkPoints")
	connectBreaks_ : ConnectBreaksPlug = PlugDescriptor("connectBreaks")
	cutLoop_ : CutLoopPlug = PlugDescriptor("cutLoop")
	distance_ : DistancePlug = PlugDescriptor("distance")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	stitch_ : StitchPlug = PlugDescriptor("stitch")
	subdivisionDensity_ : SubdivisionDensityPlug = PlugDescriptor("subdivisionDensity")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "offsetCos"
	apiTypeInt = 81
	apiTypeStr = "kOffsetCos"
	typeIdInt = 1313817427
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["checkPoints", "connectBreaks", "cutLoop", "distance", "inputCurve", "outputCurve", "stitch", "subdivisionDensity", "tolerance"]
	nodeLeafPlugs = ["checkPoints", "connectBreaks", "cutLoop", "distance", "inputCurve", "outputCurve", "stitch", "subdivisionDensity", "tolerance"]
	pass

