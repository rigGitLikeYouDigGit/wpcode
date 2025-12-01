

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class ConnectBreaksPlug(Plug):
	node : OffsetCurve = None
	pass
class CutLoopPlug(Plug):
	node : OffsetCurve = None
	pass
class CutRadiusPlug(Plug):
	node : OffsetCurve = None
	pass
class DistancePlug(Plug):
	node : OffsetCurve = None
	pass
class InputCurvePlug(Plug):
	node : OffsetCurve = None
	pass
class NormalXPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : OffsetCurve = None
	pass
class NormalYPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : OffsetCurve = None
	pass
class NormalZPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : OffsetCurve = None
	pass
class NormalPlug(Plug):
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	nrx_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	nry_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	nrz_ : NormalZPlug = PlugDescriptor("normalZ")
	node : OffsetCurve = None
	pass
class OutputCurvePlug(Plug):
	node : OffsetCurve = None
	pass
class ParameterPlug(Plug):
	node : OffsetCurve = None
	pass
class ReparameterizePlug(Plug):
	node : OffsetCurve = None
	pass
class StitchPlug(Plug):
	node : OffsetCurve = None
	pass
class SubdivisionDensityPlug(Plug):
	node : OffsetCurve = None
	pass
class TolerancePlug(Plug):
	node : OffsetCurve = None
	pass
class UseGivenNormalPlug(Plug):
	node : OffsetCurve = None
	pass
class UseParameterPlug(Plug):
	node : OffsetCurve = None
	pass
# endregion


# define node class
class OffsetCurve(AbstractBaseCreate):
	connectBreaks_ : ConnectBreaksPlug = PlugDescriptor("connectBreaks")
	cutLoop_ : CutLoopPlug = PlugDescriptor("cutLoop")
	cutRadius_ : CutRadiusPlug = PlugDescriptor("cutRadius")
	distance_ : DistancePlug = PlugDescriptor("distance")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	normal_ : NormalPlug = PlugDescriptor("normal")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	parameter_ : ParameterPlug = PlugDescriptor("parameter")
	reparameterize_ : ReparameterizePlug = PlugDescriptor("reparameterize")
	stitch_ : StitchPlug = PlugDescriptor("stitch")
	subdivisionDensity_ : SubdivisionDensityPlug = PlugDescriptor("subdivisionDensity")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")
	useGivenNormal_ : UseGivenNormalPlug = PlugDescriptor("useGivenNormal")
	useParameter_ : UseParameterPlug = PlugDescriptor("useParameter")

	# node attributes

	typeName = "offsetCurve"
	apiTypeInt = 82
	apiTypeStr = "kOffsetCurve"
	typeIdInt = 1313817429
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["connectBreaks", "cutLoop", "cutRadius", "distance", "inputCurve", "normalX", "normalY", "normalZ", "normal", "outputCurve", "parameter", "reparameterize", "stitch", "subdivisionDensity", "tolerance", "useGivenNormal", "useParameter"]
	nodeLeafPlugs = ["connectBreaks", "cutLoop", "cutRadius", "distance", "inputCurve", "normal", "outputCurve", "parameter", "reparameterize", "stitch", "subdivisionDensity", "tolerance", "useGivenNormal", "useParameter"]
	pass

