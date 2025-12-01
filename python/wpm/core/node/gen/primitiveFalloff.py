

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Transform = retriever.getNodeCls("Transform")
assert Transform
if T.TYPE_CHECKING:
	from .. import Transform

# add node doc



# region plug type defs
class EndPlug(Plug):
	node : PrimitiveFalloff = None
	pass
class NegativeSizeXPlug(Plug):
	parent : NegativeSizePlug = PlugDescriptor("negativeSize")
	node : PrimitiveFalloff = None
	pass
class NegativeSizeYPlug(Plug):
	parent : NegativeSizePlug = PlugDescriptor("negativeSize")
	node : PrimitiveFalloff = None
	pass
class NegativeSizeZPlug(Plug):
	parent : NegativeSizePlug = PlugDescriptor("negativeSize")
	node : PrimitiveFalloff = None
	pass
class NegativeSizePlug(Plug):
	negativeSizeX_ : NegativeSizeXPlug = PlugDescriptor("negativeSizeX")
	nsx_ : NegativeSizeXPlug = PlugDescriptor("negativeSizeX")
	negativeSizeY_ : NegativeSizeYPlug = PlugDescriptor("negativeSizeY")
	nsy_ : NegativeSizeYPlug = PlugDescriptor("negativeSizeY")
	negativeSizeZ_ : NegativeSizeZPlug = PlugDescriptor("negativeSizeZ")
	nsz_ : NegativeSizeZPlug = PlugDescriptor("negativeSizeZ")
	node : PrimitiveFalloff = None
	pass
class OutputWeightFunctionPlug(Plug):
	node : PrimitiveFalloff = None
	pass
class PositiveSizeXPlug(Plug):
	parent : PositiveSizePlug = PlugDescriptor("positiveSize")
	node : PrimitiveFalloff = None
	pass
class PositiveSizeYPlug(Plug):
	parent : PositiveSizePlug = PlugDescriptor("positiveSize")
	node : PrimitiveFalloff = None
	pass
class PositiveSizeZPlug(Plug):
	parent : PositiveSizePlug = PlugDescriptor("positiveSize")
	node : PrimitiveFalloff = None
	pass
class PositiveSizePlug(Plug):
	positiveSizeX_ : PositiveSizeXPlug = PlugDescriptor("positiveSizeX")
	psx_ : PositiveSizeXPlug = PlugDescriptor("positiveSizeX")
	positiveSizeY_ : PositiveSizeYPlug = PlugDescriptor("positiveSizeY")
	psy_ : PositiveSizeYPlug = PlugDescriptor("positiveSizeY")
	positiveSizeZ_ : PositiveSizeZPlug = PlugDescriptor("positiveSizeZ")
	psz_ : PositiveSizeZPlug = PlugDescriptor("positiveSizeZ")
	node : PrimitiveFalloff = None
	pass
class PrimitivePlug(Plug):
	node : PrimitiveFalloff = None
	pass
class Ramp_FloatValuePlug(Plug):
	parent : RampPlug = PlugDescriptor("ramp")
	node : PrimitiveFalloff = None
	pass
class Ramp_InterpPlug(Plug):
	parent : RampPlug = PlugDescriptor("ramp")
	node : PrimitiveFalloff = None
	pass
class Ramp_PositionPlug(Plug):
	parent : RampPlug = PlugDescriptor("ramp")
	node : PrimitiveFalloff = None
	pass
class RampPlug(Plug):
	ramp_FloatValue_ : Ramp_FloatValuePlug = PlugDescriptor("ramp_FloatValue")
	rmpfv_ : Ramp_FloatValuePlug = PlugDescriptor("ramp_FloatValue")
	ramp_Interp_ : Ramp_InterpPlug = PlugDescriptor("ramp_Interp")
	rmpi_ : Ramp_InterpPlug = PlugDescriptor("ramp_Interp")
	ramp_Position_ : Ramp_PositionPlug = PlugDescriptor("ramp_Position")
	rmpp_ : Ramp_PositionPlug = PlugDescriptor("ramp_Position")
	node : PrimitiveFalloff = None
	pass
class StartPlug(Plug):
	node : PrimitiveFalloff = None
	pass
class UseOriginalGeometryPlug(Plug):
	node : PrimitiveFalloff = None
	pass
class VertexSpacePlug(Plug):
	node : PrimitiveFalloff = None
	pass
# endregion


# define node class
class PrimitiveFalloff(Transform):
	end_ : EndPlug = PlugDescriptor("end")
	negativeSizeX_ : NegativeSizeXPlug = PlugDescriptor("negativeSizeX")
	negativeSizeY_ : NegativeSizeYPlug = PlugDescriptor("negativeSizeY")
	negativeSizeZ_ : NegativeSizeZPlug = PlugDescriptor("negativeSizeZ")
	negativeSize_ : NegativeSizePlug = PlugDescriptor("negativeSize")
	outputWeightFunction_ : OutputWeightFunctionPlug = PlugDescriptor("outputWeightFunction")
	positiveSizeX_ : PositiveSizeXPlug = PlugDescriptor("positiveSizeX")
	positiveSizeY_ : PositiveSizeYPlug = PlugDescriptor("positiveSizeY")
	positiveSizeZ_ : PositiveSizeZPlug = PlugDescriptor("positiveSizeZ")
	positiveSize_ : PositiveSizePlug = PlugDescriptor("positiveSize")
	primitive_ : PrimitivePlug = PlugDescriptor("primitive")
	ramp_FloatValue_ : Ramp_FloatValuePlug = PlugDescriptor("ramp_FloatValue")
	ramp_Interp_ : Ramp_InterpPlug = PlugDescriptor("ramp_Interp")
	ramp_Position_ : Ramp_PositionPlug = PlugDescriptor("ramp_Position")
	ramp_ : RampPlug = PlugDescriptor("ramp")
	start_ : StartPlug = PlugDescriptor("start")
	useOriginalGeometry_ : UseOriginalGeometryPlug = PlugDescriptor("useOriginalGeometry")
	vertexSpace_ : VertexSpacePlug = PlugDescriptor("vertexSpace")

	# node attributes

	typeName = "primitiveFalloff"
	apiTypeInt = 1140
	apiTypeStr = "kPrimitiveFalloff"
	typeIdInt = 1397116742
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["end", "negativeSizeX", "negativeSizeY", "negativeSizeZ", "negativeSize", "outputWeightFunction", "positiveSizeX", "positiveSizeY", "positiveSizeZ", "positiveSize", "primitive", "ramp_FloatValue", "ramp_Interp", "ramp_Position", "ramp", "start", "useOriginalGeometry", "vertexSpace"]
	nodeLeafPlugs = ["end", "negativeSize", "outputWeightFunction", "positiveSize", "primitive", "ramp", "start", "useOriginalGeometry", "vertexSpace"]
	pass

