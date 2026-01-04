

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierWorld = Catalogue.PolyModifierWorld
else:
	from .. import retriever
	PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
	assert PolyModifierWorld

# add node doc



# region plug type defs
class BridgeOffsetPlug(Plug):
	node : PolyBridgeEdge = None
	pass
class CurveTypePlug(Plug):
	node : PolyBridgeEdge = None
	pass
class DirectionPlug(Plug):
	node : PolyBridgeEdge = None
	pass
class DivisionsPlug(Plug):
	node : PolyBridgeEdge = None
	pass
class InputProfilePlug(Plug):
	node : PolyBridgeEdge = None
	pass
class ReversePlug(Plug):
	node : PolyBridgeEdge = None
	pass
class SmoothingAnglePlug(Plug):
	node : PolyBridgeEdge = None
	pass
class SourceDirectionPlug(Plug):
	node : PolyBridgeEdge = None
	pass
class StartVert1Plug(Plug):
	node : PolyBridgeEdge = None
	pass
class StartVert2Plug(Plug):
	node : PolyBridgeEdge = None
	pass
class TaperPlug(Plug):
	node : PolyBridgeEdge = None
	pass
class TaperCurve_FloatValuePlug(Plug):
	parent : TaperCurvePlug = PlugDescriptor("taperCurve")
	node : PolyBridgeEdge = None
	pass
class TaperCurve_InterpPlug(Plug):
	parent : TaperCurvePlug = PlugDescriptor("taperCurve")
	node : PolyBridgeEdge = None
	pass
class TaperCurve_PositionPlug(Plug):
	parent : TaperCurvePlug = PlugDescriptor("taperCurve")
	node : PolyBridgeEdge = None
	pass
class TaperCurvePlug(Plug):
	taperCurve_FloatValue_ : TaperCurve_FloatValuePlug = PlugDescriptor("taperCurve_FloatValue")
	cfv_ : TaperCurve_FloatValuePlug = PlugDescriptor("taperCurve_FloatValue")
	taperCurve_Interp_ : TaperCurve_InterpPlug = PlugDescriptor("taperCurve_Interp")
	ci_ : TaperCurve_InterpPlug = PlugDescriptor("taperCurve_Interp")
	taperCurve_Position_ : TaperCurve_PositionPlug = PlugDescriptor("taperCurve_Position")
	cp_ : TaperCurve_PositionPlug = PlugDescriptor("taperCurve_Position")
	node : PolyBridgeEdge = None
	pass
class TargetDirectionPlug(Plug):
	node : PolyBridgeEdge = None
	pass
class TwistPlug(Plug):
	node : PolyBridgeEdge = None
	pass
# endregion


# define node class
class PolyBridgeEdge(PolyModifierWorld):
	bridgeOffset_ : BridgeOffsetPlug = PlugDescriptor("bridgeOffset")
	curveType_ : CurveTypePlug = PlugDescriptor("curveType")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	divisions_ : DivisionsPlug = PlugDescriptor("divisions")
	inputProfile_ : InputProfilePlug = PlugDescriptor("inputProfile")
	reverse_ : ReversePlug = PlugDescriptor("reverse")
	smoothingAngle_ : SmoothingAnglePlug = PlugDescriptor("smoothingAngle")
	sourceDirection_ : SourceDirectionPlug = PlugDescriptor("sourceDirection")
	startVert1_ : StartVert1Plug = PlugDescriptor("startVert1")
	startVert2_ : StartVert2Plug = PlugDescriptor("startVert2")
	taper_ : TaperPlug = PlugDescriptor("taper")
	taperCurve_FloatValue_ : TaperCurve_FloatValuePlug = PlugDescriptor("taperCurve_FloatValue")
	taperCurve_Interp_ : TaperCurve_InterpPlug = PlugDescriptor("taperCurve_Interp")
	taperCurve_Position_ : TaperCurve_PositionPlug = PlugDescriptor("taperCurve_Position")
	taperCurve_ : TaperCurvePlug = PlugDescriptor("taperCurve")
	targetDirection_ : TargetDirectionPlug = PlugDescriptor("targetDirection")
	twist_ : TwistPlug = PlugDescriptor("twist")

	# node attributes

	typeName = "polyBridgeEdge"
	apiTypeInt = 995
	apiTypeStr = "kPolyBridgeEdge"
	typeIdInt = 1346523717
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["bridgeOffset", "curveType", "direction", "divisions", "inputProfile", "reverse", "smoothingAngle", "sourceDirection", "startVert1", "startVert2", "taper", "taperCurve_FloatValue", "taperCurve_Interp", "taperCurve_Position", "taperCurve", "targetDirection", "twist"]
	nodeLeafPlugs = ["bridgeOffset", "curveType", "direction", "divisions", "inputProfile", "reverse", "smoothingAngle", "sourceDirection", "startVert1", "startVert2", "taper", "taperCurve", "targetDirection", "twist"]
	pass

