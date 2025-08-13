

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyMoveEdge = retriever.getNodeCls("PolyMoveEdge")
assert PolyMoveEdge
if T.TYPE_CHECKING:
	from .. import PolyMoveEdge

# add node doc



# region plug type defs
class CompBoundingBoxMaxXPlug(Plug):
	parent : CompBoundingBoxMaxPlug = PlugDescriptor("compBoundingBoxMax")
	node : PolyExtrudeEdge = None
	pass
class CompBoundingBoxMaxYPlug(Plug):
	parent : CompBoundingBoxMaxPlug = PlugDescriptor("compBoundingBoxMax")
	node : PolyExtrudeEdge = None
	pass
class CompBoundingBoxMaxZPlug(Plug):
	parent : CompBoundingBoxMaxPlug = PlugDescriptor("compBoundingBoxMax")
	node : PolyExtrudeEdge = None
	pass
class CompBoundingBoxMaxPlug(Plug):
	compBoundingBoxMaxX_ : CompBoundingBoxMaxXPlug = PlugDescriptor("compBoundingBoxMaxX")
	cxx_ : CompBoundingBoxMaxXPlug = PlugDescriptor("compBoundingBoxMaxX")
	compBoundingBoxMaxY_ : CompBoundingBoxMaxYPlug = PlugDescriptor("compBoundingBoxMaxY")
	cxy_ : CompBoundingBoxMaxYPlug = PlugDescriptor("compBoundingBoxMaxY")
	compBoundingBoxMaxZ_ : CompBoundingBoxMaxZPlug = PlugDescriptor("compBoundingBoxMaxZ")
	cxz_ : CompBoundingBoxMaxZPlug = PlugDescriptor("compBoundingBoxMaxZ")
	node : PolyExtrudeEdge = None
	pass
class CompBoundingBoxMinXPlug(Plug):
	parent : CompBoundingBoxMinPlug = PlugDescriptor("compBoundingBoxMin")
	node : PolyExtrudeEdge = None
	pass
class CompBoundingBoxMinYPlug(Plug):
	parent : CompBoundingBoxMinPlug = PlugDescriptor("compBoundingBoxMin")
	node : PolyExtrudeEdge = None
	pass
class CompBoundingBoxMinZPlug(Plug):
	parent : CompBoundingBoxMinPlug = PlugDescriptor("compBoundingBoxMin")
	node : PolyExtrudeEdge = None
	pass
class CompBoundingBoxMinPlug(Plug):
	compBoundingBoxMinX_ : CompBoundingBoxMinXPlug = PlugDescriptor("compBoundingBoxMinX")
	cnx_ : CompBoundingBoxMinXPlug = PlugDescriptor("compBoundingBoxMinX")
	compBoundingBoxMinY_ : CompBoundingBoxMinYPlug = PlugDescriptor("compBoundingBoxMinY")
	cny_ : CompBoundingBoxMinYPlug = PlugDescriptor("compBoundingBoxMinY")
	compBoundingBoxMinZ_ : CompBoundingBoxMinZPlug = PlugDescriptor("compBoundingBoxMinZ")
	cnz_ : CompBoundingBoxMinZPlug = PlugDescriptor("compBoundingBoxMinZ")
	node : PolyExtrudeEdge = None
	pass
class DivisionsPlug(Plug):
	node : PolyExtrudeEdge = None
	pass
class InputProfilePlug(Plug):
	node : PolyExtrudeEdge = None
	pass
class KeepFacesTogetherPlug(Plug):
	node : PolyExtrudeEdge = None
	pass
class OffsetPlug(Plug):
	node : PolyExtrudeEdge = None
	pass
class SmoothingAnglePlug(Plug):
	node : PolyExtrudeEdge = None
	pass
class TaperPlug(Plug):
	node : PolyExtrudeEdge = None
	pass
class TaperCurve_FloatValuePlug(Plug):
	parent : TaperCurvePlug = PlugDescriptor("taperCurve")
	node : PolyExtrudeEdge = None
	pass
class TaperCurve_InterpPlug(Plug):
	parent : TaperCurvePlug = PlugDescriptor("taperCurve")
	node : PolyExtrudeEdge = None
	pass
class TaperCurve_PositionPlug(Plug):
	parent : TaperCurvePlug = PlugDescriptor("taperCurve")
	node : PolyExtrudeEdge = None
	pass
class TaperCurvePlug(Plug):
	taperCurve_FloatValue_ : TaperCurve_FloatValuePlug = PlugDescriptor("taperCurve_FloatValue")
	cfv_ : TaperCurve_FloatValuePlug = PlugDescriptor("taperCurve_FloatValue")
	taperCurve_Interp_ : TaperCurve_InterpPlug = PlugDescriptor("taperCurve_Interp")
	ci_ : TaperCurve_InterpPlug = PlugDescriptor("taperCurve_Interp")
	taperCurve_Position_ : TaperCurve_PositionPlug = PlugDescriptor("taperCurve_Position")
	cp_ : TaperCurve_PositionPlug = PlugDescriptor("taperCurve_Position")
	node : PolyExtrudeEdge = None
	pass
class ThicknessPlug(Plug):
	node : PolyExtrudeEdge = None
	pass
class TwistPlug(Plug):
	node : PolyExtrudeEdge = None
	pass
# endregion


# define node class
class PolyExtrudeEdge(PolyMoveEdge):
	compBoundingBoxMaxX_ : CompBoundingBoxMaxXPlug = PlugDescriptor("compBoundingBoxMaxX")
	compBoundingBoxMaxY_ : CompBoundingBoxMaxYPlug = PlugDescriptor("compBoundingBoxMaxY")
	compBoundingBoxMaxZ_ : CompBoundingBoxMaxZPlug = PlugDescriptor("compBoundingBoxMaxZ")
	compBoundingBoxMax_ : CompBoundingBoxMaxPlug = PlugDescriptor("compBoundingBoxMax")
	compBoundingBoxMinX_ : CompBoundingBoxMinXPlug = PlugDescriptor("compBoundingBoxMinX")
	compBoundingBoxMinY_ : CompBoundingBoxMinYPlug = PlugDescriptor("compBoundingBoxMinY")
	compBoundingBoxMinZ_ : CompBoundingBoxMinZPlug = PlugDescriptor("compBoundingBoxMinZ")
	compBoundingBoxMin_ : CompBoundingBoxMinPlug = PlugDescriptor("compBoundingBoxMin")
	divisions_ : DivisionsPlug = PlugDescriptor("divisions")
	inputProfile_ : InputProfilePlug = PlugDescriptor("inputProfile")
	keepFacesTogether_ : KeepFacesTogetherPlug = PlugDescriptor("keepFacesTogether")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	smoothingAngle_ : SmoothingAnglePlug = PlugDescriptor("smoothingAngle")
	taper_ : TaperPlug = PlugDescriptor("taper")
	taperCurve_FloatValue_ : TaperCurve_FloatValuePlug = PlugDescriptor("taperCurve_FloatValue")
	taperCurve_Interp_ : TaperCurve_InterpPlug = PlugDescriptor("taperCurve_Interp")
	taperCurve_Position_ : TaperCurve_PositionPlug = PlugDescriptor("taperCurve_Position")
	taperCurve_ : TaperCurvePlug = PlugDescriptor("taperCurve")
	thickness_ : ThicknessPlug = PlugDescriptor("thickness")
	twist_ : TwistPlug = PlugDescriptor("twist")

	# node attributes

	typeName = "polyExtrudeEdge"
	apiTypeInt = 793
	apiTypeStr = "kPolyExtrudeEdge"
	typeIdInt = 1346721861
	MFnCls = om.MFnDependencyNode
	pass

