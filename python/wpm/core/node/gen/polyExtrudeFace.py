

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyMoveFace = Catalogue.PolyMoveFace
else:
	from .. import retriever
	PolyMoveFace = retriever.getNodeCls("PolyMoveFace")
	assert PolyMoveFace

# add node doc



# region plug type defs
class CompBoundingBoxMaxXPlug(Plug):
	parent : CompBoundingBoxMaxPlug = PlugDescriptor("compBoundingBoxMax")
	node : PolyExtrudeFace = None
	pass
class CompBoundingBoxMaxYPlug(Plug):
	parent : CompBoundingBoxMaxPlug = PlugDescriptor("compBoundingBoxMax")
	node : PolyExtrudeFace = None
	pass
class CompBoundingBoxMaxZPlug(Plug):
	parent : CompBoundingBoxMaxPlug = PlugDescriptor("compBoundingBoxMax")
	node : PolyExtrudeFace = None
	pass
class CompBoundingBoxMaxPlug(Plug):
	compBoundingBoxMaxX_ : CompBoundingBoxMaxXPlug = PlugDescriptor("compBoundingBoxMaxX")
	cxx_ : CompBoundingBoxMaxXPlug = PlugDescriptor("compBoundingBoxMaxX")
	compBoundingBoxMaxY_ : CompBoundingBoxMaxYPlug = PlugDescriptor("compBoundingBoxMaxY")
	cxy_ : CompBoundingBoxMaxYPlug = PlugDescriptor("compBoundingBoxMaxY")
	compBoundingBoxMaxZ_ : CompBoundingBoxMaxZPlug = PlugDescriptor("compBoundingBoxMaxZ")
	cxz_ : CompBoundingBoxMaxZPlug = PlugDescriptor("compBoundingBoxMaxZ")
	node : PolyExtrudeFace = None
	pass
class CompBoundingBoxMinXPlug(Plug):
	parent : CompBoundingBoxMinPlug = PlugDescriptor("compBoundingBoxMin")
	node : PolyExtrudeFace = None
	pass
class CompBoundingBoxMinYPlug(Plug):
	parent : CompBoundingBoxMinPlug = PlugDescriptor("compBoundingBoxMin")
	node : PolyExtrudeFace = None
	pass
class CompBoundingBoxMinZPlug(Plug):
	parent : CompBoundingBoxMinPlug = PlugDescriptor("compBoundingBoxMin")
	node : PolyExtrudeFace = None
	pass
class CompBoundingBoxMinPlug(Plug):
	compBoundingBoxMinX_ : CompBoundingBoxMinXPlug = PlugDescriptor("compBoundingBoxMinX")
	cnx_ : CompBoundingBoxMinXPlug = PlugDescriptor("compBoundingBoxMinX")
	compBoundingBoxMinY_ : CompBoundingBoxMinYPlug = PlugDescriptor("compBoundingBoxMinY")
	cny_ : CompBoundingBoxMinYPlug = PlugDescriptor("compBoundingBoxMinY")
	compBoundingBoxMinZ_ : CompBoundingBoxMinZPlug = PlugDescriptor("compBoundingBoxMinZ")
	cnz_ : CompBoundingBoxMinZPlug = PlugDescriptor("compBoundingBoxMinZ")
	node : PolyExtrudeFace = None
	pass
class DivisionsPlug(Plug):
	node : PolyExtrudeFace = None
	pass
class InputProfilePlug(Plug):
	node : PolyExtrudeFace = None
	pass
class KeepFacesTogetherPlug(Plug):
	node : PolyExtrudeFace = None
	pass
class Maya2012Plug(Plug):
	node : PolyExtrudeFace = None
	pass
class Maya2018Plug(Plug):
	node : PolyExtrudeFace = None
	pass
class Maya2023Plug(Plug):
	node : PolyExtrudeFace = None
	pass
class NewThicknessPlug(Plug):
	node : PolyExtrudeFace = None
	pass
class ReverseAllFacesPlug(Plug):
	node : PolyExtrudeFace = None
	pass
class SmoothingAnglePlug(Plug):
	node : PolyExtrudeFace = None
	pass
class TaperPlug(Plug):
	node : PolyExtrudeFace = None
	pass
class TaperCurve_FloatValuePlug(Plug):
	parent : TaperCurvePlug = PlugDescriptor("taperCurve")
	node : PolyExtrudeFace = None
	pass
class TaperCurve_InterpPlug(Plug):
	parent : TaperCurvePlug = PlugDescriptor("taperCurve")
	node : PolyExtrudeFace = None
	pass
class TaperCurve_PositionPlug(Plug):
	parent : TaperCurvePlug = PlugDescriptor("taperCurve")
	node : PolyExtrudeFace = None
	pass
class TaperCurvePlug(Plug):
	taperCurve_FloatValue_ : TaperCurve_FloatValuePlug = PlugDescriptor("taperCurve_FloatValue")
	cfv_ : TaperCurve_FloatValuePlug = PlugDescriptor("taperCurve_FloatValue")
	taperCurve_Interp_ : TaperCurve_InterpPlug = PlugDescriptor("taperCurve_Interp")
	ci_ : TaperCurve_InterpPlug = PlugDescriptor("taperCurve_Interp")
	taperCurve_Position_ : TaperCurve_PositionPlug = PlugDescriptor("taperCurve_Position")
	cp_ : TaperCurve_PositionPlug = PlugDescriptor("taperCurve_Position")
	node : PolyExtrudeFace = None
	pass
class ThicknessPlug(Plug):
	node : PolyExtrudeFace = None
	pass
class TwistPlug(Plug):
	node : PolyExtrudeFace = None
	pass
# endregion


# define node class
class PolyExtrudeFace(PolyMoveFace):
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
	maya2012_ : Maya2012Plug = PlugDescriptor("maya2012")
	maya2018_ : Maya2018Plug = PlugDescriptor("maya2018")
	maya2023_ : Maya2023Plug = PlugDescriptor("maya2023")
	newThickness_ : NewThicknessPlug = PlugDescriptor("newThickness")
	reverseAllFaces_ : ReverseAllFacesPlug = PlugDescriptor("reverseAllFaces")
	smoothingAngle_ : SmoothingAnglePlug = PlugDescriptor("smoothingAngle")
	taper_ : TaperPlug = PlugDescriptor("taper")
	taperCurve_FloatValue_ : TaperCurve_FloatValuePlug = PlugDescriptor("taperCurve_FloatValue")
	taperCurve_Interp_ : TaperCurve_InterpPlug = PlugDescriptor("taperCurve_Interp")
	taperCurve_Position_ : TaperCurve_PositionPlug = PlugDescriptor("taperCurve_Position")
	taperCurve_ : TaperCurvePlug = PlugDescriptor("taperCurve")
	thickness_ : ThicknessPlug = PlugDescriptor("thickness")
	twist_ : TwistPlug = PlugDescriptor("twist")

	# node attributes

	typeName = "polyExtrudeFace"
	typeIdInt = 1346721862
	nodeLeafClassAttrs = ["compBoundingBoxMaxX", "compBoundingBoxMaxY", "compBoundingBoxMaxZ", "compBoundingBoxMax", "compBoundingBoxMinX", "compBoundingBoxMinY", "compBoundingBoxMinZ", "compBoundingBoxMin", "divisions", "inputProfile", "keepFacesTogether", "maya2012", "maya2018", "maya2023", "newThickness", "reverseAllFaces", "smoothingAngle", "taper", "taperCurve_FloatValue", "taperCurve_Interp", "taperCurve_Position", "taperCurve", "thickness", "twist"]
	nodeLeafPlugs = ["compBoundingBoxMax", "compBoundingBoxMin", "divisions", "inputProfile", "keepFacesTogether", "maya2012", "maya2018", "maya2023", "newThickness", "reverseAllFaces", "smoothingAngle", "taper", "taperCurve", "thickness", "twist"]
	pass

