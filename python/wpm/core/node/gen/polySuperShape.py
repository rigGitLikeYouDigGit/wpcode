

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : PolySuperShape = None
	pass
class Ellipse0Plug(Plug):
	node : PolySuperShape = None
	pass
class Ellipse1Plug(Plug):
	node : PolySuperShape = None
	pass
class EllipseMirrorPlug(Plug):
	node : PolySuperShape = None
	pass
class Harmonics0Plug(Plug):
	node : PolySuperShape = None
	pass
class Harmonics1Plug(Plug):
	node : PolySuperShape = None
	pass
class Harmonics2Plug(Plug):
	node : PolySuperShape = None
	pass
class Harmonics3Plug(Plug):
	node : PolySuperShape = None
	pass
class Harmonics4Plug(Plug):
	node : PolySuperShape = None
	pass
class Harmonics5Plug(Plug):
	node : PolySuperShape = None
	pass
class Harmonics6Plug(Plug):
	node : PolySuperShape = None
	pass
class Harmonics7Plug(Plug):
	node : PolySuperShape = None
	pass
class HeightBaselinePlug(Plug):
	node : PolySuperShape = None
	pass
class HorizontalDivisionsPlug(Plug):
	node : PolySuperShape = None
	pass
class HorizontalRevolutionsPlug(Plug):
	node : PolySuperShape = None
	pass
class InternalRadiusPlug(Plug):
	node : PolySuperShape = None
	pass
class MergeVerticesPlug(Plug):
	node : PolySuperShape = None
	pass
class OutputPlug(Plug):
	node : PolySuperShape = None
	pass
class RadiusPlug(Plug):
	node : PolySuperShape = None
	pass
class ShapePlug(Plug):
	node : PolySuperShape = None
	pass
class Ultra0Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra1Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra10Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra11Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra12Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra13Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra14Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra15Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra2Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra3Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra4Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra5Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra6Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra7Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra8Plug(Plug):
	node : PolySuperShape = None
	pass
class Ultra9Plug(Plug):
	node : PolySuperShape = None
	pass
class UltraMirrorPlug(Plug):
	node : PolySuperShape = None
	pass
class UvModePlug(Plug):
	node : PolySuperShape = None
	pass
class VerticalDivisionsPlug(Plug):
	node : PolySuperShape = None
	pass
class VerticalOffsetPlug(Plug):
	node : PolySuperShape = None
	pass
class VerticalRevolutionsPlug(Plug):
	node : PolySuperShape = None
	pass
class XOffsetPlug(Plug):
	node : PolySuperShape = None
	pass
class ZOffsetPlug(Plug):
	node : PolySuperShape = None
	pass
# endregion


# define node class
class PolySuperShape(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	ellipse0_ : Ellipse0Plug = PlugDescriptor("ellipse0")
	ellipse1_ : Ellipse1Plug = PlugDescriptor("ellipse1")
	ellipseMirror_ : EllipseMirrorPlug = PlugDescriptor("ellipseMirror")
	harmonics0_ : Harmonics0Plug = PlugDescriptor("harmonics0")
	harmonics1_ : Harmonics1Plug = PlugDescriptor("harmonics1")
	harmonics2_ : Harmonics2Plug = PlugDescriptor("harmonics2")
	harmonics3_ : Harmonics3Plug = PlugDescriptor("harmonics3")
	harmonics4_ : Harmonics4Plug = PlugDescriptor("harmonics4")
	harmonics5_ : Harmonics5Plug = PlugDescriptor("harmonics5")
	harmonics6_ : Harmonics6Plug = PlugDescriptor("harmonics6")
	harmonics7_ : Harmonics7Plug = PlugDescriptor("harmonics7")
	heightBaseline_ : HeightBaselinePlug = PlugDescriptor("heightBaseline")
	horizontalDivisions_ : HorizontalDivisionsPlug = PlugDescriptor("horizontalDivisions")
	horizontalRevolutions_ : HorizontalRevolutionsPlug = PlugDescriptor("horizontalRevolutions")
	internalRadius_ : InternalRadiusPlug = PlugDescriptor("internalRadius")
	mergeVertices_ : MergeVerticesPlug = PlugDescriptor("mergeVertices")
	output_ : OutputPlug = PlugDescriptor("output")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	shape_ : ShapePlug = PlugDescriptor("shape")
	ultra0_ : Ultra0Plug = PlugDescriptor("ultra0")
	ultra1_ : Ultra1Plug = PlugDescriptor("ultra1")
	ultra10_ : Ultra10Plug = PlugDescriptor("ultra10")
	ultra11_ : Ultra11Plug = PlugDescriptor("ultra11")
	ultra12_ : Ultra12Plug = PlugDescriptor("ultra12")
	ultra13_ : Ultra13Plug = PlugDescriptor("ultra13")
	ultra14_ : Ultra14Plug = PlugDescriptor("ultra14")
	ultra15_ : Ultra15Plug = PlugDescriptor("ultra15")
	ultra2_ : Ultra2Plug = PlugDescriptor("ultra2")
	ultra3_ : Ultra3Plug = PlugDescriptor("ultra3")
	ultra4_ : Ultra4Plug = PlugDescriptor("ultra4")
	ultra5_ : Ultra5Plug = PlugDescriptor("ultra5")
	ultra6_ : Ultra6Plug = PlugDescriptor("ultra6")
	ultra7_ : Ultra7Plug = PlugDescriptor("ultra7")
	ultra8_ : Ultra8Plug = PlugDescriptor("ultra8")
	ultra9_ : Ultra9Plug = PlugDescriptor("ultra9")
	ultraMirror_ : UltraMirrorPlug = PlugDescriptor("ultraMirror")
	uvMode_ : UvModePlug = PlugDescriptor("uvMode")
	verticalDivisions_ : VerticalDivisionsPlug = PlugDescriptor("verticalDivisions")
	verticalOffset_ : VerticalOffsetPlug = PlugDescriptor("verticalOffset")
	verticalRevolutions_ : VerticalRevolutionsPlug = PlugDescriptor("verticalRevolutions")
	xOffset_ : XOffsetPlug = PlugDescriptor("xOffset")
	zOffset_ : ZOffsetPlug = PlugDescriptor("zOffset")

	# node attributes

	typeName = "polySuperShape"
	typeIdInt = 1195051
	nodeLeafClassAttrs = ["binMembership", "ellipse0", "ellipse1", "ellipseMirror", "harmonics0", "harmonics1", "harmonics2", "harmonics3", "harmonics4", "harmonics5", "harmonics6", "harmonics7", "heightBaseline", "horizontalDivisions", "horizontalRevolutions", "internalRadius", "mergeVertices", "output", "radius", "shape", "ultra0", "ultra1", "ultra10", "ultra11", "ultra12", "ultra13", "ultra14", "ultra15", "ultra2", "ultra3", "ultra4", "ultra5", "ultra6", "ultra7", "ultra8", "ultra9", "ultraMirror", "uvMode", "verticalDivisions", "verticalOffset", "verticalRevolutions", "xOffset", "zOffset"]
	nodeLeafPlugs = ["binMembership", "ellipse0", "ellipse1", "ellipseMirror", "harmonics0", "harmonics1", "harmonics2", "harmonics3", "harmonics4", "harmonics5", "harmonics6", "harmonics7", "heightBaseline", "horizontalDivisions", "horizontalRevolutions", "internalRadius", "mergeVertices", "output", "radius", "shape", "ultra0", "ultra1", "ultra10", "ultra11", "ultra12", "ultra13", "ultra14", "ultra15", "ultra2", "ultra3", "ultra4", "ultra5", "ultra6", "ultra7", "ultra8", "ultra9", "ultraMirror", "uvMode", "verticalDivisions", "verticalOffset", "verticalRevolutions", "xOffset", "zOffset"]
	pass

