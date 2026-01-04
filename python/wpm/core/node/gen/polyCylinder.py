

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyPrimitive = Catalogue.PolyPrimitive
else:
	from .. import retriever
	PolyPrimitive = retriever.getNodeCls("PolyPrimitive")
	assert PolyPrimitive

# add node doc



# region plug type defs
class CreateUVsPlug(Plug):
	node : PolyCylinder = None
	pass
class HeightPlug(Plug):
	node : PolyCylinder = None
	pass
class Maya2022UVsPlug(Plug):
	node : PolyCylinder = None
	pass
class Maya70Plug(Plug):
	node : PolyCylinder = None
	pass
class RadiusPlug(Plug):
	node : PolyCylinder = None
	pass
class RoundCapPlug(Plug):
	node : PolyCylinder = None
	pass
class RoundCapHeightCompensationPlug(Plug):
	node : PolyCylinder = None
	pass
class SubdivisionsAxisPlug(Plug):
	node : PolyCylinder = None
	pass
class SubdivisionsCapsPlug(Plug):
	node : PolyCylinder = None
	pass
class SubdivisionsHeightPlug(Plug):
	node : PolyCylinder = None
	pass
class TexturePlug(Plug):
	node : PolyCylinder = None
	pass
# endregion


# define node class
class PolyCylinder(PolyPrimitive):
	createUVs_ : CreateUVsPlug = PlugDescriptor("createUVs")
	height_ : HeightPlug = PlugDescriptor("height")
	maya2022UVs_ : Maya2022UVsPlug = PlugDescriptor("maya2022UVs")
	maya70_ : Maya70Plug = PlugDescriptor("maya70")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	roundCap_ : RoundCapPlug = PlugDescriptor("roundCap")
	roundCapHeightCompensation_ : RoundCapHeightCompensationPlug = PlugDescriptor("roundCapHeightCompensation")
	subdivisionsAxis_ : SubdivisionsAxisPlug = PlugDescriptor("subdivisionsAxis")
	subdivisionsCaps_ : SubdivisionsCapsPlug = PlugDescriptor("subdivisionsCaps")
	subdivisionsHeight_ : SubdivisionsHeightPlug = PlugDescriptor("subdivisionsHeight")
	texture_ : TexturePlug = PlugDescriptor("texture")

	# node attributes

	typeName = "polyCylinder"
	apiTypeInt = 439
	apiTypeStr = "kPolyCylinder"
	typeIdInt = 1346591052
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["createUVs", "height", "maya2022UVs", "maya70", "radius", "roundCap", "roundCapHeightCompensation", "subdivisionsAxis", "subdivisionsCaps", "subdivisionsHeight", "texture"]
	nodeLeafPlugs = ["createUVs", "height", "maya2022UVs", "maya70", "radius", "roundCap", "roundCapHeightCompensation", "subdivisionsAxis", "subdivisionsCaps", "subdivisionsHeight", "texture"]
	pass

