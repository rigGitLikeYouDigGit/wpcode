

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
	node : PolyCone = None
	pass
class HeightPlug(Plug):
	node : PolyCone = None
	pass
class RadiusPlug(Plug):
	node : PolyCone = None
	pass
class RoundCapPlug(Plug):
	node : PolyCone = None
	pass
class SubdivisionsAxisPlug(Plug):
	node : PolyCone = None
	pass
class SubdivisionsCapPlug(Plug):
	node : PolyCone = None
	pass
class SubdivisionsHeightPlug(Plug):
	node : PolyCone = None
	pass
class TexturePlug(Plug):
	node : PolyCone = None
	pass
# endregion


# define node class
class PolyCone(PolyPrimitive):
	createUVs_ : CreateUVsPlug = PlugDescriptor("createUVs")
	height_ : HeightPlug = PlugDescriptor("height")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	roundCap_ : RoundCapPlug = PlugDescriptor("roundCap")
	subdivisionsAxis_ : SubdivisionsAxisPlug = PlugDescriptor("subdivisionsAxis")
	subdivisionsCap_ : SubdivisionsCapPlug = PlugDescriptor("subdivisionsCap")
	subdivisionsHeight_ : SubdivisionsHeightPlug = PlugDescriptor("subdivisionsHeight")
	texture_ : TexturePlug = PlugDescriptor("texture")

	# node attributes

	typeName = "polyCone"
	apiTypeInt = 437
	apiTypeStr = "kPolyCone"
	typeIdInt = 1346588494
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["createUVs", "height", "radius", "roundCap", "subdivisionsAxis", "subdivisionsCap", "subdivisionsHeight", "texture"]
	nodeLeafPlugs = ["createUVs", "height", "radius", "roundCap", "subdivisionsAxis", "subdivisionsCap", "subdivisionsHeight", "texture"]
	pass

