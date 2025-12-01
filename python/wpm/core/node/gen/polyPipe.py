

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyPrimitive = retriever.getNodeCls("PolyPrimitive")
assert PolyPrimitive
if T.TYPE_CHECKING:
	from .. import PolyPrimitive

# add node doc



# region plug type defs
class CreateUVsPlug(Plug):
	node : PolyPipe = None
	pass
class HeightPlug(Plug):
	node : PolyPipe = None
	pass
class RadiusPlug(Plug):
	node : PolyPipe = None
	pass
class RoundCapPlug(Plug):
	node : PolyPipe = None
	pass
class RoundCapHeightCompensationPlug(Plug):
	node : PolyPipe = None
	pass
class SubdivisionsAxisPlug(Plug):
	node : PolyPipe = None
	pass
class SubdivisionsCapsPlug(Plug):
	node : PolyPipe = None
	pass
class SubdivisionsHeightPlug(Plug):
	node : PolyPipe = None
	pass
class TexturePlug(Plug):
	node : PolyPipe = None
	pass
class ThicknessPlug(Plug):
	node : PolyPipe = None
	pass
# endregion


# define node class
class PolyPipe(PolyPrimitive):
	createUVs_ : CreateUVsPlug = PlugDescriptor("createUVs")
	height_ : HeightPlug = PlugDescriptor("height")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	roundCap_ : RoundCapPlug = PlugDescriptor("roundCap")
	roundCapHeightCompensation_ : RoundCapHeightCompensationPlug = PlugDescriptor("roundCapHeightCompensation")
	subdivisionsAxis_ : SubdivisionsAxisPlug = PlugDescriptor("subdivisionsAxis")
	subdivisionsCaps_ : SubdivisionsCapsPlug = PlugDescriptor("subdivisionsCaps")
	subdivisionsHeight_ : SubdivisionsHeightPlug = PlugDescriptor("subdivisionsHeight")
	texture_ : TexturePlug = PlugDescriptor("texture")
	thickness_ : ThicknessPlug = PlugDescriptor("thickness")

	# node attributes

	typeName = "polyPipe"
	apiTypeInt = 982
	apiTypeStr = "kPolyPipe"
	typeIdInt = 1347438928
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["createUVs", "height", "radius", "roundCap", "roundCapHeightCompensation", "subdivisionsAxis", "subdivisionsCaps", "subdivisionsHeight", "texture", "thickness"]
	nodeLeafPlugs = ["createUVs", "height", "radius", "roundCap", "roundCapHeightCompensation", "subdivisionsAxis", "subdivisionsCaps", "subdivisionsHeight", "texture", "thickness"]
	pass

