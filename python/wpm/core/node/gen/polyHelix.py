

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
class CoilsPlug(Plug):
	node : PolyHelix = None
	pass
class CreateUVsPlug(Plug):
	node : PolyHelix = None
	pass
class DirectionPlug(Plug):
	node : PolyHelix = None
	pass
class HeightPlug(Plug):
	node : PolyHelix = None
	pass
class Maya2022UVsPlug(Plug):
	node : PolyHelix = None
	pass
class RadiusPlug(Plug):
	node : PolyHelix = None
	pass
class RoundCapPlug(Plug):
	node : PolyHelix = None
	pass
class SubdivisionsAxisPlug(Plug):
	node : PolyHelix = None
	pass
class SubdivisionsCapsPlug(Plug):
	node : PolyHelix = None
	pass
class SubdivisionsCoilPlug(Plug):
	node : PolyHelix = None
	pass
class TexturePlug(Plug):
	node : PolyHelix = None
	pass
class UseOldInitBehaviourPlug(Plug):
	node : PolyHelix = None
	pass
class WidthPlug(Plug):
	node : PolyHelix = None
	pass
# endregion


# define node class
class PolyHelix(PolyPrimitive):
	coils_ : CoilsPlug = PlugDescriptor("coils")
	createUVs_ : CreateUVsPlug = PlugDescriptor("createUVs")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	height_ : HeightPlug = PlugDescriptor("height")
	maya2022UVs_ : Maya2022UVsPlug = PlugDescriptor("maya2022UVs")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	roundCap_ : RoundCapPlug = PlugDescriptor("roundCap")
	subdivisionsAxis_ : SubdivisionsAxisPlug = PlugDescriptor("subdivisionsAxis")
	subdivisionsCaps_ : SubdivisionsCapsPlug = PlugDescriptor("subdivisionsCaps")
	subdivisionsCoil_ : SubdivisionsCoilPlug = PlugDescriptor("subdivisionsCoil")
	texture_ : TexturePlug = PlugDescriptor("texture")
	useOldInitBehaviour_ : UseOldInitBehaviourPlug = PlugDescriptor("useOldInitBehaviour")
	width_ : WidthPlug = PlugDescriptor("width")

	# node attributes

	typeName = "polyHelix"
	apiTypeInt = 986
	apiTypeStr = "kPolyHelix"
	typeIdInt = 1212501065
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["coils", "createUVs", "direction", "height", "maya2022UVs", "radius", "roundCap", "subdivisionsAxis", "subdivisionsCaps", "subdivisionsCoil", "texture", "useOldInitBehaviour", "width"]
	nodeLeafPlugs = ["coils", "createUVs", "direction", "height", "maya2022UVs", "radius", "roundCap", "subdivisionsAxis", "subdivisionsCaps", "subdivisionsCoil", "texture", "useOldInitBehaviour", "width"]
	pass

