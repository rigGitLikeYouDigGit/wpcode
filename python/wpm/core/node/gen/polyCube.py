

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
	node : PolyCube = None
	pass
class DepthPlug(Plug):
	node : PolyCube = None
	pass
class HeightPlug(Plug):
	node : PolyCube = None
	pass
class SubdivisionsDepthPlug(Plug):
	node : PolyCube = None
	pass
class SubdivisionsHeightPlug(Plug):
	node : PolyCube = None
	pass
class SubdivisionsWidthPlug(Plug):
	node : PolyCube = None
	pass
class TexturePlug(Plug):
	node : PolyCube = None
	pass
class WidthPlug(Plug):
	node : PolyCube = None
	pass
# endregion


# define node class
class PolyCube(PolyPrimitive):
	createUVs_ : CreateUVsPlug = PlugDescriptor("createUVs")
	depth_ : DepthPlug = PlugDescriptor("depth")
	height_ : HeightPlug = PlugDescriptor("height")
	subdivisionsDepth_ : SubdivisionsDepthPlug = PlugDescriptor("subdivisionsDepth")
	subdivisionsHeight_ : SubdivisionsHeightPlug = PlugDescriptor("subdivisionsHeight")
	subdivisionsWidth_ : SubdivisionsWidthPlug = PlugDescriptor("subdivisionsWidth")
	texture_ : TexturePlug = PlugDescriptor("texture")
	width_ : WidthPlug = PlugDescriptor("width")

	# node attributes

	typeName = "polyCube"
	apiTypeInt = 438
	apiTypeStr = "kPolyCube"
	typeIdInt = 1346590018
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["createUVs", "depth", "height", "subdivisionsDepth", "subdivisionsHeight", "subdivisionsWidth", "texture", "width"]
	nodeLeafPlugs = ["createUVs", "depth", "height", "subdivisionsDepth", "subdivisionsHeight", "subdivisionsWidth", "texture", "width"]
	pass

