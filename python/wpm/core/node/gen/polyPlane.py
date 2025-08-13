

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
	node : PolyPlane = None
	pass
class HeightPlug(Plug):
	node : PolyPlane = None
	pass
class SubdivisionsHeightPlug(Plug):
	node : PolyPlane = None
	pass
class SubdivisionsWidthPlug(Plug):
	node : PolyPlane = None
	pass
class TexturePlug(Plug):
	node : PolyPlane = None
	pass
class WidthPlug(Plug):
	node : PolyPlane = None
	pass
# endregion


# define node class
class PolyPlane(PolyPrimitive):
	createUVs_ : CreateUVsPlug = PlugDescriptor("createUVs")
	height_ : HeightPlug = PlugDescriptor("height")
	subdivisionsHeight_ : SubdivisionsHeightPlug = PlugDescriptor("subdivisionsHeight")
	subdivisionsWidth_ : SubdivisionsWidthPlug = PlugDescriptor("subdivisionsWidth")
	texture_ : TexturePlug = PlugDescriptor("texture")
	width_ : WidthPlug = PlugDescriptor("width")

	# node attributes

	typeName = "polyPlane"
	typeIdInt = 1347241299
	pass

