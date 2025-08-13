

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
	node : PolyPrism = None
	pass
class LengthPlug(Plug):
	node : PolyPrism = None
	pass
class Maya2022UVsPlug(Plug):
	node : PolyPrism = None
	pass
class NumberOfSidesPlug(Plug):
	node : PolyPrism = None
	pass
class SideLengthPlug(Plug):
	node : PolyPrism = None
	pass
class SubdivisionsCapsPlug(Plug):
	node : PolyPrism = None
	pass
class SubdivisionsHeightPlug(Plug):
	node : PolyPrism = None
	pass
class TexturePlug(Plug):
	node : PolyPrism = None
	pass
# endregion


# define node class
class PolyPrism(PolyPrimitive):
	createUVs_ : CreateUVsPlug = PlugDescriptor("createUVs")
	length_ : LengthPlug = PlugDescriptor("length")
	maya2022UVs_ : Maya2022UVsPlug = PlugDescriptor("maya2022UVs")
	numberOfSides_ : NumberOfSidesPlug = PlugDescriptor("numberOfSides")
	sideLength_ : SideLengthPlug = PlugDescriptor("sideLength")
	subdivisionsCaps_ : SubdivisionsCapsPlug = PlugDescriptor("subdivisionsCaps")
	subdivisionsHeight_ : SubdivisionsHeightPlug = PlugDescriptor("subdivisionsHeight")
	texture_ : TexturePlug = PlugDescriptor("texture")

	# node attributes

	typeName = "polyPrism"
	apiTypeInt = 968
	apiTypeStr = "kPolyPrism"
	typeIdInt = 1347441225
	MFnCls = om.MFnDependencyNode
	pass

