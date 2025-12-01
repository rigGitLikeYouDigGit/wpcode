

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
	node : PolyPyramid = None
	pass
class NumberOfSidesPlug(Plug):
	node : PolyPyramid = None
	pass
class SideLengthPlug(Plug):
	node : PolyPyramid = None
	pass
class SubdivisionsCapsPlug(Plug):
	node : PolyPyramid = None
	pass
class SubdivisionsHeightPlug(Plug):
	node : PolyPyramid = None
	pass
class TexturePlug(Plug):
	node : PolyPyramid = None
	pass
# endregion


# define node class
class PolyPyramid(PolyPrimitive):
	createUVs_ : CreateUVsPlug = PlugDescriptor("createUVs")
	numberOfSides_ : NumberOfSidesPlug = PlugDescriptor("numberOfSides")
	sideLength_ : SideLengthPlug = PlugDescriptor("sideLength")
	subdivisionsCaps_ : SubdivisionsCapsPlug = PlugDescriptor("subdivisionsCaps")
	subdivisionsHeight_ : SubdivisionsHeightPlug = PlugDescriptor("subdivisionsHeight")
	texture_ : TexturePlug = PlugDescriptor("texture")

	# node attributes

	typeName = "polyPyramid"
	apiTypeInt = 969
	apiTypeStr = "kPolyPyramid"
	typeIdInt = 1347443026
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["createUVs", "numberOfSides", "sideLength", "subdivisionsCaps", "subdivisionsHeight", "texture"]
	nodeLeafPlugs = ["createUVs", "numberOfSides", "sideLength", "subdivisionsCaps", "subdivisionsHeight", "texture"]
	pass

