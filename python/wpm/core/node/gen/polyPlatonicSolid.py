

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
	node : PolyPlatonicSolid = None
	pass
class RadiusPlug(Plug):
	node : PolyPlatonicSolid = None
	pass
class SideLengthPlug(Plug):
	node : PolyPlatonicSolid = None
	pass
class SolidTypePlug(Plug):
	node : PolyPlatonicSolid = None
	pass
class TexturePlug(Plug):
	node : PolyPlatonicSolid = None
	pass
# endregion


# define node class
class PolyPlatonicSolid(PolyPrimitive):
	createUVs_ : CreateUVsPlug = PlugDescriptor("createUVs")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	sideLength_ : SideLengthPlug = PlugDescriptor("sideLength")
	solidType_ : SolidTypePlug = PlugDescriptor("solidType")
	texture_ : TexturePlug = PlugDescriptor("texture")

	# node attributes

	typeName = "polyPlatonicSolid"
	apiTypeInt = 981
	apiTypeStr = "kPolyPlatonicSolid"
	typeIdInt = 1397705801
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["createUVs", "radius", "sideLength", "solidType", "texture"]
	nodeLeafPlugs = ["createUVs", "radius", "sideLength", "solidType", "texture"]
	pass

