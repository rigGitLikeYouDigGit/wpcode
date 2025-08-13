

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
	node : PolyPrimitiveMisc = None
	pass
class PolyTypePlug(Plug):
	node : PolyPrimitiveMisc = None
	pass
class RadiusPlug(Plug):
	node : PolyPrimitiveMisc = None
	pass
class SideLengthPlug(Plug):
	node : PolyPrimitiveMisc = None
	pass
class TexturePlug(Plug):
	node : PolyPrimitiveMisc = None
	pass
# endregion


# define node class
class PolyPrimitiveMisc(PolyPrimitive):
	createUVs_ : CreateUVsPlug = PlugDescriptor("createUVs")
	polyType_ : PolyTypePlug = PlugDescriptor("polyType")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	sideLength_ : SideLengthPlug = PlugDescriptor("sideLength")
	texture_ : TexturePlug = PlugDescriptor("texture")

	# node attributes

	typeName = "polyPrimitiveMisc"
	apiTypeInt = 980
	apiTypeStr = "kPolyPrimitiveMisc"
	typeIdInt = 1296651075
	MFnCls = om.MFnDependencyNode
	pass

