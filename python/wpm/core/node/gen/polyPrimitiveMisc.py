

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
	nodeLeafClassAttrs = ["createUVs", "polyType", "radius", "sideLength", "texture"]
	nodeLeafPlugs = ["createUVs", "polyType", "radius", "sideLength", "texture"]
	pass

