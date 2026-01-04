

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
	node : PolySphere = None
	pass
class RadiusPlug(Plug):
	node : PolySphere = None
	pass
class SubdivisionsAxisPlug(Plug):
	node : PolySphere = None
	pass
class SubdivisionsHeightPlug(Plug):
	node : PolySphere = None
	pass
class TexturePlug(Plug):
	node : PolySphere = None
	pass
# endregion


# define node class
class PolySphere(PolyPrimitive):
	createUVs_ : CreateUVsPlug = PlugDescriptor("createUVs")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	subdivisionsAxis_ : SubdivisionsAxisPlug = PlugDescriptor("subdivisionsAxis")
	subdivisionsHeight_ : SubdivisionsHeightPlug = PlugDescriptor("subdivisionsHeight")
	texture_ : TexturePlug = PlugDescriptor("texture")

	# node attributes

	typeName = "polySphere"
	apiTypeInt = 441
	apiTypeStr = "kPolySphere"
	typeIdInt = 1347637320
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["createUVs", "radius", "subdivisionsAxis", "subdivisionsHeight", "texture"]
	nodeLeafPlugs = ["createUVs", "radius", "subdivisionsAxis", "subdivisionsHeight", "texture"]
	pass

