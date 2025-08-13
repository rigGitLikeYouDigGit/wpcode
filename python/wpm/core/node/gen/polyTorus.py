

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
	node : PolyTorus = None
	pass
class RadiusPlug(Plug):
	node : PolyTorus = None
	pass
class ReverseTexturePlug(Plug):
	node : PolyTorus = None
	pass
class SectionRadiusPlug(Plug):
	node : PolyTorus = None
	pass
class SubdivisionsAxisPlug(Plug):
	node : PolyTorus = None
	pass
class SubdivisionsHeightPlug(Plug):
	node : PolyTorus = None
	pass
class TexturePlug(Plug):
	node : PolyTorus = None
	pass
class TwistPlug(Plug):
	node : PolyTorus = None
	pass
# endregion


# define node class
class PolyTorus(PolyPrimitive):
	createUVs_ : CreateUVsPlug = PlugDescriptor("createUVs")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	reverseTexture_ : ReverseTexturePlug = PlugDescriptor("reverseTexture")
	sectionRadius_ : SectionRadiusPlug = PlugDescriptor("sectionRadius")
	subdivisionsAxis_ : SubdivisionsAxisPlug = PlugDescriptor("subdivisionsAxis")
	subdivisionsHeight_ : SubdivisionsHeightPlug = PlugDescriptor("subdivisionsHeight")
	texture_ : TexturePlug = PlugDescriptor("texture")
	twist_ : TwistPlug = PlugDescriptor("twist")

	# node attributes

	typeName = "polyTorus"
	apiTypeInt = 442
	apiTypeStr = "kPolyTorus"
	typeIdInt = 1347702610
	MFnCls = om.MFnDependencyNode
	pass

