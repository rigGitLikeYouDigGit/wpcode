

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyCreator = retriever.getNodeCls("PolyCreator")
assert PolyCreator
if T.TYPE_CHECKING:
	from .. import PolyCreator

# add node doc



# region plug type defs
class LoopPlug(Plug):
	node : PolyCreateFace = None
	pass
class SubdivisionPlug(Plug):
	node : PolyCreateFace = None
	pass
class TexturePlug(Plug):
	node : PolyCreateFace = None
	pass
class UvSetNamePlug(Plug):
	node : PolyCreateFace = None
	pass
class VtxxPlug(Plug):
	parent : VerticesPlug = PlugDescriptor("vertices")
	node : PolyCreateFace = None
	pass
class VtxyPlug(Plug):
	parent : VerticesPlug = PlugDescriptor("vertices")
	node : PolyCreateFace = None
	pass
class VtxzPlug(Plug):
	parent : VerticesPlug = PlugDescriptor("vertices")
	node : PolyCreateFace = None
	pass
class VerticesPlug(Plug):
	vtxx_ : VtxxPlug = PlugDescriptor("vtxx")
	vx_ : VtxxPlug = PlugDescriptor("vtxx")
	vtxy_ : VtxyPlug = PlugDescriptor("vtxy")
	vy_ : VtxyPlug = PlugDescriptor("vtxy")
	vtxz_ : VtxzPlug = PlugDescriptor("vtxz")
	vz_ : VtxzPlug = PlugDescriptor("vtxz")
	node : PolyCreateFace = None
	pass
# endregion


# define node class
class PolyCreateFace(PolyCreator):
	loop_ : LoopPlug = PlugDescriptor("loop")
	subdivision_ : SubdivisionPlug = PlugDescriptor("subdivision")
	texture_ : TexturePlug = PlugDescriptor("texture")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")
	vtxx_ : VtxxPlug = PlugDescriptor("vtxx")
	vtxy_ : VtxyPlug = PlugDescriptor("vtxy")
	vtxz_ : VtxzPlug = PlugDescriptor("vtxz")
	vertices_ : VerticesPlug = PlugDescriptor("vertices")

	# node attributes

	typeName = "polyCreateFace"
	typeIdInt = 1346589253
	pass

