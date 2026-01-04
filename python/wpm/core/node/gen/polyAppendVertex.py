

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifier = Catalogue.PolyModifier
else:
	from .. import retriever
	PolyModifier = retriever.getNodeCls("PolyModifier")
	assert PolyModifier

# add node doc



# region plug type defs
class DescPlug(Plug):
	node : PolyAppendVertex = None
	pass
class TexturePlug(Plug):
	node : PolyAppendVertex = None
	pass
class VtxxPlug(Plug):
	parent : VerticesPlug = PlugDescriptor("vertices")
	node : PolyAppendVertex = None
	pass
class VtxyPlug(Plug):
	parent : VerticesPlug = PlugDescriptor("vertices")
	node : PolyAppendVertex = None
	pass
class VtxzPlug(Plug):
	parent : VerticesPlug = PlugDescriptor("vertices")
	node : PolyAppendVertex = None
	pass
class VerticesPlug(Plug):
	vtxx_ : VtxxPlug = PlugDescriptor("vtxx")
	vx_ : VtxxPlug = PlugDescriptor("vtxx")
	vtxy_ : VtxyPlug = PlugDescriptor("vtxy")
	vy_ : VtxyPlug = PlugDescriptor("vtxy")
	vtxz_ : VtxzPlug = PlugDescriptor("vtxz")
	vz_ : VtxzPlug = PlugDescriptor("vtxz")
	node : PolyAppendVertex = None
	pass
# endregion


# define node class
class PolyAppendVertex(PolyModifier):
	desc_ : DescPlug = PlugDescriptor("desc")
	texture_ : TexturePlug = PlugDescriptor("texture")
	vtxx_ : VtxxPlug = PlugDescriptor("vtxx")
	vtxy_ : VtxyPlug = PlugDescriptor("vtxy")
	vtxz_ : VtxzPlug = PlugDescriptor("vtxz")
	vertices_ : VerticesPlug = PlugDescriptor("vertices")

	# node attributes

	typeName = "polyAppendVertex"
	apiTypeInt = 796
	apiTypeStr = "kPolyAppendVertex"
	typeIdInt = 1346457686
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["desc", "texture", "vtxx", "vtxy", "vtxz", "vertices"]
	nodeLeafPlugs = ["desc", "texture", "vertices"]
	pass

