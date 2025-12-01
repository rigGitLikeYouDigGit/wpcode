

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class DescPlug(Plug):
	node : PolyAppend = None
	pass
class SubdivisionPlug(Plug):
	node : PolyAppend = None
	pass
class Test2EdgeLoopsPlug(Plug):
	node : PolyAppend = None
	pass
class TexturePlug(Plug):
	node : PolyAppend = None
	pass
class VtxxPlug(Plug):
	parent : VerticesPlug = PlugDescriptor("vertices")
	node : PolyAppend = None
	pass
class VtxyPlug(Plug):
	parent : VerticesPlug = PlugDescriptor("vertices")
	node : PolyAppend = None
	pass
class VtxzPlug(Plug):
	parent : VerticesPlug = PlugDescriptor("vertices")
	node : PolyAppend = None
	pass
class VerticesPlug(Plug):
	vtxx_ : VtxxPlug = PlugDescriptor("vtxx")
	vx_ : VtxxPlug = PlugDescriptor("vtxx")
	vtxy_ : VtxyPlug = PlugDescriptor("vtxy")
	vy_ : VtxyPlug = PlugDescriptor("vtxy")
	vtxz_ : VtxzPlug = PlugDescriptor("vtxz")
	vz_ : VtxzPlug = PlugDescriptor("vtxz")
	node : PolyAppend = None
	pass
# endregion


# define node class
class PolyAppend(PolyModifier):
	desc_ : DescPlug = PlugDescriptor("desc")
	subdivision_ : SubdivisionPlug = PlugDescriptor("subdivision")
	test2EdgeLoops_ : Test2EdgeLoopsPlug = PlugDescriptor("test2EdgeLoops")
	texture_ : TexturePlug = PlugDescriptor("texture")
	vtxx_ : VtxxPlug = PlugDescriptor("vtxx")
	vtxy_ : VtxyPlug = PlugDescriptor("vtxy")
	vtxz_ : VtxzPlug = PlugDescriptor("vtxz")
	vertices_ : VerticesPlug = PlugDescriptor("vertices")

	# node attributes

	typeName = "polyAppend"
	apiTypeInt = 403
	apiTypeStr = "kPolyAppend"
	typeIdInt = 1346457680
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["desc", "subdivision", "test2EdgeLoops", "texture", "vtxx", "vtxy", "vtxz", "vertices"]
	nodeLeafPlugs = ["desc", "subdivision", "test2EdgeLoops", "texture", "vertices"]
	pass

