

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierWorld = Catalogue.PolyModifierWorld
else:
	from .. import retriever
	PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
	assert PolyModifierWorld

# add node doc



# region plug type defs
class AnglePlug(Plug):
	node : PolyQuad = None
	pass
class KeepGroupBorderPlug(Plug):
	node : PolyQuad = None
	pass
class KeepHardEdgesPlug(Plug):
	node : PolyQuad = None
	pass
class KeepTextureBordersPlug(Plug):
	node : PolyQuad = None
	pass
class Maya80Plug(Plug):
	node : PolyQuad = None
	pass
# endregion


# define node class
class PolyQuad(PolyModifierWorld):
	angle_ : AnglePlug = PlugDescriptor("angle")
	keepGroupBorder_ : KeepGroupBorderPlug = PlugDescriptor("keepGroupBorder")
	keepHardEdges_ : KeepHardEdgesPlug = PlugDescriptor("keepHardEdges")
	keepTextureBorders_ : KeepTextureBordersPlug = PlugDescriptor("keepTextureBorders")
	maya80_ : Maya80Plug = PlugDescriptor("maya80")

	# node attributes

	typeName = "polyQuad"
	apiTypeInt = 427
	apiTypeStr = "kPolyQuad"
	typeIdInt = 1347507521
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["angle", "keepGroupBorder", "keepHardEdges", "keepTextureBorders", "maya80"]
	nodeLeafPlugs = ["angle", "keepGroupBorder", "keepHardEdges", "keepTextureBorders", "maya80"]
	pass

