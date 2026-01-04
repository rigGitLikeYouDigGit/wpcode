

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
class DivisionsPlug(Plug):
	node : PolySubdEdge = None
	pass
class SizePlug(Plug):
	node : PolySubdEdge = None
	pass
# endregion


# define node class
class PolySubdEdge(PolyModifierWorld):
	divisions_ : DivisionsPlug = PlugDescriptor("divisions")
	size_ : SizePlug = PlugDescriptor("size")

	# node attributes

	typeName = "polySubdEdge"
	apiTypeInt = 432
	apiTypeStr = "kPolySubdEdge"
	typeIdInt = 1347638597
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["divisions", "size"]
	nodeLeafPlugs = ["divisions", "size"]
	pass

