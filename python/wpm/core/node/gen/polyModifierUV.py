

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
class UvSetNamePlug(Plug):
	node : PolyModifierUV = None
	pass
# endregion


# define node class
class PolyModifierUV(PolyModifierWorld):
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "polyModifierUV"
	typeIdInt = 1299142006
	nodeLeafClassAttrs = ["uvSetName"]
	nodeLeafPlugs = ["uvSetName"]
	pass

