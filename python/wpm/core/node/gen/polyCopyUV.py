

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierUV = Catalogue.PolyModifierUV
else:
	from .. import retriever
	PolyModifierUV = retriever.getNodeCls("PolyModifierUV")
	assert PolyModifierUV

# add node doc



# region plug type defs
class UvSetNameInputPlug(Plug):
	node : PolyCopyUV = None
	pass
# endregion


# define node class
class PolyCopyUV(PolyModifierUV):
	uvSetNameInput_ : UvSetNameInputPlug = PlugDescriptor("uvSetNameInput")

	# node attributes

	typeName = "polyCopyUV"
	typeIdInt = 1346590038
	nodeLeafClassAttrs = ["uvSetNameInput"]
	nodeLeafPlugs = ["uvSetNameInput"]
	pass

