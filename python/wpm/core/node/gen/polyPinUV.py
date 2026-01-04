

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
class OperationPlug(Plug):
	node : PolyPinUV = None
	pass
class PinPlug(Plug):
	node : PolyPinUV = None
	pass
# endregion


# define node class
class PolyPinUV(PolyModifierUV):
	operation_ : OperationPlug = PlugDescriptor("operation")
	pin_ : PinPlug = PlugDescriptor("pin")

	# node attributes

	typeName = "polyPinUV"
	apiTypeInt = 960
	apiTypeStr = "kPolyPinUV"
	typeIdInt = 1347442006
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["operation", "pin"]
	nodeLeafPlugs = ["operation", "pin"]
	pass

