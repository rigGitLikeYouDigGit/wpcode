

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ObjectFilter = retriever.getNodeCls("ObjectFilter")
assert ObjectFilter
if T.TYPE_CHECKING:
	from .. import ObjectFilter

# add node doc



# region plug type defs
class BinNamePlug(Plug):
	node : ObjectBinFilter = None
	pass
# endregion


# define node class
class ObjectBinFilter(ObjectFilter):
	binName_ : BinNamePlug = PlugDescriptor("binName")

	# node attributes

	typeName = "objectBinFilter"
	apiTypeInt = 943
	apiTypeStr = "kObjectBinFilter"
	typeIdInt = 1330333260
	MFnCls = om.MFnDependencyNode
	pass

