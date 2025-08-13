

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
class AttrTypePlug(Plug):
	node : ObjectAttrFilter = None
	pass
# endregion


# define node class
class ObjectAttrFilter(ObjectFilter):
	attrType_ : AttrTypePlug = PlugDescriptor("attrType")

	# node attributes

	typeName = "objectAttrFilter"
	apiTypeInt = 680
	apiTypeStr = "kObjectAttrFilter"
	typeIdInt = 1330004308
	MFnCls = om.MFnDependencyNode
	pass

