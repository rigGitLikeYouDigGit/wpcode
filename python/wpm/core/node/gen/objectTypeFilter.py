

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ObjectFilter = Catalogue.ObjectFilter
else:
	from .. import retriever
	ObjectFilter = retriever.getNodeCls("ObjectFilter")
	assert ObjectFilter

# add node doc



# region plug type defs
class TypePlug(Plug):
	node : ObjectTypeFilter = None
	pass
class TypeNamePlug(Plug):
	node : ObjectTypeFilter = None
	pass
# endregion


# define node class
class ObjectTypeFilter(ObjectFilter):
	type_ : TypePlug = PlugDescriptor("type")
	typeName_ : TypeNamePlug = PlugDescriptor("typeName")

	# node attributes

	typeName = "objectTypeFilter"
	apiTypeInt = 679
	apiTypeStr = "kObjectTypeFilter"
	typeIdInt = 1330923084
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["type", "typeName"]
	nodeLeafPlugs = ["type", "typeName"]
	pass

