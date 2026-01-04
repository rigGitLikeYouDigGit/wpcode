

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
class FilterTypePlug(Plug):
	node : ObjectMultiFilter = None
	pass
class ResultListPlug(Plug):
	node : ObjectMultiFilter = None
	pass
# endregion


# define node class
class ObjectMultiFilter(ObjectFilter):
	filterType_ : FilterTypePlug = PlugDescriptor("filterType")
	resultList_ : ResultListPlug = PlugDescriptor("resultList")

	# node attributes

	typeName = "objectMultiFilter"
	apiTypeInt = 677
	apiTypeStr = "kObjectMultiFilter"
	typeIdInt = 1330464332
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["filterType", "resultList"]
	nodeLeafPlugs = ["filterType", "resultList"]
	pass

