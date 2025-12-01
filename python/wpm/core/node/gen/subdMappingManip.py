

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ManipContainer = retriever.getNodeCls("ManipContainer")
assert ManipContainer
if T.TYPE_CHECKING:
	from .. import ManipContainer

# add node doc



# region plug type defs

# endregion


# define node class
class SubdMappingManip(ManipContainer):

	# node attributes

	typeName = "subdMappingManip"
	apiTypeInt = 885
	apiTypeStr = "kSubdMappingManip"
	typeIdInt = 1431523917
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

