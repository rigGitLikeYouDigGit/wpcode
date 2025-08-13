

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
class PropModManip(ManipContainer):

	# node attributes

	typeName = "propModManip"
	apiTypeInt = 178
	apiTypeStr = "kPropModManip"
	typeIdInt = 1431130189
	MFnCls = om.MFnManip3D
	pass

