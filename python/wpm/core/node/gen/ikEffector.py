

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Transform = retriever.getNodeCls("Transform")
assert Transform
if T.TYPE_CHECKING:
	from .. import Transform

# add node doc



# region plug type defs
class HandlePathPlug(Plug):
	node : IkEffector = None
	pass
class HideDisplayPlug(Plug):
	node : IkEffector = None
	pass
# endregion


# define node class
class IkEffector(Transform):
	handlePath_ : HandlePathPlug = PlugDescriptor("handlePath")
	hideDisplay_ : HideDisplayPlug = PlugDescriptor("hideDisplay")

	# node attributes

	typeName = "ikEffector"
	apiTypeInt = 119
	apiTypeStr = "kIkEffector"
	typeIdInt = 1262831174
	MFnCls = om.MFnTransform
	pass

