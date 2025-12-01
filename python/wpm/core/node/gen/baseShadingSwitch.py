

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ShadingDependNode = retriever.getNodeCls("ShadingDependNode")
assert ShadingDependNode
if T.TYPE_CHECKING:
	from .. import ShadingDependNode

# add node doc



# region plug type defs
class ObjectIdPlug(Plug):
	node : BaseShadingSwitch = None
	pass
# endregion


# define node class
class BaseShadingSwitch(ShadingDependNode):
	objectId_ : ObjectIdPlug = PlugDescriptor("objectId")

	# node attributes

	typeName = "baseShadingSwitch"
	typeIdInt = 1112758088
	nodeLeafClassAttrs = ["objectId"]
	nodeLeafPlugs = ["objectId"]
	pass

