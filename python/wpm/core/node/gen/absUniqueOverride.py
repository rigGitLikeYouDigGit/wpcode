

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbsOverride = Catalogue.AbsOverride
else:
	from .. import retriever
	AbsOverride = retriever.getNodeCls("AbsOverride")
	assert AbsOverride

# add node doc



# region plug type defs
class TargetNodeNamePlug(Plug):
	node : AbsUniqueOverride = None
	pass
# endregion


# define node class
class AbsUniqueOverride(AbsOverride):
	targetNodeName_ : TargetNodeNamePlug = PlugDescriptor("targetNodeName")

	# node attributes

	typeName = "absUniqueOverride"
	typeIdInt = 1476395936
	nodeLeafClassAttrs = ["targetNodeName"]
	nodeLeafPlugs = ["targetNodeName"]
	pass

