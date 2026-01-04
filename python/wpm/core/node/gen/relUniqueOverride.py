

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	RelOverride = Catalogue.RelOverride
else:
	from .. import retriever
	RelOverride = retriever.getNodeCls("RelOverride")
	assert RelOverride

# add node doc



# region plug type defs
class TargetNodeNamePlug(Plug):
	node : RelUniqueOverride = None
	pass
# endregion


# define node class
class RelUniqueOverride(RelOverride):
	targetNodeName_ : TargetNodeNamePlug = PlugDescriptor("targetNodeName")

	# node attributes

	typeName = "relUniqueOverride"
	typeIdInt = 1476395937
	nodeLeafClassAttrs = ["targetNodeName"]
	nodeLeafPlugs = ["targetNodeName"]
	pass

