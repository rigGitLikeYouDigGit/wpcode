

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SubdModifier = retriever.getNodeCls("SubdModifier")
assert SubdModifier
if T.TYPE_CHECKING:
	from .. import SubdModifier

# add node doc



# region plug type defs

# endregion


# define node class
class SubdAddTopology(SubdModifier):

	# node attributes

	typeName = "subdAddTopology"
	apiTypeInt = 892
	apiTypeStr = "kSubdAddTopology"
	typeIdInt = 1396790361
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

