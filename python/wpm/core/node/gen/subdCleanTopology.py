

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
class SubdCleanTopology(SubdModifier):

	# node attributes

	typeName = "subdCleanTopology"
	apiTypeInt = 893
	apiTypeStr = "kSubdCleanTopology"
	typeIdInt = 1396921433
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

