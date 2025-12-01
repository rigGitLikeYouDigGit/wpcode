

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
DagNode = retriever.getNodeCls("DagNode")
assert DagNode
if T.TYPE_CHECKING:
	from .. import DagNode

# add node doc



# region plug type defs

# endregion


# define node class
class UnknownDag(DagNode):

	# node attributes

	typeName = "unknownDag"
	apiTypeInt = 316
	apiTypeStr = "kUnknownDag"
	typeIdInt = 1431194436
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

