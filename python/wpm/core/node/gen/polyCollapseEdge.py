

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs

# endregion


# define node class
class PolyCollapseEdge(PolyModifier):

	# node attributes

	typeName = "polyCollapseEdge"
	apiTypeInt = 406
	apiTypeStr = "kPolyCollapseEdge"
	typeIdInt = 1346588485
	MFnCls = om.MFnDependencyNode
	pass

