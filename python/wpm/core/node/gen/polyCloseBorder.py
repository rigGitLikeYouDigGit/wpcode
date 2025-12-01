

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
class PolyCloseBorder(PolyModifier):

	# node attributes

	typeName = "polyCloseBorder"
	apiTypeInt = 405
	apiTypeStr = "kPolyCloseBorder"
	typeIdInt = 1346587727
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

