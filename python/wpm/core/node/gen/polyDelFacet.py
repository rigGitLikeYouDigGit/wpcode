

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
class PolyDelFacet(PolyModifier):

	# node attributes

	typeName = "polyDelFacet"
	apiTypeInt = 410
	apiTypeStr = "kPolyDelFacet"
	typeIdInt = 1346651462
	MFnCls = om.MFnDependencyNode
	pass

