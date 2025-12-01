

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyBevel = retriever.getNodeCls("PolyBevel")
assert PolyBevel
if T.TYPE_CHECKING:
	from .. import PolyBevel

# add node doc



# region plug type defs
class UseLegacyBevelAlgorithmPlug(Plug):
	node : PolyBevel2 = None
	pass
# endregion


# define node class
class PolyBevel2(PolyBevel):
	useLegacyBevelAlgorithm_ : UseLegacyBevelAlgorithmPlug = PlugDescriptor("useLegacyBevelAlgorithm")

	# node attributes

	typeName = "polyBevel2"
	apiTypeInt = 1098
	apiTypeStr = "kPolyBevel2"
	typeIdInt = 1346524722
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["useLegacyBevelAlgorithm"]
	nodeLeafPlugs = ["useLegacyBevelAlgorithm"]
	pass

