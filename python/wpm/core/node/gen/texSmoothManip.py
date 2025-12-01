

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Uv2dManip = retriever.getNodeCls("Uv2dManip")
assert Uv2dManip
if T.TYPE_CHECKING:
	from .. import Uv2dManip

# add node doc



# region plug type defs

# endregion


# define node class
class TexSmoothManip(Uv2dManip):

	# node attributes

	typeName = "texSmoothManip"
	apiTypeInt = 201
	apiTypeStr = "kTexSmoothManip"
	typeIdInt = 1414746701
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

