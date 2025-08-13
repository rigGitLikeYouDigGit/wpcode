

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
TexBaseDeformManip = retriever.getNodeCls("TexBaseDeformManip")
assert TexBaseDeformManip
if T.TYPE_CHECKING:
	from .. import TexBaseDeformManip

# add node doc



# region plug type defs

# endregion


# define node class
class TexSmudgeUVManip(TexBaseDeformManip):

	# node attributes

	typeName = "texSmudgeUVManip"
	apiTypeInt = 198
	apiTypeStr = "kTexSmudgeUVManip"
	typeIdInt = 1414745421
	MFnCls = om.MFnDependencyNode
	pass

