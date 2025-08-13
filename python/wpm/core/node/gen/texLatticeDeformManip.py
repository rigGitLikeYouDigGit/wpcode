

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
class TexLatticeDeformManip(TexBaseDeformManip):

	# node attributes

	typeName = "texLatticeDeformManip"
	apiTypeInt = 199
	apiTypeStr = "kTexLatticeDeformManip"
	typeIdInt = 1414284365
	MFnCls = om.MFnDependencyNode
	pass

