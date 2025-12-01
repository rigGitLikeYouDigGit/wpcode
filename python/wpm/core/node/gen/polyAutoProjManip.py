

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
TrsInsertManip = retriever.getNodeCls("TrsInsertManip")
assert TrsInsertManip
if T.TYPE_CHECKING:
	from .. import TrsInsertManip

# add node doc



# region plug type defs

# endregion


# define node class
class PolyAutoProjManip(TrsInsertManip):

	# node attributes

	typeName = "polyAutoProjManip"
	apiTypeInt = 967
	apiTypeStr = "kPolyAutoProjManip"
	typeIdInt = 1095782989
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

