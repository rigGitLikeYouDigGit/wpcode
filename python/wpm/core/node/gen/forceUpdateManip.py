

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
TextButtonManip = retriever.getNodeCls("TextButtonManip")
assert TextButtonManip
if T.TYPE_CHECKING:
	from .. import TextButtonManip

# add node doc



# region plug type defs

# endregion


# define node class
class ForceUpdateManip(TextButtonManip):

	# node attributes

	typeName = "forceUpdateManip"
	apiTypeInt = 695
	apiTypeStr = "kForceUpdateManip"
	typeIdInt = 1431127637
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

