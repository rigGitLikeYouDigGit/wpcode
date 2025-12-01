

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
FieldManip = retriever.getNodeCls("FieldManip")
assert FieldManip
if T.TYPE_CHECKING:
	from .. import FieldManip

# add node doc



# region plug type defs

# endregion


# define node class
class TurbulenceManip(FieldManip):

	# node attributes

	typeName = "turbulenceManip"
	typeIdInt = 1430541389
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

