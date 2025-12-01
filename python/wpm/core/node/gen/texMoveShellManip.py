

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
TransUV2dManip = retriever.getNodeCls("TransUV2dManip")
assert TransUV2dManip
if T.TYPE_CHECKING:
	from .. import TransUV2dManip

# add node doc



# region plug type defs

# endregion


# define node class
class TexMoveShellManip(TransUV2dManip):

	# node attributes

	typeName = "texMoveShellManip"
	typeIdInt = 1414353741
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

