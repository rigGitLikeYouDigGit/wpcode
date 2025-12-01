

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
BlindDataTemplate = retriever.getNodeCls("BlindDataTemplate")
assert BlindDataTemplate
if T.TYPE_CHECKING:
	from .. import BlindDataTemplate

# add node doc



# region plug type defs
class WhichOneIndexPlug(Plug):
	node : SubdHierBlind = None
	pass
# endregion


# define node class
class SubdHierBlind(BlindDataTemplate):
	whichOneIndex_ : WhichOneIndexPlug = PlugDescriptor("whichOneIndex")

	# node attributes

	typeName = "subdHierBlind"
	apiTypeInt = 801
	apiTypeStr = "kSubdHierBlind"
	typeIdInt = 1397248578
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["whichOneIndex"]
	nodeLeafPlugs = ["whichOneIndex"]
	pass

