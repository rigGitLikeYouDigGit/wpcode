

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : PostProcessList = None
	pass
class PostProcessesPlug(Plug):
	node : PostProcessList = None
	pass
# endregion


# define node class
class PostProcessList(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	postProcesses_ : PostProcessesPlug = PlugDescriptor("postProcesses")

	# node attributes

	typeName = "postProcessList"
	apiTypeInt = 464
	apiTypeStr = "kPostProcessList"
	typeIdInt = 1347441492
	MFnCls = om.MFnDependencyNode
	pass

