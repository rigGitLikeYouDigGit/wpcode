

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
	node : EditsManager = None
	pass
class EditsPlug(Plug):
	node : EditsManager = None
	pass
# endregion


# define node class
class EditsManager(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	edits_ : EditsPlug = PlugDescriptor("edits")

	# node attributes

	typeName = "editsManager"
	apiTypeInt = 1097
	apiTypeStr = "kEditsManager"
	typeIdInt = 1162692434
	MFnCls = om.MFnDependencyNode
	pass

