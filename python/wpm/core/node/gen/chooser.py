

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
	node : Chooser = None
	pass
class DisplayLevelPlug(Plug):
	node : Chooser = None
	pass
class InLevelPlug(Plug):
	node : Chooser = None
	pass
class OutputPlug(Plug):
	node : Chooser = None
	pass
# endregion


# define node class
class Chooser(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	displayLevel_ : DisplayLevelPlug = PlugDescriptor("displayLevel")
	inLevel_ : InLevelPlug = PlugDescriptor("inLevel")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "chooser"
	apiTypeInt = 772
	apiTypeStr = "kChooser"
	typeIdInt = 1128812367
	MFnCls = om.MFnDependencyNode
	pass

