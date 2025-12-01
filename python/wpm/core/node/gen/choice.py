

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
	node : Choice = None
	pass
class InputPlug(Plug):
	node : Choice = None
	pass
class OutputPlug(Plug):
	node : Choice = None
	pass
class SelectorPlug(Plug):
	node : Choice = None
	pass
# endregion


# define node class
class Choice(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	input_ : InputPlug = PlugDescriptor("input")
	output_ : OutputPlug = PlugDescriptor("output")
	selector_ : SelectorPlug = PlugDescriptor("selector")

	# node attributes

	typeName = "choice"
	apiTypeInt = 36
	apiTypeStr = "kChoice"
	typeIdInt = 1128809285
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "input", "output", "selector"]
	nodeLeafPlugs = ["binMembership", "input", "output", "selector"]
	pass

