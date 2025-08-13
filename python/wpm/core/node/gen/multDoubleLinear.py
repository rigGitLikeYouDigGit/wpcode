

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
	node : MultDoubleLinear = None
	pass
class Input1Plug(Plug):
	node : MultDoubleLinear = None
	pass
class Input2Plug(Plug):
	node : MultDoubleLinear = None
	pass
class OutputPlug(Plug):
	node : MultDoubleLinear = None
	pass
# endregion


# define node class
class MultDoubleLinear(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	input1_ : Input1Plug = PlugDescriptor("input1")
	input2_ : Input2Plug = PlugDescriptor("input2")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "multDoubleLinear"
	apiTypeInt = 774
	apiTypeStr = "kMultDoubleLinear"
	typeIdInt = 1145914444
	MFnCls = om.MFnDependencyNode
	pass

