

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
	node : AddDoubleLinear = None
	pass
class Input1Plug(Plug):
	node : AddDoubleLinear = None
	pass
class Input2Plug(Plug):
	node : AddDoubleLinear = None
	pass
class OutputPlug(Plug):
	node : AddDoubleLinear = None
	pass
# endregion


# define node class
class AddDoubleLinear(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	input1_ : Input1Plug = PlugDescriptor("input1")
	input2_ : Input2Plug = PlugDescriptor("input2")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "addDoubleLinear"
	apiTypeInt = 5
	apiTypeStr = "kAddDoubleLinear"
	typeIdInt = 1145128012
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "input1", "input2", "output"]
	nodeLeafPlugs = ["binMembership", "input1", "input2", "output"]
	pass

