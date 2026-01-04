

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : TimeFunction = None
	pass
class InputPlug(Plug):
	node : TimeFunction = None
	pass
class OutputPlug(Plug):
	node : TimeFunction = None
	pass
# endregion


# define node class
class TimeFunction(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	input_ : InputPlug = PlugDescriptor("input")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "timeFunction"
	apiTypeInt = 941
	apiTypeStr = "kTimeFunction"
	typeIdInt = 1952872558
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "input", "output"]
	nodeLeafPlugs = ["binMembership", "input", "output"]
	pass

