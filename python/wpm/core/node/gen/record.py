

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
	node : Record = None
	pass
class InputPlug(Plug):
	node : Record = None
	pass
# endregion


# define node class
class Record(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	input_ : InputPlug = PlugDescriptor("input")

	# node attributes

	typeName = "record"
	apiTypeInt = 466
	apiTypeStr = "kRecord"
	typeIdInt = 1380270916
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "input"]
	nodeLeafPlugs = ["binMembership", "input"]
	pass

