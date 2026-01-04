

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
	node : Blend = None
	pass
class CurrentPlug(Plug):
	node : Blend = None
	pass
class InputPlug(Plug):
	node : Blend = None
	pass
class OutputPlug(Plug):
	node : Blend = None
	pass
# endregion


# define node class
class Blend(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	current_ : CurrentPlug = PlugDescriptor("current")
	input_ : InputPlug = PlugDescriptor("input")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "blend"
	typeIdInt = 1094863941
	nodeLeafClassAttrs = ["binMembership", "current", "input", "output"]
	nodeLeafPlugs = ["binMembership", "current", "input", "output"]
	pass

