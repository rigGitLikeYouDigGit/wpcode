

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
	node : PolyBase = None
	pass
class OutputPlug(Plug):
	node : PolyBase = None
	pass
# endregion


# define node class
class PolyBase(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "polyBase"
	typeIdInt = 1345203758
	nodeLeafClassAttrs = ["binMembership", "output"]
	nodeLeafPlugs = ["binMembership", "output"]
	pass

