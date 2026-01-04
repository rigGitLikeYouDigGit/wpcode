

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
	node : Selector = None
	pass
class CollectionPlug(Plug):
	node : Selector = None
	pass
class InputPlug(Plug):
	node : Selector = None
	pass
class OutputPlug(Plug):
	node : Selector = None
	pass
# endregion


# define node class
class Selector(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	collection_ : CollectionPlug = PlugDescriptor("collection")
	input_ : InputPlug = PlugDescriptor("input")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "selector"
	typeIdInt = 1476395892
	nodeLeafClassAttrs = ["binMembership", "collection", "input", "output"]
	nodeLeafPlugs = ["binMembership", "collection", "input", "output"]
	pass

