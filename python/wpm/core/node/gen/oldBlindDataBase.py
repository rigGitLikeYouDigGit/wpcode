

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
	node : OldBlindDataBase = None
	pass
class TypeIdPlug(Plug):
	node : OldBlindDataBase = None
	pass
# endregion


# define node class
class OldBlindDataBase(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	typeId_ : TypeIdPlug = PlugDescriptor("typeId")

	# node attributes

	typeName = "oldBlindDataBase"
	typeIdInt = 1111770196
	nodeLeafClassAttrs = ["binMembership", "typeId"]
	nodeLeafPlugs = ["binMembership", "typeId"]
	pass

