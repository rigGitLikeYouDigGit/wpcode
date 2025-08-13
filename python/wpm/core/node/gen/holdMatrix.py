

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
	node : HoldMatrix = None
	pass
class InMatrixPlug(Plug):
	node : HoldMatrix = None
	pass
class OutMatrixPlug(Plug):
	node : HoldMatrix = None
	pass
# endregion


# define node class
class HoldMatrix(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inMatrix_ : InMatrixPlug = PlugDescriptor("inMatrix")
	outMatrix_ : OutMatrixPlug = PlugDescriptor("outMatrix")

	# node attributes

	typeName = "holdMatrix"
	typeIdInt = 1146112077
	pass

