

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
	node : PassMatrix = None
	pass
class InMatrixPlug(Plug):
	node : PassMatrix = None
	pass
class InScalePlug(Plug):
	node : PassMatrix = None
	pass
class OutMatrixPlug(Plug):
	node : PassMatrix = None
	pass
# endregion


# define node class
class PassMatrix(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inMatrix_ : InMatrixPlug = PlugDescriptor("inMatrix")
	inScale_ : InScalePlug = PlugDescriptor("inScale")
	outMatrix_ : OutMatrixPlug = PlugDescriptor("outMatrix")

	# node attributes

	typeName = "passMatrix"
	typeIdInt = 1146114893
	pass

