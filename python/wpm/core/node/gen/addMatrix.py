

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
	node : AddMatrix = None
	pass
class MatrixInPlug(Plug):
	node : AddMatrix = None
	pass
class MatrixSumPlug(Plug):
	node : AddMatrix = None
	pass
# endregion


# define node class
class AddMatrix(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	matrixIn_ : MatrixInPlug = PlugDescriptor("matrixIn")
	matrixSum_ : MatrixSumPlug = PlugDescriptor("matrixSum")

	# node attributes

	typeName = "addMatrix"
	typeIdInt = 1145130328
	pass

