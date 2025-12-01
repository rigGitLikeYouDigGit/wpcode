

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
	node : MultMatrix = None
	pass
class MatrixInPlug(Plug):
	node : MultMatrix = None
	pass
class MatrixSumPlug(Plug):
	node : MultMatrix = None
	pass
# endregion


# define node class
class MultMatrix(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	matrixIn_ : MatrixInPlug = PlugDescriptor("matrixIn")
	matrixSum_ : MatrixSumPlug = PlugDescriptor("matrixSum")

	# node attributes

	typeName = "multMatrix"
	typeIdInt = 1145918541
	nodeLeafClassAttrs = ["binMembership", "matrixIn", "matrixSum"]
	nodeLeafPlugs = ["binMembership", "matrixIn", "matrixSum"]
	pass

