

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
	node : WtAddMatrix = None
	pass
class MatrixSumPlug(Plug):
	node : WtAddMatrix = None
	pass
class MatrixInPlug(Plug):
	parent : WtMatrixPlug = PlugDescriptor("wtMatrix")
	node : WtAddMatrix = None
	pass
class WeightInPlug(Plug):
	parent : WtMatrixPlug = PlugDescriptor("wtMatrix")
	node : WtAddMatrix = None
	pass
class WtMatrixPlug(Plug):
	matrixIn_ : MatrixInPlug = PlugDescriptor("matrixIn")
	m_ : MatrixInPlug = PlugDescriptor("matrixIn")
	weightIn_ : WeightInPlug = PlugDescriptor("weightIn")
	w_ : WeightInPlug = PlugDescriptor("weightIn")
	node : WtAddMatrix = None
	pass
# endregion


# define node class
class WtAddMatrix(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	matrixSum_ : MatrixSumPlug = PlugDescriptor("matrixSum")
	matrixIn_ : MatrixInPlug = PlugDescriptor("matrixIn")
	weightIn_ : WeightInPlug = PlugDescriptor("weightIn")
	wtMatrix_ : WtMatrixPlug = PlugDescriptor("wtMatrix")

	# node attributes

	typeName = "wtAddMatrix"
	typeIdInt = 1146569037
	pass

