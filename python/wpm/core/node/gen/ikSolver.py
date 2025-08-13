

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
	node : IkSolver = None
	pass
class MaxIterationsPlug(Plug):
	node : IkSolver = None
	pass
class TolerancePlug(Plug):
	node : IkSolver = None
	pass
# endregion


# define node class
class IkSolver(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	maxIterations_ : MaxIterationsPlug = PlugDescriptor("maxIterations")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "ikSolver"
	typeIdInt = 1263752780
	pass

