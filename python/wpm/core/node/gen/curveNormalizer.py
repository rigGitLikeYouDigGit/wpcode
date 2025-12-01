

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
	node : CurveNormalizer = None
	pass
class ScalarPlug(Plug):
	node : CurveNormalizer = None
	pass
# endregion


# define node class
class CurveNormalizer(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	scalar_ : ScalarPlug = PlugDescriptor("scalar")

	# node attributes

	typeName = "curveNormalizer"
	typeIdInt = 1129206349
	nodeLeafClassAttrs = ["binMembership", "scalar"]
	nodeLeafPlugs = ["binMembership", "scalar"]
	pass

