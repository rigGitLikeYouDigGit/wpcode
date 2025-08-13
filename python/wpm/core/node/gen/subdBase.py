

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
	node : SubdBase = None
	pass
class OutSubdivPlug(Plug):
	node : SubdBase = None
	pass
# endregion


# define node class
class SubdBase(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	outSubdiv_ : OutSubdivPlug = PlugDescriptor("outSubdiv")

	# node attributes

	typeName = "subdBase"
	typeIdInt = 1395535406
	pass

