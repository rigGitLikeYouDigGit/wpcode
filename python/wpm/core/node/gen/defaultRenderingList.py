

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
	node : DefaultRenderingList = None
	pass
class RenderingPlug(Plug):
	node : DefaultRenderingList = None
	pass
# endregion


# define node class
class DefaultRenderingList(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	rendering_ : RenderingPlug = PlugDescriptor("rendering")

	# node attributes

	typeName = "defaultRenderingList"
	typeIdInt = 1146244684
	pass

