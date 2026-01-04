

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

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
	nodeLeafClassAttrs = ["binMembership", "outSubdiv"]
	nodeLeafPlugs = ["binMembership", "outSubdiv"]
	pass

