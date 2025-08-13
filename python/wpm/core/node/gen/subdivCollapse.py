

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
	node : SubdivCollapse = None
	pass
class InSubdivPlug(Plug):
	node : SubdivCollapse = None
	pass
class LevelPlug(Plug):
	node : SubdivCollapse = None
	pass
class OutSubdivPlug(Plug):
	node : SubdivCollapse = None
	pass
# endregion


# define node class
class SubdivCollapse(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inSubdiv_ : InSubdivPlug = PlugDescriptor("inSubdiv")
	level_ : LevelPlug = PlugDescriptor("level")
	outSubdiv_ : OutSubdivPlug = PlugDescriptor("outSubdiv")

	# node attributes

	typeName = "subdivCollapse"
	apiTypeInt = 805
	apiTypeStr = "kSubdivCollapse"
	typeIdInt = 1396919376
	MFnCls = om.MFnDependencyNode
	pass

