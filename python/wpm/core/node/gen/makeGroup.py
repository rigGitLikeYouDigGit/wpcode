

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
	node : MakeGroup = None
	pass
class ElemListPlug(Plug):
	node : MakeGroup = None
	pass
class GroupNamePlug(Plug):
	node : MakeGroup = None
	pass
class GroupTypePlug(Plug):
	node : MakeGroup = None
	pass
class InputComponentsPlug(Plug):
	node : MakeGroup = None
	pass
class InputGeometryPlug(Plug):
	node : MakeGroup = None
	pass
class OutputGeometryPlug(Plug):
	node : MakeGroup = None
	pass
# endregion


# define node class
class MakeGroup(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	elemList_ : ElemListPlug = PlugDescriptor("elemList")
	groupName_ : GroupNamePlug = PlugDescriptor("groupName")
	groupType_ : GroupTypePlug = PlugDescriptor("groupType")
	inputComponents_ : InputComponentsPlug = PlugDescriptor("inputComponents")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	outputGeometry_ : OutputGeometryPlug = PlugDescriptor("outputGeometry")

	# node attributes

	typeName = "makeGroup"
	apiTypeInt = 385
	apiTypeStr = "kMakeGroup"
	typeIdInt = 1347241810
	MFnCls = om.MFnDependencyNode
	pass

