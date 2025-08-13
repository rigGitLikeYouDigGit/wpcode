

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
	node : GroupParts = None
	pass
class GroupIdPlug(Plug):
	node : GroupParts = None
	pass
class InputComponentsPlug(Plug):
	node : GroupParts = None
	pass
class InputGeometryPlug(Plug):
	node : GroupParts = None
	pass
class InputRemoveComponentPlug(Plug):
	node : GroupParts = None
	pass
class OutputGeometryPlug(Plug):
	node : GroupParts = None
	pass
# endregion


# define node class
class GroupParts(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	groupId_ : GroupIdPlug = PlugDescriptor("groupId")
	inputComponents_ : InputComponentsPlug = PlugDescriptor("inputComponents")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	inputRemoveComponent_ : InputRemoveComponentPlug = PlugDescriptor("inputRemoveComponent")
	outputGeometry_ : OutputGeometryPlug = PlugDescriptor("outputGeometry")

	# node attributes

	typeName = "groupParts"
	apiTypeInt = 357
	apiTypeStr = "kGroupParts"
	typeIdInt = 1196576848
	MFnCls = om.MFnDependencyNode
	pass

