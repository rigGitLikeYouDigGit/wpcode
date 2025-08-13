

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
	node : IkSystem = None
	pass
class GlobalSnapPlug(Plug):
	node : IkSystem = None
	pass
class GlobalSolvePlug(Plug):
	node : IkSystem = None
	pass
class HandleGroupsListPlug(Plug):
	node : IkSystem = None
	pass
class HandleGroupsListDirtyFlagPlug(Plug):
	node : IkSystem = None
	pass
class HandleGroupsListSortedFlagPlug(Plug):
	node : IkSystem = None
	pass
class IkSolverPlug(Plug):
	node : IkSystem = None
	pass
class PreMaya2011IKFKBlendPlug(Plug):
	node : IkSystem = None
	pass
# endregion


# define node class
class IkSystem(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	globalSnap_ : GlobalSnapPlug = PlugDescriptor("globalSnap")
	globalSolve_ : GlobalSolvePlug = PlugDescriptor("globalSolve")
	handleGroupsList_ : HandleGroupsListPlug = PlugDescriptor("handleGroupsList")
	handleGroupsListDirtyFlag_ : HandleGroupsListDirtyFlagPlug = PlugDescriptor("handleGroupsListDirtyFlag")
	handleGroupsListSortedFlag_ : HandleGroupsListSortedFlagPlug = PlugDescriptor("handleGroupsListSortedFlag")
	ikSolver_ : IkSolverPlug = PlugDescriptor("ikSolver")
	preMaya2011IKFKBlend_ : PreMaya2011IKFKBlendPlug = PlugDescriptor("preMaya2011IKFKBlend")

	# node attributes

	typeName = "ikSystem"
	apiTypeInt = 369
	apiTypeStr = "kIkSystem"
	typeIdInt = 1263753555
	MFnCls = om.MFnDependencyNode
	pass

