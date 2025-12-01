

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
	node : DagPose = None
	pass
class BindPosePlug(Plug):
	node : DagPose = None
	pass
class GlobalPlug(Plug):
	node : DagPose = None
	pass
class MembersPlug(Plug):
	node : DagPose = None
	pass
class ParentsPlug(Plug):
	node : DagPose = None
	pass
class WorldPlug(Plug):
	node : DagPose = None
	pass
class WorldMatrixPlug(Plug):
	node : DagPose = None
	pass
class XformMatrixPlug(Plug):
	node : DagPose = None
	pass
# endregion


# define node class
class DagPose(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	bindPose_ : BindPosePlug = PlugDescriptor("bindPose")
	global_ : GlobalPlug = PlugDescriptor("global")
	members_ : MembersPlug = PlugDescriptor("members")
	parents_ : ParentsPlug = PlugDescriptor("parents")
	world_ : WorldPlug = PlugDescriptor("world")
	worldMatrix_ : WorldMatrixPlug = PlugDescriptor("worldMatrix")
	xformMatrix_ : XformMatrixPlug = PlugDescriptor("xformMatrix")

	# node attributes

	typeName = "dagPose"
	apiTypeInt = 690
	apiTypeStr = "kDagPose"
	typeIdInt = 1179668307
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "bindPose", "global", "members", "parents", "world", "worldMatrix", "xformMatrix"]
	nodeLeafPlugs = ["binMembership", "bindPose", "global", "members", "parents", "world", "worldMatrix", "xformMatrix"]
	pass

