

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
class ActivePlug(Plug):
	node : PassContributionMap = None
	pass
class BinMembershipPlug(Plug):
	node : PassContributionMap = None
	pass
class DagObjectsPlug(Plug):
	node : PassContributionMap = None
	pass
class LightPlug(Plug):
	node : PassContributionMap = None
	pass
class OwnerPlug(Plug):
	node : PassContributionMap = None
	pass
class RenderPassPlug(Plug):
	node : PassContributionMap = None
	pass
# endregion


# define node class
class PassContributionMap(_BASE_):
	active_ : ActivePlug = PlugDescriptor("active")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	dagObjects_ : DagObjectsPlug = PlugDescriptor("dagObjects")
	light_ : LightPlug = PlugDescriptor("light")
	owner_ : OwnerPlug = PlugDescriptor("owner")
	renderPass_ : RenderPassPlug = PlugDescriptor("renderPass")

	# node attributes

	typeName = "passContributionMap"
	apiTypeInt = 787
	apiTypeStr = "kPassContributionMap"
	typeIdInt = 1347633997
	MFnCls = om.MFnDependencyNode
	pass

