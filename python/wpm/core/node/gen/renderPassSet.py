

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
	node : RenderPassSet = None
	pass
class OwnerPlug(Plug):
	node : RenderPassSet = None
	pass
class RenderPassPlug(Plug):
	node : RenderPassSet = None
	pass
class RenderablePlug(Plug):
	node : RenderPassSet = None
	pass
# endregion


# define node class
class RenderPassSet(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	owner_ : OwnerPlug = PlugDescriptor("owner")
	renderPass_ : RenderPassPlug = PlugDescriptor("renderPass")
	renderable_ : RenderablePlug = PlugDescriptor("renderable")

	# node attributes

	typeName = "renderPassSet"
	apiTypeInt = 784
	apiTypeStr = "kRenderPassSet"
	typeIdInt = 1380995923
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "owner", "renderPass", "renderable"]
	nodeLeafPlugs = ["binMembership", "owner", "renderPass", "renderable"]
	pass

