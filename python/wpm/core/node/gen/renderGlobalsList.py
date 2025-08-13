

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
	node : RenderGlobalsList = None
	pass
class RenderGlobalsPlug(Plug):
	node : RenderGlobalsList = None
	pass
class RenderQualitiesPlug(Plug):
	node : RenderGlobalsList = None
	pass
class RenderResolutionsPlug(Plug):
	node : RenderGlobalsList = None
	pass
# endregion


# define node class
class RenderGlobalsList(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	renderGlobals_ : RenderGlobalsPlug = PlugDescriptor("renderGlobals")
	renderQualities_ : RenderQualitiesPlug = PlugDescriptor("renderQualities")
	renderResolutions_ : RenderResolutionsPlug = PlugDescriptor("renderResolutions")

	# node attributes

	typeName = "renderGlobalsList"
	apiTypeInt = 524
	apiTypeStr = "kRenderGlobalsList"
	typeIdInt = 1380206412
	MFnCls = om.MFnDependencyNode
	pass

