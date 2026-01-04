

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
	node : RenderLayerManager = None
	pass
class CurrentRenderLayerPlug(Plug):
	node : RenderLayerManager = None
	pass
class RenderLayerIdPlug(Plug):
	node : RenderLayerManager = None
	pass
# endregion


# define node class
class RenderLayerManager(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	currentRenderLayer_ : CurrentRenderLayerPlug = PlugDescriptor("currentRenderLayer")
	renderLayerId_ : RenderLayerIdPlug = PlugDescriptor("renderLayerId")

	# node attributes

	typeName = "renderLayerManager"
	apiTypeInt = 786
	apiTypeStr = "kRenderLayerManager"
	typeIdInt = 1380863053
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "currentRenderLayer", "renderLayerId"]
	nodeLeafPlugs = ["binMembership", "currentRenderLayer", "renderLayerId"]
	pass

