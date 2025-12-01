

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
	node : RenderSetup = None
	pass
class FirstRenderLayerPlug(Plug):
	node : RenderSetup = None
	pass
class LastRenderLayerPlug(Plug):
	node : RenderSetup = None
	pass
class ListItemsPlug(Plug):
	node : RenderSetup = None
	pass
# endregion


# define node class
class RenderSetup(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	firstRenderLayer_ : FirstRenderLayerPlug = PlugDescriptor("firstRenderLayer")
	lastRenderLayer_ : LastRenderLayerPlug = PlugDescriptor("lastRenderLayer")
	listItems_ : ListItemsPlug = PlugDescriptor("listItems")

	# node attributes

	typeName = "renderSetup"
	typeIdInt = 1476395889
	nodeLeafClassAttrs = ["binMembership", "firstRenderLayer", "lastRenderLayer", "listItems"]
	nodeLeafPlugs = ["binMembership", "firstRenderLayer", "lastRenderLayer", "listItems"]
	pass

