

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ListItem = Catalogue.ListItem
else:
	from .. import retriever
	ListItem = retriever.getNodeCls("ListItem")
	assert ListItem

# add node doc



# region plug type defs
class ContainerHighestPlug(Plug):
	node : RenderSetupLayer = None
	pass
class ContainerLowestPlug(Plug):
	node : RenderSetupLayer = None
	pass
class LegacyRenderLayerPlug(Plug):
	node : RenderSetupLayer = None
	pass
class ListItemsPlug(Plug):
	node : RenderSetupLayer = None
	pass
class NumIsolatedChildrenPlug(Plug):
	node : RenderSetupLayer = None
	pass
# endregion


# define node class
class RenderSetupLayer(ListItem):
	containerHighest_ : ContainerHighestPlug = PlugDescriptor("containerHighest")
	containerLowest_ : ContainerLowestPlug = PlugDescriptor("containerLowest")
	legacyRenderLayer_ : LegacyRenderLayerPlug = PlugDescriptor("legacyRenderLayer")
	listItems_ : ListItemsPlug = PlugDescriptor("listItems")
	numIsolatedChildren_ : NumIsolatedChildrenPlug = PlugDescriptor("numIsolatedChildren")

	# node attributes

	typeName = "renderSetupLayer"
	typeIdInt = 1476395890
	nodeLeafClassAttrs = ["containerHighest", "containerLowest", "legacyRenderLayer", "listItems", "numIsolatedChildren"]
	nodeLeafPlugs = ["containerHighest", "containerLowest", "legacyRenderLayer", "listItems", "numIsolatedChildren"]
	pass

