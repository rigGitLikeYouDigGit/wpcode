

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
	node : NodeGraphEditorBookmarkInfo = None
	pass
class NamePlug(Plug):
	node : NodeGraphEditorBookmarkInfo = None
	pass
class DependNodePlug(Plug):
	parent : NodeInfoPlug = PlugDescriptor("nodeInfo")
	node : NodeGraphEditorBookmarkInfo = None
	pass
class NodeVisualStatePlug(Plug):
	parent : NodeInfoPlug = PlugDescriptor("nodeInfo")
	node : NodeGraphEditorBookmarkInfo = None
	pass
class PositionXPlug(Plug):
	parent : NodeInfoPlug = PlugDescriptor("nodeInfo")
	node : NodeGraphEditorBookmarkInfo = None
	pass
class PositionYPlug(Plug):
	parent : NodeInfoPlug = PlugDescriptor("nodeInfo")
	node : NodeGraphEditorBookmarkInfo = None
	pass
class NodeInfoPlug(Plug):
	dependNode_ : DependNodePlug = PlugDescriptor("dependNode")
	dn_ : DependNodePlug = PlugDescriptor("dependNode")
	nodeVisualState_ : NodeVisualStatePlug = PlugDescriptor("nodeVisualState")
	nvs_ : NodeVisualStatePlug = PlugDescriptor("nodeVisualState")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	x_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	y_ : PositionYPlug = PlugDescriptor("positionY")
	node : NodeGraphEditorBookmarkInfo = None
	pass
class ViewXHPlug(Plug):
	parent : ViewRectHighPlug = PlugDescriptor("viewRectHigh")
	node : NodeGraphEditorBookmarkInfo = None
	pass
class ViewYHPlug(Plug):
	parent : ViewRectHighPlug = PlugDescriptor("viewRectHigh")
	node : NodeGraphEditorBookmarkInfo = None
	pass
class ViewRectHighPlug(Plug):
	viewXH_ : ViewXHPlug = PlugDescriptor("viewXH")
	xh_ : ViewXHPlug = PlugDescriptor("viewXH")
	viewYH_ : ViewYHPlug = PlugDescriptor("viewYH")
	yh_ : ViewYHPlug = PlugDescriptor("viewYH")
	node : NodeGraphEditorBookmarkInfo = None
	pass
class ViewXLPlug(Plug):
	parent : ViewRectLowPlug = PlugDescriptor("viewRectLow")
	node : NodeGraphEditorBookmarkInfo = None
	pass
class ViewYLPlug(Plug):
	parent : ViewRectLowPlug = PlugDescriptor("viewRectLow")
	node : NodeGraphEditorBookmarkInfo = None
	pass
class ViewRectLowPlug(Plug):
	viewXL_ : ViewXLPlug = PlugDescriptor("viewXL")
	xl_ : ViewXLPlug = PlugDescriptor("viewXL")
	viewYL_ : ViewYLPlug = PlugDescriptor("viewYL")
	yl_ : ViewYLPlug = PlugDescriptor("viewYL")
	node : NodeGraphEditorBookmarkInfo = None
	pass
# endregion


# define node class
class NodeGraphEditorBookmarkInfo(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	name_ : NamePlug = PlugDescriptor("name")
	dependNode_ : DependNodePlug = PlugDescriptor("dependNode")
	nodeVisualState_ : NodeVisualStatePlug = PlugDescriptor("nodeVisualState")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	nodeInfo_ : NodeInfoPlug = PlugDescriptor("nodeInfo")
	viewXH_ : ViewXHPlug = PlugDescriptor("viewXH")
	viewYH_ : ViewYHPlug = PlugDescriptor("viewYH")
	viewRectHigh_ : ViewRectHighPlug = PlugDescriptor("viewRectHigh")
	viewXL_ : ViewXLPlug = PlugDescriptor("viewXL")
	viewYL_ : ViewYLPlug = PlugDescriptor("viewYL")
	viewRectLow_ : ViewRectLowPlug = PlugDescriptor("viewRectLow")

	# node attributes

	typeName = "nodeGraphEditorBookmarkInfo"
	apiTypeInt = 1118
	apiTypeStr = "kNodeGraphEditorBookmarkInfo"
	typeIdInt = 1313161801
	MFnCls = om.MFnDependencyNode
	pass

