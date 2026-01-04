

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
	node : HyperView = None
	pass
class BuildDirectionPlug(Plug):
	node : HyperView = None
	pass
class DagViewPlug(Plug):
	node : HyperView = None
	pass
class DescriptionPlug(Plug):
	node : HyperView = None
	pass
class FocusNodePlug(Plug):
	node : HyperView = None
	pass
class FullNamePlug(Plug):
	node : HyperView = None
	pass
class GraphTraversalLimitPlug(Plug):
	node : HyperView = None
	pass
class HyperLayoutPlug(Plug):
	node : HyperView = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : HyperView = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : HyperView = None
	pass
class PositionPlug(Plug):
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	node : HyperView = None
	pass
class RootNodePlug(Plug):
	node : HyperView = None
	pass
class ShortNamePlug(Plug):
	node : HyperView = None
	pass
class ViewXHPlug(Plug):
	parent : ViewRectHighPlug = PlugDescriptor("viewRectHigh")
	node : HyperView = None
	pass
class ViewYHPlug(Plug):
	parent : ViewRectHighPlug = PlugDescriptor("viewRectHigh")
	node : HyperView = None
	pass
class ViewRectHighPlug(Plug):
	viewXH_ : ViewXHPlug = PlugDescriptor("viewXH")
	xh_ : ViewXHPlug = PlugDescriptor("viewXH")
	viewYH_ : ViewYHPlug = PlugDescriptor("viewYH")
	yh_ : ViewYHPlug = PlugDescriptor("viewYH")
	node : HyperView = None
	pass
class ViewXLPlug(Plug):
	parent : ViewRectLowPlug = PlugDescriptor("viewRectLow")
	node : HyperView = None
	pass
class ViewYLPlug(Plug):
	parent : ViewRectLowPlug = PlugDescriptor("viewRectLow")
	node : HyperView = None
	pass
class ViewRectLowPlug(Plug):
	viewXL_ : ViewXLPlug = PlugDescriptor("viewXL")
	xl_ : ViewXLPlug = PlugDescriptor("viewXL")
	viewYL_ : ViewYLPlug = PlugDescriptor("viewYL")
	yl_ : ViewYLPlug = PlugDescriptor("viewYL")
	node : HyperView = None
	pass
# endregion


# define node class
class HyperView(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	buildDirection_ : BuildDirectionPlug = PlugDescriptor("buildDirection")
	dagView_ : DagViewPlug = PlugDescriptor("dagView")
	description_ : DescriptionPlug = PlugDescriptor("description")
	focusNode_ : FocusNodePlug = PlugDescriptor("focusNode")
	fullName_ : FullNamePlug = PlugDescriptor("fullName")
	graphTraversalLimit_ : GraphTraversalLimitPlug = PlugDescriptor("graphTraversalLimit")
	hyperLayout_ : HyperLayoutPlug = PlugDescriptor("hyperLayout")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	position_ : PositionPlug = PlugDescriptor("position")
	rootNode_ : RootNodePlug = PlugDescriptor("rootNode")
	shortName_ : ShortNamePlug = PlugDescriptor("shortName")
	viewXH_ : ViewXHPlug = PlugDescriptor("viewXH")
	viewYH_ : ViewYHPlug = PlugDescriptor("viewYH")
	viewRectHigh_ : ViewRectHighPlug = PlugDescriptor("viewRectHigh")
	viewXL_ : ViewXLPlug = PlugDescriptor("viewXL")
	viewYL_ : ViewYLPlug = PlugDescriptor("viewYL")
	viewRectLow_ : ViewRectLowPlug = PlugDescriptor("viewRectLow")

	# node attributes

	typeName = "hyperView"
	apiTypeInt = 362
	apiTypeStr = "kHyperView"
	typeIdInt = 1145589846
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "buildDirection", "dagView", "description", "focusNode", "fullName", "graphTraversalLimit", "hyperLayout", "positionX", "positionY", "position", "rootNode", "shortName", "viewXH", "viewYH", "viewRectHigh", "viewXL", "viewYL", "viewRectLow"]
	nodeLeafPlugs = ["binMembership", "buildDirection", "dagView", "description", "focusNode", "fullName", "graphTraversalLimit", "hyperLayout", "position", "rootNode", "shortName", "viewRectHigh", "viewRectLow"]
	pass

