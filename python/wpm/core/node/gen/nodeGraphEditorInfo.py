

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
	node : NodeGraphEditorInfo = None
	pass
class CompViewXHPlug(Plug):
	parent : CompViewRectHighPlug = PlugDescriptor("compViewRectHigh")
	node : NodeGraphEditorInfo = None
	pass
class CompViewYHPlug(Plug):
	parent : CompViewRectHighPlug = PlugDescriptor("compViewRectHigh")
	node : NodeGraphEditorInfo = None
	pass
class CompViewRectHighPlug(Plug):
	parent : CompoundInfoPlug = PlugDescriptor("compoundInfo")
	compViewXH_ : CompViewXHPlug = PlugDescriptor("compViewXH")
	cxh_ : CompViewXHPlug = PlugDescriptor("compViewXH")
	compViewYH_ : CompViewYHPlug = PlugDescriptor("compViewYH")
	cyh_ : CompViewYHPlug = PlugDescriptor("compViewYH")
	node : NodeGraphEditorInfo = None
	pass
class CompViewXLPlug(Plug):
	parent : CompViewRectLowPlug = PlugDescriptor("compViewRectLow")
	node : NodeGraphEditorInfo = None
	pass
class CompViewYLPlug(Plug):
	parent : CompViewRectLowPlug = PlugDescriptor("compViewRectLow")
	node : NodeGraphEditorInfo = None
	pass
class CompViewRectLowPlug(Plug):
	parent : CompoundInfoPlug = PlugDescriptor("compoundInfo")
	compViewXL_ : CompViewXLPlug = PlugDescriptor("compViewXL")
	cxl_ : CompViewXLPlug = PlugDescriptor("compViewXL")
	compViewYL_ : CompViewYLPlug = PlugDescriptor("compViewYL")
	cyl_ : CompViewYLPlug = PlugDescriptor("compViewYL")
	node : NodeGraphEditorInfo = None
	pass
class DefaultPlug(Plug):
	node : NodeGraphEditorInfo = None
	pass
class DependNodePlug(Plug):
	parent : NodeInfoPlug = PlugDescriptor("nodeInfo")
	node : NodeGraphEditorInfo = None
	pass
class PanelHeightPlug(Plug):
	parent : PanelSizePlug = PlugDescriptor("panelSize")
	node : NodeGraphEditorInfo = None
	pass
class ParentEditorEmbeddedPlug(Plug):
	node : NodeGraphEditorInfo = None
	pass
class CompoundPathPlug(Plug):
	parent : CompoundInfoPlug = PlugDescriptor("compoundInfo")
	node : NodeGraphEditorInfo = None
	pass
class CompoundInfoPlug(Plug):
	parent : TabGraphInfoPlug = PlugDescriptor("tabGraphInfo")
	compViewRectHigh_ : CompViewRectHighPlug = PlugDescriptor("compViewRectHigh")
	cvh_ : CompViewRectHighPlug = PlugDescriptor("compViewRectHigh")
	compViewRectLow_ : CompViewRectLowPlug = PlugDescriptor("compViewRectLow")
	cvl_ : CompViewRectLowPlug = PlugDescriptor("compViewRectLow")
	compoundPath_ : CompoundPathPlug = PlugDescriptor("compoundPath")
	cp_ : CompoundPathPlug = PlugDescriptor("compoundPath")
	node : NodeGraphEditorInfo = None
	pass
class ContainerNodePlug(Plug):
	parent : TabGraphInfoPlug = PlugDescriptor("tabGraphInfo")
	node : NodeGraphEditorInfo = None
	pass
class CurrentViewPlug(Plug):
	parent : TabGraphInfoPlug = PlugDescriptor("tabGraphInfo")
	node : NodeGraphEditorInfo = None
	pass
class HasInternalLayoutPlug(Plug):
	parent : TabGraphInfoPlug = PlugDescriptor("tabGraphInfo")
	node : NodeGraphEditorInfo = None
	pass
class NodeVisualStatePlug(Plug):
	parent : NodeInfoPlug = PlugDescriptor("nodeInfo")
	node : NodeGraphEditorInfo = None
	pass
class PositionXPlug(Plug):
	parent : NodeInfoPlug = PlugDescriptor("nodeInfo")
	node : NodeGraphEditorInfo = None
	pass
class PositionYPlug(Plug):
	parent : NodeInfoPlug = PlugDescriptor("nodeInfo")
	node : NodeGraphEditorInfo = None
	pass
class NodeInfoPlug(Plug):
	parent : TabGraphInfoPlug = PlugDescriptor("tabGraphInfo")
	dependNode_ : DependNodePlug = PlugDescriptor("dependNode")
	dn_ : DependNodePlug = PlugDescriptor("dependNode")
	nodeVisualState_ : NodeVisualStatePlug = PlugDescriptor("nodeVisualState")
	nvs_ : NodeVisualStatePlug = PlugDescriptor("nodeVisualState")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	x_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	y_ : PositionYPlug = PlugDescriptor("positionY")
	node : NodeGraphEditorInfo = None
	pass
class PanelPosXPlug(Plug):
	parent : PanelPosPlug = PlugDescriptor("panelPos")
	node : NodeGraphEditorInfo = None
	pass
class PanelPosYPlug(Plug):
	parent : PanelPosPlug = PlugDescriptor("panelPos")
	node : NodeGraphEditorInfo = None
	pass
class PanelPosPlug(Plug):
	parent : TabGraphInfoPlug = PlugDescriptor("tabGraphInfo")
	panelPosX_ : PanelPosXPlug = PlugDescriptor("panelPosX")
	ppx_ : PanelPosXPlug = PlugDescriptor("panelPosX")
	panelPosY_ : PanelPosYPlug = PlugDescriptor("panelPosY")
	ppy_ : PanelPosYPlug = PlugDescriptor("panelPosY")
	node : NodeGraphEditorInfo = None
	pass
class PanelWidthPlug(Plug):
	parent : PanelSizePlug = PlugDescriptor("panelSize")
	node : NodeGraphEditorInfo = None
	pass
class PanelSizePlug(Plug):
	parent : TabGraphInfoPlug = PlugDescriptor("tabGraphInfo")
	panelHeight_ : PanelHeightPlug = PlugDescriptor("panelHeight")
	ph_ : PanelHeightPlug = PlugDescriptor("panelHeight")
	panelWidth_ : PanelWidthPlug = PlugDescriptor("panelWidth")
	pw_ : PanelWidthPlug = PlugDescriptor("panelWidth")
	node : NodeGraphEditorInfo = None
	pass
class TabNamePlug(Plug):
	parent : TabGraphInfoPlug = PlugDescriptor("tabGraphInfo")
	node : NodeGraphEditorInfo = None
	pass
class TornOffPlug(Plug):
	parent : TabGraphInfoPlug = PlugDescriptor("tabGraphInfo")
	node : NodeGraphEditorInfo = None
	pass
class ViewXHPlug(Plug):
	parent : ViewRectHighPlug = PlugDescriptor("viewRectHigh")
	node : NodeGraphEditorInfo = None
	pass
class ViewYHPlug(Plug):
	parent : ViewRectHighPlug = PlugDescriptor("viewRectHigh")
	node : NodeGraphEditorInfo = None
	pass
class ViewRectHighPlug(Plug):
	parent : TabGraphInfoPlug = PlugDescriptor("tabGraphInfo")
	viewXH_ : ViewXHPlug = PlugDescriptor("viewXH")
	xh_ : ViewXHPlug = PlugDescriptor("viewXH")
	viewYH_ : ViewYHPlug = PlugDescriptor("viewYH")
	yh_ : ViewYHPlug = PlugDescriptor("viewYH")
	node : NodeGraphEditorInfo = None
	pass
class ViewXLPlug(Plug):
	parent : ViewRectLowPlug = PlugDescriptor("viewRectLow")
	node : NodeGraphEditorInfo = None
	pass
class ViewYLPlug(Plug):
	parent : ViewRectLowPlug = PlugDescriptor("viewRectLow")
	node : NodeGraphEditorInfo = None
	pass
class ViewRectLowPlug(Plug):
	parent : TabGraphInfoPlug = PlugDescriptor("tabGraphInfo")
	viewXL_ : ViewXLPlug = PlugDescriptor("viewXL")
	xl_ : ViewXLPlug = PlugDescriptor("viewXL")
	viewYL_ : ViewYLPlug = PlugDescriptor("viewYL")
	yl_ : ViewYLPlug = PlugDescriptor("viewYL")
	node : NodeGraphEditorInfo = None
	pass
class TabGraphInfoPlug(Plug):
	compoundInfo_ : CompoundInfoPlug = PlugDescriptor("compoundInfo")
	ci_ : CompoundInfoPlug = PlugDescriptor("compoundInfo")
	containerNode_ : ContainerNodePlug = PlugDescriptor("containerNode")
	cn_ : ContainerNodePlug = PlugDescriptor("containerNode")
	currentView_ : CurrentViewPlug = PlugDescriptor("currentView")
	cv_ : CurrentViewPlug = PlugDescriptor("currentView")
	hasInternalLayout_ : HasInternalLayoutPlug = PlugDescriptor("hasInternalLayout")
	hil_ : HasInternalLayoutPlug = PlugDescriptor("hasInternalLayout")
	nodeInfo_ : NodeInfoPlug = PlugDescriptor("nodeInfo")
	ni_ : NodeInfoPlug = PlugDescriptor("nodeInfo")
	panelPos_ : PanelPosPlug = PlugDescriptor("panelPos")
	pp_ : PanelPosPlug = PlugDescriptor("panelPos")
	panelSize_ : PanelSizePlug = PlugDescriptor("panelSize")
	ps_ : PanelSizePlug = PlugDescriptor("panelSize")
	tabName_ : TabNamePlug = PlugDescriptor("tabName")
	tn_ : TabNamePlug = PlugDescriptor("tabName")
	tornOff_ : TornOffPlug = PlugDescriptor("tornOff")
	to_ : TornOffPlug = PlugDescriptor("tornOff")
	viewRectHigh_ : ViewRectHighPlug = PlugDescriptor("viewRectHigh")
	vh_ : ViewRectHighPlug = PlugDescriptor("viewRectHigh")
	viewRectLow_ : ViewRectLowPlug = PlugDescriptor("viewRectLow")
	vl_ : ViewRectLowPlug = PlugDescriptor("viewRectLow")
	node : NodeGraphEditorInfo = None
	pass
# endregion


# define node class
class NodeGraphEditorInfo(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	compViewXH_ : CompViewXHPlug = PlugDescriptor("compViewXH")
	compViewYH_ : CompViewYHPlug = PlugDescriptor("compViewYH")
	compViewRectHigh_ : CompViewRectHighPlug = PlugDescriptor("compViewRectHigh")
	compViewXL_ : CompViewXLPlug = PlugDescriptor("compViewXL")
	compViewYL_ : CompViewYLPlug = PlugDescriptor("compViewYL")
	compViewRectLow_ : CompViewRectLowPlug = PlugDescriptor("compViewRectLow")
	default_ : DefaultPlug = PlugDescriptor("default")
	dependNode_ : DependNodePlug = PlugDescriptor("dependNode")
	panelHeight_ : PanelHeightPlug = PlugDescriptor("panelHeight")
	parentEditorEmbedded_ : ParentEditorEmbeddedPlug = PlugDescriptor("parentEditorEmbedded")
	compoundPath_ : CompoundPathPlug = PlugDescriptor("compoundPath")
	compoundInfo_ : CompoundInfoPlug = PlugDescriptor("compoundInfo")
	containerNode_ : ContainerNodePlug = PlugDescriptor("containerNode")
	currentView_ : CurrentViewPlug = PlugDescriptor("currentView")
	hasInternalLayout_ : HasInternalLayoutPlug = PlugDescriptor("hasInternalLayout")
	nodeVisualState_ : NodeVisualStatePlug = PlugDescriptor("nodeVisualState")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	nodeInfo_ : NodeInfoPlug = PlugDescriptor("nodeInfo")
	panelPosX_ : PanelPosXPlug = PlugDescriptor("panelPosX")
	panelPosY_ : PanelPosYPlug = PlugDescriptor("panelPosY")
	panelPos_ : PanelPosPlug = PlugDescriptor("panelPos")
	panelWidth_ : PanelWidthPlug = PlugDescriptor("panelWidth")
	panelSize_ : PanelSizePlug = PlugDescriptor("panelSize")
	tabName_ : TabNamePlug = PlugDescriptor("tabName")
	tornOff_ : TornOffPlug = PlugDescriptor("tornOff")
	viewXH_ : ViewXHPlug = PlugDescriptor("viewXH")
	viewYH_ : ViewYHPlug = PlugDescriptor("viewYH")
	viewRectHigh_ : ViewRectHighPlug = PlugDescriptor("viewRectHigh")
	viewXL_ : ViewXLPlug = PlugDescriptor("viewXL")
	viewYL_ : ViewYLPlug = PlugDescriptor("viewYL")
	viewRectLow_ : ViewRectLowPlug = PlugDescriptor("viewRectLow")
	tabGraphInfo_ : TabGraphInfoPlug = PlugDescriptor("tabGraphInfo")

	# node attributes

	typeName = "nodeGraphEditorInfo"
	apiTypeInt = 1116
	apiTypeStr = "kNodeGraphEditorInfo"
	typeIdInt = 1313293641
	MFnCls = om.MFnDependencyNode
	pass

