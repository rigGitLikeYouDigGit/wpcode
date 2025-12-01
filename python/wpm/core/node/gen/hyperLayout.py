

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
class AllNodesFreeformPlug(Plug):
	node : HyperLayout = None
	pass
class BinMembershipPlug(Plug):
	node : HyperLayout = None
	pass
class DependNodePlug(Plug):
	parent : HyperPositionPlug = PlugDescriptor("hyperPosition")
	node : HyperLayout = None
	pass
class IsCollapsedPlug(Plug):
	parent : HyperPositionPlug = PlugDescriptor("hyperPosition")
	node : HyperLayout = None
	pass
class IsFreeformPlug(Plug):
	parent : HyperPositionPlug = PlugDescriptor("hyperPosition")
	node : HyperLayout = None
	pass
class NodeVisualStatePlug(Plug):
	parent : HyperPositionPlug = PlugDescriptor("hyperPosition")
	node : HyperLayout = None
	pass
class PositionXPlug(Plug):
	parent : HyperPositionPlug = PlugDescriptor("hyperPosition")
	node : HyperLayout = None
	pass
class PositionYPlug(Plug):
	parent : HyperPositionPlug = PlugDescriptor("hyperPosition")
	node : HyperLayout = None
	pass
class HyperPositionPlug(Plug):
	dependNode_ : DependNodePlug = PlugDescriptor("dependNode")
	dn_ : DependNodePlug = PlugDescriptor("dependNode")
	isCollapsed_ : IsCollapsedPlug = PlugDescriptor("isCollapsed")
	isc_ : IsCollapsedPlug = PlugDescriptor("isCollapsed")
	isFreeform_ : IsFreeformPlug = PlugDescriptor("isFreeform")
	isf_ : IsFreeformPlug = PlugDescriptor("isFreeform")
	nodeVisualState_ : NodeVisualStatePlug = PlugDescriptor("nodeVisualState")
	nvs_ : NodeVisualStatePlug = PlugDescriptor("nodeVisualState")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	x_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	y_ : PositionYPlug = PlugDescriptor("positionY")
	node : HyperLayout = None
	pass
class ImageNamePlug(Plug):
	node : HyperLayout = None
	pass
class ImagePositionXPlug(Plug):
	parent : ImagePositionPlug = PlugDescriptor("imagePosition")
	node : HyperLayout = None
	pass
class ImagePositionYPlug(Plug):
	parent : ImagePositionPlug = PlugDescriptor("imagePosition")
	node : HyperLayout = None
	pass
class ImagePositionPlug(Plug):
	imagePositionX_ : ImagePositionXPlug = PlugDescriptor("imagePositionX")
	ipx_ : ImagePositionXPlug = PlugDescriptor("imagePositionX")
	imagePositionY_ : ImagePositionYPlug = PlugDescriptor("imagePositionY")
	ipy_ : ImagePositionYPlug = PlugDescriptor("imagePositionY")
	node : HyperLayout = None
	pass
class ImageScalePlug(Plug):
	node : HyperLayout = None
	pass
# endregion


# define node class
class HyperLayout(_BASE_):
	allNodesFreeform_ : AllNodesFreeformPlug = PlugDescriptor("allNodesFreeform")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	dependNode_ : DependNodePlug = PlugDescriptor("dependNode")
	isCollapsed_ : IsCollapsedPlug = PlugDescriptor("isCollapsed")
	isFreeform_ : IsFreeformPlug = PlugDescriptor("isFreeform")
	nodeVisualState_ : NodeVisualStatePlug = PlugDescriptor("nodeVisualState")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	hyperPosition_ : HyperPositionPlug = PlugDescriptor("hyperPosition")
	imageName_ : ImageNamePlug = PlugDescriptor("imageName")
	imagePositionX_ : ImagePositionXPlug = PlugDescriptor("imagePositionX")
	imagePositionY_ : ImagePositionYPlug = PlugDescriptor("imagePositionY")
	imagePosition_ : ImagePositionPlug = PlugDescriptor("imagePosition")
	imageScale_ : ImageScalePlug = PlugDescriptor("imageScale")

	# node attributes

	typeName = "hyperLayout"
	apiTypeInt = 361
	apiTypeStr = "kHyperLayout"
	typeIdInt = 1213812812
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["allNodesFreeform", "binMembership", "dependNode", "isCollapsed", "isFreeform", "nodeVisualState", "positionX", "positionY", "hyperPosition", "imageName", "imagePositionX", "imagePositionY", "imagePosition", "imageScale"]
	nodeLeafPlugs = ["allNodesFreeform", "binMembership", "hyperPosition", "imageName", "imagePosition", "imageScale"]
	pass

