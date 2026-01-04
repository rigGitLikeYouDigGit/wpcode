

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
	node : ToonLineAttributes = None
	pass
class LineVisibilityPlug(Plug):
	node : ToonLineAttributes = None
	pass
class LineWidthPlug(Plug):
	node : ToonLineAttributes = None
	pass
class ViewUpdatePlug(Plug):
	node : ToonLineAttributes = None
	pass
# endregion


# define node class
class ToonLineAttributes(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	lineVisibility_ : LineVisibilityPlug = PlugDescriptor("lineVisibility")
	lineWidth_ : LineWidthPlug = PlugDescriptor("lineWidth")
	viewUpdate_ : ViewUpdatePlug = PlugDescriptor("viewUpdate")

	# node attributes

	typeName = "toonLineAttributes"
	apiTypeInt = 972
	apiTypeStr = "kToonLineAttributes"
	typeIdInt = 1414283604
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "lineVisibility", "lineWidth", "viewUpdate"]
	nodeLeafPlugs = ["binMembership", "lineVisibility", "lineWidth", "viewUpdate"]
	pass

