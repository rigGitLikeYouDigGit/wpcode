

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
	node : DisplayLayerManager = None
	pass
class CurrentDisplayLayerPlug(Plug):
	node : DisplayLayerManager = None
	pass
class DisplayLayerIdPlug(Plug):
	node : DisplayLayerManager = None
	pass
# endregion


# define node class
class DisplayLayerManager(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	currentDisplayLayer_ : CurrentDisplayLayerPlug = PlugDescriptor("currentDisplayLayer")
	displayLayerId_ : DisplayLayerIdPlug = PlugDescriptor("displayLayerId")

	# node attributes

	typeName = "displayLayerManager"
	apiTypeInt = 734
	apiTypeStr = "kDisplayLayerManager"
	typeIdInt = 1146113101
	MFnCls = om.MFnDisplayLayerManager
	nodeLeafClassAttrs = ["binMembership", "currentDisplayLayer", "displayLayerId"]
	nodeLeafPlugs = ["binMembership", "currentDisplayLayer", "displayLayerId"]
	pass

