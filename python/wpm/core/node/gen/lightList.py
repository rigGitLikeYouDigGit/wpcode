

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
	node : LightList = None
	pass
class LightsPlug(Plug):
	node : LightList = None
	pass
# endregion


# define node class
class LightList(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	lights_ : LightsPlug = PlugDescriptor("lights")

	# node attributes

	typeName = "lightList"
	apiTypeInt = 382
	apiTypeStr = "kLightList"
	typeIdInt = 1280070484
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "lights"]
	nodeLeafPlugs = ["binMembership", "lights"]
	pass

