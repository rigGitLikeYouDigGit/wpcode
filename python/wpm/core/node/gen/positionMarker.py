

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Locator = Catalogue.Locator
else:
	from .. import retriever
	Locator = retriever.getNodeCls("Locator")
	assert Locator

# add node doc



# region plug type defs
class TimePlug(Plug):
	node : PositionMarker = None
	pass
# endregion


# define node class
class PositionMarker(Locator):
	time_ : TimePlug = PlugDescriptor("time")

	# node attributes

	typeName = "positionMarker"
	apiTypeInt = 285
	apiTypeStr = "kPositionMarker"
	typeIdInt = 1347375949
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["time"]
	nodeLeafPlugs = ["time"]
	pass

