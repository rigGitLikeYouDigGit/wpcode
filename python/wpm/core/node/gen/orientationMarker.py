

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PositionMarker = retriever.getNodeCls("PositionMarker")
assert PositionMarker
if T.TYPE_CHECKING:
	from .. import PositionMarker

# add node doc



# region plug type defs
class FrontTwistPlug(Plug):
	node : OrientationMarker = None
	pass
class SideTwistPlug(Plug):
	node : OrientationMarker = None
	pass
class UpTwistPlug(Plug):
	node : OrientationMarker = None
	pass
# endregion


# define node class
class OrientationMarker(PositionMarker):
	frontTwist_ : FrontTwistPlug = PlugDescriptor("frontTwist")
	sideTwist_ : SideTwistPlug = PlugDescriptor("sideTwist")
	upTwist_ : UpTwistPlug = PlugDescriptor("upTwist")

	# node attributes

	typeName = "orientationMarker"
	apiTypeInt = 284
	apiTypeStr = "kOrientationMarker"
	typeIdInt = 1330795597
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["frontTwist", "sideTwist", "upTwist"]
	nodeLeafPlugs = ["frontTwist", "sideTwist", "upTwist"]
	pass

