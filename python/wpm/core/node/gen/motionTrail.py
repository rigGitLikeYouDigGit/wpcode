

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Snapshot = retriever.getNodeCls("Snapshot")
assert Snapshot
if T.TYPE_CHECKING:
	from .. import Snapshot

# add node doc



# region plug type defs
class AnchorTransformPlug(Plug):
	node : MotionTrail = None
	pass
class ExtraKeyframeTimesPlug(Plug):
	node : MotionTrail = None
	pass
class HasAnchorTransformPlug(Plug):
	node : MotionTrail = None
	pass
class KeyframeFlagsPlug(Plug):
	node : MotionTrail = None
	pass
class KeyframeTimesPlug(Plug):
	node : MotionTrail = None
	pass
# endregion


# define node class
class MotionTrail(Snapshot):
	anchorTransform_ : AnchorTransformPlug = PlugDescriptor("anchorTransform")
	extraKeyframeTimes_ : ExtraKeyframeTimesPlug = PlugDescriptor("extraKeyframeTimes")
	hasAnchorTransform_ : HasAnchorTransformPlug = PlugDescriptor("hasAnchorTransform")
	keyframeFlags_ : KeyframeFlagsPlug = PlugDescriptor("keyframeFlags")
	keyframeTimes_ : KeyframeTimesPlug = PlugDescriptor("keyframeTimes")

	# node attributes

	typeName = "motionTrail"
	typeIdInt = 1297044562
	nodeLeafClassAttrs = ["anchorTransform", "extraKeyframeTimes", "hasAnchorTransform", "keyframeFlags", "keyframeTimes"]
	nodeLeafPlugs = ["anchorTransform", "extraKeyframeTimes", "hasAnchorTransform", "keyframeFlags", "keyframeTimes"]
	pass

