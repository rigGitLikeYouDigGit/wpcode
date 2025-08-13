

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class AutoAnchorPlug(Plug):
	node : FfBlendSrf = None
	pass
class AutoNormalPlug(Plug):
	node : FfBlendSrf = None
	pass
class FlipLeftNormalPlug(Plug):
	node : FfBlendSrf = None
	pass
class FlipRightNormalPlug(Plug):
	node : FfBlendSrf = None
	pass
class LeftAnchorPlug(Plug):
	node : FfBlendSrf = None
	pass
class LeftCurvePlug(Plug):
	node : FfBlendSrf = None
	pass
class LeftEndPlug(Plug):
	node : FfBlendSrf = None
	pass
class LeftRailPlug(Plug):
	node : FfBlendSrf = None
	pass
class LeftStartPlug(Plug):
	node : FfBlendSrf = None
	pass
class MultipleKnotsPlug(Plug):
	node : FfBlendSrf = None
	pass
class OutputSurfacePlug(Plug):
	node : FfBlendSrf = None
	pass
class PositionTolerancePlug(Plug):
	node : FfBlendSrf = None
	pass
class ReverseLeftPlug(Plug):
	node : FfBlendSrf = None
	pass
class ReverseRightPlug(Plug):
	node : FfBlendSrf = None
	pass
class RightAnchorPlug(Plug):
	node : FfBlendSrf = None
	pass
class RightCurvePlug(Plug):
	node : FfBlendSrf = None
	pass
class RightEndPlug(Plug):
	node : FfBlendSrf = None
	pass
class RightRailPlug(Plug):
	node : FfBlendSrf = None
	pass
class RightStartPlug(Plug):
	node : FfBlendSrf = None
	pass
class TangentTolerancePlug(Plug):
	node : FfBlendSrf = None
	pass
# endregion


# define node class
class FfBlendSrf(AbstractBaseCreate):
	autoAnchor_ : AutoAnchorPlug = PlugDescriptor("autoAnchor")
	autoNormal_ : AutoNormalPlug = PlugDescriptor("autoNormal")
	flipLeftNormal_ : FlipLeftNormalPlug = PlugDescriptor("flipLeftNormal")
	flipRightNormal_ : FlipRightNormalPlug = PlugDescriptor("flipRightNormal")
	leftAnchor_ : LeftAnchorPlug = PlugDescriptor("leftAnchor")
	leftCurve_ : LeftCurvePlug = PlugDescriptor("leftCurve")
	leftEnd_ : LeftEndPlug = PlugDescriptor("leftEnd")
	leftRail_ : LeftRailPlug = PlugDescriptor("leftRail")
	leftStart_ : LeftStartPlug = PlugDescriptor("leftStart")
	multipleKnots_ : MultipleKnotsPlug = PlugDescriptor("multipleKnots")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	positionTolerance_ : PositionTolerancePlug = PlugDescriptor("positionTolerance")
	reverseLeft_ : ReverseLeftPlug = PlugDescriptor("reverseLeft")
	reverseRight_ : ReverseRightPlug = PlugDescriptor("reverseRight")
	rightAnchor_ : RightAnchorPlug = PlugDescriptor("rightAnchor")
	rightCurve_ : RightCurvePlug = PlugDescriptor("rightCurve")
	rightEnd_ : RightEndPlug = PlugDescriptor("rightEnd")
	rightRail_ : RightRailPlug = PlugDescriptor("rightRail")
	rightStart_ : RightStartPlug = PlugDescriptor("rightStart")
	tangentTolerance_ : TangentTolerancePlug = PlugDescriptor("tangentTolerance")

	# node attributes

	typeName = "ffBlendSrf"
	typeIdInt = 1312967764
	pass

