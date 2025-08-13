

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
class AutoDirectionPlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
class FlipLeftPlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
class FlipRightPlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
class LeftCurvePlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
class LeftParameterPlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
class LeftRailPlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
class MultipleKnotsPlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
class OutputSurfacePlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
class PositionTolerancePlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
class RightCurvePlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
class RightParameterPlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
class RightRailPlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
class TangentTolerancePlug(Plug):
	node : FfBlendSrfObsolete = None
	pass
# endregion


# define node class
class FfBlendSrfObsolete(AbstractBaseCreate):
	autoDirection_ : AutoDirectionPlug = PlugDescriptor("autoDirection")
	flipLeft_ : FlipLeftPlug = PlugDescriptor("flipLeft")
	flipRight_ : FlipRightPlug = PlugDescriptor("flipRight")
	leftCurve_ : LeftCurvePlug = PlugDescriptor("leftCurve")
	leftParameter_ : LeftParameterPlug = PlugDescriptor("leftParameter")
	leftRail_ : LeftRailPlug = PlugDescriptor("leftRail")
	multipleKnots_ : MultipleKnotsPlug = PlugDescriptor("multipleKnots")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	positionTolerance_ : PositionTolerancePlug = PlugDescriptor("positionTolerance")
	rightCurve_ : RightCurvePlug = PlugDescriptor("rightCurve")
	rightParameter_ : RightParameterPlug = PlugDescriptor("rightParameter")
	rightRail_ : RightRailPlug = PlugDescriptor("rightRail")
	tangentTolerance_ : TangentTolerancePlug = PlugDescriptor("tangentTolerance")

	# node attributes

	typeName = "ffBlendSrfObsolete"
	typeIdInt = 1312967763
	pass

