

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbstractBaseCreate = Catalogue.AbstractBaseCreate
else:
	from .. import retriever
	AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
	assert AbstractBaseCreate

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
	nodeLeafClassAttrs = ["autoDirection", "flipLeft", "flipRight", "leftCurve", "leftParameter", "leftRail", "multipleKnots", "outputSurface", "positionTolerance", "rightCurve", "rightParameter", "rightRail", "tangentTolerance"]
	nodeLeafPlugs = ["autoDirection", "flipLeft", "flipRight", "leftCurve", "leftParameter", "leftRail", "multipleKnots", "outputSurface", "positionTolerance", "rightCurve", "rightParameter", "rightRail", "tangentTolerance"]
	pass

