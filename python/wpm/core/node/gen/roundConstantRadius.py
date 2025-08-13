

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
class CornerSurfacePlug(Plug):
	node : RoundConstantRadius = None
	pass
class EdgeValidPlug(Plug):
	parent : EdgePlug = PlugDescriptor("edge")
	node : RoundConstantRadius = None
	pass
class InSurfIdxAPlug(Plug):
	parent : EdgePlug = PlugDescriptor("edge")
	node : RoundConstantRadius = None
	pass
class InSurfIdxBPlug(Plug):
	parent : EdgePlug = PlugDescriptor("edge")
	node : RoundConstantRadius = None
	pass
class InputCurveAPlug(Plug):
	parent : EdgePlug = PlugDescriptor("edge")
	node : RoundConstantRadius = None
	pass
class InputCurveBPlug(Plug):
	parent : EdgePlug = PlugDescriptor("edge")
	node : RoundConstantRadius = None
	pass
class EdgePlug(Plug):
	edgeValid_ : EdgeValidPlug = PlugDescriptor("edgeValid")
	ev_ : EdgeValidPlug = PlugDescriptor("edgeValid")
	inSurfIdxA_ : InSurfIdxAPlug = PlugDescriptor("inSurfIdxA")
	isa_ : InSurfIdxAPlug = PlugDescriptor("inSurfIdxA")
	inSurfIdxB_ : InSurfIdxBPlug = PlugDescriptor("inSurfIdxB")
	isb_ : InSurfIdxBPlug = PlugDescriptor("inSurfIdxB")
	inputCurveA_ : InputCurveAPlug = PlugDescriptor("inputCurveA")
	ica_ : InputCurveAPlug = PlugDescriptor("inputCurveA")
	inputCurveB_ : InputCurveBPlug = PlugDescriptor("inputCurveB")
	icb_ : InputCurveBPlug = PlugDescriptor("inputCurveB")
	node : RoundConstantRadius = None
	pass
class FilletStatusPlug(Plug):
	node : RoundConstantRadius = None
	pass
class FilletSurfacePlug(Plug):
	node : RoundConstantRadius = None
	pass
class InputSurfacePlug(Plug):
	node : RoundConstantRadius = None
	pass
class OriginalSurfacePlug(Plug):
	node : RoundConstantRadius = None
	pass
class RadiusPlug(Plug):
	node : RoundConstantRadius = None
	pass
class TolerancePlug(Plug):
	node : RoundConstantRadius = None
	pass
# endregion


# define node class
class RoundConstantRadius(AbstractBaseCreate):
	cornerSurface_ : CornerSurfacePlug = PlugDescriptor("cornerSurface")
	edgeValid_ : EdgeValidPlug = PlugDescriptor("edgeValid")
	inSurfIdxA_ : InSurfIdxAPlug = PlugDescriptor("inSurfIdxA")
	inSurfIdxB_ : InSurfIdxBPlug = PlugDescriptor("inSurfIdxB")
	inputCurveA_ : InputCurveAPlug = PlugDescriptor("inputCurveA")
	inputCurveB_ : InputCurveBPlug = PlugDescriptor("inputCurveB")
	edge_ : EdgePlug = PlugDescriptor("edge")
	filletStatus_ : FilletStatusPlug = PlugDescriptor("filletStatus")
	filletSurface_ : FilletSurfacePlug = PlugDescriptor("filletSurface")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	originalSurface_ : OriginalSurfacePlug = PlugDescriptor("originalSurface")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "roundConstantRadius"
	apiTypeInt = 645
	apiTypeStr = "kRoundConstantRadius"
	typeIdInt = 1314014034
	MFnCls = om.MFnDependencyNode
	pass

