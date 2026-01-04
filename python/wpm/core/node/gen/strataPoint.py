

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	THlocatorShape = Catalogue.THlocatorShape
else:
	from .. import retriever
	THlocatorShape = retriever.getNodeCls("THlocatorShape")
	assert THlocatorShape

# add node doc



# region plug type defs
class BalanceWheelPlug(Plug):
	node : StrataPoint = None
	pass
class StDriverClosestPoint0Plug(Plug):
	parent : StDriverClosestPointPlug = PlugDescriptor("stDriverClosestPoint")
	node : StrataPoint = None
	pass
class StDriverClosestPoint1Plug(Plug):
	parent : StDriverClosestPointPlug = PlugDescriptor("stDriverClosestPoint")
	node : StrataPoint = None
	pass
class StDriverClosestPoint2Plug(Plug):
	parent : StDriverClosestPointPlug = PlugDescriptor("stDriverClosestPoint")
	node : StrataPoint = None
	pass
class StDriverClosestPointPlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	stDriverClosestPoint0_ : StDriverClosestPoint0Plug = PlugDescriptor("stDriverClosestPoint0")
	stDriverClosestPoint0_ : StDriverClosestPoint0Plug = PlugDescriptor("stDriverClosestPoint0")
	stDriverClosestPoint1_ : StDriverClosestPoint1Plug = PlugDescriptor("stDriverClosestPoint1")
	stDriverClosestPoint1_ : StDriverClosestPoint1Plug = PlugDescriptor("stDriverClosestPoint1")
	stDriverClosestPoint2_ : StDriverClosestPoint2Plug = PlugDescriptor("stDriverClosestPoint2")
	stDriverClosestPoint2_ : StDriverClosestPoint2Plug = PlugDescriptor("stDriverClosestPoint2")
	node : StrataPoint = None
	pass
class StDriverCurvePlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverCurveLengthPlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverCurveLengthParamBlendPlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverCurveParamPlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverCurveReverseBlendPlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverLocalOffsetMatrixPlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverOutMatrixPlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverPointMatrixPlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverRefLengthCurvePlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverSurfacePlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverTypePlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverUpCurvePlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverUpdateParamsInEditModePlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverUseClosestPointPlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverWeightPlug(Plug):
	parent : StDriverPlug = PlugDescriptor("stDriver")
	node : StrataPoint = None
	pass
class StDriverPlug(Plug):
	stDriverClosestPoint_ : StDriverClosestPointPlug = PlugDescriptor("stDriverClosestPoint")
	stDriverClosestPoint_ : StDriverClosestPointPlug = PlugDescriptor("stDriverClosestPoint")
	stDriverCurve_ : StDriverCurvePlug = PlugDescriptor("stDriverCurve")
	stDriverCurve_ : StDriverCurvePlug = PlugDescriptor("stDriverCurve")
	stDriverCurveLength_ : StDriverCurveLengthPlug = PlugDescriptor("stDriverCurveLength")
	stDriverCurveLength_ : StDriverCurveLengthPlug = PlugDescriptor("stDriverCurveLength")
	stDriverCurveLengthParamBlend_ : StDriverCurveLengthParamBlendPlug = PlugDescriptor("stDriverCurveLengthParamBlend")
	stDriverCurveLengthParamBlend_ : StDriverCurveLengthParamBlendPlug = PlugDescriptor("stDriverCurveLengthParamBlend")
	stDriverCurveParam_ : StDriverCurveParamPlug = PlugDescriptor("stDriverCurveParam")
	stDriverCurveParam_ : StDriverCurveParamPlug = PlugDescriptor("stDriverCurveParam")
	stDriverCurveReverseBlend_ : StDriverCurveReverseBlendPlug = PlugDescriptor("stDriverCurveReverseBlend")
	stDriverCurveReverseBlend_ : StDriverCurveReverseBlendPlug = PlugDescriptor("stDriverCurveReverseBlend")
	stDriverLocalOffsetMatrix_ : StDriverLocalOffsetMatrixPlug = PlugDescriptor("stDriverLocalOffsetMatrix")
	stDriverLocalOffsetMatrix_ : StDriverLocalOffsetMatrixPlug = PlugDescriptor("stDriverLocalOffsetMatrix")
	stDriverOutMatrix_ : StDriverOutMatrixPlug = PlugDescriptor("stDriverOutMatrix")
	stDriverOutMatrix_ : StDriverOutMatrixPlug = PlugDescriptor("stDriverOutMatrix")
	stDriverPointMatrix_ : StDriverPointMatrixPlug = PlugDescriptor("stDriverPointMatrix")
	stDriverPointMatrix_ : StDriverPointMatrixPlug = PlugDescriptor("stDriverPointMatrix")
	stDriverRefLengthCurve_ : StDriverRefLengthCurvePlug = PlugDescriptor("stDriverRefLengthCurve")
	stDriverRefLengthCurve_ : StDriverRefLengthCurvePlug = PlugDescriptor("stDriverRefLengthCurve")
	stDriverSurface_ : StDriverSurfacePlug = PlugDescriptor("stDriverSurface")
	stDriverSurface_ : StDriverSurfacePlug = PlugDescriptor("stDriverSurface")
	stDriverType_ : StDriverTypePlug = PlugDescriptor("stDriverType")
	stDriverType_ : StDriverTypePlug = PlugDescriptor("stDriverType")
	stDriverUpCurve_ : StDriverUpCurvePlug = PlugDescriptor("stDriverUpCurve")
	stDriverUpCurve_ : StDriverUpCurvePlug = PlugDescriptor("stDriverUpCurve")
	stDriverUpdateParamsInEditMode_ : StDriverUpdateParamsInEditModePlug = PlugDescriptor("stDriverUpdateParamsInEditMode")
	stDriverUpdateParamsInEditMode_ : StDriverUpdateParamsInEditModePlug = PlugDescriptor("stDriverUpdateParamsInEditMode")
	stDriverUseClosestPoint_ : StDriverUseClosestPointPlug = PlugDescriptor("stDriverUseClosestPoint")
	stDriverUseClosestPoint_ : StDriverUseClosestPointPlug = PlugDescriptor("stDriverUseClosestPoint")
	stDriverWeight_ : StDriverWeightPlug = PlugDescriptor("stDriverWeight")
	stDriverWeight_ : StDriverWeightPlug = PlugDescriptor("stDriverWeight")
	node : StrataPoint = None
	pass
class StEditModePlug(Plug):
	node : StrataPoint = None
	pass
class StFinalDriverMatrixPlug(Plug):
	node : StrataPoint = None
	pass
class StFinalLocalMatrixPlug(Plug):
	node : StrataPoint = None
	pass
class StLinkNameToNodePlug(Plug):
	node : StrataPoint = None
	pass
class StNamePlug(Plug):
	node : StrataPoint = None
	pass
class StRadiusPlug(Plug):
	node : StrataPoint = None
	pass
class StUiDataPlug(Plug):
	node : StrataPoint = None
	pass
# endregion


# define node class
class StrataPoint(THlocatorShape):
	balanceWheel_ : BalanceWheelPlug = PlugDescriptor("balanceWheel")
	stDriverClosestPoint0_ : StDriverClosestPoint0Plug = PlugDescriptor("stDriverClosestPoint0")
	stDriverClosestPoint1_ : StDriverClosestPoint1Plug = PlugDescriptor("stDriverClosestPoint1")
	stDriverClosestPoint2_ : StDriverClosestPoint2Plug = PlugDescriptor("stDriverClosestPoint2")
	stDriverClosestPoint_ : StDriverClosestPointPlug = PlugDescriptor("stDriverClosestPoint")
	stDriverCurve_ : StDriverCurvePlug = PlugDescriptor("stDriverCurve")
	stDriverCurveLength_ : StDriverCurveLengthPlug = PlugDescriptor("stDriverCurveLength")
	stDriverCurveLengthParamBlend_ : StDriverCurveLengthParamBlendPlug = PlugDescriptor("stDriverCurveLengthParamBlend")
	stDriverCurveParam_ : StDriverCurveParamPlug = PlugDescriptor("stDriverCurveParam")
	stDriverCurveReverseBlend_ : StDriverCurveReverseBlendPlug = PlugDescriptor("stDriverCurveReverseBlend")
	stDriverLocalOffsetMatrix_ : StDriverLocalOffsetMatrixPlug = PlugDescriptor("stDriverLocalOffsetMatrix")
	stDriverOutMatrix_ : StDriverOutMatrixPlug = PlugDescriptor("stDriverOutMatrix")
	stDriverPointMatrix_ : StDriverPointMatrixPlug = PlugDescriptor("stDriverPointMatrix")
	stDriverRefLengthCurve_ : StDriverRefLengthCurvePlug = PlugDescriptor("stDriverRefLengthCurve")
	stDriverSurface_ : StDriverSurfacePlug = PlugDescriptor("stDriverSurface")
	stDriverType_ : StDriverTypePlug = PlugDescriptor("stDriverType")
	stDriverUpCurve_ : StDriverUpCurvePlug = PlugDescriptor("stDriverUpCurve")
	stDriverUpdateParamsInEditMode_ : StDriverUpdateParamsInEditModePlug = PlugDescriptor("stDriverUpdateParamsInEditMode")
	stDriverUseClosestPoint_ : StDriverUseClosestPointPlug = PlugDescriptor("stDriverUseClosestPoint")
	stDriverWeight_ : StDriverWeightPlug = PlugDescriptor("stDriverWeight")
	stDriver_ : StDriverPlug = PlugDescriptor("stDriver")
	stEditMode_ : StEditModePlug = PlugDescriptor("stEditMode")
	stFinalDriverMatrix_ : StFinalDriverMatrixPlug = PlugDescriptor("stFinalDriverMatrix")
	stFinalLocalMatrix_ : StFinalLocalMatrixPlug = PlugDescriptor("stFinalLocalMatrix")
	stLinkNameToNode_ : StLinkNameToNodePlug = PlugDescriptor("stLinkNameToNode")
	stName_ : StNamePlug = PlugDescriptor("stName")
	stRadius_ : StRadiusPlug = PlugDescriptor("stRadius")
	stUiData_ : StUiDataPlug = PlugDescriptor("stUiData")

	# node attributes

	typeName = "strataPoint"
	typeIdInt = 1191073
	nodeLeafClassAttrs = ["balanceWheel", "stDriverClosestPoint0", "stDriverClosestPoint1", "stDriverClosestPoint2", "stDriverClosestPoint", "stDriverCurve", "stDriverCurveLength", "stDriverCurveLengthParamBlend", "stDriverCurveParam", "stDriverCurveReverseBlend", "stDriverLocalOffsetMatrix", "stDriverOutMatrix", "stDriverPointMatrix", "stDriverRefLengthCurve", "stDriverSurface", "stDriverType", "stDriverUpCurve", "stDriverUpdateParamsInEditMode", "stDriverUseClosestPoint", "stDriverWeight", "stDriver", "stEditMode", "stFinalDriverMatrix", "stFinalLocalMatrix", "stLinkNameToNode", "stName", "stRadius", "stUiData"]
	nodeLeafPlugs = ["balanceWheel", "stDriver", "stEditMode", "stFinalDriverMatrix", "stFinalLocalMatrix", "stLinkNameToNode", "stName", "stRadius", "stUiData"]
	pass

