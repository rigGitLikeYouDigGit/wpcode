

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
class ArcLengthPlug(Plug):
	node : CurveInfo = None
	pass
class XValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : CurveInfo = None
	pass
class YValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : CurveInfo = None
	pass
class ZValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : CurveInfo = None
	pass
class ControlPointsPlug(Plug):
	xValue_ : XValuePlug = PlugDescriptor("xValue")
	xv_ : XValuePlug = PlugDescriptor("xValue")
	yValue_ : YValuePlug = PlugDescriptor("yValue")
	yv_ : YValuePlug = PlugDescriptor("yValue")
	zValue_ : ZValuePlug = PlugDescriptor("zValue")
	zv_ : ZValuePlug = PlugDescriptor("zValue")
	node : CurveInfo = None
	pass
class InputCurvePlug(Plug):
	node : CurveInfo = None
	pass
class KnotsPlug(Plug):
	node : CurveInfo = None
	pass
class WeightsPlug(Plug):
	node : CurveInfo = None
	pass
# endregion


# define node class
class CurveInfo(AbstractBaseCreate):
	arcLength_ : ArcLengthPlug = PlugDescriptor("arcLength")
	xValue_ : XValuePlug = PlugDescriptor("xValue")
	yValue_ : YValuePlug = PlugDescriptor("yValue")
	zValue_ : ZValuePlug = PlugDescriptor("zValue")
	controlPoints_ : ControlPointsPlug = PlugDescriptor("controlPoints")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	knots_ : KnotsPlug = PlugDescriptor("knots")
	weights_ : WeightsPlug = PlugDescriptor("weights")

	# node attributes

	typeName = "curveInfo"
	apiTypeInt = 62
	apiTypeStr = "kCurveInfo"
	typeIdInt = 1313032526
	MFnCls = om.MFnDependencyNode
	pass

