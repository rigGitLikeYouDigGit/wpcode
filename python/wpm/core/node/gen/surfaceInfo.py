

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
class XValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : SurfaceInfo = None
	pass
class YValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : SurfaceInfo = None
	pass
class ZValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : SurfaceInfo = None
	pass
class ControlPointsPlug(Plug):
	xValue_ : XValuePlug = PlugDescriptor("xValue")
	xv_ : XValuePlug = PlugDescriptor("xValue")
	yValue_ : YValuePlug = PlugDescriptor("yValue")
	yv_ : YValuePlug = PlugDescriptor("yValue")
	zValue_ : ZValuePlug = PlugDescriptor("zValue")
	zv_ : ZValuePlug = PlugDescriptor("zValue")
	node : SurfaceInfo = None
	pass
class InputSurfacePlug(Plug):
	node : SurfaceInfo = None
	pass
class KnotsUPlug(Plug):
	node : SurfaceInfo = None
	pass
class KnotsVPlug(Plug):
	node : SurfaceInfo = None
	pass
class WeightsPlug(Plug):
	node : SurfaceInfo = None
	pass
# endregion


# define node class
class SurfaceInfo(AbstractBaseCreate):
	xValue_ : XValuePlug = PlugDescriptor("xValue")
	yValue_ : YValuePlug = PlugDescriptor("yValue")
	zValue_ : ZValuePlug = PlugDescriptor("zValue")
	controlPoints_ : ControlPointsPlug = PlugDescriptor("controlPoints")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	knotsU_ : KnotsUPlug = PlugDescriptor("knotsU")
	knotsV_ : KnotsVPlug = PlugDescriptor("knotsV")
	weights_ : WeightsPlug = PlugDescriptor("weights")

	# node attributes

	typeName = "surfaceInfo"
	apiTypeInt = 103
	apiTypeStr = "kSurfaceInfo"
	typeIdInt = 1314081102
	MFnCls = om.MFnDependencyNode
	pass

