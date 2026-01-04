

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
class BothEndsPlug(Plug):
	node : ExtendCurve = None
	pass
class DistancePlug(Plug):
	node : ExtendCurve = None
	pass
class ExtendMethodPlug(Plug):
	node : ExtendCurve = None
	pass
class ExtensionTypePlug(Plug):
	node : ExtendCurve = None
	pass
class InputCurve1Plug(Plug):
	node : ExtendCurve = None
	pass
class InputCurve2Plug(Plug):
	node : ExtendCurve = None
	pass
class PointXPlug(Plug):
	parent : InputPointPlug = PlugDescriptor("inputPoint")
	node : ExtendCurve = None
	pass
class PointYPlug(Plug):
	parent : InputPointPlug = PlugDescriptor("inputPoint")
	node : ExtendCurve = None
	pass
class PointZPlug(Plug):
	parent : InputPointPlug = PlugDescriptor("inputPoint")
	node : ExtendCurve = None
	pass
class InputPointPlug(Plug):
	pointX_ : PointXPlug = PlugDescriptor("pointX")
	px_ : PointXPlug = PlugDescriptor("pointX")
	pointY_ : PointYPlug = PlugDescriptor("pointY")
	py_ : PointYPlug = PlugDescriptor("pointY")
	pointZ_ : PointZPlug = PlugDescriptor("pointZ")
	pz_ : PointZPlug = PlugDescriptor("pointZ")
	node : ExtendCurve = None
	pass
class InputSurfacePlug(Plug):
	node : ExtendCurve = None
	pass
class JoinPlug(Plug):
	node : ExtendCurve = None
	pass
class OutputCurvePlug(Plug):
	node : ExtendCurve = None
	pass
class RemoveMultipleKnotsPlug(Plug):
	node : ExtendCurve = None
	pass
class StartPlug(Plug):
	node : ExtendCurve = None
	pass
# endregion


# define node class
class ExtendCurve(AbstractBaseCreate):
	bothEnds_ : BothEndsPlug = PlugDescriptor("bothEnds")
	distance_ : DistancePlug = PlugDescriptor("distance")
	extendMethod_ : ExtendMethodPlug = PlugDescriptor("extendMethod")
	extensionType_ : ExtensionTypePlug = PlugDescriptor("extensionType")
	inputCurve1_ : InputCurve1Plug = PlugDescriptor("inputCurve1")
	inputCurve2_ : InputCurve2Plug = PlugDescriptor("inputCurve2")
	pointX_ : PointXPlug = PlugDescriptor("pointX")
	pointY_ : PointYPlug = PlugDescriptor("pointY")
	pointZ_ : PointZPlug = PlugDescriptor("pointZ")
	inputPoint_ : InputPointPlug = PlugDescriptor("inputPoint")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	join_ : JoinPlug = PlugDescriptor("join")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	removeMultipleKnots_ : RemoveMultipleKnotsPlug = PlugDescriptor("removeMultipleKnots")
	start_ : StartPlug = PlugDescriptor("start")

	# node attributes

	typeName = "extendCurve"
	apiTypeInt = 65
	apiTypeStr = "kExtendCurve"
	typeIdInt = 1313167427
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["bothEnds", "distance", "extendMethod", "extensionType", "inputCurve1", "inputCurve2", "pointX", "pointY", "pointZ", "inputPoint", "inputSurface", "join", "outputCurve", "removeMultipleKnots", "start"]
	nodeLeafPlugs = ["bothEnds", "distance", "extendMethod", "extensionType", "inputCurve1", "inputCurve2", "inputPoint", "inputSurface", "join", "outputCurve", "removeMultipleKnots", "start"]
	pass

