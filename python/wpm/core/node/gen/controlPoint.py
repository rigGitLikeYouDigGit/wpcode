

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	DeformableShape = Catalogue.DeformableShape
else:
	from .. import retriever
	DeformableShape = retriever.getNodeCls("DeformableShape")
	assert DeformableShape

# add node doc



# region plug type defs
class BlindDataNodesPlug(Plug):
	node : ControlPoint = None
	pass
class ClampedPlug(Plug):
	parent : ColorSetPlug = PlugDescriptor("colorSet")
	node : ControlPoint = None
	pass
class ColorNamePlug(Plug):
	parent : ColorSetPlug = PlugDescriptor("colorSet")
	node : ControlPoint = None
	pass
class ColorSetPointsAPlug(Plug):
	parent : ColorSetPointsPlug = PlugDescriptor("colorSetPoints")
	node : ControlPoint = None
	pass
class ColorSetPointsBPlug(Plug):
	parent : ColorSetPointsPlug = PlugDescriptor("colorSetPoints")
	node : ControlPoint = None
	pass
class ColorSetPointsGPlug(Plug):
	parent : ColorSetPointsPlug = PlugDescriptor("colorSetPoints")
	node : ControlPoint = None
	pass
class ColorSetPointsRPlug(Plug):
	parent : ColorSetPointsPlug = PlugDescriptor("colorSetPoints")
	node : ControlPoint = None
	pass
class ColorSetPointsPlug(Plug):
	parent : ColorSetPlug = PlugDescriptor("colorSet")
	colorSetPointsA_ : ColorSetPointsAPlug = PlugDescriptor("colorSetPointsA")
	clpa_ : ColorSetPointsAPlug = PlugDescriptor("colorSetPointsA")
	colorSetPointsB_ : ColorSetPointsBPlug = PlugDescriptor("colorSetPointsB")
	clpb_ : ColorSetPointsBPlug = PlugDescriptor("colorSetPointsB")
	colorSetPointsG_ : ColorSetPointsGPlug = PlugDescriptor("colorSetPointsG")
	clpg_ : ColorSetPointsGPlug = PlugDescriptor("colorSetPointsG")
	colorSetPointsR_ : ColorSetPointsRPlug = PlugDescriptor("colorSetPointsR")
	clpr_ : ColorSetPointsRPlug = PlugDescriptor("colorSetPointsR")
	node : ControlPoint = None
	pass
class RepresentationPlug(Plug):
	parent : ColorSetPlug = PlugDescriptor("colorSet")
	node : ControlPoint = None
	pass
class ColorSetPlug(Plug):
	clamped_ : ClampedPlug = PlugDescriptor("clamped")
	clam_ : ClampedPlug = PlugDescriptor("clamped")
	colorName_ : ColorNamePlug = PlugDescriptor("colorName")
	clsn_ : ColorNamePlug = PlugDescriptor("colorName")
	colorSetPoints_ : ColorSetPointsPlug = PlugDescriptor("colorSetPoints")
	clsp_ : ColorSetPointsPlug = PlugDescriptor("colorSetPoints")
	representation_ : RepresentationPlug = PlugDescriptor("representation")
	rprt_ : RepresentationPlug = PlugDescriptor("representation")
	node : ControlPoint = None
	pass
class XValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : ControlPoint = None
	pass
class YValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : ControlPoint = None
	pass
class ZValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : ControlPoint = None
	pass
class ControlPointsPlug(Plug):
	xValue_ : XValuePlug = PlugDescriptor("xValue")
	xv_ : XValuePlug = PlugDescriptor("xValue")
	yValue_ : YValuePlug = PlugDescriptor("yValue")
	yv_ : YValuePlug = PlugDescriptor("yValue")
	zValue_ : ZValuePlug = PlugDescriptor("zValue")
	zv_ : ZValuePlug = PlugDescriptor("zValue")
	node : ControlPoint = None
	pass
class CurrentColorSetPlug(Plug):
	node : ControlPoint = None
	pass
class CurrentUVSetPlug(Plug):
	node : ControlPoint = None
	pass
class DisplayColorChannelPlug(Plug):
	node : ControlPoint = None
	pass
class DisplayColorsPlug(Plug):
	node : ControlPoint = None
	pass
class DisplayImmediatePlug(Plug):
	node : ControlPoint = None
	pass
class RelativeTweakPlug(Plug):
	node : ControlPoint = None
	pass
class TweakPlug(Plug):
	node : ControlPoint = None
	pass
class TweakLocationPlug(Plug):
	node : ControlPoint = None
	pass
class UvPivotXPlug(Plug):
	parent : UvPivotPlug = PlugDescriptor("uvPivot")
	node : ControlPoint = None
	pass
class UvPivotYPlug(Plug):
	parent : UvPivotPlug = PlugDescriptor("uvPivot")
	node : ControlPoint = None
	pass
class UvPivotPlug(Plug):
	uvPivotX_ : UvPivotXPlug = PlugDescriptor("uvPivotX")
	pvx_ : UvPivotXPlug = PlugDescriptor("uvPivotX")
	uvPivotY_ : UvPivotYPlug = PlugDescriptor("uvPivotY")
	pvy_ : UvPivotYPlug = PlugDescriptor("uvPivotY")
	node : ControlPoint = None
	pass
class UvSetNamePlug(Plug):
	parent : UvSetPlug = PlugDescriptor("uvSet")
	node : ControlPoint = None
	pass
class UvSetPointsUPlug(Plug):
	parent : UvSetPointsPlug = PlugDescriptor("uvSetPoints")
	node : ControlPoint = None
	pass
class UvSetPointsVPlug(Plug):
	parent : UvSetPointsPlug = PlugDescriptor("uvSetPoints")
	node : ControlPoint = None
	pass
class UvSetPointsPlug(Plug):
	parent : UvSetPlug = PlugDescriptor("uvSet")
	uvSetPointsU_ : UvSetPointsUPlug = PlugDescriptor("uvSetPointsU")
	uvpu_ : UvSetPointsUPlug = PlugDescriptor("uvSetPointsU")
	uvSetPointsV_ : UvSetPointsVPlug = PlugDescriptor("uvSetPointsV")
	uvpv_ : UvSetPointsVPlug = PlugDescriptor("uvSetPointsV")
	node : ControlPoint = None
	pass
class UvSetTweakLocationPlug(Plug):
	parent : UvSetPlug = PlugDescriptor("uvSet")
	node : ControlPoint = None
	pass
class UvSetPlug(Plug):
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")
	uvsn_ : UvSetNamePlug = PlugDescriptor("uvSetName")
	uvSetPoints_ : UvSetPointsPlug = PlugDescriptor("uvSetPoints")
	uvsp_ : UvSetPointsPlug = PlugDescriptor("uvSetPoints")
	uvSetTweakLocation_ : UvSetTweakLocationPlug = PlugDescriptor("uvSetTweakLocation")
	uvtw_ : UvSetTweakLocationPlug = PlugDescriptor("uvSetTweakLocation")
	node : ControlPoint = None
	pass
class WeightsPlug(Plug):
	node : ControlPoint = None
	pass
# endregion


# define node class
class ControlPoint(DeformableShape):
	blindDataNodes_ : BlindDataNodesPlug = PlugDescriptor("blindDataNodes")
	clamped_ : ClampedPlug = PlugDescriptor("clamped")
	colorName_ : ColorNamePlug = PlugDescriptor("colorName")
	colorSetPointsA_ : ColorSetPointsAPlug = PlugDescriptor("colorSetPointsA")
	colorSetPointsB_ : ColorSetPointsBPlug = PlugDescriptor("colorSetPointsB")
	colorSetPointsG_ : ColorSetPointsGPlug = PlugDescriptor("colorSetPointsG")
	colorSetPointsR_ : ColorSetPointsRPlug = PlugDescriptor("colorSetPointsR")
	colorSetPoints_ : ColorSetPointsPlug = PlugDescriptor("colorSetPoints")
	representation_ : RepresentationPlug = PlugDescriptor("representation")
	colorSet_ : ColorSetPlug = PlugDescriptor("colorSet")
	xValue_ : XValuePlug = PlugDescriptor("xValue")
	yValue_ : YValuePlug = PlugDescriptor("yValue")
	zValue_ : ZValuePlug = PlugDescriptor("zValue")
	controlPoints_ : ControlPointsPlug = PlugDescriptor("controlPoints")
	currentColorSet_ : CurrentColorSetPlug = PlugDescriptor("currentColorSet")
	currentUVSet_ : CurrentUVSetPlug = PlugDescriptor("currentUVSet")
	displayColorChannel_ : DisplayColorChannelPlug = PlugDescriptor("displayColorChannel")
	displayColors_ : DisplayColorsPlug = PlugDescriptor("displayColors")
	displayImmediate_ : DisplayImmediatePlug = PlugDescriptor("displayImmediate")
	relativeTweak_ : RelativeTweakPlug = PlugDescriptor("relativeTweak")
	tweak_ : TweakPlug = PlugDescriptor("tweak")
	tweakLocation_ : TweakLocationPlug = PlugDescriptor("tweakLocation")
	uvPivotX_ : UvPivotXPlug = PlugDescriptor("uvPivotX")
	uvPivotY_ : UvPivotYPlug = PlugDescriptor("uvPivotY")
	uvPivot_ : UvPivotPlug = PlugDescriptor("uvPivot")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")
	uvSetPointsU_ : UvSetPointsUPlug = PlugDescriptor("uvSetPointsU")
	uvSetPointsV_ : UvSetPointsVPlug = PlugDescriptor("uvSetPointsV")
	uvSetPoints_ : UvSetPointsPlug = PlugDescriptor("uvSetPoints")
	uvSetTweakLocation_ : UvSetTweakLocationPlug = PlugDescriptor("uvSetTweakLocation")
	uvSet_ : UvSetPlug = PlugDescriptor("uvSet")
	weights_ : WeightsPlug = PlugDescriptor("weights")

	# node attributes

	typeName = "controlPoint"
	typeIdInt = 1145262164
	nodeLeafClassAttrs = ["blindDataNodes", "clamped", "colorName", "colorSetPointsA", "colorSetPointsB", "colorSetPointsG", "colorSetPointsR", "colorSetPoints", "representation", "colorSet", "xValue", "yValue", "zValue", "controlPoints", "currentColorSet", "currentUVSet", "displayColorChannel", "displayColors", "displayImmediate", "relativeTweak", "tweak", "tweakLocation", "uvPivotX", "uvPivotY", "uvPivot", "uvSetName", "uvSetPointsU", "uvSetPointsV", "uvSetPoints", "uvSetTweakLocation", "uvSet", "weights"]
	nodeLeafPlugs = ["blindDataNodes", "colorSet", "controlPoints", "currentColorSet", "currentUVSet", "displayColorChannel", "displayColors", "displayImmediate", "relativeTweak", "tweak", "tweakLocation", "uvPivot", "uvSet", "weights"]
	pass

