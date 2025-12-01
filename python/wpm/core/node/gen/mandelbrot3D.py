

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Texture3d = retriever.getNodeCls("Texture3d")
assert Texture3d
if T.TYPE_CHECKING:
	from .. import Texture3d

# add node doc



# region plug type defs
class AmplitudePlug(Plug):
	node : Mandelbrot3D = None
	pass
class BoxMinRadiusPlug(Plug):
	node : Mandelbrot3D = None
	pass
class BoxRadiusPlug(Plug):
	node : Mandelbrot3D = None
	pass
class BoxRatioPlug(Plug):
	node : Mandelbrot3D = None
	pass
class CenterXPlug(Plug):
	node : Mandelbrot3D = None
	pass
class CenterYPlug(Plug):
	node : Mandelbrot3D = None
	pass
class CenterZPlug(Plug):
	node : Mandelbrot3D = None
	pass
class CheckerPlug(Plug):
	node : Mandelbrot3D = None
	pass
class CircleRadiusPlug(Plug):
	node : Mandelbrot3D = None
	pass
class CircleSizeRatioPlug(Plug):
	node : Mandelbrot3D = None
	pass
class CirclesPlug(Plug):
	node : Mandelbrot3D = None
	pass
class Color_ColorBPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : Mandelbrot3D = None
	pass
class Color_ColorGPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : Mandelbrot3D = None
	pass
class Color_ColorRPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : Mandelbrot3D = None
	pass
class Color_ColorPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	color_ColorB_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	clcb_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	color_ColorG_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	clcg_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	color_ColorR_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	clcr_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	node : Mandelbrot3D = None
	pass
class Color_InterpPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Mandelbrot3D = None
	pass
class Color_PositionPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Mandelbrot3D = None
	pass
class ColorPlug(Plug):
	color_Color_ : Color_ColorPlug = PlugDescriptor("color_Color")
	clc_ : Color_ColorPlug = PlugDescriptor("color_Color")
	color_Interp_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	cli_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	color_Position_ : Color_PositionPlug = PlugDescriptor("color_Position")
	clp_ : Color_PositionPlug = PlugDescriptor("color_Position")
	node : Mandelbrot3D = None
	pass
class DepthPlug(Plug):
	node : Mandelbrot3D = None
	pass
class EscapeRadiusPlug(Plug):
	node : Mandelbrot3D = None
	pass
class FocusPlug(Plug):
	node : Mandelbrot3D = None
	pass
class ImplodePlug(Plug):
	node : Mandelbrot3D = None
	pass
class ImplodeCenterXPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : Mandelbrot3D = None
	pass
class ImplodeCenterYPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : Mandelbrot3D = None
	pass
class ImplodeCenterZPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : Mandelbrot3D = None
	pass
class ImplodeCenterPlug(Plug):
	implodeCenterX_ : ImplodeCenterXPlug = PlugDescriptor("implodeCenterX")
	imx_ : ImplodeCenterXPlug = PlugDescriptor("implodeCenterX")
	implodeCenterY_ : ImplodeCenterYPlug = PlugDescriptor("implodeCenterY")
	imy_ : ImplodeCenterYPlug = PlugDescriptor("implodeCenterY")
	implodeCenterZ_ : ImplodeCenterZPlug = PlugDescriptor("implodeCenterZ")
	imz_ : ImplodeCenterZPlug = PlugDescriptor("implodeCenterZ")
	node : Mandelbrot3D = None
	pass
class JuliaUPlug(Plug):
	node : Mandelbrot3D = None
	pass
class JuliaVPlug(Plug):
	node : Mandelbrot3D = None
	pass
class LeafEffectPlug(Plug):
	node : Mandelbrot3D = None
	pass
class LineBlendingPlug(Plug):
	node : Mandelbrot3D = None
	pass
class LineFocusPlug(Plug):
	node : Mandelbrot3D = None
	pass
class LineOffsetRatioPlug(Plug):
	node : Mandelbrot3D = None
	pass
class LineOffsetUPlug(Plug):
	node : Mandelbrot3D = None
	pass
class LineOffsetVPlug(Plug):
	node : Mandelbrot3D = None
	pass
class LobesPlug(Plug):
	node : Mandelbrot3D = None
	pass
class MandelbrotInsideMethodPlug(Plug):
	node : Mandelbrot3D = None
	pass
class MandelbrotShadeMethodPlug(Plug):
	node : Mandelbrot3D = None
	pass
class MandelbrotTypePlug(Plug):
	node : Mandelbrot3D = None
	pass
class PointsPlug(Plug):
	node : Mandelbrot3D = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Mandelbrot3D = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Mandelbrot3D = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Mandelbrot3D = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Mandelbrot3D = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Mandelbrot3D = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Mandelbrot3D = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Mandelbrot3D = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Mandelbrot3D = None
	pass
class ShiftPlug(Plug):
	node : Mandelbrot3D = None
	pass
class StalksUPlug(Plug):
	node : Mandelbrot3D = None
	pass
class StalksVPlug(Plug):
	node : Mandelbrot3D = None
	pass
class Value_FloatValuePlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : Mandelbrot3D = None
	pass
class Value_InterpPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : Mandelbrot3D = None
	pass
class Value_PositionPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : Mandelbrot3D = None
	pass
class ValuePlug(Plug):
	value_FloatValue_ : Value_FloatValuePlug = PlugDescriptor("value_FloatValue")
	vlfv_ : Value_FloatValuePlug = PlugDescriptor("value_FloatValue")
	value_Interp_ : Value_InterpPlug = PlugDescriptor("value_Interp")
	vli_ : Value_InterpPlug = PlugDescriptor("value_Interp")
	value_Position_ : Value_PositionPlug = PlugDescriptor("value_Position")
	vlp_ : Value_PositionPlug = PlugDescriptor("value_Position")
	node : Mandelbrot3D = None
	pass
class WrapAmplitudePlug(Plug):
	node : Mandelbrot3D = None
	pass
class XPixelAnglePlug(Plug):
	node : Mandelbrot3D = None
	pass
class ZoomFactorPlug(Plug):
	node : Mandelbrot3D = None
	pass
# endregion


# define node class
class Mandelbrot3D(Texture3d):
	amplitude_ : AmplitudePlug = PlugDescriptor("amplitude")
	boxMinRadius_ : BoxMinRadiusPlug = PlugDescriptor("boxMinRadius")
	boxRadius_ : BoxRadiusPlug = PlugDescriptor("boxRadius")
	boxRatio_ : BoxRatioPlug = PlugDescriptor("boxRatio")
	centerX_ : CenterXPlug = PlugDescriptor("centerX")
	centerY_ : CenterYPlug = PlugDescriptor("centerY")
	centerZ_ : CenterZPlug = PlugDescriptor("centerZ")
	checker_ : CheckerPlug = PlugDescriptor("checker")
	circleRadius_ : CircleRadiusPlug = PlugDescriptor("circleRadius")
	circleSizeRatio_ : CircleSizeRatioPlug = PlugDescriptor("circleSizeRatio")
	circles_ : CirclesPlug = PlugDescriptor("circles")
	color_ColorB_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	color_ColorG_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	color_ColorR_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	color_Color_ : Color_ColorPlug = PlugDescriptor("color_Color")
	color_Interp_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	color_Position_ : Color_PositionPlug = PlugDescriptor("color_Position")
	color_ : ColorPlug = PlugDescriptor("color")
	depth_ : DepthPlug = PlugDescriptor("depth")
	escapeRadius_ : EscapeRadiusPlug = PlugDescriptor("escapeRadius")
	focus_ : FocusPlug = PlugDescriptor("focus")
	implode_ : ImplodePlug = PlugDescriptor("implode")
	implodeCenterX_ : ImplodeCenterXPlug = PlugDescriptor("implodeCenterX")
	implodeCenterY_ : ImplodeCenterYPlug = PlugDescriptor("implodeCenterY")
	implodeCenterZ_ : ImplodeCenterZPlug = PlugDescriptor("implodeCenterZ")
	implodeCenter_ : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	juliaU_ : JuliaUPlug = PlugDescriptor("juliaU")
	juliaV_ : JuliaVPlug = PlugDescriptor("juliaV")
	leafEffect_ : LeafEffectPlug = PlugDescriptor("leafEffect")
	lineBlending_ : LineBlendingPlug = PlugDescriptor("lineBlending")
	lineFocus_ : LineFocusPlug = PlugDescriptor("lineFocus")
	lineOffsetRatio_ : LineOffsetRatioPlug = PlugDescriptor("lineOffsetRatio")
	lineOffsetU_ : LineOffsetUPlug = PlugDescriptor("lineOffsetU")
	lineOffsetV_ : LineOffsetVPlug = PlugDescriptor("lineOffsetV")
	lobes_ : LobesPlug = PlugDescriptor("lobes")
	mandelbrotInsideMethod_ : MandelbrotInsideMethodPlug = PlugDescriptor("mandelbrotInsideMethod")
	mandelbrotShadeMethod_ : MandelbrotShadeMethodPlug = PlugDescriptor("mandelbrotShadeMethod")
	mandelbrotType_ : MandelbrotTypePlug = PlugDescriptor("mandelbrotType")
	points_ : PointsPlug = PlugDescriptor("points")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	refPointObj_ : RefPointObjPlug = PlugDescriptor("refPointObj")
	shift_ : ShiftPlug = PlugDescriptor("shift")
	stalksU_ : StalksUPlug = PlugDescriptor("stalksU")
	stalksV_ : StalksVPlug = PlugDescriptor("stalksV")
	value_FloatValue_ : Value_FloatValuePlug = PlugDescriptor("value_FloatValue")
	value_Interp_ : Value_InterpPlug = PlugDescriptor("value_Interp")
	value_Position_ : Value_PositionPlug = PlugDescriptor("value_Position")
	value_ : ValuePlug = PlugDescriptor("value")
	wrapAmplitude_ : WrapAmplitudePlug = PlugDescriptor("wrapAmplitude")
	xPixelAngle_ : XPixelAnglePlug = PlugDescriptor("xPixelAngle")
	zoomFactor_ : ZoomFactorPlug = PlugDescriptor("zoomFactor")

	# node attributes

	typeName = "mandelbrot3D"
	apiTypeInt = 1085
	apiTypeStr = "kMandelbrot3D"
	typeIdInt = 1381256499
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["amplitude", "boxMinRadius", "boxRadius", "boxRatio", "centerX", "centerY", "centerZ", "checker", "circleRadius", "circleSizeRatio", "circles", "color_ColorB", "color_ColorG", "color_ColorR", "color_Color", "color_Interp", "color_Position", "color", "depth", "escapeRadius", "focus", "implode", "implodeCenterX", "implodeCenterY", "implodeCenterZ", "implodeCenter", "juliaU", "juliaV", "leafEffect", "lineBlending", "lineFocus", "lineOffsetRatio", "lineOffsetU", "lineOffsetV", "lobes", "mandelbrotInsideMethod", "mandelbrotShadeMethod", "mandelbrotType", "points", "refPointCameraX", "refPointCameraY", "refPointCameraZ", "refPointCamera", "refPointObjX", "refPointObjY", "refPointObjZ", "refPointObj", "shift", "stalksU", "stalksV", "value_FloatValue", "value_Interp", "value_Position", "value", "wrapAmplitude", "xPixelAngle", "zoomFactor"]
	nodeLeafPlugs = ["amplitude", "boxMinRadius", "boxRadius", "boxRatio", "centerX", "centerY", "centerZ", "checker", "circleRadius", "circleSizeRatio", "circles", "color", "depth", "escapeRadius", "focus", "implode", "implodeCenter", "juliaU", "juliaV", "leafEffect", "lineBlending", "lineFocus", "lineOffsetRatio", "lineOffsetU", "lineOffsetV", "lobes", "mandelbrotInsideMethod", "mandelbrotShadeMethod", "mandelbrotType", "points", "refPointCamera", "refPointObj", "shift", "stalksU", "stalksV", "value", "wrapAmplitude", "xPixelAngle", "zoomFactor"]
	pass

