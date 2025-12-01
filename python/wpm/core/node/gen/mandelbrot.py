

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Texture2d = retriever.getNodeCls("Texture2d")
assert Texture2d
if T.TYPE_CHECKING:
	from .. import Texture2d

# add node doc



# region plug type defs
class AmplitudePlug(Plug):
	node : Mandelbrot = None
	pass
class BoxMinRadiusPlug(Plug):
	node : Mandelbrot = None
	pass
class BoxRadiusPlug(Plug):
	node : Mandelbrot = None
	pass
class BoxRatioPlug(Plug):
	node : Mandelbrot = None
	pass
class CenterUPlug(Plug):
	node : Mandelbrot = None
	pass
class CenterVPlug(Plug):
	node : Mandelbrot = None
	pass
class CheckerPlug(Plug):
	node : Mandelbrot = None
	pass
class CircleRadiusPlug(Plug):
	node : Mandelbrot = None
	pass
class CircleSizeRatioPlug(Plug):
	node : Mandelbrot = None
	pass
class CirclesPlug(Plug):
	node : Mandelbrot = None
	pass
class Color_ColorBPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : Mandelbrot = None
	pass
class Color_ColorGPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : Mandelbrot = None
	pass
class Color_ColorRPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : Mandelbrot = None
	pass
class Color_ColorPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	color_ColorB_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	clcb_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	color_ColorG_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	clcg_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	color_ColorR_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	clcr_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	node : Mandelbrot = None
	pass
class Color_InterpPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Mandelbrot = None
	pass
class Color_PositionPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Mandelbrot = None
	pass
class ColorPlug(Plug):
	color_Color_ : Color_ColorPlug = PlugDescriptor("color_Color")
	clc_ : Color_ColorPlug = PlugDescriptor("color_Color")
	color_Interp_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	cli_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	color_Position_ : Color_PositionPlug = PlugDescriptor("color_Position")
	clp_ : Color_PositionPlug = PlugDescriptor("color_Position")
	node : Mandelbrot = None
	pass
class DepthPlug(Plug):
	node : Mandelbrot = None
	pass
class EscapeRadiusPlug(Plug):
	node : Mandelbrot = None
	pass
class FineOffsetUPlug(Plug):
	node : Mandelbrot = None
	pass
class FineOffsetVPlug(Plug):
	node : Mandelbrot = None
	pass
class FocusPlug(Plug):
	node : Mandelbrot = None
	pass
class ImplodePlug(Plug):
	node : Mandelbrot = None
	pass
class ImplodeCenterUPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : Mandelbrot = None
	pass
class ImplodeCenterVPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : Mandelbrot = None
	pass
class ImplodeCenterPlug(Plug):
	implodeCenterU_ : ImplodeCenterUPlug = PlugDescriptor("implodeCenterU")
	imu_ : ImplodeCenterUPlug = PlugDescriptor("implodeCenterU")
	implodeCenterV_ : ImplodeCenterVPlug = PlugDescriptor("implodeCenterV")
	imv_ : ImplodeCenterVPlug = PlugDescriptor("implodeCenterV")
	node : Mandelbrot = None
	pass
class JuliaUPlug(Plug):
	node : Mandelbrot = None
	pass
class JuliaVPlug(Plug):
	node : Mandelbrot = None
	pass
class LeafEffectPlug(Plug):
	node : Mandelbrot = None
	pass
class LineBlendingPlug(Plug):
	node : Mandelbrot = None
	pass
class LineFocusPlug(Plug):
	node : Mandelbrot = None
	pass
class LineOffsetRatioPlug(Plug):
	node : Mandelbrot = None
	pass
class LineOffsetUPlug(Plug):
	node : Mandelbrot = None
	pass
class LineOffsetVPlug(Plug):
	node : Mandelbrot = None
	pass
class LobesPlug(Plug):
	node : Mandelbrot = None
	pass
class MandelbrotInsideMethodPlug(Plug):
	node : Mandelbrot = None
	pass
class MandelbrotShadeMethodPlug(Plug):
	node : Mandelbrot = None
	pass
class MandelbrotTypePlug(Plug):
	node : Mandelbrot = None
	pass
class OrbitMapPlug(Plug):
	node : Mandelbrot = None
	pass
class OrbitMapColoringPlug(Plug):
	node : Mandelbrot = None
	pass
class OrbitMappingPlug(Plug):
	node : Mandelbrot = None
	pass
class OutUPlug(Plug):
	parent : OutUVPlug = PlugDescriptor("outUV")
	node : Mandelbrot = None
	pass
class OutVPlug(Plug):
	parent : OutUVPlug = PlugDescriptor("outUV")
	node : Mandelbrot = None
	pass
class OutUVPlug(Plug):
	outU_ : OutUPlug = PlugDescriptor("outU")
	ou_ : OutUPlug = PlugDescriptor("outU")
	outV_ : OutVPlug = PlugDescriptor("outV")
	ov_ : OutVPlug = PlugDescriptor("outV")
	node : Mandelbrot = None
	pass
class PointsPlug(Plug):
	node : Mandelbrot = None
	pass
class ShiftPlug(Plug):
	node : Mandelbrot = None
	pass
class StalksUPlug(Plug):
	node : Mandelbrot = None
	pass
class StalksVPlug(Plug):
	node : Mandelbrot = None
	pass
class Value_FloatValuePlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : Mandelbrot = None
	pass
class Value_InterpPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : Mandelbrot = None
	pass
class Value_PositionPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : Mandelbrot = None
	pass
class ValuePlug(Plug):
	value_FloatValue_ : Value_FloatValuePlug = PlugDescriptor("value_FloatValue")
	vlfv_ : Value_FloatValuePlug = PlugDescriptor("value_FloatValue")
	value_Interp_ : Value_InterpPlug = PlugDescriptor("value_Interp")
	vli_ : Value_InterpPlug = PlugDescriptor("value_Interp")
	value_Position_ : Value_PositionPlug = PlugDescriptor("value_Position")
	vlp_ : Value_PositionPlug = PlugDescriptor("value_Position")
	node : Mandelbrot = None
	pass
class WrapAmplitudePlug(Plug):
	node : Mandelbrot = None
	pass
class ZoomFactorPlug(Plug):
	node : Mandelbrot = None
	pass
# endregion


# define node class
class Mandelbrot(Texture2d):
	amplitude_ : AmplitudePlug = PlugDescriptor("amplitude")
	boxMinRadius_ : BoxMinRadiusPlug = PlugDescriptor("boxMinRadius")
	boxRadius_ : BoxRadiusPlug = PlugDescriptor("boxRadius")
	boxRatio_ : BoxRatioPlug = PlugDescriptor("boxRatio")
	centerU_ : CenterUPlug = PlugDescriptor("centerU")
	centerV_ : CenterVPlug = PlugDescriptor("centerV")
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
	fineOffsetU_ : FineOffsetUPlug = PlugDescriptor("fineOffsetU")
	fineOffsetV_ : FineOffsetVPlug = PlugDescriptor("fineOffsetV")
	focus_ : FocusPlug = PlugDescriptor("focus")
	implode_ : ImplodePlug = PlugDescriptor("implode")
	implodeCenterU_ : ImplodeCenterUPlug = PlugDescriptor("implodeCenterU")
	implodeCenterV_ : ImplodeCenterVPlug = PlugDescriptor("implodeCenterV")
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
	orbitMap_ : OrbitMapPlug = PlugDescriptor("orbitMap")
	orbitMapColoring_ : OrbitMapColoringPlug = PlugDescriptor("orbitMapColoring")
	orbitMapping_ : OrbitMappingPlug = PlugDescriptor("orbitMapping")
	outU_ : OutUPlug = PlugDescriptor("outU")
	outV_ : OutVPlug = PlugDescriptor("outV")
	outUV_ : OutUVPlug = PlugDescriptor("outUV")
	points_ : PointsPlug = PlugDescriptor("points")
	shift_ : ShiftPlug = PlugDescriptor("shift")
	stalksU_ : StalksUPlug = PlugDescriptor("stalksU")
	stalksV_ : StalksVPlug = PlugDescriptor("stalksV")
	value_FloatValue_ : Value_FloatValuePlug = PlugDescriptor("value_FloatValue")
	value_Interp_ : Value_InterpPlug = PlugDescriptor("value_Interp")
	value_Position_ : Value_PositionPlug = PlugDescriptor("value_Position")
	value_ : ValuePlug = PlugDescriptor("value")
	wrapAmplitude_ : WrapAmplitudePlug = PlugDescriptor("wrapAmplitude")
	zoomFactor_ : ZoomFactorPlug = PlugDescriptor("zoomFactor")

	# node attributes

	typeName = "mandelbrot"
	apiTypeInt = 1084
	apiTypeStr = "kMandelbrot"
	typeIdInt = 1381256513
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["amplitude", "boxMinRadius", "boxRadius", "boxRatio", "centerU", "centerV", "checker", "circleRadius", "circleSizeRatio", "circles", "color_ColorB", "color_ColorG", "color_ColorR", "color_Color", "color_Interp", "color_Position", "color", "depth", "escapeRadius", "fineOffsetU", "fineOffsetV", "focus", "implode", "implodeCenterU", "implodeCenterV", "implodeCenter", "juliaU", "juliaV", "leafEffect", "lineBlending", "lineFocus", "lineOffsetRatio", "lineOffsetU", "lineOffsetV", "lobes", "mandelbrotInsideMethod", "mandelbrotShadeMethod", "mandelbrotType", "orbitMap", "orbitMapColoring", "orbitMapping", "outU", "outV", "outUV", "points", "shift", "stalksU", "stalksV", "value_FloatValue", "value_Interp", "value_Position", "value", "wrapAmplitude", "zoomFactor"]
	nodeLeafPlugs = ["amplitude", "boxMinRadius", "boxRadius", "boxRatio", "centerU", "centerV", "checker", "circleRadius", "circleSizeRatio", "circles", "color", "depth", "escapeRadius", "fineOffsetU", "fineOffsetV", "focus", "implode", "implodeCenter", "juliaU", "juliaV", "leafEffect", "lineBlending", "lineFocus", "lineOffsetRatio", "lineOffsetU", "lineOffsetV", "lobes", "mandelbrotInsideMethod", "mandelbrotShadeMethod", "mandelbrotType", "orbitMap", "orbitMapColoring", "orbitMapping", "outUV", "points", "shift", "stalksU", "stalksV", "value", "wrapAmplitude", "zoomFactor"]
	pass

