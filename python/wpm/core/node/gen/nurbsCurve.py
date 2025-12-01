

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
CurveShape = retriever.getNodeCls("CurveShape")
assert CurveShape
if T.TYPE_CHECKING:
	from .. import CurveShape

# add node doc



# region plug type defs
class AlwaysDrawOnTopPlug(Plug):
	node : NurbsCurve = None
	pass
class CachedPlug(Plug):
	node : NurbsCurve = None
	pass
class CreatePlug(Plug):
	node : NurbsCurve = None
	pass
class DegreePlug(Plug):
	node : NurbsCurve = None
	pass
class DispCVPlug(Plug):
	node : NurbsCurve = None
	pass
class DispCurveEndPointsPlug(Plug):
	node : NurbsCurve = None
	pass
class DispEPPlug(Plug):
	node : NurbsCurve = None
	pass
class DispGeometryPlug(Plug):
	node : NurbsCurve = None
	pass
class DispHullPlug(Plug):
	node : NurbsCurve = None
	pass
class XValueEpPlug(Plug):
	parent : EditPointsPlug = PlugDescriptor("editPoints")
	node : NurbsCurve = None
	pass
class YValueEpPlug(Plug):
	parent : EditPointsPlug = PlugDescriptor("editPoints")
	node : NurbsCurve = None
	pass
class ZValueEpPlug(Plug):
	parent : EditPointsPlug = PlugDescriptor("editPoints")
	node : NurbsCurve = None
	pass
class EditPointsPlug(Plug):
	xValueEp_ : XValueEpPlug = PlugDescriptor("xValueEp")
	xve_ : XValueEpPlug = PlugDescriptor("xValueEp")
	yValueEp_ : YValueEpPlug = PlugDescriptor("yValueEp")
	yve_ : YValueEpPlug = PlugDescriptor("yValueEp")
	zValueEp_ : ZValueEpPlug = PlugDescriptor("zValueEp")
	zve_ : ZValueEpPlug = PlugDescriptor("zValueEp")
	node : NurbsCurve = None
	pass
class FormPlug(Plug):
	node : NurbsCurve = None
	pass
class HeaderPlug(Plug):
	node : NurbsCurve = None
	pass
class InPlacePlug(Plug):
	node : NurbsCurve = None
	pass
class LineWidthPlug(Plug):
	node : NurbsCurve = None
	pass
class LocalPlug(Plug):
	node : NurbsCurve = None
	pass
class MaxValuePlug(Plug):
	parent : MinMaxValuePlug = PlugDescriptor("minMaxValue")
	node : NurbsCurve = None
	pass
class MinValuePlug(Plug):
	parent : MinMaxValuePlug = PlugDescriptor("minMaxValue")
	node : NurbsCurve = None
	pass
class MinMaxValuePlug(Plug):
	maxValue_ : MaxValuePlug = PlugDescriptor("maxValue")
	max_ : MaxValuePlug = PlugDescriptor("maxValue")
	minValue_ : MinValuePlug = PlugDescriptor("minValue")
	min_ : MinValuePlug = PlugDescriptor("minValue")
	node : NurbsCurve = None
	pass
class SpansPlug(Plug):
	node : NurbsCurve = None
	pass
class TweakSizePlug(Plug):
	node : NurbsCurve = None
	pass
class WorldNormalXPlug(Plug):
	parent : WorldNormalPlug = PlugDescriptor("worldNormal")
	node : NurbsCurve = None
	pass
class WorldNormalYPlug(Plug):
	parent : WorldNormalPlug = PlugDescriptor("worldNormal")
	node : NurbsCurve = None
	pass
class WorldNormalZPlug(Plug):
	parent : WorldNormalPlug = PlugDescriptor("worldNormal")
	node : NurbsCurve = None
	pass
class WorldNormalPlug(Plug):
	worldNormalX_ : WorldNormalXPlug = PlugDescriptor("worldNormalX")
	wnx_ : WorldNormalXPlug = PlugDescriptor("worldNormalX")
	worldNormalY_ : WorldNormalYPlug = PlugDescriptor("worldNormalY")
	wny_ : WorldNormalYPlug = PlugDescriptor("worldNormalY")
	worldNormalZ_ : WorldNormalZPlug = PlugDescriptor("worldNormalZ")
	wnz_ : WorldNormalZPlug = PlugDescriptor("worldNormalZ")
	node : NurbsCurve = None
	pass
class WorldSpacePlug(Plug):
	node : NurbsCurve = None
	pass
# endregion


# define node class
class NurbsCurve(CurveShape):
	alwaysDrawOnTop_ : AlwaysDrawOnTopPlug = PlugDescriptor("alwaysDrawOnTop")
	cached_ : CachedPlug = PlugDescriptor("cached")
	create_ : CreatePlug = PlugDescriptor("create")
	degree_ : DegreePlug = PlugDescriptor("degree")
	dispCV_ : DispCVPlug = PlugDescriptor("dispCV")
	dispCurveEndPoints_ : DispCurveEndPointsPlug = PlugDescriptor("dispCurveEndPoints")
	dispEP_ : DispEPPlug = PlugDescriptor("dispEP")
	dispGeometry_ : DispGeometryPlug = PlugDescriptor("dispGeometry")
	dispHull_ : DispHullPlug = PlugDescriptor("dispHull")
	xValueEp_ : XValueEpPlug = PlugDescriptor("xValueEp")
	yValueEp_ : YValueEpPlug = PlugDescriptor("yValueEp")
	zValueEp_ : ZValueEpPlug = PlugDescriptor("zValueEp")
	editPoints_ : EditPointsPlug = PlugDescriptor("editPoints")
	form_ : FormPlug = PlugDescriptor("form")
	header_ : HeaderPlug = PlugDescriptor("header")
	inPlace_ : InPlacePlug = PlugDescriptor("inPlace")
	lineWidth_ : LineWidthPlug = PlugDescriptor("lineWidth")
	local_ : LocalPlug = PlugDescriptor("local")
	maxValue_ : MaxValuePlug = PlugDescriptor("maxValue")
	minValue_ : MinValuePlug = PlugDescriptor("minValue")
	minMaxValue_ : MinMaxValuePlug = PlugDescriptor("minMaxValue")
	spans_ : SpansPlug = PlugDescriptor("spans")
	tweakSize_ : TweakSizePlug = PlugDescriptor("tweakSize")
	worldNormalX_ : WorldNormalXPlug = PlugDescriptor("worldNormalX")
	worldNormalY_ : WorldNormalYPlug = PlugDescriptor("worldNormalY")
	worldNormalZ_ : WorldNormalZPlug = PlugDescriptor("worldNormalZ")
	worldNormal_ : WorldNormalPlug = PlugDescriptor("worldNormal")
	worldSpace_ : WorldSpacePlug = PlugDescriptor("worldSpace")

	# node attributes

	typeName = "nurbsCurve"
	apiTypeInt = 267
	apiTypeStr = "kNurbsCurve"
	typeIdInt = 1313034838
	MFnCls = om.MFnNurbsCurve
	nodeLeafClassAttrs = ["alwaysDrawOnTop", "cached", "create", "degree", "dispCV", "dispCurveEndPoints", "dispEP", "dispGeometry", "dispHull", "xValueEp", "yValueEp", "zValueEp", "editPoints", "form", "header", "inPlace", "lineWidth", "local", "maxValue", "minValue", "minMaxValue", "spans", "tweakSize", "worldNormalX", "worldNormalY", "worldNormalZ", "worldNormal", "worldSpace"]
	nodeLeafPlugs = ["alwaysDrawOnTop", "cached", "create", "degree", "dispCV", "dispCurveEndPoints", "dispEP", "dispGeometry", "dispHull", "editPoints", "form", "header", "inPlace", "lineWidth", "local", "minMaxValue", "spans", "tweakSize", "worldNormal", "worldSpace"]
	pass

