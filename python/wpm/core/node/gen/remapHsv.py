

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : RemapHsv = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RemapHsv = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RemapHsv = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RemapHsv = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : RemapHsv = None
	pass
class Hue_FloatValuePlug(Plug):
	parent : HuePlug = PlugDescriptor("hue")
	node : RemapHsv = None
	pass
class Hue_InterpPlug(Plug):
	parent : HuePlug = PlugDescriptor("hue")
	node : RemapHsv = None
	pass
class Hue_PositionPlug(Plug):
	parent : HuePlug = PlugDescriptor("hue")
	node : RemapHsv = None
	pass
class HuePlug(Plug):
	hue_FloatValue_ : Hue_FloatValuePlug = PlugDescriptor("hue_FloatValue")
	hfv_ : Hue_FloatValuePlug = PlugDescriptor("hue_FloatValue")
	hue_Interp_ : Hue_InterpPlug = PlugDescriptor("hue_Interp")
	hi_ : Hue_InterpPlug = PlugDescriptor("hue_Interp")
	hue_Position_ : Hue_PositionPlug = PlugDescriptor("hue_Position")
	hp_ : Hue_PositionPlug = PlugDescriptor("hue_Position")
	node : RemapHsv = None
	pass
class InputMaxPlug(Plug):
	node : RemapHsv = None
	pass
class InputMinPlug(Plug):
	node : RemapHsv = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : RemapHsv = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : RemapHsv = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : RemapHsv = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : RemapHsv = None
	pass
class OutputMaxPlug(Plug):
	node : RemapHsv = None
	pass
class OutputMinPlug(Plug):
	node : RemapHsv = None
	pass
class RenderPassModePlug(Plug):
	node : RemapHsv = None
	pass
class Saturation_FloatValuePlug(Plug):
	parent : SaturationPlug = PlugDescriptor("saturation")
	node : RemapHsv = None
	pass
class Saturation_InterpPlug(Plug):
	parent : SaturationPlug = PlugDescriptor("saturation")
	node : RemapHsv = None
	pass
class Saturation_PositionPlug(Plug):
	parent : SaturationPlug = PlugDescriptor("saturation")
	node : RemapHsv = None
	pass
class SaturationPlug(Plug):
	saturation_FloatValue_ : Saturation_FloatValuePlug = PlugDescriptor("saturation_FloatValue")
	sfv_ : Saturation_FloatValuePlug = PlugDescriptor("saturation_FloatValue")
	saturation_Interp_ : Saturation_InterpPlug = PlugDescriptor("saturation_Interp")
	si_ : Saturation_InterpPlug = PlugDescriptor("saturation_Interp")
	saturation_Position_ : Saturation_PositionPlug = PlugDescriptor("saturation_Position")
	sp_ : Saturation_PositionPlug = PlugDescriptor("saturation_Position")
	node : RemapHsv = None
	pass
class Value_FloatValuePlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : RemapHsv = None
	pass
class Value_InterpPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : RemapHsv = None
	pass
class Value_PositionPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : RemapHsv = None
	pass
class ValuePlug(Plug):
	value_FloatValue_ : Value_FloatValuePlug = PlugDescriptor("value_FloatValue")
	vfv_ : Value_FloatValuePlug = PlugDescriptor("value_FloatValue")
	value_Interp_ : Value_InterpPlug = PlugDescriptor("value_Interp")
	vi_ : Value_InterpPlug = PlugDescriptor("value_Interp")
	value_Position_ : Value_PositionPlug = PlugDescriptor("value_Position")
	vp_ : Value_PositionPlug = PlugDescriptor("value_Position")
	node : RemapHsv = None
	pass
# endregion


# define node class
class RemapHsv(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	hue_FloatValue_ : Hue_FloatValuePlug = PlugDescriptor("hue_FloatValue")
	hue_Interp_ : Hue_InterpPlug = PlugDescriptor("hue_Interp")
	hue_Position_ : Hue_PositionPlug = PlugDescriptor("hue_Position")
	hue_ : HuePlug = PlugDescriptor("hue")
	inputMax_ : InputMaxPlug = PlugDescriptor("inputMax")
	inputMin_ : InputMinPlug = PlugDescriptor("inputMin")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outputMax_ : OutputMaxPlug = PlugDescriptor("outputMax")
	outputMin_ : OutputMinPlug = PlugDescriptor("outputMin")
	renderPassMode_ : RenderPassModePlug = PlugDescriptor("renderPassMode")
	saturation_FloatValue_ : Saturation_FloatValuePlug = PlugDescriptor("saturation_FloatValue")
	saturation_Interp_ : Saturation_InterpPlug = PlugDescriptor("saturation_Interp")
	saturation_Position_ : Saturation_PositionPlug = PlugDescriptor("saturation_Position")
	saturation_ : SaturationPlug = PlugDescriptor("saturation")
	value_FloatValue_ : Value_FloatValuePlug = PlugDescriptor("value_FloatValue")
	value_Interp_ : Value_InterpPlug = PlugDescriptor("value_Interp")
	value_Position_ : Value_PositionPlug = PlugDescriptor("value_Position")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "remapHsv"
	apiTypeInt = 939
	apiTypeStr = "kRemapHsv"
	typeIdInt = 1380796499
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "colorB", "colorG", "colorR", "color", "hue_FloatValue", "hue_Interp", "hue_Position", "hue", "inputMax", "inputMin", "outColorB", "outColorG", "outColorR", "outColor", "outputMax", "outputMin", "renderPassMode", "saturation_FloatValue", "saturation_Interp", "saturation_Position", "saturation", "value_FloatValue", "value_Interp", "value_Position", "value"]
	nodeLeafPlugs = ["binMembership", "color", "hue", "inputMax", "inputMin", "outColor", "outputMax", "outputMin", "renderPassMode", "saturation", "value"]
	pass

