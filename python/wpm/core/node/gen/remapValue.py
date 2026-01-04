

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : RemapValue = None
	pass
class Color_ColorBPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : RemapValue = None
	pass
class Color_ColorGPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : RemapValue = None
	pass
class Color_ColorRPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : RemapValue = None
	pass
class Color_ColorPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	color_ColorB_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	clcb_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	color_ColorG_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	clcg_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	color_ColorR_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	clcr_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	node : RemapValue = None
	pass
class Color_InterpPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RemapValue = None
	pass
class Color_PositionPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RemapValue = None
	pass
class ColorPlug(Plug):
	color_Color_ : Color_ColorPlug = PlugDescriptor("color_Color")
	clc_ : Color_ColorPlug = PlugDescriptor("color_Color")
	color_Interp_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	cli_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	color_Position_ : Color_PositionPlug = PlugDescriptor("color_Position")
	clp_ : Color_PositionPlug = PlugDescriptor("color_Position")
	node : RemapValue = None
	pass
class InputMaxPlug(Plug):
	node : RemapValue = None
	pass
class InputMinPlug(Plug):
	node : RemapValue = None
	pass
class InputValuePlug(Plug):
	node : RemapValue = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : RemapValue = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : RemapValue = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : RemapValue = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : RemapValue = None
	pass
class OutValuePlug(Plug):
	node : RemapValue = None
	pass
class OutputMaxPlug(Plug):
	node : RemapValue = None
	pass
class OutputMinPlug(Plug):
	node : RemapValue = None
	pass
class Value_FloatValuePlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : RemapValue = None
	pass
class Value_InterpPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : RemapValue = None
	pass
class Value_PositionPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : RemapValue = None
	pass
class ValuePlug(Plug):
	value_FloatValue_ : Value_FloatValuePlug = PlugDescriptor("value_FloatValue")
	vlfv_ : Value_FloatValuePlug = PlugDescriptor("value_FloatValue")
	value_Interp_ : Value_InterpPlug = PlugDescriptor("value_Interp")
	vli_ : Value_InterpPlug = PlugDescriptor("value_Interp")
	value_Position_ : Value_PositionPlug = PlugDescriptor("value_Position")
	vlp_ : Value_PositionPlug = PlugDescriptor("value_Position")
	node : RemapValue = None
	pass
# endregion


# define node class
class RemapValue(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	color_ColorB_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	color_ColorG_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	color_ColorR_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	color_Color_ : Color_ColorPlug = PlugDescriptor("color_Color")
	color_Interp_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	color_Position_ : Color_PositionPlug = PlugDescriptor("color_Position")
	color_ : ColorPlug = PlugDescriptor("color")
	inputMax_ : InputMaxPlug = PlugDescriptor("inputMax")
	inputMin_ : InputMinPlug = PlugDescriptor("inputMin")
	inputValue_ : InputValuePlug = PlugDescriptor("inputValue")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outValue_ : OutValuePlug = PlugDescriptor("outValue")
	outputMax_ : OutputMaxPlug = PlugDescriptor("outputMax")
	outputMin_ : OutputMinPlug = PlugDescriptor("outputMin")
	value_FloatValue_ : Value_FloatValuePlug = PlugDescriptor("value_FloatValue")
	value_Interp_ : Value_InterpPlug = PlugDescriptor("value_Interp")
	value_Position_ : Value_PositionPlug = PlugDescriptor("value_Position")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "remapValue"
	apiTypeInt = 937
	apiTypeStr = "kRemapValue"
	typeIdInt = 1380800076
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "color_ColorB", "color_ColorG", "color_ColorR", "color_Color", "color_Interp", "color_Position", "color", "inputMax", "inputMin", "inputValue", "outColorB", "outColorG", "outColorR", "outColor", "outValue", "outputMax", "outputMin", "value_FloatValue", "value_Interp", "value_Position", "value"]
	nodeLeafPlugs = ["binMembership", "color", "inputMax", "inputMin", "inputValue", "outColor", "outValue", "outputMax", "outputMin", "value"]
	pass

