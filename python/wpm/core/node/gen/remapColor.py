

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
	node : RemapColor = None
	pass
class Blue_FloatValuePlug(Plug):
	parent : BluePlug = PlugDescriptor("blue")
	node : RemapColor = None
	pass
class Blue_InterpPlug(Plug):
	parent : BluePlug = PlugDescriptor("blue")
	node : RemapColor = None
	pass
class Blue_PositionPlug(Plug):
	parent : BluePlug = PlugDescriptor("blue")
	node : RemapColor = None
	pass
class BluePlug(Plug):
	blue_FloatValue_ : Blue_FloatValuePlug = PlugDescriptor("blue_FloatValue")
	bfv_ : Blue_FloatValuePlug = PlugDescriptor("blue_FloatValue")
	blue_Interp_ : Blue_InterpPlug = PlugDescriptor("blue_Interp")
	bi_ : Blue_InterpPlug = PlugDescriptor("blue_Interp")
	blue_Position_ : Blue_PositionPlug = PlugDescriptor("blue_Position")
	bp_ : Blue_PositionPlug = PlugDescriptor("blue_Position")
	node : RemapColor = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RemapColor = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RemapColor = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RemapColor = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : RemapColor = None
	pass
class Green_FloatValuePlug(Plug):
	parent : GreenPlug = PlugDescriptor("green")
	node : RemapColor = None
	pass
class Green_InterpPlug(Plug):
	parent : GreenPlug = PlugDescriptor("green")
	node : RemapColor = None
	pass
class Green_PositionPlug(Plug):
	parent : GreenPlug = PlugDescriptor("green")
	node : RemapColor = None
	pass
class GreenPlug(Plug):
	green_FloatValue_ : Green_FloatValuePlug = PlugDescriptor("green_FloatValue")
	gfv_ : Green_FloatValuePlug = PlugDescriptor("green_FloatValue")
	green_Interp_ : Green_InterpPlug = PlugDescriptor("green_Interp")
	gi_ : Green_InterpPlug = PlugDescriptor("green_Interp")
	green_Position_ : Green_PositionPlug = PlugDescriptor("green_Position")
	gp_ : Green_PositionPlug = PlugDescriptor("green_Position")
	node : RemapColor = None
	pass
class InputMaxPlug(Plug):
	node : RemapColor = None
	pass
class InputMinPlug(Plug):
	node : RemapColor = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : RemapColor = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : RemapColor = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : RemapColor = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : RemapColor = None
	pass
class OutputMaxPlug(Plug):
	node : RemapColor = None
	pass
class OutputMinPlug(Plug):
	node : RemapColor = None
	pass
class Red_FloatValuePlug(Plug):
	parent : RedPlug = PlugDescriptor("red")
	node : RemapColor = None
	pass
class Red_InterpPlug(Plug):
	parent : RedPlug = PlugDescriptor("red")
	node : RemapColor = None
	pass
class Red_PositionPlug(Plug):
	parent : RedPlug = PlugDescriptor("red")
	node : RemapColor = None
	pass
class RedPlug(Plug):
	red_FloatValue_ : Red_FloatValuePlug = PlugDescriptor("red_FloatValue")
	rfv_ : Red_FloatValuePlug = PlugDescriptor("red_FloatValue")
	red_Interp_ : Red_InterpPlug = PlugDescriptor("red_Interp")
	ri_ : Red_InterpPlug = PlugDescriptor("red_Interp")
	red_Position_ : Red_PositionPlug = PlugDescriptor("red_Position")
	rp_ : Red_PositionPlug = PlugDescriptor("red_Position")
	node : RemapColor = None
	pass
class RenderPassModePlug(Plug):
	node : RemapColor = None
	pass
# endregion


# define node class
class RemapColor(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	blue_FloatValue_ : Blue_FloatValuePlug = PlugDescriptor("blue_FloatValue")
	blue_Interp_ : Blue_InterpPlug = PlugDescriptor("blue_Interp")
	blue_Position_ : Blue_PositionPlug = PlugDescriptor("blue_Position")
	blue_ : BluePlug = PlugDescriptor("blue")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	green_FloatValue_ : Green_FloatValuePlug = PlugDescriptor("green_FloatValue")
	green_Interp_ : Green_InterpPlug = PlugDescriptor("green_Interp")
	green_Position_ : Green_PositionPlug = PlugDescriptor("green_Position")
	green_ : GreenPlug = PlugDescriptor("green")
	inputMax_ : InputMaxPlug = PlugDescriptor("inputMax")
	inputMin_ : InputMinPlug = PlugDescriptor("inputMin")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outputMax_ : OutputMaxPlug = PlugDescriptor("outputMax")
	outputMin_ : OutputMinPlug = PlugDescriptor("outputMin")
	red_FloatValue_ : Red_FloatValuePlug = PlugDescriptor("red_FloatValue")
	red_Interp_ : Red_InterpPlug = PlugDescriptor("red_Interp")
	red_Position_ : Red_PositionPlug = PlugDescriptor("red_Position")
	red_ : RedPlug = PlugDescriptor("red")
	renderPassMode_ : RenderPassModePlug = PlugDescriptor("renderPassMode")

	# node attributes

	typeName = "remapColor"
	apiTypeInt = 938
	apiTypeStr = "kRemapColor"
	typeIdInt = 1380795212
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "blue_FloatValue", "blue_Interp", "blue_Position", "blue", "colorB", "colorG", "colorR", "color", "green_FloatValue", "green_Interp", "green_Position", "green", "inputMax", "inputMin", "outColorB", "outColorG", "outColorR", "outColor", "outputMax", "outputMin", "red_FloatValue", "red_Interp", "red_Position", "red", "renderPassMode"]
	nodeLeafPlugs = ["binMembership", "blue", "color", "green", "inputMax", "inputMin", "outColor", "outputMax", "outputMin", "red", "renderPassMode"]
	pass

