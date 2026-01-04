

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
	node : Condition = None
	pass
class ColorIfFalseBPlug(Plug):
	parent : ColorIfFalsePlug = PlugDescriptor("colorIfFalse")
	node : Condition = None
	pass
class ColorIfFalseGPlug(Plug):
	parent : ColorIfFalsePlug = PlugDescriptor("colorIfFalse")
	node : Condition = None
	pass
class ColorIfFalseRPlug(Plug):
	parent : ColorIfFalsePlug = PlugDescriptor("colorIfFalse")
	node : Condition = None
	pass
class ColorIfFalsePlug(Plug):
	colorIfFalseB_ : ColorIfFalseBPlug = PlugDescriptor("colorIfFalseB")
	cfb_ : ColorIfFalseBPlug = PlugDescriptor("colorIfFalseB")
	colorIfFalseG_ : ColorIfFalseGPlug = PlugDescriptor("colorIfFalseG")
	cfg_ : ColorIfFalseGPlug = PlugDescriptor("colorIfFalseG")
	colorIfFalseR_ : ColorIfFalseRPlug = PlugDescriptor("colorIfFalseR")
	cfr_ : ColorIfFalseRPlug = PlugDescriptor("colorIfFalseR")
	node : Condition = None
	pass
class ColorIfTrueBPlug(Plug):
	parent : ColorIfTruePlug = PlugDescriptor("colorIfTrue")
	node : Condition = None
	pass
class ColorIfTrueGPlug(Plug):
	parent : ColorIfTruePlug = PlugDescriptor("colorIfTrue")
	node : Condition = None
	pass
class ColorIfTrueRPlug(Plug):
	parent : ColorIfTruePlug = PlugDescriptor("colorIfTrue")
	node : Condition = None
	pass
class ColorIfTruePlug(Plug):
	colorIfTrueB_ : ColorIfTrueBPlug = PlugDescriptor("colorIfTrueB")
	ctb_ : ColorIfTrueBPlug = PlugDescriptor("colorIfTrueB")
	colorIfTrueG_ : ColorIfTrueGPlug = PlugDescriptor("colorIfTrueG")
	ctg_ : ColorIfTrueGPlug = PlugDescriptor("colorIfTrueG")
	colorIfTrueR_ : ColorIfTrueRPlug = PlugDescriptor("colorIfTrueR")
	ctr_ : ColorIfTrueRPlug = PlugDescriptor("colorIfTrueR")
	node : Condition = None
	pass
class FirstTermPlug(Plug):
	node : Condition = None
	pass
class OperationPlug(Plug):
	node : Condition = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : Condition = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : Condition = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : Condition = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : Condition = None
	pass
class SecondTermPlug(Plug):
	node : Condition = None
	pass
# endregion


# define node class
class Condition(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	colorIfFalseB_ : ColorIfFalseBPlug = PlugDescriptor("colorIfFalseB")
	colorIfFalseG_ : ColorIfFalseGPlug = PlugDescriptor("colorIfFalseG")
	colorIfFalseR_ : ColorIfFalseRPlug = PlugDescriptor("colorIfFalseR")
	colorIfFalse_ : ColorIfFalsePlug = PlugDescriptor("colorIfFalse")
	colorIfTrueB_ : ColorIfTrueBPlug = PlugDescriptor("colorIfTrueB")
	colorIfTrueG_ : ColorIfTrueGPlug = PlugDescriptor("colorIfTrueG")
	colorIfTrueR_ : ColorIfTrueRPlug = PlugDescriptor("colorIfTrueR")
	colorIfTrue_ : ColorIfTruePlug = PlugDescriptor("colorIfTrue")
	firstTerm_ : FirstTermPlug = PlugDescriptor("firstTerm")
	operation_ : OperationPlug = PlugDescriptor("operation")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	secondTerm_ : SecondTermPlug = PlugDescriptor("secondTerm")

	# node attributes

	typeName = "condition"
	apiTypeInt = 37
	apiTypeStr = "kCondition"
	typeIdInt = 1380142660
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "colorIfFalseB", "colorIfFalseG", "colorIfFalseR", "colorIfFalse", "colorIfTrueB", "colorIfTrueG", "colorIfTrueR", "colorIfTrue", "firstTerm", "operation", "outColorB", "outColorG", "outColorR", "outColor", "secondTerm"]
	nodeLeafPlugs = ["binMembership", "colorIfFalse", "colorIfTrue", "firstTerm", "operation", "outColor", "secondTerm"]
	pass

