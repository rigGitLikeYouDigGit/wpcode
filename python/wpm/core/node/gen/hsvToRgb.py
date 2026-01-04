

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
	node : HsvToRgb = None
	pass
class InHsvBPlug(Plug):
	parent : InHsvPlug = PlugDescriptor("inHsv")
	node : HsvToRgb = None
	pass
class InHsvGPlug(Plug):
	parent : InHsvPlug = PlugDescriptor("inHsv")
	node : HsvToRgb = None
	pass
class InHsvRPlug(Plug):
	parent : InHsvPlug = PlugDescriptor("inHsv")
	node : HsvToRgb = None
	pass
class InHsvPlug(Plug):
	inHsvB_ : InHsvBPlug = PlugDescriptor("inHsvB")
	ib_ : InHsvBPlug = PlugDescriptor("inHsvB")
	inHsvG_ : InHsvGPlug = PlugDescriptor("inHsvG")
	ig_ : InHsvGPlug = PlugDescriptor("inHsvG")
	inHsvR_ : InHsvRPlug = PlugDescriptor("inHsvR")
	ir_ : InHsvRPlug = PlugDescriptor("inHsvR")
	node : HsvToRgb = None
	pass
class OutRgbBPlug(Plug):
	parent : OutRgbPlug = PlugDescriptor("outRgb")
	node : HsvToRgb = None
	pass
class OutRgbGPlug(Plug):
	parent : OutRgbPlug = PlugDescriptor("outRgb")
	node : HsvToRgb = None
	pass
class OutRgbRPlug(Plug):
	parent : OutRgbPlug = PlugDescriptor("outRgb")
	node : HsvToRgb = None
	pass
class OutRgbPlug(Plug):
	outRgbB_ : OutRgbBPlug = PlugDescriptor("outRgbB")
	ob_ : OutRgbBPlug = PlugDescriptor("outRgbB")
	outRgbG_ : OutRgbGPlug = PlugDescriptor("outRgbG")
	og_ : OutRgbGPlug = PlugDescriptor("outRgbG")
	outRgbR_ : OutRgbRPlug = PlugDescriptor("outRgbR")
	or_ : OutRgbRPlug = PlugDescriptor("outRgbR")
	node : HsvToRgb = None
	pass
class RenderPassModePlug(Plug):
	node : HsvToRgb = None
	pass
# endregion


# define node class
class HsvToRgb(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inHsvB_ : InHsvBPlug = PlugDescriptor("inHsvB")
	inHsvG_ : InHsvGPlug = PlugDescriptor("inHsvG")
	inHsvR_ : InHsvRPlug = PlugDescriptor("inHsvR")
	inHsv_ : InHsvPlug = PlugDescriptor("inHsv")
	outRgbB_ : OutRgbBPlug = PlugDescriptor("outRgbB")
	outRgbG_ : OutRgbGPlug = PlugDescriptor("outRgbG")
	outRgbR_ : OutRgbRPlug = PlugDescriptor("outRgbR")
	outRgb_ : OutRgbPlug = PlugDescriptor("outRgb")
	renderPassMode_ : RenderPassModePlug = PlugDescriptor("renderPassMode")

	# node attributes

	typeName = "hsvToRgb"
	apiTypeInt = 359
	apiTypeStr = "kHsvToRgb"
	typeIdInt = 1380463186
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "inHsvB", "inHsvG", "inHsvR", "inHsv", "outRgbB", "outRgbG", "outRgbR", "outRgb", "renderPassMode"]
	nodeLeafPlugs = ["binMembership", "inHsv", "outRgb", "renderPassMode"]
	pass

