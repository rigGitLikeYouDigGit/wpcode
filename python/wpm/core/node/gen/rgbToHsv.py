

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
	node : RgbToHsv = None
	pass
class InRgbBPlug(Plug):
	parent : InRgbPlug = PlugDescriptor("inRgb")
	node : RgbToHsv = None
	pass
class InRgbGPlug(Plug):
	parent : InRgbPlug = PlugDescriptor("inRgb")
	node : RgbToHsv = None
	pass
class InRgbRPlug(Plug):
	parent : InRgbPlug = PlugDescriptor("inRgb")
	node : RgbToHsv = None
	pass
class InRgbPlug(Plug):
	inRgbB_ : InRgbBPlug = PlugDescriptor("inRgbB")
	ib_ : InRgbBPlug = PlugDescriptor("inRgbB")
	inRgbG_ : InRgbGPlug = PlugDescriptor("inRgbG")
	ig_ : InRgbGPlug = PlugDescriptor("inRgbG")
	inRgbR_ : InRgbRPlug = PlugDescriptor("inRgbR")
	ir_ : InRgbRPlug = PlugDescriptor("inRgbR")
	node : RgbToHsv = None
	pass
class OutHsvHPlug(Plug):
	parent : OutHsvPlug = PlugDescriptor("outHsv")
	node : RgbToHsv = None
	pass
class OutHsvSPlug(Plug):
	parent : OutHsvPlug = PlugDescriptor("outHsv")
	node : RgbToHsv = None
	pass
class OutHsvVPlug(Plug):
	parent : OutHsvPlug = PlugDescriptor("outHsv")
	node : RgbToHsv = None
	pass
class OutHsvPlug(Plug):
	outHsvH_ : OutHsvHPlug = PlugDescriptor("outHsvH")
	oh_ : OutHsvHPlug = PlugDescriptor("outHsvH")
	outHsvS_ : OutHsvSPlug = PlugDescriptor("outHsvS")
	os_ : OutHsvSPlug = PlugDescriptor("outHsvS")
	outHsvV_ : OutHsvVPlug = PlugDescriptor("outHsvV")
	ov_ : OutHsvVPlug = PlugDescriptor("outHsvV")
	node : RgbToHsv = None
	pass
class RenderPassModePlug(Plug):
	node : RgbToHsv = None
	pass
# endregion


# define node class
class RgbToHsv(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inRgbB_ : InRgbBPlug = PlugDescriptor("inRgbB")
	inRgbG_ : InRgbGPlug = PlugDescriptor("inRgbG")
	inRgbR_ : InRgbRPlug = PlugDescriptor("inRgbR")
	inRgb_ : InRgbPlug = PlugDescriptor("inRgb")
	outHsvH_ : OutHsvHPlug = PlugDescriptor("outHsvH")
	outHsvS_ : OutHsvSPlug = PlugDescriptor("outHsvS")
	outHsvV_ : OutHsvVPlug = PlugDescriptor("outHsvV")
	outHsv_ : OutHsvPlug = PlugDescriptor("outHsv")
	renderPassMode_ : RenderPassModePlug = PlugDescriptor("renderPassMode")

	# node attributes

	typeName = "rgbToHsv"
	apiTypeInt = 469
	apiTypeStr = "kRgbToHsv"
	typeIdInt = 1381118536
	MFnCls = om.MFnDependencyNode
	pass

