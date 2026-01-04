

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
	node : ArrayMapper = None
	pass
class ComputeNodePlug(Plug):
	node : ArrayMapper = None
	pass
class ComputeNodeColorBPlug(Plug):
	parent : ComputeNodeColorPlug = PlugDescriptor("computeNodeColor")
	node : ArrayMapper = None
	pass
class ComputeNodeColorGPlug(Plug):
	parent : ComputeNodeColorPlug = PlugDescriptor("computeNodeColor")
	node : ArrayMapper = None
	pass
class ComputeNodeColorRPlug(Plug):
	parent : ComputeNodeColorPlug = PlugDescriptor("computeNodeColor")
	node : ArrayMapper = None
	pass
class ComputeNodeColorPlug(Plug):
	computeNodeColorB_ : ComputeNodeColorBPlug = PlugDescriptor("computeNodeColorB")
	cncb_ : ComputeNodeColorBPlug = PlugDescriptor("computeNodeColorB")
	computeNodeColorG_ : ComputeNodeColorGPlug = PlugDescriptor("computeNodeColorG")
	cncg_ : ComputeNodeColorGPlug = PlugDescriptor("computeNodeColorG")
	computeNodeColorR_ : ComputeNodeColorRPlug = PlugDescriptor("computeNodeColorR")
	cncr_ : ComputeNodeColorRPlug = PlugDescriptor("computeNodeColorR")
	node : ArrayMapper = None
	pass
class MaxValuePlug(Plug):
	node : ArrayMapper = None
	pass
class MinValuePlug(Plug):
	node : ArrayMapper = None
	pass
class OutColorPPPlug(Plug):
	node : ArrayMapper = None
	pass
class OutValuePPPlug(Plug):
	node : ArrayMapper = None
	pass
class TimePlug(Plug):
	node : ArrayMapper = None
	pass
class UCoordPPPlug(Plug):
	node : ArrayMapper = None
	pass
class VCoordPPPlug(Plug):
	node : ArrayMapper = None
	pass
# endregion


# define node class
class ArrayMapper(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	computeNode_ : ComputeNodePlug = PlugDescriptor("computeNode")
	computeNodeColorB_ : ComputeNodeColorBPlug = PlugDescriptor("computeNodeColorB")
	computeNodeColorG_ : ComputeNodeColorGPlug = PlugDescriptor("computeNodeColorG")
	computeNodeColorR_ : ComputeNodeColorRPlug = PlugDescriptor("computeNodeColorR")
	computeNodeColor_ : ComputeNodeColorPlug = PlugDescriptor("computeNodeColor")
	maxValue_ : MaxValuePlug = PlugDescriptor("maxValue")
	minValue_ : MinValuePlug = PlugDescriptor("minValue")
	outColorPP_ : OutColorPPPlug = PlugDescriptor("outColorPP")
	outValuePP_ : OutValuePPPlug = PlugDescriptor("outValuePP")
	time_ : TimePlug = PlugDescriptor("time")
	uCoordPP_ : UCoordPPPlug = PlugDescriptor("uCoordPP")
	vCoordPP_ : VCoordPPPlug = PlugDescriptor("vCoordPP")

	# node attributes

	typeName = "arrayMapper"
	apiTypeInt = 528
	apiTypeStr = "kArrayMapper"
	typeIdInt = 1145130320
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "computeNode", "computeNodeColorB", "computeNodeColorG", "computeNodeColorR", "computeNodeColor", "maxValue", "minValue", "outColorPP", "outValuePP", "time", "uCoordPP", "vCoordPP"]
	nodeLeafPlugs = ["binMembership", "computeNode", "computeNodeColor", "maxValue", "minValue", "outColorPP", "outValuePP", "time", "uCoordPP", "vCoordPP"]
	pass

