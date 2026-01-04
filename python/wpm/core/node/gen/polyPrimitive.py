

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyCreator = Catalogue.PolyCreator
else:
	from .. import retriever
	PolyCreator = retriever.getNodeCls("PolyCreator")
	assert PolyCreator

# add node doc



# region plug type defs
class AxisXPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : PolyPrimitive = None
	pass
class AxisYPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : PolyPrimitive = None
	pass
class AxisZPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : PolyPrimitive = None
	pass
class AxisPlug(Plug):
	axisX_ : AxisXPlug = PlugDescriptor("axisX")
	axx_ : AxisXPlug = PlugDescriptor("axisX")
	axisY_ : AxisYPlug = PlugDescriptor("axisY")
	axy_ : AxisYPlug = PlugDescriptor("axisY")
	axisZ_ : AxisZPlug = PlugDescriptor("axisZ")
	axz_ : AxisZPlug = PlugDescriptor("axisZ")
	node : PolyPrimitive = None
	pass
class ComponentTagCreatePlug(Plug):
	node : PolyPrimitive = None
	pass
class ComponentTagPrefixPlug(Plug):
	node : PolyPrimitive = None
	pass
class ComponentTagSuffixPlug(Plug):
	node : PolyPrimitive = None
	pass
class HeightBaselinePlug(Plug):
	node : PolyPrimitive = None
	pass
class ParamWarnPlug(Plug):
	node : PolyPrimitive = None
	pass
class UvSetNamePlug(Plug):
	node : PolyPrimitive = None
	pass
# endregion


# define node class
class PolyPrimitive(PolyCreator):
	axisX_ : AxisXPlug = PlugDescriptor("axisX")
	axisY_ : AxisYPlug = PlugDescriptor("axisY")
	axisZ_ : AxisZPlug = PlugDescriptor("axisZ")
	axis_ : AxisPlug = PlugDescriptor("axis")
	componentTagCreate_ : ComponentTagCreatePlug = PlugDescriptor("componentTagCreate")
	componentTagPrefix_ : ComponentTagPrefixPlug = PlugDescriptor("componentTagPrefix")
	componentTagSuffix_ : ComponentTagSuffixPlug = PlugDescriptor("componentTagSuffix")
	heightBaseline_ : HeightBaselinePlug = PlugDescriptor("heightBaseline")
	paramWarn_ : ParamWarnPlug = PlugDescriptor("paramWarn")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "polyPrimitive"
	typeIdInt = 1347441229
	nodeLeafClassAttrs = ["axisX", "axisY", "axisZ", "axis", "componentTagCreate", "componentTagPrefix", "componentTagSuffix", "heightBaseline", "paramWarn", "uvSetName"]
	nodeLeafPlugs = ["axis", "componentTagCreate", "componentTagPrefix", "componentTagSuffix", "heightBaseline", "paramWarn", "uvSetName"]
	pass

