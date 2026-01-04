

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierWorld = Catalogue.PolyModifierWorld
else:
	from .. import retriever
	PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
	assert PolyModifierWorld

# add node doc



# region plug type defs
class AxisXPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : PolyWedgeFace = None
	pass
class AxisYPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : PolyWedgeFace = None
	pass
class AxisZPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : PolyWedgeFace = None
	pass
class AxisPlug(Plug):
	axisX_ : AxisXPlug = PlugDescriptor("axisX")
	asx_ : AxisXPlug = PlugDescriptor("axisX")
	axisY_ : AxisYPlug = PlugDescriptor("axisY")
	asy_ : AxisYPlug = PlugDescriptor("axisY")
	axisZ_ : AxisZPlug = PlugDescriptor("axisZ")
	asz_ : AxisZPlug = PlugDescriptor("axisZ")
	node : PolyWedgeFace = None
	pass
class CenterXPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : PolyWedgeFace = None
	pass
class CenterYPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : PolyWedgeFace = None
	pass
class CenterZPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : PolyWedgeFace = None
	pass
class CenterPlug(Plug):
	centerX_ : CenterXPlug = PlugDescriptor("centerX")
	ctx_ : CenterXPlug = PlugDescriptor("centerX")
	centerY_ : CenterYPlug = PlugDescriptor("centerY")
	cty_ : CenterYPlug = PlugDescriptor("centerY")
	centerZ_ : CenterZPlug = PlugDescriptor("centerZ")
	ctz_ : CenterZPlug = PlugDescriptor("centerZ")
	node : PolyWedgeFace = None
	pass
class DivisionsPlug(Plug):
	node : PolyWedgeFace = None
	pass
class EdgePlug(Plug):
	node : PolyWedgeFace = None
	pass
class WedgeAnglePlug(Plug):
	node : PolyWedgeFace = None
	pass
# endregion


# define node class
class PolyWedgeFace(PolyModifierWorld):
	axisX_ : AxisXPlug = PlugDescriptor("axisX")
	axisY_ : AxisYPlug = PlugDescriptor("axisY")
	axisZ_ : AxisZPlug = PlugDescriptor("axisZ")
	axis_ : AxisPlug = PlugDescriptor("axis")
	centerX_ : CenterXPlug = PlugDescriptor("centerX")
	centerY_ : CenterYPlug = PlugDescriptor("centerY")
	centerZ_ : CenterZPlug = PlugDescriptor("centerZ")
	center_ : CenterPlug = PlugDescriptor("center")
	divisions_ : DivisionsPlug = PlugDescriptor("divisions")
	edge_ : EdgePlug = PlugDescriptor("edge")
	wedgeAngle_ : WedgeAnglePlug = PlugDescriptor("wedgeAngle")

	# node attributes

	typeName = "polyWedgeFace"
	apiTypeInt = 903
	apiTypeStr = "kPolyWedgeFace"
	typeIdInt = 1347896899
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["axisX", "axisY", "axisZ", "axis", "centerX", "centerY", "centerZ", "center", "divisions", "edge", "wedgeAngle"]
	nodeLeafPlugs = ["axis", "center", "divisions", "edge", "wedgeAngle"]
	pass

