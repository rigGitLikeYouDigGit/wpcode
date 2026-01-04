

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
	node : SetRange = None
	pass
class MaxXPlug(Plug):
	parent : MaxPlug = PlugDescriptor("max")
	node : SetRange = None
	pass
class MaxYPlug(Plug):
	parent : MaxPlug = PlugDescriptor("max")
	node : SetRange = None
	pass
class MaxZPlug(Plug):
	parent : MaxPlug = PlugDescriptor("max")
	node : SetRange = None
	pass
class MaxPlug(Plug):
	maxX_ : MaxXPlug = PlugDescriptor("maxX")
	mx_ : MaxXPlug = PlugDescriptor("maxX")
	maxY_ : MaxYPlug = PlugDescriptor("maxY")
	my_ : MaxYPlug = PlugDescriptor("maxY")
	maxZ_ : MaxZPlug = PlugDescriptor("maxZ")
	mz_ : MaxZPlug = PlugDescriptor("maxZ")
	node : SetRange = None
	pass
class MinXPlug(Plug):
	parent : MinPlug = PlugDescriptor("min")
	node : SetRange = None
	pass
class MinYPlug(Plug):
	parent : MinPlug = PlugDescriptor("min")
	node : SetRange = None
	pass
class MinZPlug(Plug):
	parent : MinPlug = PlugDescriptor("min")
	node : SetRange = None
	pass
class MinPlug(Plug):
	minX_ : MinXPlug = PlugDescriptor("minX")
	nx_ : MinXPlug = PlugDescriptor("minX")
	minY_ : MinYPlug = PlugDescriptor("minY")
	ny_ : MinYPlug = PlugDescriptor("minY")
	minZ_ : MinZPlug = PlugDescriptor("minZ")
	nz_ : MinZPlug = PlugDescriptor("minZ")
	node : SetRange = None
	pass
class OldMaxXPlug(Plug):
	parent : OldMaxPlug = PlugDescriptor("oldMax")
	node : SetRange = None
	pass
class OldMaxYPlug(Plug):
	parent : OldMaxPlug = PlugDescriptor("oldMax")
	node : SetRange = None
	pass
class OldMaxZPlug(Plug):
	parent : OldMaxPlug = PlugDescriptor("oldMax")
	node : SetRange = None
	pass
class OldMaxPlug(Plug):
	oldMaxX_ : OldMaxXPlug = PlugDescriptor("oldMaxX")
	omx_ : OldMaxXPlug = PlugDescriptor("oldMaxX")
	oldMaxY_ : OldMaxYPlug = PlugDescriptor("oldMaxY")
	omy_ : OldMaxYPlug = PlugDescriptor("oldMaxY")
	oldMaxZ_ : OldMaxZPlug = PlugDescriptor("oldMaxZ")
	omz_ : OldMaxZPlug = PlugDescriptor("oldMaxZ")
	node : SetRange = None
	pass
class OldMinXPlug(Plug):
	parent : OldMinPlug = PlugDescriptor("oldMin")
	node : SetRange = None
	pass
class OldMinYPlug(Plug):
	parent : OldMinPlug = PlugDescriptor("oldMin")
	node : SetRange = None
	pass
class OldMinZPlug(Plug):
	parent : OldMinPlug = PlugDescriptor("oldMin")
	node : SetRange = None
	pass
class OldMinPlug(Plug):
	oldMinX_ : OldMinXPlug = PlugDescriptor("oldMinX")
	onx_ : OldMinXPlug = PlugDescriptor("oldMinX")
	oldMinY_ : OldMinYPlug = PlugDescriptor("oldMinY")
	ony_ : OldMinYPlug = PlugDescriptor("oldMinY")
	oldMinZ_ : OldMinZPlug = PlugDescriptor("oldMinZ")
	onz_ : OldMinZPlug = PlugDescriptor("oldMinZ")
	node : SetRange = None
	pass
class OutValueXPlug(Plug):
	parent : OutValuePlug = PlugDescriptor("outValue")
	node : SetRange = None
	pass
class OutValueYPlug(Plug):
	parent : OutValuePlug = PlugDescriptor("outValue")
	node : SetRange = None
	pass
class OutValueZPlug(Plug):
	parent : OutValuePlug = PlugDescriptor("outValue")
	node : SetRange = None
	pass
class OutValuePlug(Plug):
	outValueX_ : OutValueXPlug = PlugDescriptor("outValueX")
	ox_ : OutValueXPlug = PlugDescriptor("outValueX")
	outValueY_ : OutValueYPlug = PlugDescriptor("outValueY")
	oy_ : OutValueYPlug = PlugDescriptor("outValueY")
	outValueZ_ : OutValueZPlug = PlugDescriptor("outValueZ")
	oz_ : OutValueZPlug = PlugDescriptor("outValueZ")
	node : SetRange = None
	pass
class ValueXPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : SetRange = None
	pass
class ValueYPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : SetRange = None
	pass
class ValueZPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : SetRange = None
	pass
class ValuePlug(Plug):
	valueX_ : ValueXPlug = PlugDescriptor("valueX")
	vx_ : ValueXPlug = PlugDescriptor("valueX")
	valueY_ : ValueYPlug = PlugDescriptor("valueY")
	vy_ : ValueYPlug = PlugDescriptor("valueY")
	valueZ_ : ValueZPlug = PlugDescriptor("valueZ")
	vz_ : ValueZPlug = PlugDescriptor("valueZ")
	node : SetRange = None
	pass
# endregion


# define node class
class SetRange(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	maxX_ : MaxXPlug = PlugDescriptor("maxX")
	maxY_ : MaxYPlug = PlugDescriptor("maxY")
	maxZ_ : MaxZPlug = PlugDescriptor("maxZ")
	max_ : MaxPlug = PlugDescriptor("max")
	minX_ : MinXPlug = PlugDescriptor("minX")
	minY_ : MinYPlug = PlugDescriptor("minY")
	minZ_ : MinZPlug = PlugDescriptor("minZ")
	min_ : MinPlug = PlugDescriptor("min")
	oldMaxX_ : OldMaxXPlug = PlugDescriptor("oldMaxX")
	oldMaxY_ : OldMaxYPlug = PlugDescriptor("oldMaxY")
	oldMaxZ_ : OldMaxZPlug = PlugDescriptor("oldMaxZ")
	oldMax_ : OldMaxPlug = PlugDescriptor("oldMax")
	oldMinX_ : OldMinXPlug = PlugDescriptor("oldMinX")
	oldMinY_ : OldMinYPlug = PlugDescriptor("oldMinY")
	oldMinZ_ : OldMinZPlug = PlugDescriptor("oldMinZ")
	oldMin_ : OldMinPlug = PlugDescriptor("oldMin")
	outValueX_ : OutValueXPlug = PlugDescriptor("outValueX")
	outValueY_ : OutValueYPlug = PlugDescriptor("outValueY")
	outValueZ_ : OutValueZPlug = PlugDescriptor("outValueZ")
	outValue_ : OutValuePlug = PlugDescriptor("outValue")
	valueX_ : ValueXPlug = PlugDescriptor("valueX")
	valueY_ : ValueYPlug = PlugDescriptor("valueY")
	valueZ_ : ValueZPlug = PlugDescriptor("valueZ")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "setRange"
	apiTypeInt = 474
	apiTypeStr = "kSetRange"
	typeIdInt = 1381125703
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "maxX", "maxY", "maxZ", "max", "minX", "minY", "minZ", "min", "oldMaxX", "oldMaxY", "oldMaxZ", "oldMax", "oldMinX", "oldMinY", "oldMinZ", "oldMin", "outValueX", "outValueY", "outValueZ", "outValue", "valueX", "valueY", "valueZ", "value"]
	nodeLeafPlugs = ["binMembership", "max", "min", "oldMax", "oldMin", "outValue", "value"]
	pass

