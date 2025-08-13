

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
class AnglePlug(Plug):
	parent : AxisAnglePlug = PlugDescriptor("axisAngle")
	node : AngleBetween = None
	pass
class AxisXPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : AngleBetween = None
	pass
class AxisYPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : AngleBetween = None
	pass
class AxisZPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : AngleBetween = None
	pass
class AxisPlug(Plug):
	parent : AxisAnglePlug = PlugDescriptor("axisAngle")
	axisX_ : AxisXPlug = PlugDescriptor("axisX")
	axx_ : AxisXPlug = PlugDescriptor("axisX")
	axisY_ : AxisYPlug = PlugDescriptor("axisY")
	axy_ : AxisYPlug = PlugDescriptor("axisY")
	axisZ_ : AxisZPlug = PlugDescriptor("axisZ")
	axz_ : AxisZPlug = PlugDescriptor("axisZ")
	node : AngleBetween = None
	pass
class AxisAnglePlug(Plug):
	angle_ : AnglePlug = PlugDescriptor("angle")
	a_ : AnglePlug = PlugDescriptor("angle")
	axis_ : AxisPlug = PlugDescriptor("axis")
	ax_ : AxisPlug = PlugDescriptor("axis")
	node : AngleBetween = None
	pass
class BinMembershipPlug(Plug):
	node : AngleBetween = None
	pass
class EulerXPlug(Plug):
	parent : EulerPlug = PlugDescriptor("euler")
	node : AngleBetween = None
	pass
class EulerYPlug(Plug):
	parent : EulerPlug = PlugDescriptor("euler")
	node : AngleBetween = None
	pass
class EulerZPlug(Plug):
	parent : EulerPlug = PlugDescriptor("euler")
	node : AngleBetween = None
	pass
class EulerPlug(Plug):
	eulerX_ : EulerXPlug = PlugDescriptor("eulerX")
	eux_ : EulerXPlug = PlugDescriptor("eulerX")
	eulerY_ : EulerYPlug = PlugDescriptor("eulerY")
	euy_ : EulerYPlug = PlugDescriptor("eulerY")
	eulerZ_ : EulerZPlug = PlugDescriptor("eulerZ")
	euz_ : EulerZPlug = PlugDescriptor("eulerZ")
	node : AngleBetween = None
	pass
class Vector1XPlug(Plug):
	parent : Vector1Plug = PlugDescriptor("vector1")
	node : AngleBetween = None
	pass
class Vector1YPlug(Plug):
	parent : Vector1Plug = PlugDescriptor("vector1")
	node : AngleBetween = None
	pass
class Vector1ZPlug(Plug):
	parent : Vector1Plug = PlugDescriptor("vector1")
	node : AngleBetween = None
	pass
class Vector1Plug(Plug):
	vector1X_ : Vector1XPlug = PlugDescriptor("vector1X")
	v1x_ : Vector1XPlug = PlugDescriptor("vector1X")
	vector1Y_ : Vector1YPlug = PlugDescriptor("vector1Y")
	v1y_ : Vector1YPlug = PlugDescriptor("vector1Y")
	vector1Z_ : Vector1ZPlug = PlugDescriptor("vector1Z")
	v1z_ : Vector1ZPlug = PlugDescriptor("vector1Z")
	node : AngleBetween = None
	pass
class Vector2XPlug(Plug):
	parent : Vector2Plug = PlugDescriptor("vector2")
	node : AngleBetween = None
	pass
class Vector2YPlug(Plug):
	parent : Vector2Plug = PlugDescriptor("vector2")
	node : AngleBetween = None
	pass
class Vector2ZPlug(Plug):
	parent : Vector2Plug = PlugDescriptor("vector2")
	node : AngleBetween = None
	pass
class Vector2Plug(Plug):
	vector2X_ : Vector2XPlug = PlugDescriptor("vector2X")
	v2x_ : Vector2XPlug = PlugDescriptor("vector2X")
	vector2Y_ : Vector2YPlug = PlugDescriptor("vector2Y")
	v2y_ : Vector2YPlug = PlugDescriptor("vector2Y")
	vector2Z_ : Vector2ZPlug = PlugDescriptor("vector2Z")
	v2z_ : Vector2ZPlug = PlugDescriptor("vector2Z")
	node : AngleBetween = None
	pass
# endregion


# define node class
class AngleBetween(_BASE_):
	angle_ : AnglePlug = PlugDescriptor("angle")
	axisX_ : AxisXPlug = PlugDescriptor("axisX")
	axisY_ : AxisYPlug = PlugDescriptor("axisY")
	axisZ_ : AxisZPlug = PlugDescriptor("axisZ")
	axis_ : AxisPlug = PlugDescriptor("axis")
	axisAngle_ : AxisAnglePlug = PlugDescriptor("axisAngle")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	eulerX_ : EulerXPlug = PlugDescriptor("eulerX")
	eulerY_ : EulerYPlug = PlugDescriptor("eulerY")
	eulerZ_ : EulerZPlug = PlugDescriptor("eulerZ")
	euler_ : EulerPlug = PlugDescriptor("euler")
	vector1X_ : Vector1XPlug = PlugDescriptor("vector1X")
	vector1Y_ : Vector1YPlug = PlugDescriptor("vector1Y")
	vector1Z_ : Vector1ZPlug = PlugDescriptor("vector1Z")
	vector1_ : Vector1Plug = PlugDescriptor("vector1")
	vector2X_ : Vector2XPlug = PlugDescriptor("vector2X")
	vector2Y_ : Vector2YPlug = PlugDescriptor("vector2Y")
	vector2Z_ : Vector2ZPlug = PlugDescriptor("vector2Z")
	vector2_ : Vector2Plug = PlugDescriptor("vector2")

	# node attributes

	typeName = "angleBetween"
	apiTypeInt = 21
	apiTypeStr = "kAngleBetween"
	typeIdInt = 1312899668
	MFnCls = om.MFnDependencyNode
	pass

