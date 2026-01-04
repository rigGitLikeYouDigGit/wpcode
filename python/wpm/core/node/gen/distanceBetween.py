

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
	node : DistanceBetween = None
	pass
class DistancePlug(Plug):
	node : DistanceBetween = None
	pass
class InMatrix1Plug(Plug):
	node : DistanceBetween = None
	pass
class InMatrix2Plug(Plug):
	node : DistanceBetween = None
	pass
class Point1XPlug(Plug):
	parent : Point1Plug = PlugDescriptor("point1")
	node : DistanceBetween = None
	pass
class Point1YPlug(Plug):
	parent : Point1Plug = PlugDescriptor("point1")
	node : DistanceBetween = None
	pass
class Point1ZPlug(Plug):
	parent : Point1Plug = PlugDescriptor("point1")
	node : DistanceBetween = None
	pass
class Point1Plug(Plug):
	point1X_ : Point1XPlug = PlugDescriptor("point1X")
	p1x_ : Point1XPlug = PlugDescriptor("point1X")
	point1Y_ : Point1YPlug = PlugDescriptor("point1Y")
	p1y_ : Point1YPlug = PlugDescriptor("point1Y")
	point1Z_ : Point1ZPlug = PlugDescriptor("point1Z")
	p1z_ : Point1ZPlug = PlugDescriptor("point1Z")
	node : DistanceBetween = None
	pass
class Point2XPlug(Plug):
	parent : Point2Plug = PlugDescriptor("point2")
	node : DistanceBetween = None
	pass
class Point2YPlug(Plug):
	parent : Point2Plug = PlugDescriptor("point2")
	node : DistanceBetween = None
	pass
class Point2ZPlug(Plug):
	parent : Point2Plug = PlugDescriptor("point2")
	node : DistanceBetween = None
	pass
class Point2Plug(Plug):
	point2X_ : Point2XPlug = PlugDescriptor("point2X")
	p2x_ : Point2XPlug = PlugDescriptor("point2X")
	point2Y_ : Point2YPlug = PlugDescriptor("point2Y")
	p2y_ : Point2YPlug = PlugDescriptor("point2Y")
	point2Z_ : Point2ZPlug = PlugDescriptor("point2Z")
	p2z_ : Point2ZPlug = PlugDescriptor("point2Z")
	node : DistanceBetween = None
	pass
# endregion


# define node class
class DistanceBetween(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	distance_ : DistancePlug = PlugDescriptor("distance")
	inMatrix1_ : InMatrix1Plug = PlugDescriptor("inMatrix1")
	inMatrix2_ : InMatrix2Plug = PlugDescriptor("inMatrix2")
	point1X_ : Point1XPlug = PlugDescriptor("point1X")
	point1Y_ : Point1YPlug = PlugDescriptor("point1Y")
	point1Z_ : Point1ZPlug = PlugDescriptor("point1Z")
	point1_ : Point1Plug = PlugDescriptor("point1")
	point2X_ : Point2XPlug = PlugDescriptor("point2X")
	point2Y_ : Point2YPlug = PlugDescriptor("point2Y")
	point2Z_ : Point2ZPlug = PlugDescriptor("point2Z")
	point2_ : Point2Plug = PlugDescriptor("point2")

	# node attributes

	typeName = "distanceBetween"
	apiTypeInt = 322
	apiTypeStr = "kDistanceBetween"
	typeIdInt = 1145324116
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "distance", "inMatrix1", "inMatrix2", "point1X", "point1Y", "point1Z", "point1", "point2X", "point2Y", "point2Z", "point2"]
	nodeLeafPlugs = ["binMembership", "distance", "inMatrix1", "inMatrix2", "point1", "point2"]
	pass

