

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
MakeCircularArc = retriever.getNodeCls("MakeCircularArc")
assert MakeCircularArc
if T.TYPE_CHECKING:
	from .. import MakeCircularArc

# add node doc



# region plug type defs
class DirectionVectorXPlug(Plug):
	parent : DirectionVectorPlug = PlugDescriptor("directionVector")
	node : MakeTwoPointCircularArc = None
	pass
class DirectionVectorYPlug(Plug):
	parent : DirectionVectorPlug = PlugDescriptor("directionVector")
	node : MakeTwoPointCircularArc = None
	pass
class DirectionVectorZPlug(Plug):
	parent : DirectionVectorPlug = PlugDescriptor("directionVector")
	node : MakeTwoPointCircularArc = None
	pass
class DirectionVectorPlug(Plug):
	directionVectorX_ : DirectionVectorXPlug = PlugDescriptor("directionVectorX")
	dvx_ : DirectionVectorXPlug = PlugDescriptor("directionVectorX")
	directionVectorY_ : DirectionVectorYPlug = PlugDescriptor("directionVectorY")
	dvy_ : DirectionVectorYPlug = PlugDescriptor("directionVectorY")
	directionVectorZ_ : DirectionVectorZPlug = PlugDescriptor("directionVectorZ")
	dvz_ : DirectionVectorZPlug = PlugDescriptor("directionVectorZ")
	node : MakeTwoPointCircularArc = None
	pass
class Point1XPlug(Plug):
	parent : Point1Plug = PlugDescriptor("point1")
	node : MakeTwoPointCircularArc = None
	pass
class Point1YPlug(Plug):
	parent : Point1Plug = PlugDescriptor("point1")
	node : MakeTwoPointCircularArc = None
	pass
class Point1ZPlug(Plug):
	parent : Point1Plug = PlugDescriptor("point1")
	node : MakeTwoPointCircularArc = None
	pass
class Point1Plug(Plug):
	point1X_ : Point1XPlug = PlugDescriptor("point1X")
	p1x_ : Point1XPlug = PlugDescriptor("point1X")
	point1Y_ : Point1YPlug = PlugDescriptor("point1Y")
	p1y_ : Point1YPlug = PlugDescriptor("point1Y")
	point1Z_ : Point1ZPlug = PlugDescriptor("point1Z")
	p1z_ : Point1ZPlug = PlugDescriptor("point1Z")
	node : MakeTwoPointCircularArc = None
	pass
class Point2XPlug(Plug):
	parent : Point2Plug = PlugDescriptor("point2")
	node : MakeTwoPointCircularArc = None
	pass
class Point2YPlug(Plug):
	parent : Point2Plug = PlugDescriptor("point2")
	node : MakeTwoPointCircularArc = None
	pass
class Point2ZPlug(Plug):
	parent : Point2Plug = PlugDescriptor("point2")
	node : MakeTwoPointCircularArc = None
	pass
class Point2Plug(Plug):
	point2X_ : Point2XPlug = PlugDescriptor("point2X")
	p2x_ : Point2XPlug = PlugDescriptor("point2X")
	point2Y_ : Point2YPlug = PlugDescriptor("point2Y")
	p2y_ : Point2YPlug = PlugDescriptor("point2Y")
	point2Z_ : Point2ZPlug = PlugDescriptor("point2Z")
	p2z_ : Point2ZPlug = PlugDescriptor("point2Z")
	node : MakeTwoPointCircularArc = None
	pass
class RadiusPlug(Plug):
	node : MakeTwoPointCircularArc = None
	pass
class ToggleArcPlug(Plug):
	node : MakeTwoPointCircularArc = None
	pass
# endregion


# define node class
class MakeTwoPointCircularArc(MakeCircularArc):
	directionVectorX_ : DirectionVectorXPlug = PlugDescriptor("directionVectorX")
	directionVectorY_ : DirectionVectorYPlug = PlugDescriptor("directionVectorY")
	directionVectorZ_ : DirectionVectorZPlug = PlugDescriptor("directionVectorZ")
	directionVector_ : DirectionVectorPlug = PlugDescriptor("directionVector")
	point1X_ : Point1XPlug = PlugDescriptor("point1X")
	point1Y_ : Point1YPlug = PlugDescriptor("point1Y")
	point1Z_ : Point1ZPlug = PlugDescriptor("point1Z")
	point1_ : Point1Plug = PlugDescriptor("point1")
	point2X_ : Point2XPlug = PlugDescriptor("point2X")
	point2Y_ : Point2YPlug = PlugDescriptor("point2Y")
	point2Z_ : Point2ZPlug = PlugDescriptor("point2Z")
	point2_ : Point2Plug = PlugDescriptor("point2")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	toggleArc_ : ToggleArcPlug = PlugDescriptor("toggleArc")

	# node attributes

	typeName = "makeTwoPointCircularArc"
	typeIdInt = 1311916865
	pass

