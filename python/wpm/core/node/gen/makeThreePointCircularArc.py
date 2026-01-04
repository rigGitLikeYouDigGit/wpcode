

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	MakeCircularArc = Catalogue.MakeCircularArc
else:
	from .. import retriever
	MakeCircularArc = retriever.getNodeCls("MakeCircularArc")
	assert MakeCircularArc

# add node doc



# region plug type defs
class Point1XPlug(Plug):
	parent : Point1Plug = PlugDescriptor("point1")
	node : MakeThreePointCircularArc = None
	pass
class Point1YPlug(Plug):
	parent : Point1Plug = PlugDescriptor("point1")
	node : MakeThreePointCircularArc = None
	pass
class Point1ZPlug(Plug):
	parent : Point1Plug = PlugDescriptor("point1")
	node : MakeThreePointCircularArc = None
	pass
class Point1Plug(Plug):
	point1X_ : Point1XPlug = PlugDescriptor("point1X")
	p1x_ : Point1XPlug = PlugDescriptor("point1X")
	point1Y_ : Point1YPlug = PlugDescriptor("point1Y")
	p1y_ : Point1YPlug = PlugDescriptor("point1Y")
	point1Z_ : Point1ZPlug = PlugDescriptor("point1Z")
	p1z_ : Point1ZPlug = PlugDescriptor("point1Z")
	node : MakeThreePointCircularArc = None
	pass
class Point2XPlug(Plug):
	parent : Point2Plug = PlugDescriptor("point2")
	node : MakeThreePointCircularArc = None
	pass
class Point2YPlug(Plug):
	parent : Point2Plug = PlugDescriptor("point2")
	node : MakeThreePointCircularArc = None
	pass
class Point2ZPlug(Plug):
	parent : Point2Plug = PlugDescriptor("point2")
	node : MakeThreePointCircularArc = None
	pass
class Point2Plug(Plug):
	point2X_ : Point2XPlug = PlugDescriptor("point2X")
	p2x_ : Point2XPlug = PlugDescriptor("point2X")
	point2Y_ : Point2YPlug = PlugDescriptor("point2Y")
	p2y_ : Point2YPlug = PlugDescriptor("point2Y")
	point2Z_ : Point2ZPlug = PlugDescriptor("point2Z")
	p2z_ : Point2ZPlug = PlugDescriptor("point2Z")
	node : MakeThreePointCircularArc = None
	pass
class Point3XPlug(Plug):
	parent : Point3Plug = PlugDescriptor("point3")
	node : MakeThreePointCircularArc = None
	pass
class Point3YPlug(Plug):
	parent : Point3Plug = PlugDescriptor("point3")
	node : MakeThreePointCircularArc = None
	pass
class Point3ZPlug(Plug):
	parent : Point3Plug = PlugDescriptor("point3")
	node : MakeThreePointCircularArc = None
	pass
class Point3Plug(Plug):
	point3X_ : Point3XPlug = PlugDescriptor("point3X")
	p3x_ : Point3XPlug = PlugDescriptor("point3X")
	point3Y_ : Point3YPlug = PlugDescriptor("point3Y")
	p3y_ : Point3YPlug = PlugDescriptor("point3Y")
	point3Z_ : Point3ZPlug = PlugDescriptor("point3Z")
	p3z_ : Point3ZPlug = PlugDescriptor("point3Z")
	node : MakeThreePointCircularArc = None
	pass
class RadiusPlug(Plug):
	node : MakeThreePointCircularArc = None
	pass
# endregion


# define node class
class MakeThreePointCircularArc(MakeCircularArc):
	point1X_ : Point1XPlug = PlugDescriptor("point1X")
	point1Y_ : Point1YPlug = PlugDescriptor("point1Y")
	point1Z_ : Point1ZPlug = PlugDescriptor("point1Z")
	point1_ : Point1Plug = PlugDescriptor("point1")
	point2X_ : Point2XPlug = PlugDescriptor("point2X")
	point2Y_ : Point2YPlug = PlugDescriptor("point2Y")
	point2Z_ : Point2ZPlug = PlugDescriptor("point2Z")
	point2_ : Point2Plug = PlugDescriptor("point2")
	point3X_ : Point3XPlug = PlugDescriptor("point3X")
	point3Y_ : Point3YPlug = PlugDescriptor("point3Y")
	point3Z_ : Point3ZPlug = PlugDescriptor("point3Z")
	point3_ : Point3Plug = PlugDescriptor("point3")
	radius_ : RadiusPlug = PlugDescriptor("radius")

	# node attributes

	typeName = "makeThreePointCircularArc"
	typeIdInt = 1311982401
	nodeLeafClassAttrs = ["point1X", "point1Y", "point1Z", "point1", "point2X", "point2Y", "point2Z", "point2", "point3X", "point3Y", "point3Z", "point3", "radius"]
	nodeLeafPlugs = ["point1", "point2", "point3", "radius"]
	pass

