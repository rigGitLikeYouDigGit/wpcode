

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class CenterXPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : MakeNurbCircle = None
	pass
class CenterYPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : MakeNurbCircle = None
	pass
class CenterZPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : MakeNurbCircle = None
	pass
class CenterPlug(Plug):
	centerX_ : CenterXPlug = PlugDescriptor("centerX")
	cx_ : CenterXPlug = PlugDescriptor("centerX")
	centerY_ : CenterYPlug = PlugDescriptor("centerY")
	cy_ : CenterYPlug = PlugDescriptor("centerY")
	centerZ_ : CenterZPlug = PlugDescriptor("centerZ")
	cz_ : CenterZPlug = PlugDescriptor("centerZ")
	node : MakeNurbCircle = None
	pass
class DegreePlug(Plug):
	node : MakeNurbCircle = None
	pass
class FirstPointXPlug(Plug):
	parent : FirstPlug = PlugDescriptor("first")
	node : MakeNurbCircle = None
	pass
class FirstPointYPlug(Plug):
	parent : FirstPlug = PlugDescriptor("first")
	node : MakeNurbCircle = None
	pass
class FirstPointZPlug(Plug):
	parent : FirstPlug = PlugDescriptor("first")
	node : MakeNurbCircle = None
	pass
class FirstPlug(Plug):
	firstPointX_ : FirstPointXPlug = PlugDescriptor("firstPointX")
	fpx_ : FirstPointXPlug = PlugDescriptor("firstPointX")
	firstPointY_ : FirstPointYPlug = PlugDescriptor("firstPointY")
	fpy_ : FirstPointYPlug = PlugDescriptor("firstPointY")
	firstPointZ_ : FirstPointZPlug = PlugDescriptor("firstPointZ")
	fpz_ : FirstPointZPlug = PlugDescriptor("firstPointZ")
	node : MakeNurbCircle = None
	pass
class FixCenterPlug(Plug):
	node : MakeNurbCircle = None
	pass
class NormalXPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : MakeNurbCircle = None
	pass
class NormalYPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : MakeNurbCircle = None
	pass
class NormalZPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : MakeNurbCircle = None
	pass
class NormalPlug(Plug):
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	nrx_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	nry_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	nrz_ : NormalZPlug = PlugDescriptor("normalZ")
	node : MakeNurbCircle = None
	pass
class OutputCurvePlug(Plug):
	node : MakeNurbCircle = None
	pass
class RadiusPlug(Plug):
	node : MakeNurbCircle = None
	pass
class SectionsPlug(Plug):
	node : MakeNurbCircle = None
	pass
class SweepPlug(Plug):
	node : MakeNurbCircle = None
	pass
class TolerancePlug(Plug):
	node : MakeNurbCircle = None
	pass
class UseTolerancePlug(Plug):
	node : MakeNurbCircle = None
	pass
# endregion


# define node class
class MakeNurbCircle(AbstractBaseCreate):
	centerX_ : CenterXPlug = PlugDescriptor("centerX")
	centerY_ : CenterYPlug = PlugDescriptor("centerY")
	centerZ_ : CenterZPlug = PlugDescriptor("centerZ")
	center_ : CenterPlug = PlugDescriptor("center")
	degree_ : DegreePlug = PlugDescriptor("degree")
	firstPointX_ : FirstPointXPlug = PlugDescriptor("firstPointX")
	firstPointY_ : FirstPointYPlug = PlugDescriptor("firstPointY")
	firstPointZ_ : FirstPointZPlug = PlugDescriptor("firstPointZ")
	first_ : FirstPlug = PlugDescriptor("first")
	fixCenter_ : FixCenterPlug = PlugDescriptor("fixCenter")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	normal_ : NormalPlug = PlugDescriptor("normal")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	sections_ : SectionsPlug = PlugDescriptor("sections")
	sweep_ : SweepPlug = PlugDescriptor("sweep")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")
	useTolerance_ : UseTolerancePlug = PlugDescriptor("useTolerance")

	# node attributes

	typeName = "makeNurbCircle"
	typeIdInt = 1313034819
	nodeLeafClassAttrs = ["centerX", "centerY", "centerZ", "center", "degree", "firstPointX", "firstPointY", "firstPointZ", "first", "fixCenter", "normalX", "normalY", "normalZ", "normal", "outputCurve", "radius", "sections", "sweep", "tolerance", "useTolerance"]
	nodeLeafPlugs = ["center", "degree", "first", "fixCenter", "normal", "outputCurve", "radius", "sections", "sweep", "tolerance", "useTolerance"]
	pass

