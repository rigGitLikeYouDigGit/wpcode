

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Primitive = retriever.getNodeCls("Primitive")
assert Primitive
if T.TYPE_CHECKING:
	from .. import Primitive

# add node doc



# region plug type defs
class AbsoluteSweepDifferencePlug(Plug):
	node : RevolvedPrimitive = None
	pass
class BottomCapCurvePlug(Plug):
	node : RevolvedPrimitive = None
	pass
class DegreePlug(Plug):
	node : RevolvedPrimitive = None
	pass
class EndSweepPlug(Plug):
	node : RevolvedPrimitive = None
	pass
class HeightRatioPlug(Plug):
	node : RevolvedPrimitive = None
	pass
class RadiusPlug(Plug):
	node : RevolvedPrimitive = None
	pass
class SectionsPlug(Plug):
	node : RevolvedPrimitive = None
	pass
class SpansPlug(Plug):
	node : RevolvedPrimitive = None
	pass
class StartSweepPlug(Plug):
	node : RevolvedPrimitive = None
	pass
class TolerancePlug(Plug):
	node : RevolvedPrimitive = None
	pass
class TopCapCurvePlug(Plug):
	node : RevolvedPrimitive = None
	pass
class UseTolerancePlug(Plug):
	node : RevolvedPrimitive = None
	pass
# endregion


# define node class
class RevolvedPrimitive(Primitive):
	absoluteSweepDifference_ : AbsoluteSweepDifferencePlug = PlugDescriptor("absoluteSweepDifference")
	bottomCapCurve_ : BottomCapCurvePlug = PlugDescriptor("bottomCapCurve")
	degree_ : DegreePlug = PlugDescriptor("degree")
	endSweep_ : EndSweepPlug = PlugDescriptor("endSweep")
	heightRatio_ : HeightRatioPlug = PlugDescriptor("heightRatio")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	sections_ : SectionsPlug = PlugDescriptor("sections")
	spans_ : SpansPlug = PlugDescriptor("spans")
	startSweep_ : StartSweepPlug = PlugDescriptor("startSweep")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")
	topCapCurve_ : TopCapCurvePlug = PlugDescriptor("topCapCurve")
	useTolerance_ : UseTolerancePlug = PlugDescriptor("useTolerance")

	# node attributes

	typeName = "revolvedPrimitive"
	typeIdInt = 1314017363
	pass

