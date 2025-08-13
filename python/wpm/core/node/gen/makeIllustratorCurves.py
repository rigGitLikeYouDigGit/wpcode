

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
class CountPlug(Plug):
	node : MakeIllustratorCurves = None
	pass
class IllustratorFilenamePlug(Plug):
	node : MakeIllustratorCurves = None
	pass
class OutputCurvesPlug(Plug):
	node : MakeIllustratorCurves = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : MakeIllustratorCurves = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : MakeIllustratorCurves = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : MakeIllustratorCurves = None
	pass
class PositionPlug(Plug):
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	pz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : MakeIllustratorCurves = None
	pass
class ReloadPlug(Plug):
	node : MakeIllustratorCurves = None
	pass
class ScaleFactorPlug(Plug):
	node : MakeIllustratorCurves = None
	pass
class TolerancePlug(Plug):
	node : MakeIllustratorCurves = None
	pass
# endregion


# define node class
class MakeIllustratorCurves(AbstractBaseCreate):
	count_ : CountPlug = PlugDescriptor("count")
	illustratorFilename_ : IllustratorFilenamePlug = PlugDescriptor("illustratorFilename")
	outputCurves_ : OutputCurvesPlug = PlugDescriptor("outputCurves")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	reload_ : ReloadPlug = PlugDescriptor("reload")
	scaleFactor_ : ScaleFactorPlug = PlugDescriptor("scaleFactor")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "makeIllustratorCurves"
	typeIdInt = 1313687875
	pass

