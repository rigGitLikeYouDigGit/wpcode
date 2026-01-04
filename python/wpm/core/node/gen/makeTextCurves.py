

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbstractBaseCreate = Catalogue.AbstractBaseCreate
else:
	from .. import retriever
	AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
	assert AbstractBaseCreate

# add node doc



# region plug type defs
class CountPlug(Plug):
	node : MakeTextCurves = None
	pass
class DeprecatedFontNamePlug(Plug):
	node : MakeTextCurves = None
	pass
class FontPlug(Plug):
	node : MakeTextCurves = None
	pass
class OutputCurvePlug(Plug):
	node : MakeTextCurves = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : MakeTextCurves = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : MakeTextCurves = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : MakeTextCurves = None
	pass
class PositionPlug(Plug):
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	pz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : MakeTextCurves = None
	pass
class TextPlug(Plug):
	node : MakeTextCurves = None
	pass
# endregion


# define node class
class MakeTextCurves(AbstractBaseCreate):
	count_ : CountPlug = PlugDescriptor("count")
	deprecatedFontName_ : DeprecatedFontNamePlug = PlugDescriptor("deprecatedFontName")
	font_ : FontPlug = PlugDescriptor("font")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	text_ : TextPlug = PlugDescriptor("text")

	# node attributes

	typeName = "makeTextCurves"
	typeIdInt = 1314150467
	nodeLeafClassAttrs = ["count", "deprecatedFontName", "font", "outputCurve", "positionX", "positionY", "positionZ", "position", "text"]
	nodeLeafPlugs = ["count", "deprecatedFontName", "font", "outputCurve", "position", "text"]
	pass

