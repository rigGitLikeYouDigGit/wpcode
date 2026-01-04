

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ManipContainer = Catalogue.ManipContainer
else:
	from .. import retriever
	ManipContainer = retriever.getNodeCls("ManipContainer")
	assert ManipContainer

# add node doc



# region plug type defs
class IconFilePlug(Plug):
	node : ButtonManip = None
	pass
class OffsetXPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : ButtonManip = None
	pass
class OffsetYPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : ButtonManip = None
	pass
class OffsetZPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : ButtonManip = None
	pass
class OffsetPlug(Plug):
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	ofx_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	ofy_ : OffsetYPlug = PlugDescriptor("offsetY")
	offsetZ_ : OffsetZPlug = PlugDescriptor("offsetZ")
	ofz_ : OffsetZPlug = PlugDescriptor("offsetZ")
	node : ButtonManip = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : ButtonManip = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : ButtonManip = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : ButtonManip = None
	pass
class PositionPlug(Plug):
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	pz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : ButtonManip = None
	pass
class ScriptPlug(Plug):
	node : ButtonManip = None
	pass
# endregion


# define node class
class ButtonManip(ManipContainer):
	iconFile_ : IconFilePlug = PlugDescriptor("iconFile")
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	offsetZ_ : OffsetZPlug = PlugDescriptor("offsetZ")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	script_ : ScriptPlug = PlugDescriptor("script")

	# node attributes

	typeName = "buttonManip"
	apiTypeInt = 153
	apiTypeStr = "kButtonManip"
	typeIdInt = 1431126612
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = ["iconFile", "offsetX", "offsetY", "offsetZ", "offset", "positionX", "positionY", "positionZ", "position", "script"]
	nodeLeafPlugs = ["iconFile", "offset", "position", "script"]
	pass

