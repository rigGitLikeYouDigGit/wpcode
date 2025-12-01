

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Field = retriever.getNodeCls("Field")
assert Field
if T.TYPE_CHECKING:
	from .. import Field

# add node doc



# region plug type defs
class DirectionXPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : GravityField = None
	pass
class DirectionYPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : GravityField = None
	pass
class DirectionZPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : GravityField = None
	pass
class DirectionPlug(Plug):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	dx_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	dy_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	dz_ : DirectionZPlug = PlugDescriptor("directionZ")
	node : GravityField = None
	pass
# endregion


# define node class
class GravityField(Field):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	direction_ : DirectionPlug = PlugDescriptor("direction")

	# node attributes

	typeName = "gravityField"
	typeIdInt = 1497846337
	nodeLeafClassAttrs = ["directionX", "directionY", "directionZ", "direction"]
	nodeLeafPlugs = ["direction"]
	pass

