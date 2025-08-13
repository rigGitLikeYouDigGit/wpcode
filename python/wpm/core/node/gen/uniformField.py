

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
	node : UniformField = None
	pass
class DirectionYPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : UniformField = None
	pass
class DirectionZPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : UniformField = None
	pass
class DirectionPlug(Plug):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	dx_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	dy_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	dz_ : DirectionZPlug = PlugDescriptor("directionZ")
	node : UniformField = None
	pass
class InheritFactorPlug(Plug):
	node : UniformField = None
	pass
# endregion


# define node class
class UniformField(Field):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	inheritFactor_ : InheritFactorPlug = PlugDescriptor("inheritFactor")

	# node attributes

	typeName = "uniformField"
	typeIdInt = 1498762825
	pass

