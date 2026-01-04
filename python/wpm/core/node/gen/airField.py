

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Field = Catalogue.Field
else:
	from .. import retriever
	Field = retriever.getNodeCls("Field")
	assert Field

# add node doc



# region plug type defs
class ComponentOnlyPlug(Plug):
	node : AirField = None
	pass
class DirectionXPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : AirField = None
	pass
class DirectionYPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : AirField = None
	pass
class DirectionZPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : AirField = None
	pass
class DirectionPlug(Plug):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	dx_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	dy_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	dz_ : DirectionZPlug = PlugDescriptor("directionZ")
	node : AirField = None
	pass
class EnableSpreadPlug(Plug):
	node : AirField = None
	pass
class InheritRotationPlug(Plug):
	node : AirField = None
	pass
class InheritVelocityPlug(Plug):
	node : AirField = None
	pass
class SpeedPlug(Plug):
	node : AirField = None
	pass
class SpreadPlug(Plug):
	node : AirField = None
	pass
# endregion


# define node class
class AirField(Field):
	componentOnly_ : ComponentOnlyPlug = PlugDescriptor("componentOnly")
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	enableSpread_ : EnableSpreadPlug = PlugDescriptor("enableSpread")
	inheritRotation_ : InheritRotationPlug = PlugDescriptor("inheritRotation")
	inheritVelocity_ : InheritVelocityPlug = PlugDescriptor("inheritVelocity")
	speed_ : SpeedPlug = PlugDescriptor("speed")
	spread_ : SpreadPlug = PlugDescriptor("spread")

	# node attributes

	typeName = "airField"
	typeIdInt = 1497450834
	nodeLeafClassAttrs = ["componentOnly", "directionX", "directionY", "directionZ", "direction", "enableSpread", "inheritRotation", "inheritVelocity", "speed", "spread"]
	nodeLeafPlugs = ["componentOnly", "direction", "enableSpread", "inheritRotation", "inheritVelocity", "speed", "spread"]
	pass

