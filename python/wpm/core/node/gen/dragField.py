

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
class CurrentTimePlug(Plug):
	node : DragField = None
	pass
class DirectionXPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : DragField = None
	pass
class DirectionYPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : DragField = None
	pass
class DirectionZPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : DragField = None
	pass
class DirectionPlug(Plug):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	dx_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	dy_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	dz_ : DirectionZPlug = PlugDescriptor("directionZ")
	node : DragField = None
	pass
class InheritVelocityPlug(Plug):
	node : DragField = None
	pass
class MotionAttenuationPlug(Plug):
	node : DragField = None
	pass
class SpeedAttenuationPlug(Plug):
	node : DragField = None
	pass
class UseDirectionPlug(Plug):
	node : DragField = None
	pass
# endregion


# define node class
class DragField(Field):
	currentTime_ : CurrentTimePlug = PlugDescriptor("currentTime")
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	inheritVelocity_ : InheritVelocityPlug = PlugDescriptor("inheritVelocity")
	motionAttenuation_ : MotionAttenuationPlug = PlugDescriptor("motionAttenuation")
	speedAttenuation_ : SpeedAttenuationPlug = PlugDescriptor("speedAttenuation")
	useDirection_ : UseDirectionPlug = PlugDescriptor("useDirection")

	# node attributes

	typeName = "dragField"
	typeIdInt = 1497649735
	nodeLeafClassAttrs = ["currentTime", "directionX", "directionY", "directionZ", "direction", "inheritVelocity", "motionAttenuation", "speedAttenuation", "useDirection"]
	nodeLeafPlugs = ["currentTime", "direction", "inheritVelocity", "motionAttenuation", "speedAttenuation", "useDirection"]
	pass

