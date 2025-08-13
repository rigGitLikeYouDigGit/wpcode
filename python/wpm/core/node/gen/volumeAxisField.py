

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
class AlongAxisPlug(Plug):
	node : VolumeAxisField = None
	pass
class AroundAxisPlug(Plug):
	node : VolumeAxisField = None
	pass
class AwayFromAxisPlug(Plug):
	node : VolumeAxisField = None
	pass
class AwayFromCenterPlug(Plug):
	node : VolumeAxisField = None
	pass
class DetailTurbulencePlug(Plug):
	node : VolumeAxisField = None
	pass
class DirectionXPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : VolumeAxisField = None
	pass
class DirectionYPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : VolumeAxisField = None
	pass
class DirectionZPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : VolumeAxisField = None
	pass
class DirectionPlug(Plug):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	dx_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	dy_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	dz_ : DirectionZPlug = PlugDescriptor("directionZ")
	node : VolumeAxisField = None
	pass
class DirectionalSpeedPlug(Plug):
	node : VolumeAxisField = None
	pass
class DisplaySpeedPlug(Plug):
	node : VolumeAxisField = None
	pass
class InvertAttenuationPlug(Plug):
	node : VolumeAxisField = None
	pass
class TimePlug(Plug):
	node : VolumeAxisField = None
	pass
class TurbulencePlug(Plug):
	node : VolumeAxisField = None
	pass
class TurbulenceFrequencyXPlug(Plug):
	parent : TurbulenceFrequencyPlug = PlugDescriptor("turbulenceFrequency")
	node : VolumeAxisField = None
	pass
class TurbulenceFrequencyYPlug(Plug):
	parent : TurbulenceFrequencyPlug = PlugDescriptor("turbulenceFrequency")
	node : VolumeAxisField = None
	pass
class TurbulenceFrequencyZPlug(Plug):
	parent : TurbulenceFrequencyPlug = PlugDescriptor("turbulenceFrequency")
	node : VolumeAxisField = None
	pass
class TurbulenceFrequencyPlug(Plug):
	turbulenceFrequencyX_ : TurbulenceFrequencyXPlug = PlugDescriptor("turbulenceFrequencyX")
	tfx_ : TurbulenceFrequencyXPlug = PlugDescriptor("turbulenceFrequencyX")
	turbulenceFrequencyY_ : TurbulenceFrequencyYPlug = PlugDescriptor("turbulenceFrequencyY")
	tfy_ : TurbulenceFrequencyYPlug = PlugDescriptor("turbulenceFrequencyY")
	turbulenceFrequencyZ_ : TurbulenceFrequencyZPlug = PlugDescriptor("turbulenceFrequencyZ")
	tfz_ : TurbulenceFrequencyZPlug = PlugDescriptor("turbulenceFrequencyZ")
	node : VolumeAxisField = None
	pass
class TurbulenceOffsetXPlug(Plug):
	parent : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	node : VolumeAxisField = None
	pass
class TurbulenceOffsetYPlug(Plug):
	parent : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	node : VolumeAxisField = None
	pass
class TurbulenceOffsetZPlug(Plug):
	parent : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	node : VolumeAxisField = None
	pass
class TurbulenceOffsetPlug(Plug):
	turbulenceOffsetX_ : TurbulenceOffsetXPlug = PlugDescriptor("turbulenceOffsetX")
	tox_ : TurbulenceOffsetXPlug = PlugDescriptor("turbulenceOffsetX")
	turbulenceOffsetY_ : TurbulenceOffsetYPlug = PlugDescriptor("turbulenceOffsetY")
	toy_ : TurbulenceOffsetYPlug = PlugDescriptor("turbulenceOffsetY")
	turbulenceOffsetZ_ : TurbulenceOffsetZPlug = PlugDescriptor("turbulenceOffsetZ")
	toz_ : TurbulenceOffsetZPlug = PlugDescriptor("turbulenceOffsetZ")
	node : VolumeAxisField = None
	pass
class TurbulenceSpeedPlug(Plug):
	node : VolumeAxisField = None
	pass
# endregion


# define node class
class VolumeAxisField(Field):
	alongAxis_ : AlongAxisPlug = PlugDescriptor("alongAxis")
	aroundAxis_ : AroundAxisPlug = PlugDescriptor("aroundAxis")
	awayFromAxis_ : AwayFromAxisPlug = PlugDescriptor("awayFromAxis")
	awayFromCenter_ : AwayFromCenterPlug = PlugDescriptor("awayFromCenter")
	detailTurbulence_ : DetailTurbulencePlug = PlugDescriptor("detailTurbulence")
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	directionalSpeed_ : DirectionalSpeedPlug = PlugDescriptor("directionalSpeed")
	displaySpeed_ : DisplaySpeedPlug = PlugDescriptor("displaySpeed")
	invertAttenuation_ : InvertAttenuationPlug = PlugDescriptor("invertAttenuation")
	time_ : TimePlug = PlugDescriptor("time")
	turbulence_ : TurbulencePlug = PlugDescriptor("turbulence")
	turbulenceFrequencyX_ : TurbulenceFrequencyXPlug = PlugDescriptor("turbulenceFrequencyX")
	turbulenceFrequencyY_ : TurbulenceFrequencyYPlug = PlugDescriptor("turbulenceFrequencyY")
	turbulenceFrequencyZ_ : TurbulenceFrequencyZPlug = PlugDescriptor("turbulenceFrequencyZ")
	turbulenceFrequency_ : TurbulenceFrequencyPlug = PlugDescriptor("turbulenceFrequency")
	turbulenceOffsetX_ : TurbulenceOffsetXPlug = PlugDescriptor("turbulenceOffsetX")
	turbulenceOffsetY_ : TurbulenceOffsetYPlug = PlugDescriptor("turbulenceOffsetY")
	turbulenceOffsetZ_ : TurbulenceOffsetZPlug = PlugDescriptor("turbulenceOffsetZ")
	turbulenceOffset_ : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	turbulenceSpeed_ : TurbulenceSpeedPlug = PlugDescriptor("turbulenceSpeed")

	# node attributes

	typeName = "volumeAxisField"
	typeIdInt = 1498830918
	pass

