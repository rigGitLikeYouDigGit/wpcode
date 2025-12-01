

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PointLight = retriever.getNodeCls("PointLight")
assert PointLight
if T.TYPE_CHECKING:
	from .. import PointLight

# add node doc



# region plug type defs
class ArcPlug(Plug):
	node : VolumeLight = None
	pass
class ColorRange_ColorBPlug(Plug):
	parent : ColorRange_ColorPlug = PlugDescriptor("colorRange_Color")
	node : VolumeLight = None
	pass
class ColorRange_ColorGPlug(Plug):
	parent : ColorRange_ColorPlug = PlugDescriptor("colorRange_Color")
	node : VolumeLight = None
	pass
class ColorRange_ColorRPlug(Plug):
	parent : ColorRange_ColorPlug = PlugDescriptor("colorRange_Color")
	node : VolumeLight = None
	pass
class ColorRange_ColorPlug(Plug):
	parent : ColorRangePlug = PlugDescriptor("colorRange")
	colorRange_ColorB_ : ColorRange_ColorBPlug = PlugDescriptor("colorRange_ColorB")
	crgcb_ : ColorRange_ColorBPlug = PlugDescriptor("colorRange_ColorB")
	colorRange_ColorG_ : ColorRange_ColorGPlug = PlugDescriptor("colorRange_ColorG")
	crgcg_ : ColorRange_ColorGPlug = PlugDescriptor("colorRange_ColorG")
	colorRange_ColorR_ : ColorRange_ColorRPlug = PlugDescriptor("colorRange_ColorR")
	crgcr_ : ColorRange_ColorRPlug = PlugDescriptor("colorRange_ColorR")
	node : VolumeLight = None
	pass
class ColorRange_InterpPlug(Plug):
	parent : ColorRangePlug = PlugDescriptor("colorRange")
	node : VolumeLight = None
	pass
class ColorRange_PositionPlug(Plug):
	parent : ColorRangePlug = PlugDescriptor("colorRange")
	node : VolumeLight = None
	pass
class ColorRangePlug(Plug):
	colorRange_Color_ : ColorRange_ColorPlug = PlugDescriptor("colorRange_Color")
	crgc_ : ColorRange_ColorPlug = PlugDescriptor("colorRange_Color")
	colorRange_Interp_ : ColorRange_InterpPlug = PlugDescriptor("colorRange_Interp")
	crgi_ : ColorRange_InterpPlug = PlugDescriptor("colorRange_Interp")
	colorRange_Position_ : ColorRange_PositionPlug = PlugDescriptor("colorRange_Position")
	crgp_ : ColorRange_PositionPlug = PlugDescriptor("colorRange_Position")
	node : VolumeLight = None
	pass
class ConeEndRadiusPlug(Plug):
	node : VolumeLight = None
	pass
class EmitAmbientPlug(Plug):
	node : VolumeLight = None
	pass
class LightAnglePlug(Plug):
	node : VolumeLight = None
	pass
class LightShapePlug(Plug):
	node : VolumeLight = None
	pass
class Penumbra_FloatValuePlug(Plug):
	parent : PenumbraPlug = PlugDescriptor("penumbra")
	node : VolumeLight = None
	pass
class Penumbra_InterpPlug(Plug):
	parent : PenumbraPlug = PlugDescriptor("penumbra")
	node : VolumeLight = None
	pass
class Penumbra_PositionPlug(Plug):
	parent : PenumbraPlug = PlugDescriptor("penumbra")
	node : VolumeLight = None
	pass
class PenumbraPlug(Plug):
	penumbra_FloatValue_ : Penumbra_FloatValuePlug = PlugDescriptor("penumbra_FloatValue")
	penfv_ : Penumbra_FloatValuePlug = PlugDescriptor("penumbra_FloatValue")
	penumbra_Interp_ : Penumbra_InterpPlug = PlugDescriptor("penumbra_Interp")
	peni_ : Penumbra_InterpPlug = PlugDescriptor("penumbra_Interp")
	penumbra_Position_ : Penumbra_PositionPlug = PlugDescriptor("penumbra_Position")
	penp_ : Penumbra_PositionPlug = PlugDescriptor("penumbra_Position")
	node : VolumeLight = None
	pass
class VolumeLightDirPlug(Plug):
	node : VolumeLight = None
	pass
# endregion


# define node class
class VolumeLight(PointLight):
	arc_ : ArcPlug = PlugDescriptor("arc")
	colorRange_ColorB_ : ColorRange_ColorBPlug = PlugDescriptor("colorRange_ColorB")
	colorRange_ColorG_ : ColorRange_ColorGPlug = PlugDescriptor("colorRange_ColorG")
	colorRange_ColorR_ : ColorRange_ColorRPlug = PlugDescriptor("colorRange_ColorR")
	colorRange_Color_ : ColorRange_ColorPlug = PlugDescriptor("colorRange_Color")
	colorRange_Interp_ : ColorRange_InterpPlug = PlugDescriptor("colorRange_Interp")
	colorRange_Position_ : ColorRange_PositionPlug = PlugDescriptor("colorRange_Position")
	colorRange_ : ColorRangePlug = PlugDescriptor("colorRange")
	coneEndRadius_ : ConeEndRadiusPlug = PlugDescriptor("coneEndRadius")
	emitAmbient_ : EmitAmbientPlug = PlugDescriptor("emitAmbient")
	lightAngle_ : LightAnglePlug = PlugDescriptor("lightAngle")
	lightShape_ : LightShapePlug = PlugDescriptor("lightShape")
	penumbra_FloatValue_ : Penumbra_FloatValuePlug = PlugDescriptor("penumbra_FloatValue")
	penumbra_Interp_ : Penumbra_InterpPlug = PlugDescriptor("penumbra_Interp")
	penumbra_Position_ : Penumbra_PositionPlug = PlugDescriptor("penumbra_Position")
	penumbra_ : PenumbraPlug = PlugDescriptor("penumbra")
	volumeLightDir_ : VolumeLightDirPlug = PlugDescriptor("volumeLightDir")

	# node attributes

	typeName = "volumeLight"
	apiTypeInt = 897
	apiTypeStr = "kVolumeLight"
	typeIdInt = 1448037452
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["arc", "colorRange_ColorB", "colorRange_ColorG", "colorRange_ColorR", "colorRange_Color", "colorRange_Interp", "colorRange_Position", "colorRange", "coneEndRadius", "emitAmbient", "lightAngle", "lightShape", "penumbra_FloatValue", "penumbra_Interp", "penumbra_Position", "penumbra", "volumeLightDir"]
	nodeLeafPlugs = ["arc", "colorRange", "coneEndRadius", "emitAmbient", "lightAngle", "lightShape", "penumbra", "volumeLightDir"]
	pass

