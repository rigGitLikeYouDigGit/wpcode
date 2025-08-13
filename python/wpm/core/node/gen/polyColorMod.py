

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class AlphaScale_FloatValuePlug(Plug):
	parent : AlphaScalePlug = PlugDescriptor("alphaScale")
	node : PolyColorMod = None
	pass
class AlphaScale_InterpPlug(Plug):
	parent : AlphaScalePlug = PlugDescriptor("alphaScale")
	node : PolyColorMod = None
	pass
class AlphaScale_PositionPlug(Plug):
	parent : AlphaScalePlug = PlugDescriptor("alphaScale")
	node : PolyColorMod = None
	pass
class AlphaScalePlug(Plug):
	alphaScale_FloatValue_ : AlphaScale_FloatValuePlug = PlugDescriptor("alphaScale_FloatValue")
	afv_ : AlphaScale_FloatValuePlug = PlugDescriptor("alphaScale_FloatValue")
	alphaScale_Interp_ : AlphaScale_InterpPlug = PlugDescriptor("alphaScale_Interp")
	ai_ : AlphaScale_InterpPlug = PlugDescriptor("alphaScale_Interp")
	alphaScale_Position_ : AlphaScale_PositionPlug = PlugDescriptor("alphaScale_Position")
	ap_ : AlphaScale_PositionPlug = PlugDescriptor("alphaScale_Position")
	node : PolyColorMod = None
	pass
class BaseColorNamePlug(Plug):
	node : PolyColorMod = None
	pass
class BlueScale_FloatValuePlug(Plug):
	parent : BlueScalePlug = PlugDescriptor("blueScale")
	node : PolyColorMod = None
	pass
class BlueScale_InterpPlug(Plug):
	parent : BlueScalePlug = PlugDescriptor("blueScale")
	node : PolyColorMod = None
	pass
class BlueScale_PositionPlug(Plug):
	parent : BlueScalePlug = PlugDescriptor("blueScale")
	node : PolyColorMod = None
	pass
class BlueScalePlug(Plug):
	blueScale_FloatValue_ : BlueScale_FloatValuePlug = PlugDescriptor("blueScale_FloatValue")
	bfv_ : BlueScale_FloatValuePlug = PlugDescriptor("blueScale_FloatValue")
	blueScale_Interp_ : BlueScale_InterpPlug = PlugDescriptor("blueScale_Interp")
	bi_ : BlueScale_InterpPlug = PlugDescriptor("blueScale_Interp")
	blueScale_Position_ : BlueScale_PositionPlug = PlugDescriptor("blueScale_Position")
	bp_ : BlueScale_PositionPlug = PlugDescriptor("blueScale_Position")
	node : PolyColorMod = None
	pass
class GreenScale_FloatValuePlug(Plug):
	parent : GreenScalePlug = PlugDescriptor("greenScale")
	node : PolyColorMod = None
	pass
class GreenScale_InterpPlug(Plug):
	parent : GreenScalePlug = PlugDescriptor("greenScale")
	node : PolyColorMod = None
	pass
class GreenScale_PositionPlug(Plug):
	parent : GreenScalePlug = PlugDescriptor("greenScale")
	node : PolyColorMod = None
	pass
class GreenScalePlug(Plug):
	greenScale_FloatValue_ : GreenScale_FloatValuePlug = PlugDescriptor("greenScale_FloatValue")
	gfv_ : GreenScale_FloatValuePlug = PlugDescriptor("greenScale_FloatValue")
	greenScale_Interp_ : GreenScale_InterpPlug = PlugDescriptor("greenScale_Interp")
	gi_ : GreenScale_InterpPlug = PlugDescriptor("greenScale_Interp")
	greenScale_Position_ : GreenScale_PositionPlug = PlugDescriptor("greenScale_Position")
	gp_ : GreenScale_PositionPlug = PlugDescriptor("greenScale_Position")
	node : PolyColorMod = None
	pass
class HuevPlug(Plug):
	node : PolyColorMod = None
	pass
class IntensityScale_FloatValuePlug(Plug):
	parent : IntensityScalePlug = PlugDescriptor("intensityScale")
	node : PolyColorMod = None
	pass
class IntensityScale_InterpPlug(Plug):
	parent : IntensityScalePlug = PlugDescriptor("intensityScale")
	node : PolyColorMod = None
	pass
class IntensityScale_PositionPlug(Plug):
	parent : IntensityScalePlug = PlugDescriptor("intensityScale")
	node : PolyColorMod = None
	pass
class IntensityScalePlug(Plug):
	intensityScale_FloatValue_ : IntensityScale_FloatValuePlug = PlugDescriptor("intensityScale_FloatValue")
	nfv_ : IntensityScale_FloatValuePlug = PlugDescriptor("intensityScale_FloatValue")
	intensityScale_Interp_ : IntensityScale_InterpPlug = PlugDescriptor("intensityScale_Interp")
	ni_ : IntensityScale_InterpPlug = PlugDescriptor("intensityScale_Interp")
	intensityScale_Position_ : IntensityScale_PositionPlug = PlugDescriptor("intensityScale_Position")
	np_ : IntensityScale_PositionPlug = PlugDescriptor("intensityScale_Position")
	node : PolyColorMod = None
	pass
class RedScale_FloatValuePlug(Plug):
	parent : RedScalePlug = PlugDescriptor("redScale")
	node : PolyColorMod = None
	pass
class RedScale_InterpPlug(Plug):
	parent : RedScalePlug = PlugDescriptor("redScale")
	node : PolyColorMod = None
	pass
class RedScale_PositionPlug(Plug):
	parent : RedScalePlug = PlugDescriptor("redScale")
	node : PolyColorMod = None
	pass
class RedScalePlug(Plug):
	redScale_FloatValue_ : RedScale_FloatValuePlug = PlugDescriptor("redScale_FloatValue")
	rfv_ : RedScale_FloatValuePlug = PlugDescriptor("redScale_FloatValue")
	redScale_Interp_ : RedScale_InterpPlug = PlugDescriptor("redScale_Interp")
	ri_ : RedScale_InterpPlug = PlugDescriptor("redScale_Interp")
	redScale_Position_ : RedScale_PositionPlug = PlugDescriptor("redScale_Position")
	rp_ : RedScale_PositionPlug = PlugDescriptor("redScale_Position")
	node : PolyColorMod = None
	pass
class SatvPlug(Plug):
	node : PolyColorMod = None
	pass
class ValuePlug(Plug):
	node : PolyColorMod = None
	pass
# endregion


# define node class
class PolyColorMod(PolyModifier):
	alphaScale_FloatValue_ : AlphaScale_FloatValuePlug = PlugDescriptor("alphaScale_FloatValue")
	alphaScale_Interp_ : AlphaScale_InterpPlug = PlugDescriptor("alphaScale_Interp")
	alphaScale_Position_ : AlphaScale_PositionPlug = PlugDescriptor("alphaScale_Position")
	alphaScale_ : AlphaScalePlug = PlugDescriptor("alphaScale")
	baseColorName_ : BaseColorNamePlug = PlugDescriptor("baseColorName")
	blueScale_FloatValue_ : BlueScale_FloatValuePlug = PlugDescriptor("blueScale_FloatValue")
	blueScale_Interp_ : BlueScale_InterpPlug = PlugDescriptor("blueScale_Interp")
	blueScale_Position_ : BlueScale_PositionPlug = PlugDescriptor("blueScale_Position")
	blueScale_ : BlueScalePlug = PlugDescriptor("blueScale")
	greenScale_FloatValue_ : GreenScale_FloatValuePlug = PlugDescriptor("greenScale_FloatValue")
	greenScale_Interp_ : GreenScale_InterpPlug = PlugDescriptor("greenScale_Interp")
	greenScale_Position_ : GreenScale_PositionPlug = PlugDescriptor("greenScale_Position")
	greenScale_ : GreenScalePlug = PlugDescriptor("greenScale")
	huev_ : HuevPlug = PlugDescriptor("huev")
	intensityScale_FloatValue_ : IntensityScale_FloatValuePlug = PlugDescriptor("intensityScale_FloatValue")
	intensityScale_Interp_ : IntensityScale_InterpPlug = PlugDescriptor("intensityScale_Interp")
	intensityScale_Position_ : IntensityScale_PositionPlug = PlugDescriptor("intensityScale_Position")
	intensityScale_ : IntensityScalePlug = PlugDescriptor("intensityScale")
	redScale_FloatValue_ : RedScale_FloatValuePlug = PlugDescriptor("redScale_FloatValue")
	redScale_Interp_ : RedScale_InterpPlug = PlugDescriptor("redScale_Interp")
	redScale_Position_ : RedScale_PositionPlug = PlugDescriptor("redScale_Position")
	redScale_ : RedScalePlug = PlugDescriptor("redScale")
	satv_ : SatvPlug = PlugDescriptor("satv")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "polyColorMod"
	apiTypeInt = 740
	apiTypeStr = "kPolyColorMod"
	typeIdInt = 1346587983
	MFnCls = om.MFnDependencyNode
	pass

