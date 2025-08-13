

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Reflect = retriever.getNodeCls("Reflect")
assert Reflect
if T.TYPE_CHECKING:
	from .. import Reflect

# add node doc



# region plug type defs
class ColorScale_ColorBPlug(Plug):
	parent : ColorScale_ColorPlug = PlugDescriptor("colorScale_Color")
	node : HairTubeShader = None
	pass
class ColorScale_ColorGPlug(Plug):
	parent : ColorScale_ColorPlug = PlugDescriptor("colorScale_Color")
	node : HairTubeShader = None
	pass
class ColorScale_ColorRPlug(Plug):
	parent : ColorScale_ColorPlug = PlugDescriptor("colorScale_Color")
	node : HairTubeShader = None
	pass
class ColorScale_ColorPlug(Plug):
	parent : ColorScalePlug = PlugDescriptor("colorScale")
	colorScale_ColorB_ : ColorScale_ColorBPlug = PlugDescriptor("colorScale_ColorB")
	clscb_ : ColorScale_ColorBPlug = PlugDescriptor("colorScale_ColorB")
	colorScale_ColorG_ : ColorScale_ColorGPlug = PlugDescriptor("colorScale_ColorG")
	clscg_ : ColorScale_ColorGPlug = PlugDescriptor("colorScale_ColorG")
	colorScale_ColorR_ : ColorScale_ColorRPlug = PlugDescriptor("colorScale_ColorR")
	clscr_ : ColorScale_ColorRPlug = PlugDescriptor("colorScale_ColorR")
	node : HairTubeShader = None
	pass
class ColorScale_InterpPlug(Plug):
	parent : ColorScalePlug = PlugDescriptor("colorScale")
	node : HairTubeShader = None
	pass
class ColorScale_PositionPlug(Plug):
	parent : ColorScalePlug = PlugDescriptor("colorScale")
	node : HairTubeShader = None
	pass
class ColorScalePlug(Plug):
	colorScale_Color_ : ColorScale_ColorPlug = PlugDescriptor("colorScale_Color")
	clsc_ : ColorScale_ColorPlug = PlugDescriptor("colorScale_Color")
	colorScale_Interp_ : ColorScale_InterpPlug = PlugDescriptor("colorScale_Interp")
	clsi_ : ColorScale_InterpPlug = PlugDescriptor("colorScale_Interp")
	colorScale_Position_ : ColorScale_PositionPlug = PlugDescriptor("colorScale_Position")
	clsp_ : ColorScale_PositionPlug = PlugDescriptor("colorScale_Position")
	node : HairTubeShader = None
	pass
class ScatterPlug(Plug):
	node : HairTubeShader = None
	pass
class ScatterPowerPlug(Plug):
	node : HairTubeShader = None
	pass
class SpecularPowerPlug(Plug):
	node : HairTubeShader = None
	pass
class SpecularShiftPlug(Plug):
	node : HairTubeShader = None
	pass
class TangentUCameraXPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : HairTubeShader = None
	pass
class TangentUCameraYPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : HairTubeShader = None
	pass
class TangentUCameraZPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : HairTubeShader = None
	pass
class TangentUCameraPlug(Plug):
	tangentUCameraX_ : TangentUCameraXPlug = PlugDescriptor("tangentUCameraX")
	utnx_ : TangentUCameraXPlug = PlugDescriptor("tangentUCameraX")
	tangentUCameraY_ : TangentUCameraYPlug = PlugDescriptor("tangentUCameraY")
	utny_ : TangentUCameraYPlug = PlugDescriptor("tangentUCameraY")
	tangentUCameraZ_ : TangentUCameraZPlug = PlugDescriptor("tangentUCameraZ")
	utnz_ : TangentUCameraZPlug = PlugDescriptor("tangentUCameraZ")
	node : HairTubeShader = None
	pass
class TangentVCameraXPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : HairTubeShader = None
	pass
class TangentVCameraYPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : HairTubeShader = None
	pass
class TangentVCameraZPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : HairTubeShader = None
	pass
class TangentVCameraPlug(Plug):
	tangentVCameraX_ : TangentVCameraXPlug = PlugDescriptor("tangentVCameraX")
	vtnx_ : TangentVCameraXPlug = PlugDescriptor("tangentVCameraX")
	tangentVCameraY_ : TangentVCameraYPlug = PlugDescriptor("tangentVCameraY")
	vtny_ : TangentVCameraYPlug = PlugDescriptor("tangentVCameraY")
	tangentVCameraZ_ : TangentVCameraZPlug = PlugDescriptor("tangentVCameraZ")
	vtnz_ : TangentVCameraZPlug = PlugDescriptor("tangentVCameraZ")
	node : HairTubeShader = None
	pass
class TubeDirectionPlug(Plug):
	node : HairTubeShader = None
	pass
class UCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : HairTubeShader = None
	pass
class VCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : HairTubeShader = None
	pass
class UvCoordPlug(Plug):
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	uvu_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	uvv_ : VCoordPlug = PlugDescriptor("vCoord")
	node : HairTubeShader = None
	pass
# endregion


# define node class
class HairTubeShader(Reflect):
	colorScale_ColorB_ : ColorScale_ColorBPlug = PlugDescriptor("colorScale_ColorB")
	colorScale_ColorG_ : ColorScale_ColorGPlug = PlugDescriptor("colorScale_ColorG")
	colorScale_ColorR_ : ColorScale_ColorRPlug = PlugDescriptor("colorScale_ColorR")
	colorScale_Color_ : ColorScale_ColorPlug = PlugDescriptor("colorScale_Color")
	colorScale_Interp_ : ColorScale_InterpPlug = PlugDescriptor("colorScale_Interp")
	colorScale_Position_ : ColorScale_PositionPlug = PlugDescriptor("colorScale_Position")
	colorScale_ : ColorScalePlug = PlugDescriptor("colorScale")
	scatter_ : ScatterPlug = PlugDescriptor("scatter")
	scatterPower_ : ScatterPowerPlug = PlugDescriptor("scatterPower")
	specularPower_ : SpecularPowerPlug = PlugDescriptor("specularPower")
	specularShift_ : SpecularShiftPlug = PlugDescriptor("specularShift")
	tangentUCameraX_ : TangentUCameraXPlug = PlugDescriptor("tangentUCameraX")
	tangentUCameraY_ : TangentUCameraYPlug = PlugDescriptor("tangentUCameraY")
	tangentUCameraZ_ : TangentUCameraZPlug = PlugDescriptor("tangentUCameraZ")
	tangentUCamera_ : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	tangentVCameraX_ : TangentVCameraXPlug = PlugDescriptor("tangentVCameraX")
	tangentVCameraY_ : TangentVCameraYPlug = PlugDescriptor("tangentVCameraY")
	tangentVCameraZ_ : TangentVCameraZPlug = PlugDescriptor("tangentVCameraZ")
	tangentVCamera_ : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	tubeDirection_ : TubeDirectionPlug = PlugDescriptor("tubeDirection")
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	uvCoord_ : UvCoordPlug = PlugDescriptor("uvCoord")

	# node attributes

	typeName = "hairTubeShader"
	apiTypeInt = 947
	apiTypeStr = "kHairTubeShader"
	typeIdInt = 1380471874
	MFnCls = om.MFnDependencyNode
	pass

