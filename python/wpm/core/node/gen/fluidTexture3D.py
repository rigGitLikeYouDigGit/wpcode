

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	FluidShape = Catalogue.FluidShape
else:
	from .. import retriever
	FluidShape = retriever.getNodeCls("FluidShape")
	assert FluidShape

# add node doc



# region plug type defs
class AlphaGainPlug(Plug):
	node : FluidTexture3D = None
	pass
class AlphaOffsetPlug(Plug):
	node : FluidTexture3D = None
	pass
class DefaultColorBPlug(Plug):
	parent : DefaultColorPlug = PlugDescriptor("defaultColor")
	node : FluidTexture3D = None
	pass
class DefaultColorGPlug(Plug):
	parent : DefaultColorPlug = PlugDescriptor("defaultColor")
	node : FluidTexture3D = None
	pass
class DefaultColorRPlug(Plug):
	parent : DefaultColorPlug = PlugDescriptor("defaultColor")
	node : FluidTexture3D = None
	pass
class DefaultColorPlug(Plug):
	defaultColorB_ : DefaultColorBPlug = PlugDescriptor("defaultColorB")
	dcb_ : DefaultColorBPlug = PlugDescriptor("defaultColorB")
	defaultColorG_ : DefaultColorGPlug = PlugDescriptor("defaultColorG")
	dcg_ : DefaultColorGPlug = PlugDescriptor("defaultColorG")
	defaultColorR_ : DefaultColorRPlug = PlugDescriptor("defaultColorR")
	dcr_ : DefaultColorRPlug = PlugDescriptor("defaultColorR")
	node : FluidTexture3D = None
	pass
class OutAlphaPlug(Plug):
	node : FluidTexture3D = None
	pass
class OucxPlug(Plug):
	parent : OutCoordPlug = PlugDescriptor("outCoord")
	node : FluidTexture3D = None
	pass
class OucyPlug(Plug):
	parent : OutCoordPlug = PlugDescriptor("outCoord")
	node : FluidTexture3D = None
	pass
class OuczPlug(Plug):
	parent : OutCoordPlug = PlugDescriptor("outCoord")
	node : FluidTexture3D = None
	pass
class OutCoordPlug(Plug):
	oucx_ : OucxPlug = PlugDescriptor("oucx")
	ocx_ : OucxPlug = PlugDescriptor("oucx")
	oucy_ : OucyPlug = PlugDescriptor("oucy")
	ocy_ : OucyPlug = PlugDescriptor("oucy")
	oucz_ : OuczPlug = PlugDescriptor("oucz")
	ocz_ : OuczPlug = PlugDescriptor("oucz")
	node : FluidTexture3D = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : FluidTexture3D = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : FluidTexture3D = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : FluidTexture3D = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : FluidTexture3D = None
	pass
# endregion


# define node class
class FluidTexture3D(FluidShape):
	alphaGain_ : AlphaGainPlug = PlugDescriptor("alphaGain")
	alphaOffset_ : AlphaOffsetPlug = PlugDescriptor("alphaOffset")
	defaultColorB_ : DefaultColorBPlug = PlugDescriptor("defaultColorB")
	defaultColorG_ : DefaultColorGPlug = PlugDescriptor("defaultColorG")
	defaultColorR_ : DefaultColorRPlug = PlugDescriptor("defaultColorR")
	defaultColor_ : DefaultColorPlug = PlugDescriptor("defaultColor")
	outAlpha_ : OutAlphaPlug = PlugDescriptor("outAlpha")
	oucx_ : OucxPlug = PlugDescriptor("oucx")
	oucy_ : OucyPlug = PlugDescriptor("oucy")
	oucz_ : OuczPlug = PlugDescriptor("oucz")
	outCoord_ : OutCoordPlug = PlugDescriptor("outCoord")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")

	# node attributes

	typeName = "fluidTexture3D"
	apiTypeInt = 908
	apiTypeStr = "kFluidTexture3D"
	typeIdInt = 1179407448
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["alphaGain", "alphaOffset", "defaultColorB", "defaultColorG", "defaultColorR", "defaultColor", "outAlpha", "oucx", "oucy", "oucz", "outCoord", "refPointCameraX", "refPointCameraY", "refPointCameraZ", "refPointCamera"]
	nodeLeafPlugs = ["alphaGain", "alphaOffset", "defaultColor", "outAlpha", "outCoord", "refPointCamera"]
	pass

