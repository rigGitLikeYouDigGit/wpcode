

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ShadingDependNode = retriever.getNodeCls("ShadingDependNode")
assert ShadingDependNode
if T.TYPE_CHECKING:
	from .. import ShadingDependNode

# add node doc



# region plug type defs
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : ShadingMap = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : ShadingMap = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : ShadingMap = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : ShadingMap = None
	pass
class GlowColorBPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : ShadingMap = None
	pass
class GlowColorGPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : ShadingMap = None
	pass
class GlowColorRPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : ShadingMap = None
	pass
class GlowColorPlug(Plug):
	glowColorB_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	gb_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	glowColorG_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	gg_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	glowColorR_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	gr_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	node : ShadingMap = None
	pass
class MapFunctionUPlug(Plug):
	node : ShadingMap = None
	pass
class MapFunctionVPlug(Plug):
	node : ShadingMap = None
	pass
class MatteOpacityPlug(Plug):
	node : ShadingMap = None
	pass
class MatteOpacityModePlug(Plug):
	node : ShadingMap = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : ShadingMap = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : ShadingMap = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : ShadingMap = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : ShadingMap = None
	pass
class OutGlowColorBPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : ShadingMap = None
	pass
class OutGlowColorGPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : ShadingMap = None
	pass
class OutGlowColorRPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : ShadingMap = None
	pass
class OutGlowColorPlug(Plug):
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	ogb_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	ogg_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	ogr_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	node : ShadingMap = None
	pass
class OutMatteOpacityBPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : ShadingMap = None
	pass
class OutMatteOpacityGPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : ShadingMap = None
	pass
class OutMatteOpacityRPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : ShadingMap = None
	pass
class OutMatteOpacityPlug(Plug):
	outMatteOpacityB_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	omob_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	outMatteOpacityG_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	omog_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	outMatteOpacityR_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	omor_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	node : ShadingMap = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : ShadingMap = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : ShadingMap = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : ShadingMap = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : ShadingMap = None
	pass
class RenderPassModePlug(Plug):
	node : ShadingMap = None
	pass
class ShadingMapColorBPlug(Plug):
	parent : ShadingMapColorPlug = PlugDescriptor("shadingMapColor")
	node : ShadingMap = None
	pass
class ShadingMapColorGPlug(Plug):
	parent : ShadingMapColorPlug = PlugDescriptor("shadingMapColor")
	node : ShadingMap = None
	pass
class ShadingMapColorRPlug(Plug):
	parent : ShadingMapColorPlug = PlugDescriptor("shadingMapColor")
	node : ShadingMap = None
	pass
class ShadingMapColorPlug(Plug):
	shadingMapColorB_ : ShadingMapColorBPlug = PlugDescriptor("shadingMapColorB")
	scb_ : ShadingMapColorBPlug = PlugDescriptor("shadingMapColorB")
	shadingMapColorG_ : ShadingMapColorGPlug = PlugDescriptor("shadingMapColorG")
	scg_ : ShadingMapColorGPlug = PlugDescriptor("shadingMapColorG")
	shadingMapColorR_ : ShadingMapColorRPlug = PlugDescriptor("shadingMapColorR")
	scr_ : ShadingMapColorRPlug = PlugDescriptor("shadingMapColorR")
	node : ShadingMap = None
	pass
class TransparencyBPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : ShadingMap = None
	pass
class TransparencyGPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : ShadingMap = None
	pass
class TransparencyRPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : ShadingMap = None
	pass
class TransparencyPlug(Plug):
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	itb_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	itg_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	itr_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	node : ShadingMap = None
	pass
class UCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : ShadingMap = None
	pass
class VCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : ShadingMap = None
	pass
class UvCoordPlug(Plug):
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	uu_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	vv_ : VCoordPlug = PlugDescriptor("vCoord")
	node : ShadingMap = None
	pass
# endregion


# define node class
class ShadingMap(ShadingDependNode):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	glowColorB_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	glowColorG_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	glowColorR_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	glowColor_ : GlowColorPlug = PlugDescriptor("glowColor")
	mapFunctionU_ : MapFunctionUPlug = PlugDescriptor("mapFunctionU")
	mapFunctionV_ : MapFunctionVPlug = PlugDescriptor("mapFunctionV")
	matteOpacity_ : MatteOpacityPlug = PlugDescriptor("matteOpacity")
	matteOpacityMode_ : MatteOpacityModePlug = PlugDescriptor("matteOpacityMode")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	outGlowColor_ : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	outMatteOpacityB_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	outMatteOpacityG_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	outMatteOpacityR_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	outMatteOpacity_ : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	outTransparency_ : OutTransparencyPlug = PlugDescriptor("outTransparency")
	renderPassMode_ : RenderPassModePlug = PlugDescriptor("renderPassMode")
	shadingMapColorB_ : ShadingMapColorBPlug = PlugDescriptor("shadingMapColorB")
	shadingMapColorG_ : ShadingMapColorGPlug = PlugDescriptor("shadingMapColorG")
	shadingMapColorR_ : ShadingMapColorRPlug = PlugDescriptor("shadingMapColorR")
	shadingMapColor_ : ShadingMapColorPlug = PlugDescriptor("shadingMapColor")
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	transparency_ : TransparencyPlug = PlugDescriptor("transparency")
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	uvCoord_ : UvCoordPlug = PlugDescriptor("uvCoord")

	# node attributes

	typeName = "shadingMap"
	apiTypeInt = 477
	apiTypeStr = "kShadingMap"
	typeIdInt = 1396985168
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["colorB", "colorG", "colorR", "color", "glowColorB", "glowColorG", "glowColorR", "glowColor", "mapFunctionU", "mapFunctionV", "matteOpacity", "matteOpacityMode", "outColorB", "outColorG", "outColorR", "outColor", "outGlowColorB", "outGlowColorG", "outGlowColorR", "outGlowColor", "outMatteOpacityB", "outMatteOpacityG", "outMatteOpacityR", "outMatteOpacity", "outTransparencyB", "outTransparencyG", "outTransparencyR", "outTransparency", "renderPassMode", "shadingMapColorB", "shadingMapColorG", "shadingMapColorR", "shadingMapColor", "transparencyB", "transparencyG", "transparencyR", "transparency", "uCoord", "vCoord", "uvCoord"]
	nodeLeafPlugs = ["color", "glowColor", "mapFunctionU", "mapFunctionV", "matteOpacity", "matteOpacityMode", "outColor", "outGlowColor", "outMatteOpacity", "outTransparency", "renderPassMode", "shadingMapColor", "transparency", "uvCoord"]
	pass

