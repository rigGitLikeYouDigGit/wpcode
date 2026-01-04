

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ShadingDependNode = Catalogue.ShadingDependNode
else:
	from .. import retriever
	ShadingDependNode = retriever.getNodeCls("ShadingDependNode")
	assert ShadingDependNode

# add node doc



# region plug type defs
class AlphaGainPlug(Plug):
	node : Texture2d = None
	pass
class AlphaIsLuminancePlug(Plug):
	node : Texture2d = None
	pass
class AlphaOffsetPlug(Plug):
	node : Texture2d = None
	pass
class ColorGainBPlug(Plug):
	parent : ColorGainPlug = PlugDescriptor("colorGain")
	node : Texture2d = None
	pass
class ColorGainGPlug(Plug):
	parent : ColorGainPlug = PlugDescriptor("colorGain")
	node : Texture2d = None
	pass
class ColorGainRPlug(Plug):
	parent : ColorGainPlug = PlugDescriptor("colorGain")
	node : Texture2d = None
	pass
class ColorGainPlug(Plug):
	colorGainB_ : ColorGainBPlug = PlugDescriptor("colorGainB")
	cgb_ : ColorGainBPlug = PlugDescriptor("colorGainB")
	colorGainG_ : ColorGainGPlug = PlugDescriptor("colorGainG")
	cgg_ : ColorGainGPlug = PlugDescriptor("colorGainG")
	colorGainR_ : ColorGainRPlug = PlugDescriptor("colorGainR")
	cgr_ : ColorGainRPlug = PlugDescriptor("colorGainR")
	node : Texture2d = None
	pass
class ColorOffsetBPlug(Plug):
	parent : ColorOffsetPlug = PlugDescriptor("colorOffset")
	node : Texture2d = None
	pass
class ColorOffsetGPlug(Plug):
	parent : ColorOffsetPlug = PlugDescriptor("colorOffset")
	node : Texture2d = None
	pass
class ColorOffsetRPlug(Plug):
	parent : ColorOffsetPlug = PlugDescriptor("colorOffset")
	node : Texture2d = None
	pass
class ColorOffsetPlug(Plug):
	colorOffsetB_ : ColorOffsetBPlug = PlugDescriptor("colorOffsetB")
	cob_ : ColorOffsetBPlug = PlugDescriptor("colorOffsetB")
	colorOffsetG_ : ColorOffsetGPlug = PlugDescriptor("colorOffsetG")
	cog_ : ColorOffsetGPlug = PlugDescriptor("colorOffsetG")
	colorOffsetR_ : ColorOffsetRPlug = PlugDescriptor("colorOffsetR")
	cor_ : ColorOffsetRPlug = PlugDescriptor("colorOffsetR")
	node : Texture2d = None
	pass
class DefaultColorBPlug(Plug):
	parent : DefaultColorPlug = PlugDescriptor("defaultColor")
	node : Texture2d = None
	pass
class DefaultColorGPlug(Plug):
	parent : DefaultColorPlug = PlugDescriptor("defaultColor")
	node : Texture2d = None
	pass
class DefaultColorRPlug(Plug):
	parent : DefaultColorPlug = PlugDescriptor("defaultColor")
	node : Texture2d = None
	pass
class DefaultColorPlug(Plug):
	defaultColorB_ : DefaultColorBPlug = PlugDescriptor("defaultColorB")
	dcb_ : DefaultColorBPlug = PlugDescriptor("defaultColorB")
	defaultColorG_ : DefaultColorGPlug = PlugDescriptor("defaultColorG")
	dcg_ : DefaultColorGPlug = PlugDescriptor("defaultColorG")
	defaultColorR_ : DefaultColorRPlug = PlugDescriptor("defaultColorR")
	dcr_ : DefaultColorRPlug = PlugDescriptor("defaultColorR")
	node : Texture2d = None
	pass
class FilterPlug(Plug):
	node : Texture2d = None
	pass
class FilterOffsetPlug(Plug):
	node : Texture2d = None
	pass
class InvertPlug(Plug):
	node : Texture2d = None
	pass
class OutAlphaPlug(Plug):
	node : Texture2d = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : Texture2d = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : Texture2d = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : Texture2d = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : Texture2d = None
	pass
class UCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : Texture2d = None
	pass
class VCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : Texture2d = None
	pass
class UvCoordPlug(Plug):
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	u_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	v_ : VCoordPlug = PlugDescriptor("vCoord")
	node : Texture2d = None
	pass
class UvFilterSizeXPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : Texture2d = None
	pass
class UvFilterSizeYPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : Texture2d = None
	pass
class UvFilterSizePlug(Plug):
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	fsx_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	fsy_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	node : Texture2d = None
	pass
# endregion


# define node class
class Texture2d(ShadingDependNode):
	alphaGain_ : AlphaGainPlug = PlugDescriptor("alphaGain")
	alphaIsLuminance_ : AlphaIsLuminancePlug = PlugDescriptor("alphaIsLuminance")
	alphaOffset_ : AlphaOffsetPlug = PlugDescriptor("alphaOffset")
	colorGainB_ : ColorGainBPlug = PlugDescriptor("colorGainB")
	colorGainG_ : ColorGainGPlug = PlugDescriptor("colorGainG")
	colorGainR_ : ColorGainRPlug = PlugDescriptor("colorGainR")
	colorGain_ : ColorGainPlug = PlugDescriptor("colorGain")
	colorOffsetB_ : ColorOffsetBPlug = PlugDescriptor("colorOffsetB")
	colorOffsetG_ : ColorOffsetGPlug = PlugDescriptor("colorOffsetG")
	colorOffsetR_ : ColorOffsetRPlug = PlugDescriptor("colorOffsetR")
	colorOffset_ : ColorOffsetPlug = PlugDescriptor("colorOffset")
	defaultColorB_ : DefaultColorBPlug = PlugDescriptor("defaultColorB")
	defaultColorG_ : DefaultColorGPlug = PlugDescriptor("defaultColorG")
	defaultColorR_ : DefaultColorRPlug = PlugDescriptor("defaultColorR")
	defaultColor_ : DefaultColorPlug = PlugDescriptor("defaultColor")
	filter_ : FilterPlug = PlugDescriptor("filter")
	filterOffset_ : FilterOffsetPlug = PlugDescriptor("filterOffset")
	invert_ : InvertPlug = PlugDescriptor("invert")
	outAlpha_ : OutAlphaPlug = PlugDescriptor("outAlpha")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	uvCoord_ : UvCoordPlug = PlugDescriptor("uvCoord")
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	uvFilterSize_ : UvFilterSizePlug = PlugDescriptor("uvFilterSize")

	# node attributes

	typeName = "texture2d"
	typeIdInt = 1381259314
	nodeLeafClassAttrs = ["alphaGain", "alphaIsLuminance", "alphaOffset", "colorGainB", "colorGainG", "colorGainR", "colorGain", "colorOffsetB", "colorOffsetG", "colorOffsetR", "colorOffset", "defaultColorB", "defaultColorG", "defaultColorR", "defaultColor", "filter", "filterOffset", "invert", "outAlpha", "outColorB", "outColorG", "outColorR", "outColor", "uCoord", "vCoord", "uvCoord", "uvFilterSizeX", "uvFilterSizeY", "uvFilterSize"]
	nodeLeafPlugs = ["alphaGain", "alphaIsLuminance", "alphaOffset", "colorGain", "colorOffset", "defaultColor", "filter", "filterOffset", "invert", "outAlpha", "outColor", "uvCoord", "uvFilterSize"]
	pass

