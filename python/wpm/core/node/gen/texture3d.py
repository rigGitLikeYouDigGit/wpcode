

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
	node : Texture3d = None
	pass
class AlphaIsLuminancePlug(Plug):
	node : Texture3d = None
	pass
class AlphaOffsetPlug(Plug):
	node : Texture3d = None
	pass
class BlendPlug(Plug):
	node : Texture3d = None
	pass
class ColorGainBPlug(Plug):
	parent : ColorGainPlug = PlugDescriptor("colorGain")
	node : Texture3d = None
	pass
class ColorGainGPlug(Plug):
	parent : ColorGainPlug = PlugDescriptor("colorGain")
	node : Texture3d = None
	pass
class ColorGainRPlug(Plug):
	parent : ColorGainPlug = PlugDescriptor("colorGain")
	node : Texture3d = None
	pass
class ColorGainPlug(Plug):
	colorGainB_ : ColorGainBPlug = PlugDescriptor("colorGainB")
	cgb_ : ColorGainBPlug = PlugDescriptor("colorGainB")
	colorGainG_ : ColorGainGPlug = PlugDescriptor("colorGainG")
	cgg_ : ColorGainGPlug = PlugDescriptor("colorGainG")
	colorGainR_ : ColorGainRPlug = PlugDescriptor("colorGainR")
	cgr_ : ColorGainRPlug = PlugDescriptor("colorGainR")
	node : Texture3d = None
	pass
class ColorOffsetBPlug(Plug):
	parent : ColorOffsetPlug = PlugDescriptor("colorOffset")
	node : Texture3d = None
	pass
class ColorOffsetGPlug(Plug):
	parent : ColorOffsetPlug = PlugDescriptor("colorOffset")
	node : Texture3d = None
	pass
class ColorOffsetRPlug(Plug):
	parent : ColorOffsetPlug = PlugDescriptor("colorOffset")
	node : Texture3d = None
	pass
class ColorOffsetPlug(Plug):
	colorOffsetB_ : ColorOffsetBPlug = PlugDescriptor("colorOffsetB")
	cob_ : ColorOffsetBPlug = PlugDescriptor("colorOffsetB")
	colorOffsetG_ : ColorOffsetGPlug = PlugDescriptor("colorOffsetG")
	cog_ : ColorOffsetGPlug = PlugDescriptor("colorOffsetG")
	colorOffsetR_ : ColorOffsetRPlug = PlugDescriptor("colorOffsetR")
	cor_ : ColorOffsetRPlug = PlugDescriptor("colorOffsetR")
	node : Texture3d = None
	pass
class DefaultColorBPlug(Plug):
	parent : DefaultColorPlug = PlugDescriptor("defaultColor")
	node : Texture3d = None
	pass
class DefaultColorGPlug(Plug):
	parent : DefaultColorPlug = PlugDescriptor("defaultColor")
	node : Texture3d = None
	pass
class DefaultColorRPlug(Plug):
	parent : DefaultColorPlug = PlugDescriptor("defaultColor")
	node : Texture3d = None
	pass
class DefaultColorPlug(Plug):
	defaultColorB_ : DefaultColorBPlug = PlugDescriptor("defaultColorB")
	dcb_ : DefaultColorBPlug = PlugDescriptor("defaultColorB")
	defaultColorG_ : DefaultColorGPlug = PlugDescriptor("defaultColorG")
	dcg_ : DefaultColorGPlug = PlugDescriptor("defaultColorG")
	defaultColorR_ : DefaultColorRPlug = PlugDescriptor("defaultColorR")
	dcr_ : DefaultColorRPlug = PlugDescriptor("defaultColorR")
	node : Texture3d = None
	pass
class FilterPlug(Plug):
	node : Texture3d = None
	pass
class FilterOffsetPlug(Plug):
	node : Texture3d = None
	pass
class FilterSizeXPlug(Plug):
	parent : FilterSizePlug = PlugDescriptor("filterSize")
	node : Texture3d = None
	pass
class FilterSizeYPlug(Plug):
	parent : FilterSizePlug = PlugDescriptor("filterSize")
	node : Texture3d = None
	pass
class FilterSizeZPlug(Plug):
	parent : FilterSizePlug = PlugDescriptor("filterSize")
	node : Texture3d = None
	pass
class FilterSizePlug(Plug):
	filterSizeX_ : FilterSizeXPlug = PlugDescriptor("filterSizeX")
	fsx_ : FilterSizeXPlug = PlugDescriptor("filterSizeX")
	filterSizeY_ : FilterSizeYPlug = PlugDescriptor("filterSizeY")
	fsy_ : FilterSizeYPlug = PlugDescriptor("filterSizeY")
	filterSizeZ_ : FilterSizeZPlug = PlugDescriptor("filterSizeZ")
	fsz_ : FilterSizeZPlug = PlugDescriptor("filterSizeZ")
	node : Texture3d = None
	pass
class InvertPlug(Plug):
	node : Texture3d = None
	pass
class LocalPlug(Plug):
	node : Texture3d = None
	pass
class MatrixEyeToWorldPlug(Plug):
	node : Texture3d = None
	pass
class OutAlphaPlug(Plug):
	node : Texture3d = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : Texture3d = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : Texture3d = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : Texture3d = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : Texture3d = None
	pass
class PlacementMatrixPlug(Plug):
	node : Texture3d = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Texture3d = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Texture3d = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Texture3d = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : Texture3d = None
	pass
class PointObjXPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : Texture3d = None
	pass
class PointObjYPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : Texture3d = None
	pass
class PointObjZPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : Texture3d = None
	pass
class PointObjPlug(Plug):
	pointObjX_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pox_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pointObjY_ : PointObjYPlug = PlugDescriptor("pointObjY")
	poy_ : PointObjYPlug = PlugDescriptor("pointObjY")
	pointObjZ_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	poz_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	node : Texture3d = None
	pass
class WrapPlug(Plug):
	node : Texture3d = None
	pass
# endregion


# define node class
class Texture3d(ShadingDependNode):
	alphaGain_ : AlphaGainPlug = PlugDescriptor("alphaGain")
	alphaIsLuminance_ : AlphaIsLuminancePlug = PlugDescriptor("alphaIsLuminance")
	alphaOffset_ : AlphaOffsetPlug = PlugDescriptor("alphaOffset")
	blend_ : BlendPlug = PlugDescriptor("blend")
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
	filterSizeX_ : FilterSizeXPlug = PlugDescriptor("filterSizeX")
	filterSizeY_ : FilterSizeYPlug = PlugDescriptor("filterSizeY")
	filterSizeZ_ : FilterSizeZPlug = PlugDescriptor("filterSizeZ")
	filterSize_ : FilterSizePlug = PlugDescriptor("filterSize")
	invert_ : InvertPlug = PlugDescriptor("invert")
	local_ : LocalPlug = PlugDescriptor("local")
	matrixEyeToWorld_ : MatrixEyeToWorldPlug = PlugDescriptor("matrixEyeToWorld")
	outAlpha_ : OutAlphaPlug = PlugDescriptor("outAlpha")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	placementMatrix_ : PlacementMatrixPlug = PlugDescriptor("placementMatrix")
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pointCamera_ : PointCameraPlug = PlugDescriptor("pointCamera")
	pointObjX_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pointObjY_ : PointObjYPlug = PlugDescriptor("pointObjY")
	pointObjZ_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	pointObj_ : PointObjPlug = PlugDescriptor("pointObj")
	wrap_ : WrapPlug = PlugDescriptor("wrap")

	# node attributes

	typeName = "texture3d"
	typeIdInt = 1381259315
	nodeLeafClassAttrs = ["alphaGain", "alphaIsLuminance", "alphaOffset", "blend", "colorGainB", "colorGainG", "colorGainR", "colorGain", "colorOffsetB", "colorOffsetG", "colorOffsetR", "colorOffset", "defaultColorB", "defaultColorG", "defaultColorR", "defaultColor", "filter", "filterOffset", "filterSizeX", "filterSizeY", "filterSizeZ", "filterSize", "invert", "local", "matrixEyeToWorld", "outAlpha", "outColorB", "outColorG", "outColorR", "outColor", "placementMatrix", "pointCameraX", "pointCameraY", "pointCameraZ", "pointCamera", "pointObjX", "pointObjY", "pointObjZ", "pointObj", "wrap"]
	nodeLeafPlugs = ["alphaGain", "alphaIsLuminance", "alphaOffset", "blend", "colorGain", "colorOffset", "defaultColor", "filter", "filterOffset", "filterSize", "invert", "local", "matrixEyeToWorld", "outAlpha", "outColor", "placementMatrix", "pointCamera", "pointObj", "wrap"]
	pass

