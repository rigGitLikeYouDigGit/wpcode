

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Texture3d = retriever.getNodeCls("Texture3d")
assert Texture3d
if T.TYPE_CHECKING:
	from .. import Texture3d

# add node doc



# region plug type defs
class AgePlug(Plug):
	node : Wood = None
	pass
class AmplitudeXPlug(Plug):
	node : Wood = None
	pass
class AmplitudeYPlug(Plug):
	node : Wood = None
	pass
class CenterUPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : Wood = None
	pass
class CenterVPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : Wood = None
	pass
class CenterPlug(Plug):
	centerU_ : CenterUPlug = PlugDescriptor("centerU")
	cu_ : CenterUPlug = PlugDescriptor("centerU")
	centerV_ : CenterVPlug = PlugDescriptor("centerV")
	cv_ : CenterVPlug = PlugDescriptor("centerV")
	node : Wood = None
	pass
class DepthMaxPlug(Plug):
	parent : DepthPlug = PlugDescriptor("depth")
	node : Wood = None
	pass
class DepthMinPlug(Plug):
	parent : DepthPlug = PlugDescriptor("depth")
	node : Wood = None
	pass
class DepthPlug(Plug):
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	dmx_ : DepthMaxPlug = PlugDescriptor("depthMax")
	depthMin_ : DepthMinPlug = PlugDescriptor("depthMin")
	dmn_ : DepthMinPlug = PlugDescriptor("depthMin")
	node : Wood = None
	pass
class FillerColorBPlug(Plug):
	parent : FillerColorPlug = PlugDescriptor("fillerColor")
	node : Wood = None
	pass
class FillerColorGPlug(Plug):
	parent : FillerColorPlug = PlugDescriptor("fillerColor")
	node : Wood = None
	pass
class FillerColorRPlug(Plug):
	parent : FillerColorPlug = PlugDescriptor("fillerColor")
	node : Wood = None
	pass
class FillerColorPlug(Plug):
	fillerColorB_ : FillerColorBPlug = PlugDescriptor("fillerColorB")
	fcb_ : FillerColorBPlug = PlugDescriptor("fillerColorB")
	fillerColorG_ : FillerColorGPlug = PlugDescriptor("fillerColorG")
	fcg_ : FillerColorGPlug = PlugDescriptor("fillerColorG")
	fillerColorR_ : FillerColorRPlug = PlugDescriptor("fillerColorR")
	fcr_ : FillerColorRPlug = PlugDescriptor("fillerColorR")
	node : Wood = None
	pass
class GrainColorBPlug(Plug):
	parent : GrainColorPlug = PlugDescriptor("grainColor")
	node : Wood = None
	pass
class GrainColorGPlug(Plug):
	parent : GrainColorPlug = PlugDescriptor("grainColor")
	node : Wood = None
	pass
class GrainColorRPlug(Plug):
	parent : GrainColorPlug = PlugDescriptor("grainColor")
	node : Wood = None
	pass
class GrainColorPlug(Plug):
	grainColorB_ : GrainColorBPlug = PlugDescriptor("grainColorB")
	gcb_ : GrainColorBPlug = PlugDescriptor("grainColorB")
	grainColorG_ : GrainColorGPlug = PlugDescriptor("grainColorG")
	gcg_ : GrainColorGPlug = PlugDescriptor("grainColorG")
	grainColorR_ : GrainColorRPlug = PlugDescriptor("grainColorR")
	gcr_ : GrainColorRPlug = PlugDescriptor("grainColorR")
	node : Wood = None
	pass
class GrainContrastPlug(Plug):
	node : Wood = None
	pass
class GrainSpacingPlug(Plug):
	node : Wood = None
	pass
class LayerSizePlug(Plug):
	node : Wood = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Wood = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Wood = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Wood = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : Wood = None
	pass
class RandomnessPlug(Plug):
	node : Wood = None
	pass
class RatioPlug(Plug):
	node : Wood = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Wood = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Wood = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Wood = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Wood = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Wood = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Wood = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Wood = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Wood = None
	pass
class RipplesXPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : Wood = None
	pass
class RipplesYPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : Wood = None
	pass
class RipplesZPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : Wood = None
	pass
class RipplesPlug(Plug):
	ripplesX_ : RipplesXPlug = PlugDescriptor("ripplesX")
	rx_ : RipplesXPlug = PlugDescriptor("ripplesX")
	ripplesY_ : RipplesYPlug = PlugDescriptor("ripplesY")
	ry_ : RipplesYPlug = PlugDescriptor("ripplesY")
	ripplesZ_ : RipplesZPlug = PlugDescriptor("ripplesZ")
	rz_ : RipplesZPlug = PlugDescriptor("ripplesZ")
	node : Wood = None
	pass
class VeinColorBPlug(Plug):
	parent : VeinColorPlug = PlugDescriptor("veinColor")
	node : Wood = None
	pass
class VeinColorGPlug(Plug):
	parent : VeinColorPlug = PlugDescriptor("veinColor")
	node : Wood = None
	pass
class VeinColorRPlug(Plug):
	parent : VeinColorPlug = PlugDescriptor("veinColor")
	node : Wood = None
	pass
class VeinColorPlug(Plug):
	veinColorB_ : VeinColorBPlug = PlugDescriptor("veinColorB")
	vcb_ : VeinColorBPlug = PlugDescriptor("veinColorB")
	veinColorG_ : VeinColorGPlug = PlugDescriptor("veinColorG")
	vcg_ : VeinColorGPlug = PlugDescriptor("veinColorG")
	veinColorR_ : VeinColorRPlug = PlugDescriptor("veinColorR")
	vcr_ : VeinColorRPlug = PlugDescriptor("veinColorR")
	node : Wood = None
	pass
class VeinSpreadPlug(Plug):
	node : Wood = None
	pass
class XPixelAnglePlug(Plug):
	node : Wood = None
	pass
# endregion


# define node class
class Wood(Texture3d):
	age_ : AgePlug = PlugDescriptor("age")
	amplitudeX_ : AmplitudeXPlug = PlugDescriptor("amplitudeX")
	amplitudeY_ : AmplitudeYPlug = PlugDescriptor("amplitudeY")
	centerU_ : CenterUPlug = PlugDescriptor("centerU")
	centerV_ : CenterVPlug = PlugDescriptor("centerV")
	center_ : CenterPlug = PlugDescriptor("center")
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	depthMin_ : DepthMinPlug = PlugDescriptor("depthMin")
	depth_ : DepthPlug = PlugDescriptor("depth")
	fillerColorB_ : FillerColorBPlug = PlugDescriptor("fillerColorB")
	fillerColorG_ : FillerColorGPlug = PlugDescriptor("fillerColorG")
	fillerColorR_ : FillerColorRPlug = PlugDescriptor("fillerColorR")
	fillerColor_ : FillerColorPlug = PlugDescriptor("fillerColor")
	grainColorB_ : GrainColorBPlug = PlugDescriptor("grainColorB")
	grainColorG_ : GrainColorGPlug = PlugDescriptor("grainColorG")
	grainColorR_ : GrainColorRPlug = PlugDescriptor("grainColorR")
	grainColor_ : GrainColorPlug = PlugDescriptor("grainColor")
	grainContrast_ : GrainContrastPlug = PlugDescriptor("grainContrast")
	grainSpacing_ : GrainSpacingPlug = PlugDescriptor("grainSpacing")
	layerSize_ : LayerSizePlug = PlugDescriptor("layerSize")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	randomness_ : RandomnessPlug = PlugDescriptor("randomness")
	ratio_ : RatioPlug = PlugDescriptor("ratio")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	refPointObj_ : RefPointObjPlug = PlugDescriptor("refPointObj")
	ripplesX_ : RipplesXPlug = PlugDescriptor("ripplesX")
	ripplesY_ : RipplesYPlug = PlugDescriptor("ripplesY")
	ripplesZ_ : RipplesZPlug = PlugDescriptor("ripplesZ")
	ripples_ : RipplesPlug = PlugDescriptor("ripples")
	veinColorB_ : VeinColorBPlug = PlugDescriptor("veinColorB")
	veinColorG_ : VeinColorGPlug = PlugDescriptor("veinColorG")
	veinColorR_ : VeinColorRPlug = PlugDescriptor("veinColorR")
	veinColor_ : VeinColorPlug = PlugDescriptor("veinColor")
	veinSpread_ : VeinSpreadPlug = PlugDescriptor("veinSpread")
	xPixelAngle_ : XPixelAnglePlug = PlugDescriptor("xPixelAngle")

	# node attributes

	typeName = "wood"
	apiTypeInt = 519
	apiTypeStr = "kWood"
	typeIdInt = 1381259076
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["age", "amplitudeX", "amplitudeY", "centerU", "centerV", "center", "depthMax", "depthMin", "depth", "fillerColorB", "fillerColorG", "fillerColorR", "fillerColor", "grainColorB", "grainColorG", "grainColorR", "grainColor", "grainContrast", "grainSpacing", "layerSize", "normalCameraX", "normalCameraY", "normalCameraZ", "normalCamera", "randomness", "ratio", "refPointCameraX", "refPointCameraY", "refPointCameraZ", "refPointCamera", "refPointObjX", "refPointObjY", "refPointObjZ", "refPointObj", "ripplesX", "ripplesY", "ripplesZ", "ripples", "veinColorB", "veinColorG", "veinColorR", "veinColor", "veinSpread", "xPixelAngle"]
	nodeLeafPlugs = ["age", "amplitudeX", "amplitudeY", "center", "depth", "fillerColor", "grainColor", "grainContrast", "grainSpacing", "layerSize", "normalCamera", "randomness", "ratio", "refPointCamera", "refPointObj", "ripples", "veinColor", "veinSpread", "xPixelAngle"]
	pass

