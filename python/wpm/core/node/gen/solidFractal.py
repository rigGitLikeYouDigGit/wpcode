

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Texture3d = Catalogue.Texture3d
else:
	from .. import retriever
	Texture3d = retriever.getNodeCls("Texture3d")
	assert Texture3d

# add node doc



# region plug type defs
class AmplitudePlug(Plug):
	node : SolidFractal = None
	pass
class AnimatedPlug(Plug):
	node : SolidFractal = None
	pass
class BiasPlug(Plug):
	node : SolidFractal = None
	pass
class DepthMaxPlug(Plug):
	parent : DepthPlug = PlugDescriptor("depth")
	node : SolidFractal = None
	pass
class DepthMinPlug(Plug):
	parent : DepthPlug = PlugDescriptor("depth")
	node : SolidFractal = None
	pass
class DepthPlug(Plug):
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	dmx_ : DepthMaxPlug = PlugDescriptor("depthMax")
	depthMin_ : DepthMinPlug = PlugDescriptor("depthMin")
	dmn_ : DepthMinPlug = PlugDescriptor("depthMin")
	node : SolidFractal = None
	pass
class FrequencyRatioPlug(Plug):
	node : SolidFractal = None
	pass
class InflectionPlug(Plug):
	node : SolidFractal = None
	pass
class RatioPlug(Plug):
	node : SolidFractal = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : SolidFractal = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : SolidFractal = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : SolidFractal = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : SolidFractal = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : SolidFractal = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : SolidFractal = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : SolidFractal = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : SolidFractal = None
	pass
class RipplesXPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : SolidFractal = None
	pass
class RipplesYPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : SolidFractal = None
	pass
class RipplesZPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : SolidFractal = None
	pass
class RipplesPlug(Plug):
	ripplesX_ : RipplesXPlug = PlugDescriptor("ripplesX")
	rx_ : RipplesXPlug = PlugDescriptor("ripplesX")
	ripplesY_ : RipplesYPlug = PlugDescriptor("ripplesY")
	ry_ : RipplesYPlug = PlugDescriptor("ripplesY")
	ripplesZ_ : RipplesZPlug = PlugDescriptor("ripplesZ")
	rz_ : RipplesZPlug = PlugDescriptor("ripplesZ")
	node : SolidFractal = None
	pass
class ThresholdPlug(Plug):
	node : SolidFractal = None
	pass
class TimePlug(Plug):
	node : SolidFractal = None
	pass
class TimeRatioPlug(Plug):
	node : SolidFractal = None
	pass
class XPixelAnglePlug(Plug):
	node : SolidFractal = None
	pass
# endregion


# define node class
class SolidFractal(Texture3d):
	amplitude_ : AmplitudePlug = PlugDescriptor("amplitude")
	animated_ : AnimatedPlug = PlugDescriptor("animated")
	bias_ : BiasPlug = PlugDescriptor("bias")
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	depthMin_ : DepthMinPlug = PlugDescriptor("depthMin")
	depth_ : DepthPlug = PlugDescriptor("depth")
	frequencyRatio_ : FrequencyRatioPlug = PlugDescriptor("frequencyRatio")
	inflection_ : InflectionPlug = PlugDescriptor("inflection")
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
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")
	time_ : TimePlug = PlugDescriptor("time")
	timeRatio_ : TimeRatioPlug = PlugDescriptor("timeRatio")
	xPixelAngle_ : XPixelAnglePlug = PlugDescriptor("xPixelAngle")

	# node attributes

	typeName = "solidFractal"
	apiTypeInt = 516
	apiTypeStr = "kSolidFractal"
	typeIdInt = 1381254707
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["amplitude", "animated", "bias", "depthMax", "depthMin", "depth", "frequencyRatio", "inflection", "ratio", "refPointCameraX", "refPointCameraY", "refPointCameraZ", "refPointCamera", "refPointObjX", "refPointObjY", "refPointObjZ", "refPointObj", "ripplesX", "ripplesY", "ripplesZ", "ripples", "threshold", "time", "timeRatio", "xPixelAngle"]
	nodeLeafPlugs = ["amplitude", "animated", "bias", "depth", "frequencyRatio", "inflection", "ratio", "refPointCamera", "refPointObj", "ripples", "threshold", "time", "timeRatio", "xPixelAngle"]
	pass

