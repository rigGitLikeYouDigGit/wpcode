

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
class AmplitudePlug(Plug):
	node : Marble = None
	pass
class ContrastPlug(Plug):
	node : Marble = None
	pass
class DepthMaxPlug(Plug):
	parent : DepthPlug = PlugDescriptor("depth")
	node : Marble = None
	pass
class DepthMinPlug(Plug):
	parent : DepthPlug = PlugDescriptor("depth")
	node : Marble = None
	pass
class DepthPlug(Plug):
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	dmx_ : DepthMaxPlug = PlugDescriptor("depthMax")
	depthMin_ : DepthMinPlug = PlugDescriptor("depthMin")
	dmn_ : DepthMinPlug = PlugDescriptor("depthMin")
	node : Marble = None
	pass
class DiffusionPlug(Plug):
	node : Marble = None
	pass
class FillerColorBPlug(Plug):
	parent : FillerColorPlug = PlugDescriptor("fillerColor")
	node : Marble = None
	pass
class FillerColorGPlug(Plug):
	parent : FillerColorPlug = PlugDescriptor("fillerColor")
	node : Marble = None
	pass
class FillerColorRPlug(Plug):
	parent : FillerColorPlug = PlugDescriptor("fillerColor")
	node : Marble = None
	pass
class FillerColorPlug(Plug):
	fillerColorB_ : FillerColorBPlug = PlugDescriptor("fillerColorB")
	fcb_ : FillerColorBPlug = PlugDescriptor("fillerColorB")
	fillerColorG_ : FillerColorGPlug = PlugDescriptor("fillerColorG")
	fcg_ : FillerColorGPlug = PlugDescriptor("fillerColorG")
	fillerColorR_ : FillerColorRPlug = PlugDescriptor("fillerColorR")
	fcr_ : FillerColorRPlug = PlugDescriptor("fillerColorR")
	node : Marble = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Marble = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Marble = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Marble = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : Marble = None
	pass
class RatioPlug(Plug):
	node : Marble = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Marble = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Marble = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Marble = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Marble = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Marble = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Marble = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Marble = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Marble = None
	pass
class RipplesXPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : Marble = None
	pass
class RipplesYPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : Marble = None
	pass
class RipplesZPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : Marble = None
	pass
class RipplesPlug(Plug):
	ripplesX_ : RipplesXPlug = PlugDescriptor("ripplesX")
	rx_ : RipplesXPlug = PlugDescriptor("ripplesX")
	ripplesY_ : RipplesYPlug = PlugDescriptor("ripplesY")
	ry_ : RipplesYPlug = PlugDescriptor("ripplesY")
	ripplesZ_ : RipplesZPlug = PlugDescriptor("ripplesZ")
	rz_ : RipplesZPlug = PlugDescriptor("ripplesZ")
	node : Marble = None
	pass
class VeinColorBPlug(Plug):
	parent : VeinColorPlug = PlugDescriptor("veinColor")
	node : Marble = None
	pass
class VeinColorGPlug(Plug):
	parent : VeinColorPlug = PlugDescriptor("veinColor")
	node : Marble = None
	pass
class VeinColorRPlug(Plug):
	parent : VeinColorPlug = PlugDescriptor("veinColor")
	node : Marble = None
	pass
class VeinColorPlug(Plug):
	veinColorB_ : VeinColorBPlug = PlugDescriptor("veinColorB")
	vcb_ : VeinColorBPlug = PlugDescriptor("veinColorB")
	veinColorG_ : VeinColorGPlug = PlugDescriptor("veinColorG")
	vcg_ : VeinColorGPlug = PlugDescriptor("veinColorG")
	veinColorR_ : VeinColorRPlug = PlugDescriptor("veinColorR")
	vcr_ : VeinColorRPlug = PlugDescriptor("veinColorR")
	node : Marble = None
	pass
class VeinWidthPlug(Plug):
	node : Marble = None
	pass
class XPixelAnglePlug(Plug):
	node : Marble = None
	pass
# endregion


# define node class
class Marble(Texture3d):
	amplitude_ : AmplitudePlug = PlugDescriptor("amplitude")
	contrast_ : ContrastPlug = PlugDescriptor("contrast")
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	depthMin_ : DepthMinPlug = PlugDescriptor("depthMin")
	depth_ : DepthPlug = PlugDescriptor("depth")
	diffusion_ : DiffusionPlug = PlugDescriptor("diffusion")
	fillerColorB_ : FillerColorBPlug = PlugDescriptor("fillerColorB")
	fillerColorG_ : FillerColorGPlug = PlugDescriptor("fillerColorG")
	fillerColorR_ : FillerColorRPlug = PlugDescriptor("fillerColorR")
	fillerColor_ : FillerColorPlug = PlugDescriptor("fillerColor")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
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
	veinWidth_ : VeinWidthPlug = PlugDescriptor("veinWidth")
	xPixelAngle_ : XPixelAnglePlug = PlugDescriptor("xPixelAngle")

	# node attributes

	typeName = "marble"
	apiTypeInt = 513
	apiTypeStr = "kMarble"
	typeIdInt = 1381256530
	MFnCls = om.MFnDependencyNode
	pass

