

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
	node : Cloud = None
	pass
class CenterThreshPlug(Plug):
	node : Cloud = None
	pass
class Color1BPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Cloud = None
	pass
class Color1GPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Cloud = None
	pass
class Color1RPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Cloud = None
	pass
class Color1Plug(Plug):
	color1B_ : Color1BPlug = PlugDescriptor("color1B")
	c1b_ : Color1BPlug = PlugDescriptor("color1B")
	color1G_ : Color1GPlug = PlugDescriptor("color1G")
	c1g_ : Color1GPlug = PlugDescriptor("color1G")
	color1R_ : Color1RPlug = PlugDescriptor("color1R")
	c1r_ : Color1RPlug = PlugDescriptor("color1R")
	node : Cloud = None
	pass
class Color2BPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Cloud = None
	pass
class Color2GPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Cloud = None
	pass
class Color2RPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Cloud = None
	pass
class Color2Plug(Plug):
	color2B_ : Color2BPlug = PlugDescriptor("color2B")
	c2b_ : Color2BPlug = PlugDescriptor("color2B")
	color2G_ : Color2GPlug = PlugDescriptor("color2G")
	c2g_ : Color2GPlug = PlugDescriptor("color2G")
	color2R_ : Color2RPlug = PlugDescriptor("color2R")
	c2r_ : Color2RPlug = PlugDescriptor("color2R")
	node : Cloud = None
	pass
class ContrastPlug(Plug):
	node : Cloud = None
	pass
class DepthMaxPlug(Plug):
	parent : DepthPlug = PlugDescriptor("depth")
	node : Cloud = None
	pass
class DepthMinPlug(Plug):
	parent : DepthPlug = PlugDescriptor("depth")
	node : Cloud = None
	pass
class DepthPlug(Plug):
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	dmx_ : DepthMaxPlug = PlugDescriptor("depthMax")
	depthMin_ : DepthMinPlug = PlugDescriptor("depthMin")
	dmn_ : DepthMinPlug = PlugDescriptor("depthMin")
	node : Cloud = None
	pass
class EdgeThreshPlug(Plug):
	node : Cloud = None
	pass
class EyeToTextureMatrixPlug(Plug):
	node : Cloud = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Cloud = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Cloud = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Cloud = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : Cloud = None
	pass
class RatioPlug(Plug):
	node : Cloud = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Cloud = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Cloud = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Cloud = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Cloud = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Cloud = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Cloud = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Cloud = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Cloud = None
	pass
class RipplesXPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : Cloud = None
	pass
class RipplesYPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : Cloud = None
	pass
class RipplesZPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : Cloud = None
	pass
class RipplesPlug(Plug):
	ripplesX_ : RipplesXPlug = PlugDescriptor("ripplesX")
	rx_ : RipplesXPlug = PlugDescriptor("ripplesX")
	ripplesY_ : RipplesYPlug = PlugDescriptor("ripplesY")
	ry_ : RipplesYPlug = PlugDescriptor("ripplesY")
	ripplesZ_ : RipplesZPlug = PlugDescriptor("ripplesZ")
	rz_ : RipplesZPlug = PlugDescriptor("ripplesZ")
	node : Cloud = None
	pass
class SoftEdgesPlug(Plug):
	node : Cloud = None
	pass
class TranspRangePlug(Plug):
	node : Cloud = None
	pass
class XPixelAnglePlug(Plug):
	node : Cloud = None
	pass
# endregion


# define node class
class Cloud(Texture3d):
	amplitude_ : AmplitudePlug = PlugDescriptor("amplitude")
	centerThresh_ : CenterThreshPlug = PlugDescriptor("centerThresh")
	color1B_ : Color1BPlug = PlugDescriptor("color1B")
	color1G_ : Color1GPlug = PlugDescriptor("color1G")
	color1R_ : Color1RPlug = PlugDescriptor("color1R")
	color1_ : Color1Plug = PlugDescriptor("color1")
	color2B_ : Color2BPlug = PlugDescriptor("color2B")
	color2G_ : Color2GPlug = PlugDescriptor("color2G")
	color2R_ : Color2RPlug = PlugDescriptor("color2R")
	color2_ : Color2Plug = PlugDescriptor("color2")
	contrast_ : ContrastPlug = PlugDescriptor("contrast")
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	depthMin_ : DepthMinPlug = PlugDescriptor("depthMin")
	depth_ : DepthPlug = PlugDescriptor("depth")
	edgeThresh_ : EdgeThreshPlug = PlugDescriptor("edgeThresh")
	eyeToTextureMatrix_ : EyeToTextureMatrixPlug = PlugDescriptor("eyeToTextureMatrix")
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
	softEdges_ : SoftEdgesPlug = PlugDescriptor("softEdges")
	transpRange_ : TranspRangePlug = PlugDescriptor("transpRange")
	xPixelAngle_ : XPixelAnglePlug = PlugDescriptor("xPixelAngle")

	# node attributes

	typeName = "cloud"
	apiTypeInt = 509
	apiTypeStr = "kCloud"
	typeIdInt = 1381253956
	MFnCls = om.MFnDependencyNode
	pass

