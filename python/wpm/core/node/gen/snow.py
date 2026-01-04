

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
class DepthDecayPlug(Plug):
	node : Snow = None
	pass
class EyeToTextureMatrixPlug(Plug):
	node : Snow = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Snow = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Snow = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Snow = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : Snow = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Snow = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Snow = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Snow = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Snow = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Snow = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Snow = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Snow = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Snow = None
	pass
class SnowColorBPlug(Plug):
	parent : SnowColorPlug = PlugDescriptor("snowColor")
	node : Snow = None
	pass
class SnowColorGPlug(Plug):
	parent : SnowColorPlug = PlugDescriptor("snowColor")
	node : Snow = None
	pass
class SnowColorRPlug(Plug):
	parent : SnowColorPlug = PlugDescriptor("snowColor")
	node : Snow = None
	pass
class SnowColorPlug(Plug):
	snowColorB_ : SnowColorBPlug = PlugDescriptor("snowColorB")
	snb_ : SnowColorBPlug = PlugDescriptor("snowColorB")
	snowColorG_ : SnowColorGPlug = PlugDescriptor("snowColorG")
	sng_ : SnowColorGPlug = PlugDescriptor("snowColorG")
	snowColorR_ : SnowColorRPlug = PlugDescriptor("snowColorR")
	snr_ : SnowColorRPlug = PlugDescriptor("snowColorR")
	node : Snow = None
	pass
class SurfaceColorBPlug(Plug):
	parent : SurfaceColorPlug = PlugDescriptor("surfaceColor")
	node : Snow = None
	pass
class SurfaceColorGPlug(Plug):
	parent : SurfaceColorPlug = PlugDescriptor("surfaceColor")
	node : Snow = None
	pass
class SurfaceColorRPlug(Plug):
	parent : SurfaceColorPlug = PlugDescriptor("surfaceColor")
	node : Snow = None
	pass
class SurfaceColorPlug(Plug):
	surfaceColorB_ : SurfaceColorBPlug = PlugDescriptor("surfaceColorB")
	sub_ : SurfaceColorBPlug = PlugDescriptor("surfaceColorB")
	surfaceColorG_ : SurfaceColorGPlug = PlugDescriptor("surfaceColorG")
	sug_ : SurfaceColorGPlug = PlugDescriptor("surfaceColorG")
	surfaceColorR_ : SurfaceColorRPlug = PlugDescriptor("surfaceColorR")
	sur_ : SurfaceColorRPlug = PlugDescriptor("surfaceColorR")
	node : Snow = None
	pass
class ThicknessPlug(Plug):
	node : Snow = None
	pass
class ThresholdPlug(Plug):
	node : Snow = None
	pass
# endregion


# define node class
class Snow(Texture3d):
	depthDecay_ : DepthDecayPlug = PlugDescriptor("depthDecay")
	eyeToTextureMatrix_ : EyeToTextureMatrixPlug = PlugDescriptor("eyeToTextureMatrix")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	refPointObj_ : RefPointObjPlug = PlugDescriptor("refPointObj")
	snowColorB_ : SnowColorBPlug = PlugDescriptor("snowColorB")
	snowColorG_ : SnowColorGPlug = PlugDescriptor("snowColorG")
	snowColorR_ : SnowColorRPlug = PlugDescriptor("snowColorR")
	snowColor_ : SnowColorPlug = PlugDescriptor("snowColor")
	surfaceColorB_ : SurfaceColorBPlug = PlugDescriptor("surfaceColorB")
	surfaceColorG_ : SurfaceColorGPlug = PlugDescriptor("surfaceColorG")
	surfaceColorR_ : SurfaceColorRPlug = PlugDescriptor("surfaceColorR")
	surfaceColor_ : SurfaceColorPlug = PlugDescriptor("surfaceColor")
	thickness_ : ThicknessPlug = PlugDescriptor("thickness")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")

	# node attributes

	typeName = "snow"
	apiTypeInt = 515
	apiTypeStr = "kSnow"
	typeIdInt = 1381258062
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["depthDecay", "eyeToTextureMatrix", "normalCameraX", "normalCameraY", "normalCameraZ", "normalCamera", "refPointCameraX", "refPointCameraY", "refPointCameraZ", "refPointCamera", "refPointObjX", "refPointObjY", "refPointObjZ", "refPointObj", "snowColorB", "snowColorG", "snowColorR", "snowColor", "surfaceColorB", "surfaceColorG", "surfaceColorR", "surfaceColor", "thickness", "threshold"]
	nodeLeafPlugs = ["depthDecay", "eyeToTextureMatrix", "normalCamera", "refPointCamera", "refPointObj", "snowColor", "surfaceColor", "thickness", "threshold"]
	pass

