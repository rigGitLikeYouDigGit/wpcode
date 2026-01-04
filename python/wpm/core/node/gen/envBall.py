

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	TextureEnv = Catalogue.TextureEnv
else:
	from .. import retriever
	TextureEnv = retriever.getNodeCls("TextureEnv")
	assert TextureEnv

# add node doc



# region plug type defs
class BackPlug(Plug):
	node : EnvBall = None
	pass
class BottomPlug(Plug):
	node : EnvBall = None
	pass
class ElevationPlug(Plug):
	node : EnvBall = None
	pass
class EyeSpacePlug(Plug):
	node : EnvBall = None
	pass
class FrontPlug(Plug):
	node : EnvBall = None
	pass
class ImageBPlug(Plug):
	parent : ImagePlug = PlugDescriptor("image")
	node : EnvBall = None
	pass
class ImageGPlug(Plug):
	parent : ImagePlug = PlugDescriptor("image")
	node : EnvBall = None
	pass
class ImageRPlug(Plug):
	parent : ImagePlug = PlugDescriptor("image")
	node : EnvBall = None
	pass
class ImagePlug(Plug):
	imageB_ : ImageBPlug = PlugDescriptor("imageB")
	sob_ : ImageBPlug = PlugDescriptor("imageB")
	imageG_ : ImageGPlug = PlugDescriptor("imageG")
	sog_ : ImageGPlug = PlugDescriptor("imageG")
	imageR_ : ImageRPlug = PlugDescriptor("imageR")
	sor_ : ImageRPlug = PlugDescriptor("imageR")
	node : EnvBall = None
	pass
class InclinationPlug(Plug):
	node : EnvBall = None
	pass
class InfoBitsPlug(Plug):
	node : EnvBall = None
	pass
class LeftPlug(Plug):
	node : EnvBall = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvBall = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvBall = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvBall = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : EnvBall = None
	pass
class ReflectPlug(Plug):
	node : EnvBall = None
	pass
class RightPlug(Plug):
	node : EnvBall = None
	pass
class SkyRadiusPlug(Plug):
	node : EnvBall = None
	pass
class TopPlug(Plug):
	node : EnvBall = None
	pass
# endregion


# define node class
class EnvBall(TextureEnv):
	back_ : BackPlug = PlugDescriptor("back")
	bottom_ : BottomPlug = PlugDescriptor("bottom")
	elevation_ : ElevationPlug = PlugDescriptor("elevation")
	eyeSpace_ : EyeSpacePlug = PlugDescriptor("eyeSpace")
	front_ : FrontPlug = PlugDescriptor("front")
	imageB_ : ImageBPlug = PlugDescriptor("imageB")
	imageG_ : ImageGPlug = PlugDescriptor("imageG")
	imageR_ : ImageRPlug = PlugDescriptor("imageR")
	image_ : ImagePlug = PlugDescriptor("image")
	inclination_ : InclinationPlug = PlugDescriptor("inclination")
	infoBits_ : InfoBitsPlug = PlugDescriptor("infoBits")
	left_ : LeftPlug = PlugDescriptor("left")
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pointCamera_ : PointCameraPlug = PlugDescriptor("pointCamera")
	reflect_ : ReflectPlug = PlugDescriptor("reflect")
	right_ : RightPlug = PlugDescriptor("right")
	skyRadius_ : SkyRadiusPlug = PlugDescriptor("skyRadius")
	top_ : TopPlug = PlugDescriptor("top")

	# node attributes

	typeName = "envBall"
	apiTypeInt = 491
	apiTypeStr = "kEnvBall"
	typeIdInt = 1380270668
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["back", "bottom", "elevation", "eyeSpace", "front", "imageB", "imageG", "imageR", "image", "inclination", "infoBits", "left", "pointCameraX", "pointCameraY", "pointCameraZ", "pointCamera", "reflect", "right", "skyRadius", "top"]
	nodeLeafPlugs = ["back", "bottom", "elevation", "eyeSpace", "front", "image", "inclination", "infoBits", "left", "pointCamera", "reflect", "right", "skyRadius", "top"]
	pass

