

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
class FlipPlug(Plug):
	node : EnvSphere = None
	pass
class ImageBPlug(Plug):
	parent : ImagePlug = PlugDescriptor("image")
	node : EnvSphere = None
	pass
class ImageGPlug(Plug):
	parent : ImagePlug = PlugDescriptor("image")
	node : EnvSphere = None
	pass
class ImageRPlug(Plug):
	parent : ImagePlug = PlugDescriptor("image")
	node : EnvSphere = None
	pass
class ImagePlug(Plug):
	imageB_ : ImageBPlug = PlugDescriptor("imageB")
	sob_ : ImageBPlug = PlugDescriptor("imageB")
	imageG_ : ImageGPlug = PlugDescriptor("imageG")
	sog_ : ImageGPlug = PlugDescriptor("imageG")
	imageR_ : ImageRPlug = PlugDescriptor("imageR")
	sor_ : ImageRPlug = PlugDescriptor("imageR")
	node : EnvSphere = None
	pass
class InfoBitsPlug(Plug):
	node : EnvSphere = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvSphere = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvSphere = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvSphere = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : EnvSphere = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : EnvSphere = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : EnvSphere = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : EnvSphere = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : EnvSphere = None
	pass
class ShearUPlug(Plug):
	parent : ShearUVPlug = PlugDescriptor("shearUV")
	node : EnvSphere = None
	pass
class ShearVPlug(Plug):
	parent : ShearUVPlug = PlugDescriptor("shearUV")
	node : EnvSphere = None
	pass
class ShearUVPlug(Plug):
	shearU_ : ShearUPlug = PlugDescriptor("shearU")
	su_ : ShearUPlug = PlugDescriptor("shearU")
	shearV_ : ShearVPlug = PlugDescriptor("shearV")
	sv_ : ShearVPlug = PlugDescriptor("shearV")
	node : EnvSphere = None
	pass
# endregion


# define node class
class EnvSphere(TextureEnv):
	flip_ : FlipPlug = PlugDescriptor("flip")
	imageB_ : ImageBPlug = PlugDescriptor("imageB")
	imageG_ : ImageGPlug = PlugDescriptor("imageG")
	imageR_ : ImageRPlug = PlugDescriptor("imageR")
	image_ : ImagePlug = PlugDescriptor("image")
	infoBits_ : InfoBitsPlug = PlugDescriptor("infoBits")
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pointCamera_ : PointCameraPlug = PlugDescriptor("pointCamera")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	shearU_ : ShearUPlug = PlugDescriptor("shearU")
	shearV_ : ShearVPlug = PlugDescriptor("shearV")
	shearUV_ : ShearUVPlug = PlugDescriptor("shearUV")

	# node attributes

	typeName = "envSphere"
	apiTypeInt = 495
	apiTypeStr = "kEnvSphere"
	typeIdInt = 1380275024
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["flip", "imageB", "imageG", "imageR", "image", "infoBits", "pointCameraX", "pointCameraY", "pointCameraZ", "pointCamera", "refPointCameraX", "refPointCameraY", "refPointCameraZ", "refPointCamera", "shearU", "shearV", "shearUV"]
	nodeLeafPlugs = ["flip", "image", "infoBits", "pointCamera", "refPointCamera", "shearUV"]
	pass

