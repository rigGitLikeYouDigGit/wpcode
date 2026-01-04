

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	WeightGeometryFilter = Catalogue.WeightGeometryFilter
else:
	from .. import retriever
	WeightGeometryFilter = retriever.getNodeCls("WeightGeometryFilter")
	assert WeightGeometryFilter

# add node doc



# region plug type defs
class CacheSetupPlug(Plug):
	node : TextureDeformer = None
	pass
class DirectionPlug(Plug):
	node : TextureDeformer = None
	pass
class HandleMatrixPlug(Plug):
	node : TextureDeformer = None
	pass
class HandleVisibilityPlug(Plug):
	node : TextureDeformer = None
	pass
class OffsetPlug(Plug):
	node : TextureDeformer = None
	pass
class PointSpacePlug(Plug):
	node : TextureDeformer = None
	pass
class StrengthPlug(Plug):
	node : TextureDeformer = None
	pass
class TextureBPlug(Plug):
	parent : TexturePlug = PlugDescriptor("texture")
	node : TextureDeformer = None
	pass
class TextureGPlug(Plug):
	parent : TexturePlug = PlugDescriptor("texture")
	node : TextureDeformer = None
	pass
class TextureRPlug(Plug):
	parent : TexturePlug = PlugDescriptor("texture")
	node : TextureDeformer = None
	pass
class TexturePlug(Plug):
	textureB_ : TextureBPlug = PlugDescriptor("textureB")
	tb_ : TextureBPlug = PlugDescriptor("textureB")
	textureG_ : TextureGPlug = PlugDescriptor("textureG")
	tg_ : TextureGPlug = PlugDescriptor("textureG")
	textureR_ : TextureRPlug = PlugDescriptor("textureR")
	tr_ : TextureRPlug = PlugDescriptor("textureR")
	node : TextureDeformer = None
	pass
class VectorOffsetXPlug(Plug):
	parent : VectorOffsetPlug = PlugDescriptor("vectorOffset")
	node : TextureDeformer = None
	pass
class VectorOffsetYPlug(Plug):
	parent : VectorOffsetPlug = PlugDescriptor("vectorOffset")
	node : TextureDeformer = None
	pass
class VectorOffsetZPlug(Plug):
	parent : VectorOffsetPlug = PlugDescriptor("vectorOffset")
	node : TextureDeformer = None
	pass
class VectorOffsetPlug(Plug):
	vectorOffsetX_ : VectorOffsetXPlug = PlugDescriptor("vectorOffsetX")
	vox_ : VectorOffsetXPlug = PlugDescriptor("vectorOffsetX")
	vectorOffsetY_ : VectorOffsetYPlug = PlugDescriptor("vectorOffsetY")
	voy_ : VectorOffsetYPlug = PlugDescriptor("vectorOffsetY")
	vectorOffsetZ_ : VectorOffsetZPlug = PlugDescriptor("vectorOffsetZ")
	voz_ : VectorOffsetZPlug = PlugDescriptor("vectorOffsetZ")
	node : TextureDeformer = None
	pass
class VectorSpacePlug(Plug):
	node : TextureDeformer = None
	pass
class VectorStrengthXPlug(Plug):
	parent : VectorStrengthPlug = PlugDescriptor("vectorStrength")
	node : TextureDeformer = None
	pass
class VectorStrengthYPlug(Plug):
	parent : VectorStrengthPlug = PlugDescriptor("vectorStrength")
	node : TextureDeformer = None
	pass
class VectorStrengthZPlug(Plug):
	parent : VectorStrengthPlug = PlugDescriptor("vectorStrength")
	node : TextureDeformer = None
	pass
class VectorStrengthPlug(Plug):
	vectorStrengthX_ : VectorStrengthXPlug = PlugDescriptor("vectorStrengthX")
	vsx_ : VectorStrengthXPlug = PlugDescriptor("vectorStrengthX")
	vectorStrengthY_ : VectorStrengthYPlug = PlugDescriptor("vectorStrengthY")
	vsy_ : VectorStrengthYPlug = PlugDescriptor("vectorStrengthY")
	vectorStrengthZ_ : VectorStrengthZPlug = PlugDescriptor("vectorStrengthZ")
	vsz_ : VectorStrengthZPlug = PlugDescriptor("vectorStrengthZ")
	node : TextureDeformer = None
	pass
# endregion


# define node class
class TextureDeformer(WeightGeometryFilter):
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	handleMatrix_ : HandleMatrixPlug = PlugDescriptor("handleMatrix")
	handleVisibility_ : HandleVisibilityPlug = PlugDescriptor("handleVisibility")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	pointSpace_ : PointSpacePlug = PlugDescriptor("pointSpace")
	strength_ : StrengthPlug = PlugDescriptor("strength")
	textureB_ : TextureBPlug = PlugDescriptor("textureB")
	textureG_ : TextureGPlug = PlugDescriptor("textureG")
	textureR_ : TextureRPlug = PlugDescriptor("textureR")
	texture_ : TexturePlug = PlugDescriptor("texture")
	vectorOffsetX_ : VectorOffsetXPlug = PlugDescriptor("vectorOffsetX")
	vectorOffsetY_ : VectorOffsetYPlug = PlugDescriptor("vectorOffsetY")
	vectorOffsetZ_ : VectorOffsetZPlug = PlugDescriptor("vectorOffsetZ")
	vectorOffset_ : VectorOffsetPlug = PlugDescriptor("vectorOffset")
	vectorSpace_ : VectorSpacePlug = PlugDescriptor("vectorSpace")
	vectorStrengthX_ : VectorStrengthXPlug = PlugDescriptor("vectorStrengthX")
	vectorStrengthY_ : VectorStrengthYPlug = PlugDescriptor("vectorStrengthY")
	vectorStrengthZ_ : VectorStrengthZPlug = PlugDescriptor("vectorStrengthZ")
	vectorStrength_ : VectorStrengthPlug = PlugDescriptor("vectorStrength")

	# node attributes

	typeName = "textureDeformer"
	apiTypeInt = 343
	apiTypeStr = "kTextureDeformer"
	typeIdInt = 1415070790
	MFnCls = om.MFnGeometryFilter
	nodeLeafClassAttrs = ["cacheSetup", "direction", "handleMatrix", "handleVisibility", "offset", "pointSpace", "strength", "textureB", "textureG", "textureR", "texture", "vectorOffsetX", "vectorOffsetY", "vectorOffsetZ", "vectorOffset", "vectorSpace", "vectorStrengthX", "vectorStrengthY", "vectorStrengthZ", "vectorStrength"]
	nodeLeafPlugs = ["cacheSetup", "direction", "handleMatrix", "handleVisibility", "offset", "pointSpace", "strength", "texture", "vectorOffset", "vectorSpace", "vectorStrength"]
	pass

