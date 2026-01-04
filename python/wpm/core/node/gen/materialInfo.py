

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : MaterialInfo = None
	pass
class MaterialPlug(Plug):
	node : MaterialInfo = None
	pass
class ShadingGroupPlug(Plug):
	node : MaterialInfo = None
	pass
class TexturePlug(Plug):
	node : MaterialInfo = None
	pass
class TextureChannelPlug(Plug):
	node : MaterialInfo = None
	pass
class TextureFilterPlug(Plug):
	node : MaterialInfo = None
	pass
class TextureNamePlug(Plug):
	node : MaterialInfo = None
	pass
class TexturePlugPlug(Plug):
	node : MaterialInfo = None
	pass
# endregion


# define node class
class MaterialInfo(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	material_ : MaterialPlug = PlugDescriptor("material")
	shadingGroup_ : ShadingGroupPlug = PlugDescriptor("shadingGroup")
	texture_ : TexturePlug = PlugDescriptor("texture")
	textureChannel_ : TextureChannelPlug = PlugDescriptor("textureChannel")
	textureFilter_ : TextureFilterPlug = PlugDescriptor("textureFilter")
	textureName_ : TextureNamePlug = PlugDescriptor("textureName")
	texturePlug_ : TexturePlugPlug = PlugDescriptor("texturePlug")

	# node attributes

	typeName = "materialInfo"
	apiTypeInt = 392
	apiTypeStr = "kMaterialInfo"
	typeIdInt = 1145918537
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "material", "shadingGroup", "texture", "textureChannel", "textureFilter", "textureName", "texturePlug"]
	nodeLeafPlugs = ["binMembership", "material", "shadingGroup", "texture", "textureChannel", "textureFilter", "textureName", "texturePlug"]
	pass

