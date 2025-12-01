

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : HwReflectionMap = None
	pass
class CubeBackTextureNamePlug(Plug):
	node : HwReflectionMap = None
	pass
class CubeBottomTextureNamePlug(Plug):
	node : HwReflectionMap = None
	pass
class CubeFrontTextureNamePlug(Plug):
	node : HwReflectionMap = None
	pass
class CubeLeftTextureNamePlug(Plug):
	node : HwReflectionMap = None
	pass
class CubeMapPlug(Plug):
	node : HwReflectionMap = None
	pass
class CubeRightTextureNamePlug(Plug):
	node : HwReflectionMap = None
	pass
class CubeTopTextureNamePlug(Plug):
	node : HwReflectionMap = None
	pass
class DecalModePlug(Plug):
	node : HwReflectionMap = None
	pass
class SphereMapTextureNamePlug(Plug):
	node : HwReflectionMap = None
	pass
class TextureHasChangedPlug(Plug):
	node : HwReflectionMap = None
	pass
# endregion


# define node class
class HwReflectionMap(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	cubeBackTextureName_ : CubeBackTextureNamePlug = PlugDescriptor("cubeBackTextureName")
	cubeBottomTextureName_ : CubeBottomTextureNamePlug = PlugDescriptor("cubeBottomTextureName")
	cubeFrontTextureName_ : CubeFrontTextureNamePlug = PlugDescriptor("cubeFrontTextureName")
	cubeLeftTextureName_ : CubeLeftTextureNamePlug = PlugDescriptor("cubeLeftTextureName")
	cubeMap_ : CubeMapPlug = PlugDescriptor("cubeMap")
	cubeRightTextureName_ : CubeRightTextureNamePlug = PlugDescriptor("cubeRightTextureName")
	cubeTopTextureName_ : CubeTopTextureNamePlug = PlugDescriptor("cubeTopTextureName")
	decalMode_ : DecalModePlug = PlugDescriptor("decalMode")
	sphereMapTextureName_ : SphereMapTextureNamePlug = PlugDescriptor("sphereMapTextureName")
	textureHasChanged_ : TextureHasChangedPlug = PlugDescriptor("textureHasChanged")

	# node attributes

	typeName = "hwReflectionMap"
	typeIdInt = 1213682253
	nodeLeafClassAttrs = ["binMembership", "cubeBackTextureName", "cubeBottomTextureName", "cubeFrontTextureName", "cubeLeftTextureName", "cubeMap", "cubeRightTextureName", "cubeTopTextureName", "decalMode", "sphereMapTextureName", "textureHasChanged"]
	nodeLeafPlugs = ["binMembership", "cubeBackTextureName", "cubeBottomTextureName", "cubeFrontTextureName", "cubeLeftTextureName", "cubeMap", "cubeRightTextureName", "cubeTopTextureName", "decalMode", "sphereMapTextureName", "textureHasChanged"]
	pass

