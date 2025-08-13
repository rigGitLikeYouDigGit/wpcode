

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ShadingDependNode = retriever.getNodeCls("ShadingDependNode")
assert ShadingDependNode
if T.TYPE_CHECKING:
	from .. import ShadingDependNode

# add node doc



# region plug type defs
class FilterSizeXPlug(Plug):
	parent : FilterSizePlug = PlugDescriptor("filterSize")
	node : TextureEnv = None
	pass
class FilterSizeYPlug(Plug):
	parent : FilterSizePlug = PlugDescriptor("filterSize")
	node : TextureEnv = None
	pass
class FilterSizeZPlug(Plug):
	parent : FilterSizePlug = PlugDescriptor("filterSize")
	node : TextureEnv = None
	pass
class FilterSizePlug(Plug):
	filterSizeX_ : FilterSizeXPlug = PlugDescriptor("filterSizeX")
	fsx_ : FilterSizeXPlug = PlugDescriptor("filterSizeX")
	filterSizeY_ : FilterSizeYPlug = PlugDescriptor("filterSizeY")
	fsy_ : FilterSizeYPlug = PlugDescriptor("filterSizeY")
	filterSizeZ_ : FilterSizeZPlug = PlugDescriptor("filterSizeZ")
	fsz_ : FilterSizeZPlug = PlugDescriptor("filterSizeZ")
	node : TextureEnv = None
	pass
class MatrixEyeToWorldPlug(Plug):
	node : TextureEnv = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : TextureEnv = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : TextureEnv = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : TextureEnv = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : TextureEnv = None
	pass
class ObjectTypePlug(Plug):
	node : TextureEnv = None
	pass
class OutAlphaPlug(Plug):
	node : TextureEnv = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : TextureEnv = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : TextureEnv = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : TextureEnv = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : TextureEnv = None
	pass
class PlacementMatrixPlug(Plug):
	node : TextureEnv = None
	pass
class RayDirectionXPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : TextureEnv = None
	pass
class RayDirectionYPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : TextureEnv = None
	pass
class RayDirectionZPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : TextureEnv = None
	pass
class RayDirectionPlug(Plug):
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rx_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	ry_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rz_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	node : TextureEnv = None
	pass
class UCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : TextureEnv = None
	pass
class VCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : TextureEnv = None
	pass
class UvCoordPlug(Plug):
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	u_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	v_ : VCoordPlug = PlugDescriptor("vCoord")
	node : TextureEnv = None
	pass
class UvFilterSizeXPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : TextureEnv = None
	pass
class UvFilterSizeYPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : TextureEnv = None
	pass
class UvFilterSizePlug(Plug):
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	ufx_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	ufy_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	node : TextureEnv = None
	pass
class XPixelAnglePlug(Plug):
	node : TextureEnv = None
	pass
# endregion


# define node class
class TextureEnv(ShadingDependNode):
	filterSizeX_ : FilterSizeXPlug = PlugDescriptor("filterSizeX")
	filterSizeY_ : FilterSizeYPlug = PlugDescriptor("filterSizeY")
	filterSizeZ_ : FilterSizeZPlug = PlugDescriptor("filterSizeZ")
	filterSize_ : FilterSizePlug = PlugDescriptor("filterSize")
	matrixEyeToWorld_ : MatrixEyeToWorldPlug = PlugDescriptor("matrixEyeToWorld")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	objectType_ : ObjectTypePlug = PlugDescriptor("objectType")
	outAlpha_ : OutAlphaPlug = PlugDescriptor("outAlpha")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	placementMatrix_ : PlacementMatrixPlug = PlugDescriptor("placementMatrix")
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rayDirection_ : RayDirectionPlug = PlugDescriptor("rayDirection")
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	uvCoord_ : UvCoordPlug = PlugDescriptor("uvCoord")
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	uvFilterSize_ : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	xPixelAngle_ : XPixelAnglePlug = PlugDescriptor("xPixelAngle")

	# node attributes

	typeName = "textureEnv"
	typeIdInt = 1381259333
	pass

