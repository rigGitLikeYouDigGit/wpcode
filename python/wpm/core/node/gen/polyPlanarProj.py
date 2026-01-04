

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierWorld = Catalogue.PolyModifierWorld
else:
	from .. import retriever
	PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
	assert PolyModifierWorld

# add node doc



# region plug type defs
class CameraMatrixPlug(Plug):
	node : PolyPlanarProj = None
	pass
class CompIdPlug(Plug):
	node : PolyPlanarProj = None
	pass
class ImageCenterXPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : PolyPlanarProj = None
	pass
class ImageCenterYPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : PolyPlanarProj = None
	pass
class ImageCenterPlug(Plug):
	imageCenterX_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	icx_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	imageCenterY_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	icy_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	node : PolyPlanarProj = None
	pass
class ImageScaleUPlug(Plug):
	parent : ImageScalePlug = PlugDescriptor("imageScale")
	node : PolyPlanarProj = None
	pass
class ImageScaleVPlug(Plug):
	parent : ImageScalePlug = PlugDescriptor("imageScale")
	node : PolyPlanarProj = None
	pass
class ImageScalePlug(Plug):
	imageScaleU_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	isu_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	imageScaleV_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	isv_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	node : PolyPlanarProj = None
	pass
class IsPerspectivePlug(Plug):
	node : PolyPlanarProj = None
	pass
class PortBottomPlug(Plug):
	node : PolyPlanarProj = None
	pass
class PortLeftPlug(Plug):
	node : PolyPlanarProj = None
	pass
class PortRightPlug(Plug):
	node : PolyPlanarProj = None
	pass
class PortTopPlug(Plug):
	node : PolyPlanarProj = None
	pass
class ProjectionCenterXPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : PolyPlanarProj = None
	pass
class ProjectionCenterYPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : PolyPlanarProj = None
	pass
class ProjectionCenterZPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : PolyPlanarProj = None
	pass
class ProjectionCenterPlug(Plug):
	projectionCenterX_ : ProjectionCenterXPlug = PlugDescriptor("projectionCenterX")
	pcx_ : ProjectionCenterXPlug = PlugDescriptor("projectionCenterX")
	projectionCenterY_ : ProjectionCenterYPlug = PlugDescriptor("projectionCenterY")
	pcy_ : ProjectionCenterYPlug = PlugDescriptor("projectionCenterY")
	projectionCenterZ_ : ProjectionCenterZPlug = PlugDescriptor("projectionCenterZ")
	pcz_ : ProjectionCenterZPlug = PlugDescriptor("projectionCenterZ")
	node : PolyPlanarProj = None
	pass
class ProjectionHeightPlug(Plug):
	parent : ProjectionScalePlug = PlugDescriptor("projectionScale")
	node : PolyPlanarProj = None
	pass
class ProjectionWidthPlug(Plug):
	parent : ProjectionScalePlug = PlugDescriptor("projectionScale")
	node : PolyPlanarProj = None
	pass
class ProjectionScalePlug(Plug):
	projectionHeight_ : ProjectionHeightPlug = PlugDescriptor("projectionHeight")
	ph_ : ProjectionHeightPlug = PlugDescriptor("projectionHeight")
	projectionWidth_ : ProjectionWidthPlug = PlugDescriptor("projectionWidth")
	pw_ : ProjectionWidthPlug = PlugDescriptor("projectionWidth")
	node : PolyPlanarProj = None
	pass
class RadiusPlug(Plug):
	node : PolyPlanarProj = None
	pass
class RotateXPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyPlanarProj = None
	pass
class RotateYPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyPlanarProj = None
	pass
class RotateZPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyPlanarProj = None
	pass
class RotatePlug(Plug):
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rx_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	ry_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rz_ : RotateZPlug = PlugDescriptor("rotateZ")
	node : PolyPlanarProj = None
	pass
class RotationAnglePlug(Plug):
	node : PolyPlanarProj = None
	pass
class UvSetNamePlug(Plug):
	node : PolyPlanarProj = None
	pass
# endregion


# define node class
class PolyPlanarProj(PolyModifierWorld):
	cameraMatrix_ : CameraMatrixPlug = PlugDescriptor("cameraMatrix")
	compId_ : CompIdPlug = PlugDescriptor("compId")
	imageCenterX_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	imageCenterY_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	imageCenter_ : ImageCenterPlug = PlugDescriptor("imageCenter")
	imageScaleU_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	imageScaleV_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	imageScale_ : ImageScalePlug = PlugDescriptor("imageScale")
	isPerspective_ : IsPerspectivePlug = PlugDescriptor("isPerspective")
	portBottom_ : PortBottomPlug = PlugDescriptor("portBottom")
	portLeft_ : PortLeftPlug = PlugDescriptor("portLeft")
	portRight_ : PortRightPlug = PlugDescriptor("portRight")
	portTop_ : PortTopPlug = PlugDescriptor("portTop")
	projectionCenterX_ : ProjectionCenterXPlug = PlugDescriptor("projectionCenterX")
	projectionCenterY_ : ProjectionCenterYPlug = PlugDescriptor("projectionCenterY")
	projectionCenterZ_ : ProjectionCenterZPlug = PlugDescriptor("projectionCenterZ")
	projectionCenter_ : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	projectionHeight_ : ProjectionHeightPlug = PlugDescriptor("projectionHeight")
	projectionWidth_ : ProjectionWidthPlug = PlugDescriptor("projectionWidth")
	projectionScale_ : ProjectionScalePlug = PlugDescriptor("projectionScale")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rotate_ : RotatePlug = PlugDescriptor("rotate")
	rotationAngle_ : RotationAnglePlug = PlugDescriptor("rotationAngle")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "polyPlanarProj"
	typeIdInt = 1347439696
	nodeLeafClassAttrs = ["cameraMatrix", "compId", "imageCenterX", "imageCenterY", "imageCenter", "imageScaleU", "imageScaleV", "imageScale", "isPerspective", "portBottom", "portLeft", "portRight", "portTop", "projectionCenterX", "projectionCenterY", "projectionCenterZ", "projectionCenter", "projectionHeight", "projectionWidth", "projectionScale", "radius", "rotateX", "rotateY", "rotateZ", "rotate", "rotationAngle", "uvSetName"]
	nodeLeafPlugs = ["cameraMatrix", "compId", "imageCenter", "imageScale", "isPerspective", "portBottom", "portLeft", "portRight", "portTop", "projectionCenter", "projectionScale", "radius", "rotate", "rotationAngle", "uvSetName"]
	pass

