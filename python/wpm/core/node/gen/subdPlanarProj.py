

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SubdModifierWorld = retriever.getNodeCls("SubdModifierWorld")
assert SubdModifierWorld
if T.TYPE_CHECKING:
	from .. import SubdModifierWorld

# add node doc



# region plug type defs
class CompIdPlug(Plug):
	node : SubdPlanarProj = None
	pass
class ImageCenterXPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : SubdPlanarProj = None
	pass
class ImageCenterYPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : SubdPlanarProj = None
	pass
class ImageCenterPlug(Plug):
	imageCenterX_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	icx_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	imageCenterY_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	icy_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	node : SubdPlanarProj = None
	pass
class ImageScaleUPlug(Plug):
	parent : ImageScalePlug = PlugDescriptor("imageScale")
	node : SubdPlanarProj = None
	pass
class ImageScaleVPlug(Plug):
	parent : ImageScalePlug = PlugDescriptor("imageScale")
	node : SubdPlanarProj = None
	pass
class ImageScalePlug(Plug):
	imageScaleU_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	isu_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	imageScaleV_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	isv_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	node : SubdPlanarProj = None
	pass
class ProjectionCenterXPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : SubdPlanarProj = None
	pass
class ProjectionCenterYPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : SubdPlanarProj = None
	pass
class ProjectionCenterZPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : SubdPlanarProj = None
	pass
class ProjectionCenterPlug(Plug):
	projectionCenterX_ : ProjectionCenterXPlug = PlugDescriptor("projectionCenterX")
	pcx_ : ProjectionCenterXPlug = PlugDescriptor("projectionCenterX")
	projectionCenterY_ : ProjectionCenterYPlug = PlugDescriptor("projectionCenterY")
	pcy_ : ProjectionCenterYPlug = PlugDescriptor("projectionCenterY")
	projectionCenterZ_ : ProjectionCenterZPlug = PlugDescriptor("projectionCenterZ")
	pcz_ : ProjectionCenterZPlug = PlugDescriptor("projectionCenterZ")
	node : SubdPlanarProj = None
	pass
class ProjectionHeightPlug(Plug):
	parent : ProjectionScalePlug = PlugDescriptor("projectionScale")
	node : SubdPlanarProj = None
	pass
class ProjectionWidthPlug(Plug):
	parent : ProjectionScalePlug = PlugDescriptor("projectionScale")
	node : SubdPlanarProj = None
	pass
class ProjectionScalePlug(Plug):
	projectionHeight_ : ProjectionHeightPlug = PlugDescriptor("projectionHeight")
	ph_ : ProjectionHeightPlug = PlugDescriptor("projectionHeight")
	projectionWidth_ : ProjectionWidthPlug = PlugDescriptor("projectionWidth")
	pw_ : ProjectionWidthPlug = PlugDescriptor("projectionWidth")
	node : SubdPlanarProj = None
	pass
class RadiusPlug(Plug):
	node : SubdPlanarProj = None
	pass
class RotateXPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : SubdPlanarProj = None
	pass
class RotateYPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : SubdPlanarProj = None
	pass
class RotateZPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : SubdPlanarProj = None
	pass
class RotatePlug(Plug):
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rx_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	ry_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rz_ : RotateZPlug = PlugDescriptor("rotateZ")
	node : SubdPlanarProj = None
	pass
class RotationAnglePlug(Plug):
	node : SubdPlanarProj = None
	pass
# endregion


# define node class
class SubdPlanarProj(SubdModifierWorld):
	compId_ : CompIdPlug = PlugDescriptor("compId")
	imageCenterX_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	imageCenterY_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	imageCenter_ : ImageCenterPlug = PlugDescriptor("imageCenter")
	imageScaleU_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	imageScaleV_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	imageScale_ : ImageScalePlug = PlugDescriptor("imageScale")
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

	# node attributes

	typeName = "subdPlanarProj"
	typeIdInt = 1397771344
	nodeLeafClassAttrs = ["compId", "imageCenterX", "imageCenterY", "imageCenter", "imageScaleU", "imageScaleV", "imageScale", "projectionCenterX", "projectionCenterY", "projectionCenterZ", "projectionCenter", "projectionHeight", "projectionWidth", "projectionScale", "radius", "rotateX", "rotateY", "rotateZ", "rotate", "rotationAngle"]
	nodeLeafPlugs = ["compId", "imageCenter", "imageScale", "projectionCenter", "projectionScale", "radius", "rotate", "rotationAngle"]
	pass

