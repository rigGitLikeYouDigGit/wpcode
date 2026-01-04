

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
class CompIdPlug(Plug):
	node : PolyCylProj = None
	pass
class ImageCenterXPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : PolyCylProj = None
	pass
class ImageCenterYPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : PolyCylProj = None
	pass
class ImageCenterPlug(Plug):
	imageCenterX_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	icx_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	imageCenterY_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	icy_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	node : PolyCylProj = None
	pass
class ImageScaleUPlug(Plug):
	parent : ImageScalePlug = PlugDescriptor("imageScale")
	node : PolyCylProj = None
	pass
class ImageScaleVPlug(Plug):
	parent : ImageScalePlug = PlugDescriptor("imageScale")
	node : PolyCylProj = None
	pass
class ImageScalePlug(Plug):
	imageScaleU_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	isu_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	imageScaleV_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	isv_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	node : PolyCylProj = None
	pass
class ProjectionCenterXPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : PolyCylProj = None
	pass
class ProjectionCenterYPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : PolyCylProj = None
	pass
class ProjectionCenterZPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : PolyCylProj = None
	pass
class ProjectionCenterPlug(Plug):
	projectionCenterX_ : ProjectionCenterXPlug = PlugDescriptor("projectionCenterX")
	pcx_ : ProjectionCenterXPlug = PlugDescriptor("projectionCenterX")
	projectionCenterY_ : ProjectionCenterYPlug = PlugDescriptor("projectionCenterY")
	pcy_ : ProjectionCenterYPlug = PlugDescriptor("projectionCenterY")
	projectionCenterZ_ : ProjectionCenterZPlug = PlugDescriptor("projectionCenterZ")
	pcz_ : ProjectionCenterZPlug = PlugDescriptor("projectionCenterZ")
	node : PolyCylProj = None
	pass
class ProjectionHeightPlug(Plug):
	parent : ProjectionScalePlug = PlugDescriptor("projectionScale")
	node : PolyCylProj = None
	pass
class ProjectionHorizontalSweepPlug(Plug):
	parent : ProjectionScalePlug = PlugDescriptor("projectionScale")
	node : PolyCylProj = None
	pass
class ProjectionScalePlug(Plug):
	projectionHeight_ : ProjectionHeightPlug = PlugDescriptor("projectionHeight")
	ph_ : ProjectionHeightPlug = PlugDescriptor("projectionHeight")
	projectionHorizontalSweep_ : ProjectionHorizontalSweepPlug = PlugDescriptor("projectionHorizontalSweep")
	phs_ : ProjectionHorizontalSweepPlug = PlugDescriptor("projectionHorizontalSweep")
	node : PolyCylProj = None
	pass
class RadiusPlug(Plug):
	node : PolyCylProj = None
	pass
class RotateXPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyCylProj = None
	pass
class RotateYPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyCylProj = None
	pass
class RotateZPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyCylProj = None
	pass
class RotatePlug(Plug):
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rx_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	ry_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rz_ : RotateZPlug = PlugDescriptor("rotateZ")
	node : PolyCylProj = None
	pass
class RotationAnglePlug(Plug):
	node : PolyCylProj = None
	pass
class SeamCorrectPlug(Plug):
	node : PolyCylProj = None
	pass
class UseOldPolyProjectionPlug(Plug):
	node : PolyCylProj = None
	pass
class UvSetNamePlug(Plug):
	node : PolyCylProj = None
	pass
# endregion


# define node class
class PolyCylProj(PolyModifierWorld):
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
	projectionHorizontalSweep_ : ProjectionHorizontalSweepPlug = PlugDescriptor("projectionHorizontalSweep")
	projectionScale_ : ProjectionScalePlug = PlugDescriptor("projectionScale")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rotate_ : RotatePlug = PlugDescriptor("rotate")
	rotationAngle_ : RotationAnglePlug = PlugDescriptor("rotationAngle")
	seamCorrect_ : SeamCorrectPlug = PlugDescriptor("seamCorrect")
	useOldPolyProjection_ : UseOldPolyProjectionPlug = PlugDescriptor("useOldPolyProjection")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "polyCylProj"
	apiTypeInt = 408
	apiTypeStr = "kPolyCylProj"
	typeIdInt = 1346591056
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["compId", "imageCenterX", "imageCenterY", "imageCenter", "imageScaleU", "imageScaleV", "imageScale", "projectionCenterX", "projectionCenterY", "projectionCenterZ", "projectionCenter", "projectionHeight", "projectionHorizontalSweep", "projectionScale", "radius", "rotateX", "rotateY", "rotateZ", "rotate", "rotationAngle", "seamCorrect", "useOldPolyProjection", "uvSetName"]
	nodeLeafPlugs = ["compId", "imageCenter", "imageScale", "projectionCenter", "projectionScale", "radius", "rotate", "rotationAngle", "seamCorrect", "useOldPolyProjection", "uvSetName"]
	pass

