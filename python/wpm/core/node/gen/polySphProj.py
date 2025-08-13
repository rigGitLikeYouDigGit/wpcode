

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
assert PolyModifierWorld
if T.TYPE_CHECKING:
	from .. import PolyModifierWorld

# add node doc



# region plug type defs
class CompIdPlug(Plug):
	node : PolySphProj = None
	pass
class ImageCenterXPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : PolySphProj = None
	pass
class ImageCenterYPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : PolySphProj = None
	pass
class ImageCenterPlug(Plug):
	imageCenterX_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	icx_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	imageCenterY_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	icy_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	node : PolySphProj = None
	pass
class ImageScaleUPlug(Plug):
	parent : ImageScalePlug = PlugDescriptor("imageScale")
	node : PolySphProj = None
	pass
class ImageScaleVPlug(Plug):
	parent : ImageScalePlug = PlugDescriptor("imageScale")
	node : PolySphProj = None
	pass
class ImageScalePlug(Plug):
	imageScaleU_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	isu_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	imageScaleV_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	isv_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	node : PolySphProj = None
	pass
class ProjectionCenterXPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : PolySphProj = None
	pass
class ProjectionCenterYPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : PolySphProj = None
	pass
class ProjectionCenterZPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : PolySphProj = None
	pass
class ProjectionCenterPlug(Plug):
	projectionCenterX_ : ProjectionCenterXPlug = PlugDescriptor("projectionCenterX")
	pcx_ : ProjectionCenterXPlug = PlugDescriptor("projectionCenterX")
	projectionCenterY_ : ProjectionCenterYPlug = PlugDescriptor("projectionCenterY")
	pcy_ : ProjectionCenterYPlug = PlugDescriptor("projectionCenterY")
	projectionCenterZ_ : ProjectionCenterZPlug = PlugDescriptor("projectionCenterZ")
	pcz_ : ProjectionCenterZPlug = PlugDescriptor("projectionCenterZ")
	node : PolySphProj = None
	pass
class ProjectionHorizontalSweepPlug(Plug):
	parent : ProjectionScalePlug = PlugDescriptor("projectionScale")
	node : PolySphProj = None
	pass
class ProjectionVerticalSweepPlug(Plug):
	parent : ProjectionScalePlug = PlugDescriptor("projectionScale")
	node : PolySphProj = None
	pass
class ProjectionScalePlug(Plug):
	projectionHorizontalSweep_ : ProjectionHorizontalSweepPlug = PlugDescriptor("projectionHorizontalSweep")
	phs_ : ProjectionHorizontalSweepPlug = PlugDescriptor("projectionHorizontalSweep")
	projectionVerticalSweep_ : ProjectionVerticalSweepPlug = PlugDescriptor("projectionVerticalSweep")
	pvs_ : ProjectionVerticalSweepPlug = PlugDescriptor("projectionVerticalSweep")
	node : PolySphProj = None
	pass
class RadiusPlug(Plug):
	node : PolySphProj = None
	pass
class RotateXPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolySphProj = None
	pass
class RotateYPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolySphProj = None
	pass
class RotateZPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolySphProj = None
	pass
class RotatePlug(Plug):
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rx_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	ry_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rz_ : RotateZPlug = PlugDescriptor("rotateZ")
	node : PolySphProj = None
	pass
class RotationAnglePlug(Plug):
	node : PolySphProj = None
	pass
class SeamCorrectPlug(Plug):
	node : PolySphProj = None
	pass
class UseOldPolyProjectionPlug(Plug):
	node : PolySphProj = None
	pass
class UvSetNamePlug(Plug):
	node : PolySphProj = None
	pass
# endregion


# define node class
class PolySphProj(PolyModifierWorld):
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
	projectionHorizontalSweep_ : ProjectionHorizontalSweepPlug = PlugDescriptor("projectionHorizontalSweep")
	projectionVerticalSweep_ : ProjectionVerticalSweepPlug = PlugDescriptor("projectionVerticalSweep")
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

	typeName = "polySphProj"
	apiTypeInt = 430
	apiTypeStr = "kPolySphProj"
	typeIdInt = 1347637328
	MFnCls = om.MFnDependencyNode
	pass

