

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
class ImageCenterXPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : PolyProj = None
	pass
class ImageCenterYPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : PolyProj = None
	pass
class ImageCenterPlug(Plug):
	imageCenterX_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	icx_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	imageCenterY_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	icy_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	node : PolyProj = None
	pass
class ImageScaleUPlug(Plug):
	parent : ImageScalePlug = PlugDescriptor("imageScale")
	node : PolyProj = None
	pass
class ImageScaleVPlug(Plug):
	parent : ImageScalePlug = PlugDescriptor("imageScale")
	node : PolyProj = None
	pass
class ImageScalePlug(Plug):
	imageScaleU_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	isu_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	imageScaleV_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	isv_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	node : PolyProj = None
	pass
class ProjTypePlug(Plug):
	node : PolyProj = None
	pass
class ProjectionCenterXPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : PolyProj = None
	pass
class ProjectionCenterYPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : PolyProj = None
	pass
class ProjectionCenterZPlug(Plug):
	parent : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	node : PolyProj = None
	pass
class ProjectionCenterPlug(Plug):
	projectionCenterX_ : ProjectionCenterXPlug = PlugDescriptor("projectionCenterX")
	pcx_ : ProjectionCenterXPlug = PlugDescriptor("projectionCenterX")
	projectionCenterY_ : ProjectionCenterYPlug = PlugDescriptor("projectionCenterY")
	pcy_ : ProjectionCenterYPlug = PlugDescriptor("projectionCenterY")
	projectionCenterZ_ : ProjectionCenterZPlug = PlugDescriptor("projectionCenterZ")
	pcz_ : ProjectionCenterZPlug = PlugDescriptor("projectionCenterZ")
	node : PolyProj = None
	pass
class ProjectionScaleUPlug(Plug):
	parent : ProjectionScalePlug = PlugDescriptor("projectionScale")
	node : PolyProj = None
	pass
class ProjectionScaleVPlug(Plug):
	parent : ProjectionScalePlug = PlugDescriptor("projectionScale")
	node : PolyProj = None
	pass
class ProjectionScalePlug(Plug):
	projectionScaleU_ : ProjectionScaleUPlug = PlugDescriptor("projectionScaleU")
	psu_ : ProjectionScaleUPlug = PlugDescriptor("projectionScaleU")
	projectionScaleV_ : ProjectionScaleVPlug = PlugDescriptor("projectionScaleV")
	psv_ : ProjectionScaleVPlug = PlugDescriptor("projectionScaleV")
	node : PolyProj = None
	pass
class RadiusPlug(Plug):
	node : PolyProj = None
	pass
class RotateXPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyProj = None
	pass
class RotateYPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyProj = None
	pass
class RotateZPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyProj = None
	pass
class RotatePlug(Plug):
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rx_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	ry_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rz_ : RotateZPlug = PlugDescriptor("rotateZ")
	node : PolyProj = None
	pass
class RotationAnglePlug(Plug):
	node : PolyProj = None
	pass
class WorldInverseMatrixPlug(Plug):
	node : PolyProj = None
	pass
# endregion


# define node class
class PolyProj(PolyModifierWorld):
	imageCenterX_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	imageCenterY_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	imageCenter_ : ImageCenterPlug = PlugDescriptor("imageCenter")
	imageScaleU_ : ImageScaleUPlug = PlugDescriptor("imageScaleU")
	imageScaleV_ : ImageScaleVPlug = PlugDescriptor("imageScaleV")
	imageScale_ : ImageScalePlug = PlugDescriptor("imageScale")
	projType_ : ProjTypePlug = PlugDescriptor("projType")
	projectionCenterX_ : ProjectionCenterXPlug = PlugDescriptor("projectionCenterX")
	projectionCenterY_ : ProjectionCenterYPlug = PlugDescriptor("projectionCenterY")
	projectionCenterZ_ : ProjectionCenterZPlug = PlugDescriptor("projectionCenterZ")
	projectionCenter_ : ProjectionCenterPlug = PlugDescriptor("projectionCenter")
	projectionScaleU_ : ProjectionScaleUPlug = PlugDescriptor("projectionScaleU")
	projectionScaleV_ : ProjectionScaleVPlug = PlugDescriptor("projectionScaleV")
	projectionScale_ : ProjectionScalePlug = PlugDescriptor("projectionScale")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rotate_ : RotatePlug = PlugDescriptor("rotate")
	rotationAngle_ : RotationAnglePlug = PlugDescriptor("rotationAngle")
	worldInverseMatrix_ : WorldInverseMatrixPlug = PlugDescriptor("worldInverseMatrix")

	# node attributes

	typeName = "polyProj"
	apiTypeInt = 426
	apiTypeStr = "kPolyProj"
	typeIdInt = 1347441231
	MFnCls = om.MFnDependencyNode
	pass

