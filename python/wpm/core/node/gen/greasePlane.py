

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ImagePlane = retriever.getNodeCls("ImagePlane")
assert ImagePlane
if T.TYPE_CHECKING:
	from .. import ImagePlane

# add node doc



# region plug type defs
class GreaseSequencePlug(Plug):
	node : GreasePlane = None
	pass
class RenderPlanePlug(Plug):
	node : GreasePlane = None
	pass
class RenderPlaneRotateXPlug(Plug):
	parent : RenderPlaneRotatePlug = PlugDescriptor("renderPlaneRotate")
	node : GreasePlane = None
	pass
class RenderPlaneRotateYPlug(Plug):
	parent : RenderPlaneRotatePlug = PlugDescriptor("renderPlaneRotate")
	node : GreasePlane = None
	pass
class RenderPlaneRotateZPlug(Plug):
	parent : RenderPlaneRotatePlug = PlugDescriptor("renderPlaneRotate")
	node : GreasePlane = None
	pass
class RenderPlaneRotatePlug(Plug):
	renderPlaneRotateX_ : RenderPlaneRotateXPlug = PlugDescriptor("renderPlaneRotateX")
	rprx_ : RenderPlaneRotateXPlug = PlugDescriptor("renderPlaneRotateX")
	renderPlaneRotateY_ : RenderPlaneRotateYPlug = PlugDescriptor("renderPlaneRotateY")
	rpry_ : RenderPlaneRotateYPlug = PlugDescriptor("renderPlaneRotateY")
	renderPlaneRotateZ_ : RenderPlaneRotateZPlug = PlugDescriptor("renderPlaneRotateZ")
	rprz_ : RenderPlaneRotateZPlug = PlugDescriptor("renderPlaneRotateZ")
	node : GreasePlane = None
	pass
class RenderPlaneScaleXPlug(Plug):
	parent : RenderPlaneScalePlug = PlugDescriptor("renderPlaneScale")
	node : GreasePlane = None
	pass
class RenderPlaneScaleYPlug(Plug):
	parent : RenderPlaneScalePlug = PlugDescriptor("renderPlaneScale")
	node : GreasePlane = None
	pass
class RenderPlaneScaleZPlug(Plug):
	parent : RenderPlaneScalePlug = PlugDescriptor("renderPlaneScale")
	node : GreasePlane = None
	pass
class RenderPlaneScalePlug(Plug):
	renderPlaneScaleX_ : RenderPlaneScaleXPlug = PlugDescriptor("renderPlaneScaleX")
	rpsx_ : RenderPlaneScaleXPlug = PlugDescriptor("renderPlaneScaleX")
	renderPlaneScaleY_ : RenderPlaneScaleYPlug = PlugDescriptor("renderPlaneScaleY")
	rpsy_ : RenderPlaneScaleYPlug = PlugDescriptor("renderPlaneScaleY")
	renderPlaneScaleZ_ : RenderPlaneScaleZPlug = PlugDescriptor("renderPlaneScaleZ")
	rpsz_ : RenderPlaneScaleZPlug = PlugDescriptor("renderPlaneScaleZ")
	node : GreasePlane = None
	pass
class RenderPlaneTranslateXPlug(Plug):
	parent : RenderPlaneTranslatePlug = PlugDescriptor("renderPlaneTranslate")
	node : GreasePlane = None
	pass
class RenderPlaneTranslateYPlug(Plug):
	parent : RenderPlaneTranslatePlug = PlugDescriptor("renderPlaneTranslate")
	node : GreasePlane = None
	pass
class RenderPlaneTranslateZPlug(Plug):
	parent : RenderPlaneTranslatePlug = PlugDescriptor("renderPlaneTranslate")
	node : GreasePlane = None
	pass
class RenderPlaneTranslatePlug(Plug):
	renderPlaneTranslateX_ : RenderPlaneTranslateXPlug = PlugDescriptor("renderPlaneTranslateX")
	rptx_ : RenderPlaneTranslateXPlug = PlugDescriptor("renderPlaneTranslateX")
	renderPlaneTranslateY_ : RenderPlaneTranslateYPlug = PlugDescriptor("renderPlaneTranslateY")
	rpty_ : RenderPlaneTranslateYPlug = PlugDescriptor("renderPlaneTranslateY")
	renderPlaneTranslateZ_ : RenderPlaneTranslateZPlug = PlugDescriptor("renderPlaneTranslateZ")
	rptz_ : RenderPlaneTranslateZPlug = PlugDescriptor("renderPlaneTranslateZ")
	node : GreasePlane = None
	pass
class SourceDepthPlug(Plug):
	node : GreasePlane = None
	pass
class SourcePlanePlug(Plug):
	node : GreasePlane = None
	pass
class SourcePlaneRotateXPlug(Plug):
	parent : SourcePlaneRotatePlug = PlugDescriptor("sourcePlaneRotate")
	node : GreasePlane = None
	pass
class SourcePlaneRotateYPlug(Plug):
	parent : SourcePlaneRotatePlug = PlugDescriptor("sourcePlaneRotate")
	node : GreasePlane = None
	pass
class SourcePlaneRotateZPlug(Plug):
	parent : SourcePlaneRotatePlug = PlugDescriptor("sourcePlaneRotate")
	node : GreasePlane = None
	pass
class SourcePlaneRotatePlug(Plug):
	sourcePlaneRotateX_ : SourcePlaneRotateXPlug = PlugDescriptor("sourcePlaneRotateX")
	sprx_ : SourcePlaneRotateXPlug = PlugDescriptor("sourcePlaneRotateX")
	sourcePlaneRotateY_ : SourcePlaneRotateYPlug = PlugDescriptor("sourcePlaneRotateY")
	spry_ : SourcePlaneRotateYPlug = PlugDescriptor("sourcePlaneRotateY")
	sourcePlaneRotateZ_ : SourcePlaneRotateZPlug = PlugDescriptor("sourcePlaneRotateZ")
	sprz_ : SourcePlaneRotateZPlug = PlugDescriptor("sourcePlaneRotateZ")
	node : GreasePlane = None
	pass
class SourcePlaneScaleXPlug(Plug):
	parent : SourcePlaneScalePlug = PlugDescriptor("sourcePlaneScale")
	node : GreasePlane = None
	pass
class SourcePlaneScaleYPlug(Plug):
	parent : SourcePlaneScalePlug = PlugDescriptor("sourcePlaneScale")
	node : GreasePlane = None
	pass
class SourcePlaneScaleZPlug(Plug):
	parent : SourcePlaneScalePlug = PlugDescriptor("sourcePlaneScale")
	node : GreasePlane = None
	pass
class SourcePlaneScalePlug(Plug):
	sourcePlaneScaleX_ : SourcePlaneScaleXPlug = PlugDescriptor("sourcePlaneScaleX")
	spsx_ : SourcePlaneScaleXPlug = PlugDescriptor("sourcePlaneScaleX")
	sourcePlaneScaleY_ : SourcePlaneScaleYPlug = PlugDescriptor("sourcePlaneScaleY")
	spsy_ : SourcePlaneScaleYPlug = PlugDescriptor("sourcePlaneScaleY")
	sourcePlaneScaleZ_ : SourcePlaneScaleZPlug = PlugDescriptor("sourcePlaneScaleZ")
	spsz_ : SourcePlaneScaleZPlug = PlugDescriptor("sourcePlaneScaleZ")
	node : GreasePlane = None
	pass
class SourcePlaneTranslateXPlug(Plug):
	parent : SourcePlaneTranslatePlug = PlugDescriptor("sourcePlaneTranslate")
	node : GreasePlane = None
	pass
class SourcePlaneTranslateYPlug(Plug):
	parent : SourcePlaneTranslatePlug = PlugDescriptor("sourcePlaneTranslate")
	node : GreasePlane = None
	pass
class SourcePlaneTranslateZPlug(Plug):
	parent : SourcePlaneTranslatePlug = PlugDescriptor("sourcePlaneTranslate")
	node : GreasePlane = None
	pass
class SourcePlaneTranslatePlug(Plug):
	sourcePlaneTranslateX_ : SourcePlaneTranslateXPlug = PlugDescriptor("sourcePlaneTranslateX")
	sptx_ : SourcePlaneTranslateXPlug = PlugDescriptor("sourcePlaneTranslateX")
	sourcePlaneTranslateY_ : SourcePlaneTranslateYPlug = PlugDescriptor("sourcePlaneTranslateY")
	spty_ : SourcePlaneTranslateYPlug = PlugDescriptor("sourcePlaneTranslateY")
	sourcePlaneTranslateZ_ : SourcePlaneTranslateZPlug = PlugDescriptor("sourcePlaneTranslateZ")
	sptz_ : SourcePlaneTranslateZPlug = PlugDescriptor("sourcePlaneTranslateZ")
	node : GreasePlane = None
	pass
# endregion


# define node class
class GreasePlane(ImagePlane):
	greaseSequence_ : GreaseSequencePlug = PlugDescriptor("greaseSequence")
	renderPlane_ : RenderPlanePlug = PlugDescriptor("renderPlane")
	renderPlaneRotateX_ : RenderPlaneRotateXPlug = PlugDescriptor("renderPlaneRotateX")
	renderPlaneRotateY_ : RenderPlaneRotateYPlug = PlugDescriptor("renderPlaneRotateY")
	renderPlaneRotateZ_ : RenderPlaneRotateZPlug = PlugDescriptor("renderPlaneRotateZ")
	renderPlaneRotate_ : RenderPlaneRotatePlug = PlugDescriptor("renderPlaneRotate")
	renderPlaneScaleX_ : RenderPlaneScaleXPlug = PlugDescriptor("renderPlaneScaleX")
	renderPlaneScaleY_ : RenderPlaneScaleYPlug = PlugDescriptor("renderPlaneScaleY")
	renderPlaneScaleZ_ : RenderPlaneScaleZPlug = PlugDescriptor("renderPlaneScaleZ")
	renderPlaneScale_ : RenderPlaneScalePlug = PlugDescriptor("renderPlaneScale")
	renderPlaneTranslateX_ : RenderPlaneTranslateXPlug = PlugDescriptor("renderPlaneTranslateX")
	renderPlaneTranslateY_ : RenderPlaneTranslateYPlug = PlugDescriptor("renderPlaneTranslateY")
	renderPlaneTranslateZ_ : RenderPlaneTranslateZPlug = PlugDescriptor("renderPlaneTranslateZ")
	renderPlaneTranslate_ : RenderPlaneTranslatePlug = PlugDescriptor("renderPlaneTranslate")
	sourceDepth_ : SourceDepthPlug = PlugDescriptor("sourceDepth")
	sourcePlane_ : SourcePlanePlug = PlugDescriptor("sourcePlane")
	sourcePlaneRotateX_ : SourcePlaneRotateXPlug = PlugDescriptor("sourcePlaneRotateX")
	sourcePlaneRotateY_ : SourcePlaneRotateYPlug = PlugDescriptor("sourcePlaneRotateY")
	sourcePlaneRotateZ_ : SourcePlaneRotateZPlug = PlugDescriptor("sourcePlaneRotateZ")
	sourcePlaneRotate_ : SourcePlaneRotatePlug = PlugDescriptor("sourcePlaneRotate")
	sourcePlaneScaleX_ : SourcePlaneScaleXPlug = PlugDescriptor("sourcePlaneScaleX")
	sourcePlaneScaleY_ : SourcePlaneScaleYPlug = PlugDescriptor("sourcePlaneScaleY")
	sourcePlaneScaleZ_ : SourcePlaneScaleZPlug = PlugDescriptor("sourcePlaneScaleZ")
	sourcePlaneScale_ : SourcePlaneScalePlug = PlugDescriptor("sourcePlaneScale")
	sourcePlaneTranslateX_ : SourcePlaneTranslateXPlug = PlugDescriptor("sourcePlaneTranslateX")
	sourcePlaneTranslateY_ : SourcePlaneTranslateYPlug = PlugDescriptor("sourcePlaneTranslateY")
	sourcePlaneTranslateZ_ : SourcePlaneTranslateZPlug = PlugDescriptor("sourcePlaneTranslateZ")
	sourcePlaneTranslate_ : SourcePlaneTranslatePlug = PlugDescriptor("sourcePlaneTranslate")

	# node attributes

	typeName = "greasePlane"
	apiTypeInt = 1086
	apiTypeStr = "kGreasePlane"
	typeIdInt = 1145524300
	MFnCls = om.MFnDagNode
	pass

