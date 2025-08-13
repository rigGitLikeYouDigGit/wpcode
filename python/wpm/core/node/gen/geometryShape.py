

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Shape = retriever.getNodeCls("Shape")
assert Shape
if T.TYPE_CHECKING:
	from .. import Shape

# add node doc



# region plug type defs
class AntialiasingLevelPlug(Plug):
	node : GeometryShape = None
	pass
class AsBackgroundPlug(Plug):
	node : GeometryShape = None
	pass
class CastsShadowsPlug(Plug):
	node : GeometryShape = None
	pass
class CompObjectGrpCompListPlug(Plug):
	parent : CompObjectGroupsPlug = PlugDescriptor("compObjectGroups")
	node : GeometryShape = None
	pass
class CompObjectGroupsPlug(Plug):
	parent : CompInstObjGroupsPlug = PlugDescriptor("compInstObjGroups")
	compObjectGroupId_ : CompObjectGroupIdPlug = PlugDescriptor("compObjectGroupId")
	cgid_ : CompObjectGroupIdPlug = PlugDescriptor("compObjectGroupId")
	compObjectGrpCompList_ : CompObjectGrpCompListPlug = PlugDescriptor("compObjectGrpCompList")
	cgcl_ : CompObjectGrpCompListPlug = PlugDescriptor("compObjectGrpCompList")
	node : GeometryShape = None
	pass
class CompInstObjGroupsPlug(Plug):
	compObjectGroups_ : CompObjectGroupsPlug = PlugDescriptor("compObjectGroups")
	cog_ : CompObjectGroupsPlug = PlugDescriptor("compObjectGroups")
	node : GeometryShape = None
	pass
class CompObjectGroupIdPlug(Plug):
	parent : CompObjectGroupsPlug = PlugDescriptor("compObjectGroups")
	node : GeometryShape = None
	pass
class ComponentTagContentsPlug(Plug):
	parent : ComponentTagsPlug = PlugDescriptor("componentTags")
	node : GeometryShape = None
	pass
class ComponentTagNamePlug(Plug):
	parent : ComponentTagsPlug = PlugDescriptor("componentTags")
	node : GeometryShape = None
	pass
class ComponentTagsPlug(Plug):
	componentTagContents_ : ComponentTagContentsPlug = PlugDescriptor("componentTagContents")
	gtagcmp_ : ComponentTagContentsPlug = PlugDescriptor("componentTagContents")
	componentTagName_ : ComponentTagNamePlug = PlugDescriptor("componentTagName")
	gtagnm_ : ComponentTagNamePlug = PlugDescriptor("componentTagName")
	node : GeometryShape = None
	pass
class DepthJitterPlug(Plug):
	node : GeometryShape = None
	pass
class GeometryAntialiasingOverridePlug(Plug):
	node : GeometryShape = None
	pass
class HardwareFogMultiplierPlug(Plug):
	node : GeometryShape = None
	pass
class IgnoreSelfShadowingPlug(Plug):
	node : GeometryShape = None
	pass
class InstMaterialAssignPlug(Plug):
	node : GeometryShape = None
	pass
class MaxShadingSamplesPlug(Plug):
	node : GeometryShape = None
	pass
class MaxVisibilitySamplesPlug(Plug):
	node : GeometryShape = None
	pass
class MaxVisibilitySamplesOverridePlug(Plug):
	node : GeometryShape = None
	pass
class MotionBlurPlug(Plug):
	node : GeometryShape = None
	pass
class PickTexturePlug(Plug):
	node : GeometryShape = None
	pass
class PrimaryVisibilityPlug(Plug):
	node : GeometryShape = None
	pass
class ReceiveShadowsPlug(Plug):
	node : GeometryShape = None
	pass
class ReferenceObjectPlug(Plug):
	node : GeometryShape = None
	pass
class RenderTypePlug(Plug):
	node : GeometryShape = None
	pass
class RenderVolumePlug(Plug):
	node : GeometryShape = None
	pass
class ShadingSamplesPlug(Plug):
	node : GeometryShape = None
	pass
class ShadingSamplesOverridePlug(Plug):
	node : GeometryShape = None
	pass
class VisibleFractionPlug(Plug):
	node : GeometryShape = None
	pass
class VisibleInReflectionsPlug(Plug):
	node : GeometryShape = None
	pass
class VisibleInRefractionsPlug(Plug):
	node : GeometryShape = None
	pass
class VolumeSamplesPlug(Plug):
	node : GeometryShape = None
	pass
class VolumeSamplesOverridePlug(Plug):
	node : GeometryShape = None
	pass
# endregion


# define node class
class GeometryShape(Shape):
	antialiasingLevel_ : AntialiasingLevelPlug = PlugDescriptor("antialiasingLevel")
	asBackground_ : AsBackgroundPlug = PlugDescriptor("asBackground")
	castsShadows_ : CastsShadowsPlug = PlugDescriptor("castsShadows")
	compObjectGrpCompList_ : CompObjectGrpCompListPlug = PlugDescriptor("compObjectGrpCompList")
	compObjectGroups_ : CompObjectGroupsPlug = PlugDescriptor("compObjectGroups")
	compInstObjGroups_ : CompInstObjGroupsPlug = PlugDescriptor("compInstObjGroups")
	compObjectGroupId_ : CompObjectGroupIdPlug = PlugDescriptor("compObjectGroupId")
	componentTagContents_ : ComponentTagContentsPlug = PlugDescriptor("componentTagContents")
	componentTagName_ : ComponentTagNamePlug = PlugDescriptor("componentTagName")
	componentTags_ : ComponentTagsPlug = PlugDescriptor("componentTags")
	depthJitter_ : DepthJitterPlug = PlugDescriptor("depthJitter")
	geometryAntialiasingOverride_ : GeometryAntialiasingOverridePlug = PlugDescriptor("geometryAntialiasingOverride")
	hardwareFogMultiplier_ : HardwareFogMultiplierPlug = PlugDescriptor("hardwareFogMultiplier")
	ignoreSelfShadowing_ : IgnoreSelfShadowingPlug = PlugDescriptor("ignoreSelfShadowing")
	instMaterialAssign_ : InstMaterialAssignPlug = PlugDescriptor("instMaterialAssign")
	maxShadingSamples_ : MaxShadingSamplesPlug = PlugDescriptor("maxShadingSamples")
	maxVisibilitySamples_ : MaxVisibilitySamplesPlug = PlugDescriptor("maxVisibilitySamples")
	maxVisibilitySamplesOverride_ : MaxVisibilitySamplesOverridePlug = PlugDescriptor("maxVisibilitySamplesOverride")
	motionBlur_ : MotionBlurPlug = PlugDescriptor("motionBlur")
	pickTexture_ : PickTexturePlug = PlugDescriptor("pickTexture")
	primaryVisibility_ : PrimaryVisibilityPlug = PlugDescriptor("primaryVisibility")
	receiveShadows_ : ReceiveShadowsPlug = PlugDescriptor("receiveShadows")
	referenceObject_ : ReferenceObjectPlug = PlugDescriptor("referenceObject")
	renderType_ : RenderTypePlug = PlugDescriptor("renderType")
	renderVolume_ : RenderVolumePlug = PlugDescriptor("renderVolume")
	shadingSamples_ : ShadingSamplesPlug = PlugDescriptor("shadingSamples")
	shadingSamplesOverride_ : ShadingSamplesOverridePlug = PlugDescriptor("shadingSamplesOverride")
	visibleFraction_ : VisibleFractionPlug = PlugDescriptor("visibleFraction")
	visibleInReflections_ : VisibleInReflectionsPlug = PlugDescriptor("visibleInReflections")
	visibleInRefractions_ : VisibleInRefractionsPlug = PlugDescriptor("visibleInRefractions")
	volumeSamples_ : VolumeSamplesPlug = PlugDescriptor("volumeSamples")
	volumeSamplesOverride_ : VolumeSamplesOverridePlug = PlugDescriptor("volumeSamplesOverride")

	# node attributes

	typeName = "geometryShape"
	typeIdInt = 1196640336
	pass

