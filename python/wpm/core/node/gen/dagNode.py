

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Entity = retriever.getNodeCls("Entity")
assert Entity
if T.TYPE_CHECKING:
	from .. import Entity

# add node doc



# region plug type defs
class BoundingBoxMaxXPlug(Plug):
	parent : BoundingBoxMaxPlug = PlugDescriptor("boundingBoxMax")
	node : DagNode = None
	pass
class BoundingBoxMaxYPlug(Plug):
	parent : BoundingBoxMaxPlug = PlugDescriptor("boundingBoxMax")
	node : DagNode = None
	pass
class BoundingBoxMaxZPlug(Plug):
	parent : BoundingBoxMaxPlug = PlugDescriptor("boundingBoxMax")
	node : DagNode = None
	pass
class BoundingBoxMaxPlug(Plug):
	parent : BoundingBoxPlug = PlugDescriptor("boundingBox")
	boundingBoxMaxX_ : BoundingBoxMaxXPlug = PlugDescriptor("boundingBoxMaxX")
	bbxx_ : BoundingBoxMaxXPlug = PlugDescriptor("boundingBoxMaxX")
	boundingBoxMaxY_ : BoundingBoxMaxYPlug = PlugDescriptor("boundingBoxMaxY")
	bbxy_ : BoundingBoxMaxYPlug = PlugDescriptor("boundingBoxMaxY")
	boundingBoxMaxZ_ : BoundingBoxMaxZPlug = PlugDescriptor("boundingBoxMaxZ")
	bbxz_ : BoundingBoxMaxZPlug = PlugDescriptor("boundingBoxMaxZ")
	node : DagNode = None
	pass
class BoundingBoxMinXPlug(Plug):
	parent : BoundingBoxMinPlug = PlugDescriptor("boundingBoxMin")
	node : DagNode = None
	pass
class BoundingBoxMinYPlug(Plug):
	parent : BoundingBoxMinPlug = PlugDescriptor("boundingBoxMin")
	node : DagNode = None
	pass
class BoundingBoxMinZPlug(Plug):
	parent : BoundingBoxMinPlug = PlugDescriptor("boundingBoxMin")
	node : DagNode = None
	pass
class BoundingBoxMinPlug(Plug):
	parent : BoundingBoxPlug = PlugDescriptor("boundingBox")
	boundingBoxMinX_ : BoundingBoxMinXPlug = PlugDescriptor("boundingBoxMinX")
	bbnx_ : BoundingBoxMinXPlug = PlugDescriptor("boundingBoxMinX")
	boundingBoxMinY_ : BoundingBoxMinYPlug = PlugDescriptor("boundingBoxMinY")
	bbny_ : BoundingBoxMinYPlug = PlugDescriptor("boundingBoxMinY")
	boundingBoxMinZ_ : BoundingBoxMinZPlug = PlugDescriptor("boundingBoxMinZ")
	bbnz_ : BoundingBoxMinZPlug = PlugDescriptor("boundingBoxMinZ")
	node : DagNode = None
	pass
class BoundingBoxSizeXPlug(Plug):
	parent : BoundingBoxSizePlug = PlugDescriptor("boundingBoxSize")
	node : DagNode = None
	pass
class BoundingBoxSizeYPlug(Plug):
	parent : BoundingBoxSizePlug = PlugDescriptor("boundingBoxSize")
	node : DagNode = None
	pass
class BoundingBoxSizeZPlug(Plug):
	parent : BoundingBoxSizePlug = PlugDescriptor("boundingBoxSize")
	node : DagNode = None
	pass
class BoundingBoxSizePlug(Plug):
	parent : BoundingBoxPlug = PlugDescriptor("boundingBox")
	boundingBoxSizeX_ : BoundingBoxSizeXPlug = PlugDescriptor("boundingBoxSizeX")
	bbsx_ : BoundingBoxSizeXPlug = PlugDescriptor("boundingBoxSizeX")
	boundingBoxSizeY_ : BoundingBoxSizeYPlug = PlugDescriptor("boundingBoxSizeY")
	bbsy_ : BoundingBoxSizeYPlug = PlugDescriptor("boundingBoxSizeY")
	boundingBoxSizeZ_ : BoundingBoxSizeZPlug = PlugDescriptor("boundingBoxSizeZ")
	bbsz_ : BoundingBoxSizeZPlug = PlugDescriptor("boundingBoxSizeZ")
	node : DagNode = None
	pass
class BoundingBoxPlug(Plug):
	boundingBoxMax_ : BoundingBoxMaxPlug = PlugDescriptor("boundingBoxMax")
	bbmx_ : BoundingBoxMaxPlug = PlugDescriptor("boundingBoxMax")
	boundingBoxMin_ : BoundingBoxMinPlug = PlugDescriptor("boundingBoxMin")
	bbmn_ : BoundingBoxMinPlug = PlugDescriptor("boundingBoxMin")
	boundingBoxSize_ : BoundingBoxSizePlug = PlugDescriptor("boundingBoxSize")
	bbsi_ : BoundingBoxSizePlug = PlugDescriptor("boundingBoxSize")
	node : DagNode = None
	pass
class BoundingBoxCenterXPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : DagNode = None
	pass
class BoundingBoxCenterYPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : DagNode = None
	pass
class BoundingBoxCenterZPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : DagNode = None
	pass
class CenterPlug(Plug):
	boundingBoxCenterX_ : BoundingBoxCenterXPlug = PlugDescriptor("boundingBoxCenterX")
	bcx_ : BoundingBoxCenterXPlug = PlugDescriptor("boundingBoxCenterX")
	boundingBoxCenterY_ : BoundingBoxCenterYPlug = PlugDescriptor("boundingBoxCenterY")
	bcy_ : BoundingBoxCenterYPlug = PlugDescriptor("boundingBoxCenterY")
	boundingBoxCenterZ_ : BoundingBoxCenterZPlug = PlugDescriptor("boundingBoxCenterZ")
	bcz_ : BoundingBoxCenterZPlug = PlugDescriptor("boundingBoxCenterZ")
	node : DagNode = None
	pass
class HideOnPlaybackPlug(Plug):
	parent : DrawOverridePlug = PlugDescriptor("drawOverride")
	node : DagNode = None
	pass
class OverrideColorPlug(Plug):
	parent : DrawOverridePlug = PlugDescriptor("drawOverride")
	node : DagNode = None
	pass
class OverrideColorAPlug(Plug):
	parent : DrawOverridePlug = PlugDescriptor("drawOverride")
	node : DagNode = None
	pass
class OverrideColorRGBPlug(Plug):
	parent : DrawOverridePlug = PlugDescriptor("drawOverride")
	overrideColorB_ : OverrideColorBPlug = PlugDescriptor("overrideColorB")
	ovcb_ : OverrideColorBPlug = PlugDescriptor("overrideColorB")
	overrideColorG_ : OverrideColorGPlug = PlugDescriptor("overrideColorG")
	ovcg_ : OverrideColorGPlug = PlugDescriptor("overrideColorG")
	overrideColorR_ : OverrideColorRPlug = PlugDescriptor("overrideColorR")
	ovcr_ : OverrideColorRPlug = PlugDescriptor("overrideColorR")
	node : DagNode = None
	pass
class OverrideDisplayTypePlug(Plug):
	parent : DrawOverridePlug = PlugDescriptor("drawOverride")
	node : DagNode = None
	pass
class OverrideEnabledPlug(Plug):
	parent : DrawOverridePlug = PlugDescriptor("drawOverride")
	node : DagNode = None
	pass
class OverrideLevelOfDetailPlug(Plug):
	parent : DrawOverridePlug = PlugDescriptor("drawOverride")
	node : DagNode = None
	pass
class OverridePlaybackPlug(Plug):
	parent : DrawOverridePlug = PlugDescriptor("drawOverride")
	node : DagNode = None
	pass
class OverrideRGBColorsPlug(Plug):
	parent : DrawOverridePlug = PlugDescriptor("drawOverride")
	node : DagNode = None
	pass
class OverrideShadingPlug(Plug):
	parent : DrawOverridePlug = PlugDescriptor("drawOverride")
	node : DagNode = None
	pass
class OverrideTexturingPlug(Plug):
	parent : DrawOverridePlug = PlugDescriptor("drawOverride")
	node : DagNode = None
	pass
class OverrideVisibilityPlug(Plug):
	parent : DrawOverridePlug = PlugDescriptor("drawOverride")
	node : DagNode = None
	pass
class DrawOverridePlug(Plug):
	hideOnPlayback_ : HideOnPlaybackPlug = PlugDescriptor("hideOnPlayback")
	hpb_ : HideOnPlaybackPlug = PlugDescriptor("hideOnPlayback")
	overrideColor_ : OverrideColorPlug = PlugDescriptor("overrideColor")
	ovc_ : OverrideColorPlug = PlugDescriptor("overrideColor")
	overrideColorA_ : OverrideColorAPlug = PlugDescriptor("overrideColorA")
	ovca_ : OverrideColorAPlug = PlugDescriptor("overrideColorA")
	overrideColorRGB_ : OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	ovrgb_ : OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	overrideDisplayType_ : OverrideDisplayTypePlug = PlugDescriptor("overrideDisplayType")
	ovdt_ : OverrideDisplayTypePlug = PlugDescriptor("overrideDisplayType")
	overrideEnabled_ : OverrideEnabledPlug = PlugDescriptor("overrideEnabled")
	ove_ : OverrideEnabledPlug = PlugDescriptor("overrideEnabled")
	overrideLevelOfDetail_ : OverrideLevelOfDetailPlug = PlugDescriptor("overrideLevelOfDetail")
	ovlod_ : OverrideLevelOfDetailPlug = PlugDescriptor("overrideLevelOfDetail")
	overridePlayback_ : OverridePlaybackPlug = PlugDescriptor("overridePlayback")
	ovp_ : OverridePlaybackPlug = PlugDescriptor("overridePlayback")
	overrideRGBColors_ : OverrideRGBColorsPlug = PlugDescriptor("overrideRGBColors")
	ovrgbf_ : OverrideRGBColorsPlug = PlugDescriptor("overrideRGBColors")
	overrideShading_ : OverrideShadingPlug = PlugDescriptor("overrideShading")
	ovs_ : OverrideShadingPlug = PlugDescriptor("overrideShading")
	overrideTexturing_ : OverrideTexturingPlug = PlugDescriptor("overrideTexturing")
	ovt_ : OverrideTexturingPlug = PlugDescriptor("overrideTexturing")
	overrideVisibility_ : OverrideVisibilityPlug = PlugDescriptor("overrideVisibility")
	ovv_ : OverrideVisibilityPlug = PlugDescriptor("overrideVisibility")
	node : DagNode = None
	pass
class GhostColorPostBPlug(Plug):
	parent : GhostColorPostPlug = PlugDescriptor("ghostColorPost")
	node : DagNode = None
	pass
class GhostColorPostGPlug(Plug):
	parent : GhostColorPostPlug = PlugDescriptor("ghostColorPost")
	node : DagNode = None
	pass
class GhostColorPostRPlug(Plug):
	parent : GhostColorPostPlug = PlugDescriptor("ghostColorPost")
	node : DagNode = None
	pass
class GhostColorPostPlug(Plug):
	ghostColorPostB_ : GhostColorPostBPlug = PlugDescriptor("ghostColorPostB")
	gab_ : GhostColorPostBPlug = PlugDescriptor("ghostColorPostB")
	ghostColorPostG_ : GhostColorPostGPlug = PlugDescriptor("ghostColorPostG")
	gag_ : GhostColorPostGPlug = PlugDescriptor("ghostColorPostG")
	ghostColorPostR_ : GhostColorPostRPlug = PlugDescriptor("ghostColorPostR")
	gar_ : GhostColorPostRPlug = PlugDescriptor("ghostColorPostR")
	node : DagNode = None
	pass
class GhostColorPreBPlug(Plug):
	parent : GhostColorPrePlug = PlugDescriptor("ghostColorPre")
	node : DagNode = None
	pass
class GhostColorPreGPlug(Plug):
	parent : GhostColorPrePlug = PlugDescriptor("ghostColorPre")
	node : DagNode = None
	pass
class GhostColorPreRPlug(Plug):
	parent : GhostColorPrePlug = PlugDescriptor("ghostColorPre")
	node : DagNode = None
	pass
class GhostColorPrePlug(Plug):
	ghostColorPreB_ : GhostColorPreBPlug = PlugDescriptor("ghostColorPreB")
	gpb_ : GhostColorPreBPlug = PlugDescriptor("ghostColorPreB")
	ghostColorPreG_ : GhostColorPreGPlug = PlugDescriptor("ghostColorPreG")
	gpg_ : GhostColorPreGPlug = PlugDescriptor("ghostColorPreG")
	ghostColorPreR_ : GhostColorPreRPlug = PlugDescriptor("ghostColorPreR")
	grr_ : GhostColorPreRPlug = PlugDescriptor("ghostColorPreR")
	node : DagNode = None
	pass
class GhostPostFramesPlug(Plug):
	parent : GhostCustomStepsPlug = PlugDescriptor("ghostCustomSteps")
	node : DagNode = None
	pass
class GhostPreFramesPlug(Plug):
	parent : GhostCustomStepsPlug = PlugDescriptor("ghostCustomSteps")
	node : DagNode = None
	pass
class GhostsStepPlug(Plug):
	parent : GhostCustomStepsPlug = PlugDescriptor("ghostCustomSteps")
	node : DagNode = None
	pass
class GhostCustomStepsPlug(Plug):
	ghostPostFrames_ : GhostPostFramesPlug = PlugDescriptor("ghostPostFrames")
	gpof_ : GhostPostFramesPlug = PlugDescriptor("ghostPostFrames")
	ghostPreFrames_ : GhostPreFramesPlug = PlugDescriptor("ghostPreFrames")
	gprf_ : GhostPreFramesPlug = PlugDescriptor("ghostPreFrames")
	ghostsStep_ : GhostsStepPlug = PlugDescriptor("ghostsStep")
	gstp_ : GhostsStepPlug = PlugDescriptor("ghostsStep")
	node : DagNode = None
	pass
class GhostDriverPlug(Plug):
	node : DagNode = None
	pass
class GhostFramesPlug(Plug):
	node : DagNode = None
	pass
class GhostFarOpacityPlug(Plug):
	parent : GhostOpacityRangePlug = PlugDescriptor("ghostOpacityRange")
	node : DagNode = None
	pass
class GhostNearOpacityPlug(Plug):
	parent : GhostOpacityRangePlug = PlugDescriptor("ghostOpacityRange")
	node : DagNode = None
	pass
class GhostOpacityRangePlug(Plug):
	ghostFarOpacity_ : GhostFarOpacityPlug = PlugDescriptor("ghostFarOpacity")
	gfro_ : GhostFarOpacityPlug = PlugDescriptor("ghostFarOpacity")
	ghostNearOpacity_ : GhostNearOpacityPlug = PlugDescriptor("ghostNearOpacity")
	gnro_ : GhostNearOpacityPlug = PlugDescriptor("ghostNearOpacity")
	node : DagNode = None
	pass
class GhostUseDriverPlug(Plug):
	node : DagNode = None
	pass
class GhostingPlug(Plug):
	node : DagNode = None
	pass
class GhostingModePlug(Plug):
	node : DagNode = None
	pass
class HiddenInOutlinerPlug(Plug):
	node : DagNode = None
	pass
class ObjectGrpColorPlug(Plug):
	parent : ObjectGroupsPlug = PlugDescriptor("objectGroups")
	node : DagNode = None
	pass
class ObjectGrpCompListPlug(Plug):
	parent : ObjectGroupsPlug = PlugDescriptor("objectGroups")
	node : DagNode = None
	pass
class ObjectGroupsPlug(Plug):
	parent : InstObjGroupsPlug = PlugDescriptor("instObjGroups")
	objectGroupId_ : ObjectGroupIdPlug = PlugDescriptor("objectGroupId")
	gid_ : ObjectGroupIdPlug = PlugDescriptor("objectGroupId")
	objectGrpColor_ : ObjectGrpColorPlug = PlugDescriptor("objectGrpColor")
	gco_ : ObjectGrpColorPlug = PlugDescriptor("objectGrpColor")
	objectGrpCompList_ : ObjectGrpCompListPlug = PlugDescriptor("objectGrpCompList")
	gcl_ : ObjectGrpCompListPlug = PlugDescriptor("objectGrpCompList")
	node : DagNode = None
	pass
class InstObjGroupsPlug(Plug):
	objectGroups_ : ObjectGroupsPlug = PlugDescriptor("objectGroups")
	og_ : ObjectGroupsPlug = PlugDescriptor("objectGroups")
	node : DagNode = None
	pass
class IntermediateObjectPlug(Plug):
	node : DagNode = None
	pass
class InverseMatrixPlug(Plug):
	node : DagNode = None
	pass
class LodVisibilityPlug(Plug):
	node : DagNode = None
	pass
class MatrixPlug(Plug):
	node : DagNode = None
	pass
class ObjectColorPlug(Plug):
	node : DagNode = None
	pass
class ObjectColorBPlug(Plug):
	parent : ObjectColorRGBPlug = PlugDescriptor("objectColorRGB")
	node : DagNode = None
	pass
class ObjectColorGPlug(Plug):
	parent : ObjectColorRGBPlug = PlugDescriptor("objectColorRGB")
	node : DagNode = None
	pass
class ObjectColorRPlug(Plug):
	parent : ObjectColorRGBPlug = PlugDescriptor("objectColorRGB")
	node : DagNode = None
	pass
class ObjectColorRGBPlug(Plug):
	objectColorB_ : ObjectColorBPlug = PlugDescriptor("objectColorB")
	obcb_ : ObjectColorBPlug = PlugDescriptor("objectColorB")
	objectColorG_ : ObjectColorGPlug = PlugDescriptor("objectColorG")
	obcg_ : ObjectColorGPlug = PlugDescriptor("objectColorG")
	objectColorR_ : ObjectColorRPlug = PlugDescriptor("objectColorR")
	obcr_ : ObjectColorRPlug = PlugDescriptor("objectColorR")
	node : DagNode = None
	pass
class ObjectGroupIdPlug(Plug):
	parent : ObjectGroupsPlug = PlugDescriptor("objectGroups")
	node : DagNode = None
	pass
class OutlinerColorBPlug(Plug):
	parent : OutlinerColorPlug = PlugDescriptor("outlinerColor")
	node : DagNode = None
	pass
class OutlinerColorGPlug(Plug):
	parent : OutlinerColorPlug = PlugDescriptor("outlinerColor")
	node : DagNode = None
	pass
class OutlinerColorRPlug(Plug):
	parent : OutlinerColorPlug = PlugDescriptor("outlinerColor")
	node : DagNode = None
	pass
class OutlinerColorPlug(Plug):
	outlinerColorB_ : OutlinerColorBPlug = PlugDescriptor("outlinerColorB")
	oclrb_ : OutlinerColorBPlug = PlugDescriptor("outlinerColorB")
	outlinerColorG_ : OutlinerColorGPlug = PlugDescriptor("outlinerColorG")
	oclrg_ : OutlinerColorGPlug = PlugDescriptor("outlinerColorG")
	outlinerColorR_ : OutlinerColorRPlug = PlugDescriptor("outlinerColorR")
	oclrr_ : OutlinerColorRPlug = PlugDescriptor("outlinerColorR")
	node : DagNode = None
	pass
class OverrideColorBPlug(Plug):
	parent : OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	node : DagNode = None
	pass
class OverrideColorGPlug(Plug):
	parent : OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	node : DagNode = None
	pass
class OverrideColorRPlug(Plug):
	parent : OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	node : DagNode = None
	pass
class ParentInverseMatrixPlug(Plug):
	node : DagNode = None
	pass
class ParentMatrixPlug(Plug):
	node : DagNode = None
	pass
class IdentificationPlug(Plug):
	parent : RenderInfoPlug = PlugDescriptor("renderInfo")
	node : DagNode = None
	pass
class LayerOverrideColorPlug(Plug):
	parent : RenderInfoPlug = PlugDescriptor("renderInfo")
	node : DagNode = None
	pass
class LayerRenderablePlug(Plug):
	parent : RenderInfoPlug = PlugDescriptor("renderInfo")
	node : DagNode = None
	pass
class RenderInfoPlug(Plug):
	identification_ : IdentificationPlug = PlugDescriptor("identification")
	rlid_ : IdentificationPlug = PlugDescriptor("identification")
	layerOverrideColor_ : LayerOverrideColorPlug = PlugDescriptor("layerOverrideColor")
	lovc_ : LayerOverrideColorPlug = PlugDescriptor("layerOverrideColor")
	layerRenderable_ : LayerRenderablePlug = PlugDescriptor("layerRenderable")
	rndr_ : LayerRenderablePlug = PlugDescriptor("layerRenderable")
	node : DagNode = None
	pass
class RenderLayerColorPlug(Plug):
	parent : RenderLayerInfoPlug = PlugDescriptor("renderLayerInfo")
	node : DagNode = None
	pass
class RenderLayerIdPlug(Plug):
	parent : RenderLayerInfoPlug = PlugDescriptor("renderLayerInfo")
	node : DagNode = None
	pass
class RenderLayerRenderablePlug(Plug):
	parent : RenderLayerInfoPlug = PlugDescriptor("renderLayerInfo")
	node : DagNode = None
	pass
class RenderLayerInfoPlug(Plug):
	renderLayerColor_ : RenderLayerColorPlug = PlugDescriptor("renderLayerColor")
	rlc_ : RenderLayerColorPlug = PlugDescriptor("renderLayerColor")
	renderLayerId_ : RenderLayerIdPlug = PlugDescriptor("renderLayerId")
	rli_ : RenderLayerIdPlug = PlugDescriptor("renderLayerId")
	renderLayerRenderable_ : RenderLayerRenderablePlug = PlugDescriptor("renderLayerRenderable")
	rlr_ : RenderLayerRenderablePlug = PlugDescriptor("renderLayerRenderable")
	node : DagNode = None
	pass
class SelectionChildHighlightingPlug(Plug):
	node : DagNode = None
	pass
class TemplatePlug(Plug):
	node : DagNode = None
	pass
class UseObjectColorPlug(Plug):
	node : DagNode = None
	pass
class UseOutlinerColorPlug(Plug):
	node : DagNode = None
	pass
class VisibilityPlug(Plug):
	node : DagNode = None
	pass
class WireColorBPlug(Plug):
	parent : WireColorRGBPlug = PlugDescriptor("wireColorRGB")
	node : DagNode = None
	pass
class WireColorGPlug(Plug):
	parent : WireColorRGBPlug = PlugDescriptor("wireColorRGB")
	node : DagNode = None
	pass
class WireColorRPlug(Plug):
	parent : WireColorRGBPlug = PlugDescriptor("wireColorRGB")
	node : DagNode = None
	pass
class WireColorRGBPlug(Plug):
	wireColorB_ : WireColorBPlug = PlugDescriptor("wireColorB")
	wfcb_ : WireColorBPlug = PlugDescriptor("wireColorB")
	wireColorG_ : WireColorGPlug = PlugDescriptor("wireColorG")
	wfcg_ : WireColorGPlug = PlugDescriptor("wireColorG")
	wireColorR_ : WireColorRPlug = PlugDescriptor("wireColorR")
	wfcr_ : WireColorRPlug = PlugDescriptor("wireColorR")
	node : DagNode = None
	pass
class WorldInverseMatrixPlug(Plug):
	node : DagNode = None
	pass
class WorldMatrixPlug(Plug):
	node : DagNode = None
	pass
# endregion


# define node class
class DagNode(Entity):
	boundingBoxMaxX_ : BoundingBoxMaxXPlug = PlugDescriptor("boundingBoxMaxX")
	boundingBoxMaxY_ : BoundingBoxMaxYPlug = PlugDescriptor("boundingBoxMaxY")
	boundingBoxMaxZ_ : BoundingBoxMaxZPlug = PlugDescriptor("boundingBoxMaxZ")
	boundingBoxMax_ : BoundingBoxMaxPlug = PlugDescriptor("boundingBoxMax")
	boundingBoxMinX_ : BoundingBoxMinXPlug = PlugDescriptor("boundingBoxMinX")
	boundingBoxMinY_ : BoundingBoxMinYPlug = PlugDescriptor("boundingBoxMinY")
	boundingBoxMinZ_ : BoundingBoxMinZPlug = PlugDescriptor("boundingBoxMinZ")
	boundingBoxMin_ : BoundingBoxMinPlug = PlugDescriptor("boundingBoxMin")
	boundingBoxSizeX_ : BoundingBoxSizeXPlug = PlugDescriptor("boundingBoxSizeX")
	boundingBoxSizeY_ : BoundingBoxSizeYPlug = PlugDescriptor("boundingBoxSizeY")
	boundingBoxSizeZ_ : BoundingBoxSizeZPlug = PlugDescriptor("boundingBoxSizeZ")
	boundingBoxSize_ : BoundingBoxSizePlug = PlugDescriptor("boundingBoxSize")
	boundingBox_ : BoundingBoxPlug = PlugDescriptor("boundingBox")
	boundingBoxCenterX_ : BoundingBoxCenterXPlug = PlugDescriptor("boundingBoxCenterX")
	boundingBoxCenterY_ : BoundingBoxCenterYPlug = PlugDescriptor("boundingBoxCenterY")
	boundingBoxCenterZ_ : BoundingBoxCenterZPlug = PlugDescriptor("boundingBoxCenterZ")
	center_ : CenterPlug = PlugDescriptor("center")
	hideOnPlayback_ : HideOnPlaybackPlug = PlugDescriptor("hideOnPlayback")
	overrideColor_ : OverrideColorPlug = PlugDescriptor("overrideColor")
	overrideColorA_ : OverrideColorAPlug = PlugDescriptor("overrideColorA")
	overrideColorRGB_ : OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	overrideDisplayType_ : OverrideDisplayTypePlug = PlugDescriptor("overrideDisplayType")
	overrideEnabled_ : OverrideEnabledPlug = PlugDescriptor("overrideEnabled")
	overrideLevelOfDetail_ : OverrideLevelOfDetailPlug = PlugDescriptor("overrideLevelOfDetail")
	overridePlayback_ : OverridePlaybackPlug = PlugDescriptor("overridePlayback")
	overrideRGBColors_ : OverrideRGBColorsPlug = PlugDescriptor("overrideRGBColors")
	overrideShading_ : OverrideShadingPlug = PlugDescriptor("overrideShading")
	overrideTexturing_ : OverrideTexturingPlug = PlugDescriptor("overrideTexturing")
	overrideVisibility_ : OverrideVisibilityPlug = PlugDescriptor("overrideVisibility")
	drawOverride_ : DrawOverridePlug = PlugDescriptor("drawOverride")
	ghostColorPostB_ : GhostColorPostBPlug = PlugDescriptor("ghostColorPostB")
	ghostColorPostG_ : GhostColorPostGPlug = PlugDescriptor("ghostColorPostG")
	ghostColorPostR_ : GhostColorPostRPlug = PlugDescriptor("ghostColorPostR")
	ghostColorPost_ : GhostColorPostPlug = PlugDescriptor("ghostColorPost")
	ghostColorPreB_ : GhostColorPreBPlug = PlugDescriptor("ghostColorPreB")
	ghostColorPreG_ : GhostColorPreGPlug = PlugDescriptor("ghostColorPreG")
	ghostColorPreR_ : GhostColorPreRPlug = PlugDescriptor("ghostColorPreR")
	ghostColorPre_ : GhostColorPrePlug = PlugDescriptor("ghostColorPre")
	ghostPostFrames_ : GhostPostFramesPlug = PlugDescriptor("ghostPostFrames")
	ghostPreFrames_ : GhostPreFramesPlug = PlugDescriptor("ghostPreFrames")
	ghostsStep_ : GhostsStepPlug = PlugDescriptor("ghostsStep")
	ghostCustomSteps_ : GhostCustomStepsPlug = PlugDescriptor("ghostCustomSteps")
	ghostDriver_ : GhostDriverPlug = PlugDescriptor("ghostDriver")
	ghostFrames_ : GhostFramesPlug = PlugDescriptor("ghostFrames")
	ghostFarOpacity_ : GhostFarOpacityPlug = PlugDescriptor("ghostFarOpacity")
	ghostNearOpacity_ : GhostNearOpacityPlug = PlugDescriptor("ghostNearOpacity")
	ghostOpacityRange_ : GhostOpacityRangePlug = PlugDescriptor("ghostOpacityRange")
	ghostUseDriver_ : GhostUseDriverPlug = PlugDescriptor("ghostUseDriver")
	ghosting_ : GhostingPlug = PlugDescriptor("ghosting")
	ghostingMode_ : GhostingModePlug = PlugDescriptor("ghostingMode")
	hiddenInOutliner_ : HiddenInOutlinerPlug = PlugDescriptor("hiddenInOutliner")
	objectGrpColor_ : ObjectGrpColorPlug = PlugDescriptor("objectGrpColor")
	objectGrpCompList_ : ObjectGrpCompListPlug = PlugDescriptor("objectGrpCompList")
	objectGroups_ : ObjectGroupsPlug = PlugDescriptor("objectGroups")
	instObjGroups_ : InstObjGroupsPlug = PlugDescriptor("instObjGroups")
	intermediateObject_ : IntermediateObjectPlug = PlugDescriptor("intermediateObject")
	inverseMatrix_ : InverseMatrixPlug = PlugDescriptor("inverseMatrix")
	lodVisibility_ : LodVisibilityPlug = PlugDescriptor("lodVisibility")
	matrix_ : MatrixPlug = PlugDescriptor("matrix")
	objectColor_ : ObjectColorPlug = PlugDescriptor("objectColor")
	objectColorB_ : ObjectColorBPlug = PlugDescriptor("objectColorB")
	objectColorG_ : ObjectColorGPlug = PlugDescriptor("objectColorG")
	objectColorR_ : ObjectColorRPlug = PlugDescriptor("objectColorR")
	objectColorRGB_ : ObjectColorRGBPlug = PlugDescriptor("objectColorRGB")
	objectGroupId_ : ObjectGroupIdPlug = PlugDescriptor("objectGroupId")
	outlinerColorB_ : OutlinerColorBPlug = PlugDescriptor("outlinerColorB")
	outlinerColorG_ : OutlinerColorGPlug = PlugDescriptor("outlinerColorG")
	outlinerColorR_ : OutlinerColorRPlug = PlugDescriptor("outlinerColorR")
	outlinerColor_ : OutlinerColorPlug = PlugDescriptor("outlinerColor")
	overrideColorB_ : OverrideColorBPlug = PlugDescriptor("overrideColorB")
	overrideColorG_ : OverrideColorGPlug = PlugDescriptor("overrideColorG")
	overrideColorR_ : OverrideColorRPlug = PlugDescriptor("overrideColorR")
	parentInverseMatrix_ : ParentInverseMatrixPlug = PlugDescriptor("parentInverseMatrix")
	parentMatrix_ : ParentMatrixPlug = PlugDescriptor("parentMatrix")
	identification_ : IdentificationPlug = PlugDescriptor("identification")
	layerOverrideColor_ : LayerOverrideColorPlug = PlugDescriptor("layerOverrideColor")
	layerRenderable_ : LayerRenderablePlug = PlugDescriptor("layerRenderable")
	renderInfo_ : RenderInfoPlug = PlugDescriptor("renderInfo")
	renderLayerColor_ : RenderLayerColorPlug = PlugDescriptor("renderLayerColor")
	renderLayerId_ : RenderLayerIdPlug = PlugDescriptor("renderLayerId")
	renderLayerRenderable_ : RenderLayerRenderablePlug = PlugDescriptor("renderLayerRenderable")
	renderLayerInfo_ : RenderLayerInfoPlug = PlugDescriptor("renderLayerInfo")
	selectionChildHighlighting_ : SelectionChildHighlightingPlug = PlugDescriptor("selectionChildHighlighting")
	template_ : TemplatePlug = PlugDescriptor("template")
	useObjectColor_ : UseObjectColorPlug = PlugDescriptor("useObjectColor")
	useOutlinerColor_ : UseOutlinerColorPlug = PlugDescriptor("useOutlinerColor")
	visibility_ : VisibilityPlug = PlugDescriptor("visibility")
	wireColorB_ : WireColorBPlug = PlugDescriptor("wireColorB")
	wireColorG_ : WireColorGPlug = PlugDescriptor("wireColorG")
	wireColorR_ : WireColorRPlug = PlugDescriptor("wireColorR")
	wireColorRGB_ : WireColorRGBPlug = PlugDescriptor("wireColorRGB")
	worldInverseMatrix_ : WorldInverseMatrixPlug = PlugDescriptor("worldInverseMatrix")
	worldMatrix_ : WorldMatrixPlug = PlugDescriptor("worldMatrix")

	# node attributes

	typeName = "dagNode"
	apiTypeInt = 107
	apiTypeStr = "kDagNode"
	typeIdInt = 1145128782
	MFnCls = om.MFnDagNode
	pass

