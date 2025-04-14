

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core import cmds, om, WN, WPlug
#from wpm.core.node.base import WN, PlugDescriptor, Plug
from wpm.core.node.base import *

# add any extra imports
from .dagNode import DagNode

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node:Transform

	pass


class BlackBoxPlug(Plug):
	node:Transform

	pass


class BorderConnectionsPlug(Plug):
	node:Transform

	pass


class BoundingBoxMaxXPlug(Plug):
	parent:BoundingBoxMaxPlug = PlugDescriptor("boundingBoxMax")
	node:Transform

	pass


class BoundingBoxMaxYPlug(Plug):
	parent:BoundingBoxMaxPlug = PlugDescriptor("boundingBoxMax")
	node:Transform

	pass


class BoundingBoxMaxZPlug(Plug):
	parent:BoundingBoxMaxPlug = PlugDescriptor("boundingBoxMax")
	node:Transform

	pass


class BoundingBoxMaxPlug(Plug):
	parent:BoundingBoxPlug = PlugDescriptor("boundingBox")
	boundingBoxMaxX_:BoundingBoxMaxXPlug = PlugDescriptor("boundingBoxMaxX")
	bbxx_:BoundingBoxMaxXPlug = PlugDescriptor("boundingBoxMaxX")
	boundingBoxMaxY_:BoundingBoxMaxYPlug = PlugDescriptor("boundingBoxMaxY")
	bbxy_:BoundingBoxMaxYPlug = PlugDescriptor("boundingBoxMaxY")
	boundingBoxMaxZ_:BoundingBoxMaxZPlug = PlugDescriptor("boundingBoxMaxZ")
	bbxz_:BoundingBoxMaxZPlug = PlugDescriptor("boundingBoxMaxZ")
	node:Transform

	pass


class BoundingBoxMinXPlug(Plug):
	parent:BoundingBoxMinPlug = PlugDescriptor("boundingBoxMin")
	node:Transform

	pass


class BoundingBoxMinYPlug(Plug):
	parent:BoundingBoxMinPlug = PlugDescriptor("boundingBoxMin")
	node:Transform

	pass


class BoundingBoxMinZPlug(Plug):
	parent:BoundingBoxMinPlug = PlugDescriptor("boundingBoxMin")
	node:Transform

	pass


class BoundingBoxMinPlug(Plug):
	parent:BoundingBoxPlug = PlugDescriptor("boundingBox")
	boundingBoxMinX_:BoundingBoxMinXPlug = PlugDescriptor("boundingBoxMinX")
	bbnx_:BoundingBoxMinXPlug = PlugDescriptor("boundingBoxMinX")
	boundingBoxMinY_:BoundingBoxMinYPlug = PlugDescriptor("boundingBoxMinY")
	bbny_:BoundingBoxMinYPlug = PlugDescriptor("boundingBoxMinY")
	boundingBoxMinZ_:BoundingBoxMinZPlug = PlugDescriptor("boundingBoxMinZ")
	bbnz_:BoundingBoxMinZPlug = PlugDescriptor("boundingBoxMinZ")
	node:Transform

	pass


class BoundingBoxSizeXPlug(Plug):
	parent:BoundingBoxSizePlug = PlugDescriptor("boundingBoxSize")
	node:Transform

	pass


class BoundingBoxSizeYPlug(Plug):
	parent:BoundingBoxSizePlug = PlugDescriptor("boundingBoxSize")
	node:Transform

	pass


class BoundingBoxSizeZPlug(Plug):
	parent:BoundingBoxSizePlug = PlugDescriptor("boundingBoxSize")
	node:Transform

	pass


class BoundingBoxSizePlug(Plug):
	parent:BoundingBoxPlug = PlugDescriptor("boundingBox")
	boundingBoxSizeX_:BoundingBoxSizeXPlug = PlugDescriptor("boundingBoxSizeX")
	bbsx_:BoundingBoxSizeXPlug = PlugDescriptor("boundingBoxSizeX")
	boundingBoxSizeY_:BoundingBoxSizeYPlug = PlugDescriptor("boundingBoxSizeY")
	bbsy_:BoundingBoxSizeYPlug = PlugDescriptor("boundingBoxSizeY")
	boundingBoxSizeZ_:BoundingBoxSizeZPlug = PlugDescriptor("boundingBoxSizeZ")
	bbsz_:BoundingBoxSizeZPlug = PlugDescriptor("boundingBoxSizeZ")
	node:Transform

	pass


class BoundingBoxPlug(Plug):
	boundingBoxMax_:BoundingBoxMaxPlug = PlugDescriptor("boundingBoxMax")
	bbmx_:BoundingBoxMaxPlug = PlugDescriptor("boundingBoxMax")
	boundingBoxMin_:BoundingBoxMinPlug = PlugDescriptor("boundingBoxMin")
	bbmn_:BoundingBoxMinPlug = PlugDescriptor("boundingBoxMin")
	boundingBoxSize_:BoundingBoxSizePlug = PlugDescriptor("boundingBoxSize")
	bbsi_:BoundingBoxSizePlug = PlugDescriptor("boundingBoxSize")
	node:Transform

	pass


class CachingPlug(Plug):
	node:Transform

	pass


class BoundingBoxCenterXPlug(Plug):
	parent:CenterPlug = PlugDescriptor("center")
	node:Transform

	pass


class BoundingBoxCenterYPlug(Plug):
	parent:CenterPlug = PlugDescriptor("center")
	node:Transform

	pass


class BoundingBoxCenterZPlug(Plug):
	parent:CenterPlug = PlugDescriptor("center")
	node:Transform

	pass


class CenterPlug(Plug):
	boundingBoxCenterX_:BoundingBoxCenterXPlug = PlugDescriptor("boundingBoxCenterX")
	bcx_:BoundingBoxCenterXPlug = PlugDescriptor("boundingBoxCenterX")
	boundingBoxCenterY_:BoundingBoxCenterYPlug = PlugDescriptor("boundingBoxCenterY")
	bcy_:BoundingBoxCenterYPlug = PlugDescriptor("boundingBoxCenterY")
	boundingBoxCenterZ_:BoundingBoxCenterZPlug = PlugDescriptor("boundingBoxCenterZ")
	bcz_:BoundingBoxCenterZPlug = PlugDescriptor("boundingBoxCenterZ")
	node:Transform

	pass


class ContainerTypePlug(Plug):
	node:Transform

	pass


class CreationDatePlug(Plug):
	node:Transform

	pass


class CreatorPlug(Plug):
	node:Transform

	pass


class CustomTreatmentPlug(Plug):
	node:Transform

	pass


class DagLocalInverseMatrixPlug(Plug):
	node:Transform

	pass


class DagLocalMatrixPlug(Plug):
	node:Transform

	pass


class DisplayHandlePlug(Plug):
	node:Transform

	pass


class DisplayLocalAxisPlug(Plug):
	node:Transform

	pass


class DisplayRotatePivotPlug(Plug):
	node:Transform

	pass


class DisplayScalePivotPlug(Plug):
	node:Transform

	pass


class HideOnPlaybackPlug(Plug):
	parent:DrawOverridePlug = PlugDescriptor("drawOverride")
	node:Transform

	pass


class OverrideColorPlug(Plug):
	parent:DrawOverridePlug = PlugDescriptor("drawOverride")
	node:Transform

	pass


class OverrideColorAPlug(Plug):
	parent:DrawOverridePlug = PlugDescriptor("drawOverride")
	node:Transform

	pass


class OverrideColorRGBPlug(Plug):
	parent:DrawOverridePlug = PlugDescriptor("drawOverride")
	overrideColorB_:OverrideColorBPlug = PlugDescriptor("overrideColorB")
	ovcb_:OverrideColorBPlug = PlugDescriptor("overrideColorB")
	overrideColorG_:OverrideColorGPlug = PlugDescriptor("overrideColorG")
	ovcg_:OverrideColorGPlug = PlugDescriptor("overrideColorG")
	overrideColorR_:OverrideColorRPlug = PlugDescriptor("overrideColorR")
	ovcr_:OverrideColorRPlug = PlugDescriptor("overrideColorR")
	node:Transform

	pass


class OverrideDisplayTypePlug(Plug):
	parent:DrawOverridePlug = PlugDescriptor("drawOverride")
	node:Transform

	pass


class OverrideEnabledPlug(Plug):
	parent:DrawOverridePlug = PlugDescriptor("drawOverride")
	node:Transform

	pass


class OverrideLevelOfDetailPlug(Plug):
	parent:DrawOverridePlug = PlugDescriptor("drawOverride")
	node:Transform

	pass


class OverridePlaybackPlug(Plug):
	parent:DrawOverridePlug = PlugDescriptor("drawOverride")
	node:Transform

	pass


class OverrideRGBColorsPlug(Plug):
	parent:DrawOverridePlug = PlugDescriptor("drawOverride")
	node:Transform

	pass


class OverrideShadingPlug(Plug):
	parent:DrawOverridePlug = PlugDescriptor("drawOverride")
	node:Transform

	pass


class OverrideTexturingPlug(Plug):
	parent:DrawOverridePlug = PlugDescriptor("drawOverride")
	node:Transform

	pass


class OverrideVisibilityPlug(Plug):
	parent:DrawOverridePlug = PlugDescriptor("drawOverride")
	node:Transform

	pass


class DrawOverridePlug(Plug):
	hideOnPlayback_:HideOnPlaybackPlug = PlugDescriptor("hideOnPlayback")
	hpb_:HideOnPlaybackPlug = PlugDescriptor("hideOnPlayback")
	overrideColor_:OverrideColorPlug = PlugDescriptor("overrideColor")
	ovc_:OverrideColorPlug = PlugDescriptor("overrideColor")
	overrideColorA_:OverrideColorAPlug = PlugDescriptor("overrideColorA")
	ovca_:OverrideColorAPlug = PlugDescriptor("overrideColorA")
	overrideColorRGB_:OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	ovrgb_:OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	overrideDisplayType_:OverrideDisplayTypePlug = PlugDescriptor("overrideDisplayType")
	ovdt_:OverrideDisplayTypePlug = PlugDescriptor("overrideDisplayType")
	overrideEnabled_:OverrideEnabledPlug = PlugDescriptor("overrideEnabled")
	ove_:OverrideEnabledPlug = PlugDescriptor("overrideEnabled")
	overrideLevelOfDetail_:OverrideLevelOfDetailPlug = PlugDescriptor("overrideLevelOfDetail")
	ovlod_:OverrideLevelOfDetailPlug = PlugDescriptor("overrideLevelOfDetail")
	overridePlayback_:OverridePlaybackPlug = PlugDescriptor("overridePlayback")
	ovp_:OverridePlaybackPlug = PlugDescriptor("overridePlayback")
	overrideRGBColors_:OverrideRGBColorsPlug = PlugDescriptor("overrideRGBColors")
	ovrgbf_:OverrideRGBColorsPlug = PlugDescriptor("overrideRGBColors")
	overrideShading_:OverrideShadingPlug = PlugDescriptor("overrideShading")
	ovs_:OverrideShadingPlug = PlugDescriptor("overrideShading")
	overrideTexturing_:OverrideTexturingPlug = PlugDescriptor("overrideTexturing")
	ovt_:OverrideTexturingPlug = PlugDescriptor("overrideTexturing")
	overrideVisibility_:OverrideVisibilityPlug = PlugDescriptor("overrideVisibility")
	ovv_:OverrideVisibilityPlug = PlugDescriptor("overrideVisibility")
	node:Transform

	pass


class DynamicsPlug(Plug):
	node:Transform

	pass


class FrozenPlug(Plug):
	node:Transform

	pass


class GeometryPlug(Plug):
	node:Transform

	pass


class GhostColorPostBPlug(Plug):
	parent:GhostColorPostPlug = PlugDescriptor("ghostColorPost")
	node:Transform

	pass


class GhostColorPostGPlug(Plug):
	parent:GhostColorPostPlug = PlugDescriptor("ghostColorPost")
	node:Transform

	pass


class GhostColorPostRPlug(Plug):
	parent:GhostColorPostPlug = PlugDescriptor("ghostColorPost")
	node:Transform

	pass


class GhostColorPostPlug(Plug):
	ghostColorPostB_:GhostColorPostBPlug = PlugDescriptor("ghostColorPostB")
	gab_:GhostColorPostBPlug = PlugDescriptor("ghostColorPostB")
	ghostColorPostG_:GhostColorPostGPlug = PlugDescriptor("ghostColorPostG")
	gag_:GhostColorPostGPlug = PlugDescriptor("ghostColorPostG")
	ghostColorPostR_:GhostColorPostRPlug = PlugDescriptor("ghostColorPostR")
	gar_:GhostColorPostRPlug = PlugDescriptor("ghostColorPostR")
	node:Transform

	pass


class GhostColorPreBPlug(Plug):
	parent:GhostColorPrePlug = PlugDescriptor("ghostColorPre")
	node:Transform

	pass


class GhostColorPreGPlug(Plug):
	parent:GhostColorPrePlug = PlugDescriptor("ghostColorPre")
	node:Transform

	pass


class GhostColorPreRPlug(Plug):
	parent:GhostColorPrePlug = PlugDescriptor("ghostColorPre")
	node:Transform

	pass


class GhostColorPrePlug(Plug):
	ghostColorPreB_:GhostColorPreBPlug = PlugDescriptor("ghostColorPreB")
	gpb_:GhostColorPreBPlug = PlugDescriptor("ghostColorPreB")
	ghostColorPreG_:GhostColorPreGPlug = PlugDescriptor("ghostColorPreG")
	gpg_:GhostColorPreGPlug = PlugDescriptor("ghostColorPreG")
	ghostColorPreR_:GhostColorPreRPlug = PlugDescriptor("ghostColorPreR")
	grr_:GhostColorPreRPlug = PlugDescriptor("ghostColorPreR")
	node:Transform

	pass


class GhostPostFramesPlug(Plug):
	parent:GhostCustomStepsPlug = PlugDescriptor("ghostCustomSteps")
	node:Transform

	pass


class GhostPreFramesPlug(Plug):
	parent:GhostCustomStepsPlug = PlugDescriptor("ghostCustomSteps")
	node:Transform

	pass


class GhostsStepPlug(Plug):
	parent:GhostCustomStepsPlug = PlugDescriptor("ghostCustomSteps")
	node:Transform

	pass


class GhostCustomStepsPlug(Plug):
	ghostPostFrames_:GhostPostFramesPlug = PlugDescriptor("ghostPostFrames")
	gpof_:GhostPostFramesPlug = PlugDescriptor("ghostPostFrames")
	ghostPreFrames_:GhostPreFramesPlug = PlugDescriptor("ghostPreFrames")
	gprf_:GhostPreFramesPlug = PlugDescriptor("ghostPreFrames")
	ghostsStep_:GhostsStepPlug = PlugDescriptor("ghostsStep")
	gstp_:GhostsStepPlug = PlugDescriptor("ghostsStep")
	node:Transform

	pass


class GhostDriverPlug(Plug):
	node:Transform

	pass


class GhostFramesPlug(Plug):
	node:Transform

	pass


class GhostFarOpacityPlug(Plug):
	parent:GhostOpacityRangePlug = PlugDescriptor("ghostOpacityRange")
	node:Transform

	pass


class GhostNearOpacityPlug(Plug):
	parent:GhostOpacityRangePlug = PlugDescriptor("ghostOpacityRange")
	node:Transform

	pass


class GhostOpacityRangePlug(Plug):
	ghostFarOpacity_:GhostFarOpacityPlug = PlugDescriptor("ghostFarOpacity")
	gfro_:GhostFarOpacityPlug = PlugDescriptor("ghostFarOpacity")
	ghostNearOpacity_:GhostNearOpacityPlug = PlugDescriptor("ghostNearOpacity")
	gnro_:GhostNearOpacityPlug = PlugDescriptor("ghostNearOpacity")
	node:Transform

	pass


class GhostUseDriverPlug(Plug):
	node:Transform

	pass


class GhostingPlug(Plug):
	node:Transform

	pass


class GhostingModePlug(Plug):
	node:Transform

	pass


class HiddenInOutlinerPlug(Plug):
	node:Transform

	pass


class HyperLayoutPlug(Plug):
	node:Transform

	pass


class IconNamePlug(Plug):
	node:Transform

	pass


class InheritsTransformPlug(Plug):
	node:Transform

	pass


class ObjectGrpColorPlug(Plug):
	parent:ObjectGroupsPlug = PlugDescriptor("objectGroups")
	node:Transform

	pass


class ObjectGrpCompListPlug(Plug):
	parent:ObjectGroupsPlug = PlugDescriptor("objectGroups")
	node:Transform

	pass


class ObjectGroupsPlug(Plug):
	parent:InstObjGroupsPlug = PlugDescriptor("instObjGroups")
	objectGroupId_:ObjectGroupIdPlug = PlugDescriptor("objectGroupId")
	gid_:ObjectGroupIdPlug = PlugDescriptor("objectGroupId")
	objectGrpColor_:ObjectGrpColorPlug = PlugDescriptor("objectGrpColor")
	gco_:ObjectGrpColorPlug = PlugDescriptor("objectGrpColor")
	objectGrpCompList_:ObjectGrpCompListPlug = PlugDescriptor("objectGrpCompList")
	gcl_:ObjectGrpCompListPlug = PlugDescriptor("objectGrpCompList")
	node:Transform

	pass


class InstObjGroupsPlug(Plug):
	objectGroups_:ObjectGroupsPlug = PlugDescriptor("objectGroups")
	og_:ObjectGroupsPlug = PlugDescriptor("objectGroups")
	node:Transform

	pass


class IntermediateObjectPlug(Plug):
	node:Transform

	pass


class InverseMatrixPlug(Plug):
	node:Transform

	pass


class IsCollapsedPlug(Plug):
	node:Transform

	pass


class IsHierarchicalConnectionPlug(Plug):
	node:Transform

	pass


class IsHistoricallyInterestingPlug(Plug):
	node:Transform

	pass


class LodVisibilityPlug(Plug):
	node:Transform

	pass


class MatrixPlug(Plug):
	node:Transform

	pass


class MaxRotXLimitPlug(Plug):
	parent:MaxRotLimitPlug = PlugDescriptor("maxRotLimit")
	node:Transform

	pass


class MaxRotYLimitPlug(Plug):
	parent:MaxRotLimitPlug = PlugDescriptor("maxRotLimit")
	node:Transform

	pass


class MaxRotZLimitPlug(Plug):
	parent:MaxRotLimitPlug = PlugDescriptor("maxRotLimit")
	node:Transform

	pass


class MaxRotLimitPlug(Plug):
	maxRotXLimit_:MaxRotXLimitPlug = PlugDescriptor("maxRotXLimit")
	xrxl_:MaxRotXLimitPlug = PlugDescriptor("maxRotXLimit")
	maxRotYLimit_:MaxRotYLimitPlug = PlugDescriptor("maxRotYLimit")
	xryl_:MaxRotYLimitPlug = PlugDescriptor("maxRotYLimit")
	maxRotZLimit_:MaxRotZLimitPlug = PlugDescriptor("maxRotZLimit")
	xrzl_:MaxRotZLimitPlug = PlugDescriptor("maxRotZLimit")
	node:Transform

	pass


class MaxRotXLimitEnablePlug(Plug):
	parent:MaxRotLimitEnablePlug = PlugDescriptor("maxRotLimitEnable")
	node:Transform

	pass


class MaxRotYLimitEnablePlug(Plug):
	parent:MaxRotLimitEnablePlug = PlugDescriptor("maxRotLimitEnable")
	node:Transform

	pass


class MaxRotZLimitEnablePlug(Plug):
	parent:MaxRotLimitEnablePlug = PlugDescriptor("maxRotLimitEnable")
	node:Transform

	pass


class MaxRotLimitEnablePlug(Plug):
	maxRotXLimitEnable_:MaxRotXLimitEnablePlug = PlugDescriptor("maxRotXLimitEnable")
	xrxe_:MaxRotXLimitEnablePlug = PlugDescriptor("maxRotXLimitEnable")
	maxRotYLimitEnable_:MaxRotYLimitEnablePlug = PlugDescriptor("maxRotYLimitEnable")
	xrye_:MaxRotYLimitEnablePlug = PlugDescriptor("maxRotYLimitEnable")
	maxRotZLimitEnable_:MaxRotZLimitEnablePlug = PlugDescriptor("maxRotZLimitEnable")
	xrze_:MaxRotZLimitEnablePlug = PlugDescriptor("maxRotZLimitEnable")
	node:Transform

	pass


class MaxScaleXLimitPlug(Plug):
	parent:MaxScaleLimitPlug = PlugDescriptor("maxScaleLimit")
	node:Transform

	pass


class MaxScaleYLimitPlug(Plug):
	parent:MaxScaleLimitPlug = PlugDescriptor("maxScaleLimit")
	node:Transform

	pass


class MaxScaleZLimitPlug(Plug):
	parent:MaxScaleLimitPlug = PlugDescriptor("maxScaleLimit")
	node:Transform

	pass


class MaxScaleLimitPlug(Plug):
	maxScaleXLimit_:MaxScaleXLimitPlug = PlugDescriptor("maxScaleXLimit")
	xsxl_:MaxScaleXLimitPlug = PlugDescriptor("maxScaleXLimit")
	maxScaleYLimit_:MaxScaleYLimitPlug = PlugDescriptor("maxScaleYLimit")
	xsyl_:MaxScaleYLimitPlug = PlugDescriptor("maxScaleYLimit")
	maxScaleZLimit_:MaxScaleZLimitPlug = PlugDescriptor("maxScaleZLimit")
	xszl_:MaxScaleZLimitPlug = PlugDescriptor("maxScaleZLimit")
	node:Transform

	pass


class MaxScaleXLimitEnablePlug(Plug):
	parent:MaxScaleLimitEnablePlug = PlugDescriptor("maxScaleLimitEnable")
	node:Transform

	pass


class MaxScaleYLimitEnablePlug(Plug):
	parent:MaxScaleLimitEnablePlug = PlugDescriptor("maxScaleLimitEnable")
	node:Transform

	pass


class MaxScaleZLimitEnablePlug(Plug):
	parent:MaxScaleLimitEnablePlug = PlugDescriptor("maxScaleLimitEnable")
	node:Transform

	pass


class MaxScaleLimitEnablePlug(Plug):
	maxScaleXLimitEnable_:MaxScaleXLimitEnablePlug = PlugDescriptor("maxScaleXLimitEnable")
	xsxe_:MaxScaleXLimitEnablePlug = PlugDescriptor("maxScaleXLimitEnable")
	maxScaleYLimitEnable_:MaxScaleYLimitEnablePlug = PlugDescriptor("maxScaleYLimitEnable")
	xsye_:MaxScaleYLimitEnablePlug = PlugDescriptor("maxScaleYLimitEnable")
	maxScaleZLimitEnable_:MaxScaleZLimitEnablePlug = PlugDescriptor("maxScaleZLimitEnable")
	xsze_:MaxScaleZLimitEnablePlug = PlugDescriptor("maxScaleZLimitEnable")
	node:Transform

	pass


class MaxTransXLimitPlug(Plug):
	parent:MaxTransLimitPlug = PlugDescriptor("maxTransLimit")
	node:Transform

	pass


class MaxTransYLimitPlug(Plug):
	parent:MaxTransLimitPlug = PlugDescriptor("maxTransLimit")
	node:Transform

	pass


class MaxTransZLimitPlug(Plug):
	parent:MaxTransLimitPlug = PlugDescriptor("maxTransLimit")
	node:Transform

	pass


class MaxTransLimitPlug(Plug):
	maxTransXLimit_:MaxTransXLimitPlug = PlugDescriptor("maxTransXLimit")
	xtxl_:MaxTransXLimitPlug = PlugDescriptor("maxTransXLimit")
	maxTransYLimit_:MaxTransYLimitPlug = PlugDescriptor("maxTransYLimit")
	xtyl_:MaxTransYLimitPlug = PlugDescriptor("maxTransYLimit")
	maxTransZLimit_:MaxTransZLimitPlug = PlugDescriptor("maxTransZLimit")
	xtzl_:MaxTransZLimitPlug = PlugDescriptor("maxTransZLimit")
	node:Transform

	pass


class MaxTransXLimitEnablePlug(Plug):
	parent:MaxTransLimitEnablePlug = PlugDescriptor("maxTransLimitEnable")
	node:Transform

	pass


class MaxTransYLimitEnablePlug(Plug):
	parent:MaxTransLimitEnablePlug = PlugDescriptor("maxTransLimitEnable")
	node:Transform

	pass


class MaxTransZLimitEnablePlug(Plug):
	parent:MaxTransLimitEnablePlug = PlugDescriptor("maxTransLimitEnable")
	node:Transform

	pass


class MaxTransLimitEnablePlug(Plug):
	maxTransXLimitEnable_:MaxTransXLimitEnablePlug = PlugDescriptor("maxTransXLimitEnable")
	xtxe_:MaxTransXLimitEnablePlug = PlugDescriptor("maxTransXLimitEnable")
	maxTransYLimitEnable_:MaxTransYLimitEnablePlug = PlugDescriptor("maxTransYLimitEnable")
	xtye_:MaxTransYLimitEnablePlug = PlugDescriptor("maxTransYLimitEnable")
	maxTransZLimitEnable_:MaxTransZLimitEnablePlug = PlugDescriptor("maxTransZLimitEnable")
	xtze_:MaxTransZLimitEnablePlug = PlugDescriptor("maxTransZLimitEnable")
	node:Transform

	pass


class MessagePlug(Plug):
	node:Transform

	pass


class MinRotXLimitPlug(Plug):
	parent:MinRotLimitPlug = PlugDescriptor("minRotLimit")
	node:Transform

	pass


class MinRotYLimitPlug(Plug):
	parent:MinRotLimitPlug = PlugDescriptor("minRotLimit")
	node:Transform

	pass


class MinRotZLimitPlug(Plug):
	parent:MinRotLimitPlug = PlugDescriptor("minRotLimit")
	node:Transform

	pass


class MinRotLimitPlug(Plug):
	minRotXLimit_:MinRotXLimitPlug = PlugDescriptor("minRotXLimit")
	mrxl_:MinRotXLimitPlug = PlugDescriptor("minRotXLimit")
	minRotYLimit_:MinRotYLimitPlug = PlugDescriptor("minRotYLimit")
	mryl_:MinRotYLimitPlug = PlugDescriptor("minRotYLimit")
	minRotZLimit_:MinRotZLimitPlug = PlugDescriptor("minRotZLimit")
	mrzl_:MinRotZLimitPlug = PlugDescriptor("minRotZLimit")
	node:Transform

	pass


class MinRotXLimitEnablePlug(Plug):
	parent:MinRotLimitEnablePlug = PlugDescriptor("minRotLimitEnable")
	node:Transform

	pass


class MinRotYLimitEnablePlug(Plug):
	parent:MinRotLimitEnablePlug = PlugDescriptor("minRotLimitEnable")
	node:Transform

	pass


class MinRotZLimitEnablePlug(Plug):
	parent:MinRotLimitEnablePlug = PlugDescriptor("minRotLimitEnable")
	node:Transform

	pass


class MinRotLimitEnablePlug(Plug):
	minRotXLimitEnable_:MinRotXLimitEnablePlug = PlugDescriptor("minRotXLimitEnable")
	mrxe_:MinRotXLimitEnablePlug = PlugDescriptor("minRotXLimitEnable")
	minRotYLimitEnable_:MinRotYLimitEnablePlug = PlugDescriptor("minRotYLimitEnable")
	mrye_:MinRotYLimitEnablePlug = PlugDescriptor("minRotYLimitEnable")
	minRotZLimitEnable_:MinRotZLimitEnablePlug = PlugDescriptor("minRotZLimitEnable")
	mrze_:MinRotZLimitEnablePlug = PlugDescriptor("minRotZLimitEnable")
	node:Transform

	pass


class MinScaleXLimitPlug(Plug):
	parent:MinScaleLimitPlug = PlugDescriptor("minScaleLimit")
	node:Transform

	pass


class MinScaleYLimitPlug(Plug):
	parent:MinScaleLimitPlug = PlugDescriptor("minScaleLimit")
	node:Transform

	pass


class MinScaleZLimitPlug(Plug):
	parent:MinScaleLimitPlug = PlugDescriptor("minScaleLimit")
	node:Transform

	pass


class MinScaleLimitPlug(Plug):
	minScaleXLimit_:MinScaleXLimitPlug = PlugDescriptor("minScaleXLimit")
	msxl_:MinScaleXLimitPlug = PlugDescriptor("minScaleXLimit")
	minScaleYLimit_:MinScaleYLimitPlug = PlugDescriptor("minScaleYLimit")
	msyl_:MinScaleYLimitPlug = PlugDescriptor("minScaleYLimit")
	minScaleZLimit_:MinScaleZLimitPlug = PlugDescriptor("minScaleZLimit")
	mszl_:MinScaleZLimitPlug = PlugDescriptor("minScaleZLimit")
	node:Transform

	pass


class MinScaleXLimitEnablePlug(Plug):
	parent:MinScaleLimitEnablePlug = PlugDescriptor("minScaleLimitEnable")
	node:Transform

	pass


class MinScaleYLimitEnablePlug(Plug):
	parent:MinScaleLimitEnablePlug = PlugDescriptor("minScaleLimitEnable")
	node:Transform

	pass


class MinScaleZLimitEnablePlug(Plug):
	parent:MinScaleLimitEnablePlug = PlugDescriptor("minScaleLimitEnable")
	node:Transform

	pass


class MinScaleLimitEnablePlug(Plug):
	minScaleXLimitEnable_:MinScaleXLimitEnablePlug = PlugDescriptor("minScaleXLimitEnable")
	msxe_:MinScaleXLimitEnablePlug = PlugDescriptor("minScaleXLimitEnable")
	minScaleYLimitEnable_:MinScaleYLimitEnablePlug = PlugDescriptor("minScaleYLimitEnable")
	msye_:MinScaleYLimitEnablePlug = PlugDescriptor("minScaleYLimitEnable")
	minScaleZLimitEnable_:MinScaleZLimitEnablePlug = PlugDescriptor("minScaleZLimitEnable")
	msze_:MinScaleZLimitEnablePlug = PlugDescriptor("minScaleZLimitEnable")
	node:Transform

	pass


class MinTransXLimitPlug(Plug):
	parent:MinTransLimitPlug = PlugDescriptor("minTransLimit")
	node:Transform

	pass


class MinTransYLimitPlug(Plug):
	parent:MinTransLimitPlug = PlugDescriptor("minTransLimit")
	node:Transform

	pass


class MinTransZLimitPlug(Plug):
	parent:MinTransLimitPlug = PlugDescriptor("minTransLimit")
	node:Transform

	pass


class MinTransLimitPlug(Plug):
	minTransXLimit_:MinTransXLimitPlug = PlugDescriptor("minTransXLimit")
	mtxl_:MinTransXLimitPlug = PlugDescriptor("minTransXLimit")
	minTransYLimit_:MinTransYLimitPlug = PlugDescriptor("minTransYLimit")
	mtyl_:MinTransYLimitPlug = PlugDescriptor("minTransYLimit")
	minTransZLimit_:MinTransZLimitPlug = PlugDescriptor("minTransZLimit")
	mtzl_:MinTransZLimitPlug = PlugDescriptor("minTransZLimit")
	node:Transform

	pass


class MinTransXLimitEnablePlug(Plug):
	parent:MinTransLimitEnablePlug = PlugDescriptor("minTransLimitEnable")
	node:Transform

	pass


class MinTransYLimitEnablePlug(Plug):
	parent:MinTransLimitEnablePlug = PlugDescriptor("minTransLimitEnable")
	node:Transform

	pass


class MinTransZLimitEnablePlug(Plug):
	parent:MinTransLimitEnablePlug = PlugDescriptor("minTransLimitEnable")
	node:Transform

	pass


class MinTransLimitEnablePlug(Plug):
	minTransXLimitEnable_:MinTransXLimitEnablePlug = PlugDescriptor("minTransXLimitEnable")
	mtxe_:MinTransXLimitEnablePlug = PlugDescriptor("minTransXLimitEnable")
	minTransYLimitEnable_:MinTransYLimitEnablePlug = PlugDescriptor("minTransYLimitEnable")
	mtye_:MinTransYLimitEnablePlug = PlugDescriptor("minTransYLimitEnable")
	minTransZLimitEnable_:MinTransZLimitEnablePlug = PlugDescriptor("minTransZLimitEnable")
	mtze_:MinTransZLimitEnablePlug = PlugDescriptor("minTransZLimitEnable")
	node:Transform

	pass


class NodeStatePlug(Plug):
	node:Transform

	pass


class ObjectColorPlug(Plug):
	node:Transform

	pass


class ObjectColorBPlug(Plug):
	parent:ObjectColorRGBPlug = PlugDescriptor("objectColorRGB")
	node:Transform

	pass


class ObjectColorGPlug(Plug):
	parent:ObjectColorRGBPlug = PlugDescriptor("objectColorRGB")
	node:Transform

	pass


class ObjectColorRPlug(Plug):
	parent:ObjectColorRGBPlug = PlugDescriptor("objectColorRGB")
	node:Transform

	pass


class ObjectColorRGBPlug(Plug):
	objectColorB_:ObjectColorBPlug = PlugDescriptor("objectColorB")
	obcb_:ObjectColorBPlug = PlugDescriptor("objectColorB")
	objectColorG_:ObjectColorGPlug = PlugDescriptor("objectColorG")
	obcg_:ObjectColorGPlug = PlugDescriptor("objectColorG")
	objectColorR_:ObjectColorRPlug = PlugDescriptor("objectColorR")
	obcr_:ObjectColorRPlug = PlugDescriptor("objectColorR")
	node:Transform

	pass


class ObjectGroupIdPlug(Plug):
	parent:ObjectGroupsPlug = PlugDescriptor("objectGroups")
	node:Transform

	pass


class OffsetParentMatrixPlug(Plug):
	node:Transform

	pass


class OutlinerColorBPlug(Plug):
	parent:OutlinerColorPlug = PlugDescriptor("outlinerColor")
	node:Transform

	pass


class OutlinerColorGPlug(Plug):
	parent:OutlinerColorPlug = PlugDescriptor("outlinerColor")
	node:Transform

	pass


class OutlinerColorRPlug(Plug):
	parent:OutlinerColorPlug = PlugDescriptor("outlinerColor")
	node:Transform

	pass


class OutlinerColorPlug(Plug):
	outlinerColorB_:OutlinerColorBPlug = PlugDescriptor("outlinerColorB")
	oclrb_:OutlinerColorBPlug = PlugDescriptor("outlinerColorB")
	outlinerColorG_:OutlinerColorGPlug = PlugDescriptor("outlinerColorG")
	oclrg_:OutlinerColorGPlug = PlugDescriptor("outlinerColorG")
	outlinerColorR_:OutlinerColorRPlug = PlugDescriptor("outlinerColorR")
	oclrr_:OutlinerColorRPlug = PlugDescriptor("outlinerColorR")
	node:Transform

	pass


class OverrideColorBPlug(Plug):
	parent:OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	node:Transform

	pass


class OverrideColorGPlug(Plug):
	parent:OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	node:Transform

	pass


class OverrideColorRPlug(Plug):
	parent:OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	node:Transform

	pass


class ParentInverseMatrixPlug(Plug):
	node:Transform

	pass


class ParentMatrixPlug(Plug):
	node:Transform

	pass


class IsHierarchicalNodePlug(Plug):
	parent:PublishedNodeInfoPlug = PlugDescriptor("publishedNodeInfo")
	node:Transform

	pass


class PublishedNodePlug(Plug):
	parent:PublishedNodeInfoPlug = PlugDescriptor("publishedNodeInfo")
	node:Transform

	pass


class PublishedNodeTypePlug(Plug):
	parent:PublishedNodeInfoPlug = PlugDescriptor("publishedNodeInfo")
	node:Transform

	pass


class PublishedNodeInfoPlug(Plug):
	isHierarchicalNode_:IsHierarchicalNodePlug = PlugDescriptor("isHierarchicalNode")
	ihn_:IsHierarchicalNodePlug = PlugDescriptor("isHierarchicalNode")
	publishedNode_:PublishedNodePlug = PlugDescriptor("publishedNode")
	pnod_:PublishedNodePlug = PlugDescriptor("publishedNode")
	publishedNodeType_:PublishedNodeTypePlug = PlugDescriptor("publishedNodeType")
	pntp_:PublishedNodeTypePlug = PlugDescriptor("publishedNodeType")
	node:Transform

	pass


class IdentificationPlug(Plug):
	parent:RenderInfoPlug = PlugDescriptor("renderInfo")
	node:Transform

	pass


class LayerOverrideColorPlug(Plug):
	parent:RenderInfoPlug = PlugDescriptor("renderInfo")
	node:Transform

	pass


class LayerRenderablePlug(Plug):
	parent:RenderInfoPlug = PlugDescriptor("renderInfo")
	node:Transform

	pass


class RenderInfoPlug(Plug):
	identification_:IdentificationPlug = PlugDescriptor("identification")
	rlid_:IdentificationPlug = PlugDescriptor("identification")
	layerOverrideColor_:LayerOverrideColorPlug = PlugDescriptor("layerOverrideColor")
	lovc_:LayerOverrideColorPlug = PlugDescriptor("layerOverrideColor")
	layerRenderable_:LayerRenderablePlug = PlugDescriptor("layerRenderable")
	rndr_:LayerRenderablePlug = PlugDescriptor("layerRenderable")
	node:Transform

	pass


class RenderLayerColorPlug(Plug):
	parent:RenderLayerInfoPlug = PlugDescriptor("renderLayerInfo")
	node:Transform

	pass


class RenderLayerIdPlug(Plug):
	parent:RenderLayerInfoPlug = PlugDescriptor("renderLayerInfo")
	node:Transform

	pass


class RenderLayerRenderablePlug(Plug):
	parent:RenderLayerInfoPlug = PlugDescriptor("renderLayerInfo")
	node:Transform

	pass


class RenderLayerInfoPlug(Plug):
	renderLayerColor_:RenderLayerColorPlug = PlugDescriptor("renderLayerColor")
	rlc_:RenderLayerColorPlug = PlugDescriptor("renderLayerColor")
	renderLayerId_:RenderLayerIdPlug = PlugDescriptor("renderLayerId")
	rli_:RenderLayerIdPlug = PlugDescriptor("renderLayerId")
	renderLayerRenderable_:RenderLayerRenderablePlug = PlugDescriptor("renderLayerRenderable")
	rlr_:RenderLayerRenderablePlug = PlugDescriptor("renderLayerRenderable")
	node:Transform

	pass


class RmbCommandPlug(Plug):
	node:Transform

	pass


class RotateXPlug(Plug):
	parent:RotatePlug = PlugDescriptor("rotate")
	node:Transform

	pass


class RotateYPlug(Plug):
	parent:RotatePlug = PlugDescriptor("rotate")
	node:Transform

	pass


class RotateZPlug(Plug):
	parent:RotatePlug = PlugDescriptor("rotate")
	node:Transform

	pass


class RotatePlug(Plug):
	rotateX_:RotateXPlug = PlugDescriptor("rotateX")
	rx_:RotateXPlug = PlugDescriptor("rotateX")
	rotateY_:RotateYPlug = PlugDescriptor("rotateY")
	ry_:RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_:RotateZPlug = PlugDescriptor("rotateZ")
	rz_:RotateZPlug = PlugDescriptor("rotateZ")
	node:Transform

	pass


class RotateAxisXPlug(Plug):
	parent:RotateAxisPlug = PlugDescriptor("rotateAxis")
	node:Transform

	pass


class RotateAxisYPlug(Plug):
	parent:RotateAxisPlug = PlugDescriptor("rotateAxis")
	node:Transform

	pass


class RotateAxisZPlug(Plug):
	parent:RotateAxisPlug = PlugDescriptor("rotateAxis")
	node:Transform

	pass


class RotateAxisPlug(Plug):
	rotateAxisX_:RotateAxisXPlug = PlugDescriptor("rotateAxisX")
	rax_:RotateAxisXPlug = PlugDescriptor("rotateAxisX")
	rotateAxisY_:RotateAxisYPlug = PlugDescriptor("rotateAxisY")
	ray_:RotateAxisYPlug = PlugDescriptor("rotateAxisY")
	rotateAxisZ_:RotateAxisZPlug = PlugDescriptor("rotateAxisZ")
	raz_:RotateAxisZPlug = PlugDescriptor("rotateAxisZ")
	node:Transform

	pass


class RotateOrderPlug(Plug):
	node:Transform

	pass


class RotatePivotXPlug(Plug):
	parent:RotatePivotPlug = PlugDescriptor("rotatePivot")
	node:Transform

	pass


class RotatePivotYPlug(Plug):
	parent:RotatePivotPlug = PlugDescriptor("rotatePivot")
	node:Transform

	pass


class RotatePivotZPlug(Plug):
	parent:RotatePivotPlug = PlugDescriptor("rotatePivot")
	node:Transform

	pass


class RotatePivotPlug(Plug):
	rotatePivotX_:RotatePivotXPlug = PlugDescriptor("rotatePivotX")
	rpx_:RotatePivotXPlug = PlugDescriptor("rotatePivotX")
	rotatePivotY_:RotatePivotYPlug = PlugDescriptor("rotatePivotY")
	rpy_:RotatePivotYPlug = PlugDescriptor("rotatePivotY")
	rotatePivotZ_:RotatePivotZPlug = PlugDescriptor("rotatePivotZ")
	rpz_:RotatePivotZPlug = PlugDescriptor("rotatePivotZ")
	node:Transform

	pass


class RotatePivotTranslateXPlug(Plug):
	parent:RotatePivotTranslatePlug = PlugDescriptor("rotatePivotTranslate")
	node:Transform

	pass


class RotatePivotTranslateYPlug(Plug):
	parent:RotatePivotTranslatePlug = PlugDescriptor("rotatePivotTranslate")
	node:Transform

	pass


class RotatePivotTranslateZPlug(Plug):
	parent:RotatePivotTranslatePlug = PlugDescriptor("rotatePivotTranslate")
	node:Transform

	pass


class RotatePivotTranslatePlug(Plug):
	rotatePivotTranslateX_:RotatePivotTranslateXPlug = PlugDescriptor("rotatePivotTranslateX")
	rptx_:RotatePivotTranslateXPlug = PlugDescriptor("rotatePivotTranslateX")
	rotatePivotTranslateY_:RotatePivotTranslateYPlug = PlugDescriptor("rotatePivotTranslateY")
	rpty_:RotatePivotTranslateYPlug = PlugDescriptor("rotatePivotTranslateY")
	rotatePivotTranslateZ_:RotatePivotTranslateZPlug = PlugDescriptor("rotatePivotTranslateZ")
	rptz_:RotatePivotTranslateZPlug = PlugDescriptor("rotatePivotTranslateZ")
	node:Transform

	pass


class RotateQuaternionWPlug(Plug):
	parent:RotateQuaternionPlug = PlugDescriptor("rotateQuaternion")
	node:Transform

	pass


class RotateQuaternionXPlug(Plug):
	parent:RotateQuaternionPlug = PlugDescriptor("rotateQuaternion")
	node:Transform

	pass


class RotateQuaternionYPlug(Plug):
	parent:RotateQuaternionPlug = PlugDescriptor("rotateQuaternion")
	node:Transform

	pass


class RotateQuaternionZPlug(Plug):
	parent:RotateQuaternionPlug = PlugDescriptor("rotateQuaternion")
	node:Transform

	pass


class RotateQuaternionPlug(Plug):
	rotateQuaternionW_:RotateQuaternionWPlug = PlugDescriptor("rotateQuaternionW")
	rqw_:RotateQuaternionWPlug = PlugDescriptor("rotateQuaternionW")
	rotateQuaternionX_:RotateQuaternionXPlug = PlugDescriptor("rotateQuaternionX")
	rqx_:RotateQuaternionXPlug = PlugDescriptor("rotateQuaternionX")
	rotateQuaternionY_:RotateQuaternionYPlug = PlugDescriptor("rotateQuaternionY")
	rqy_:RotateQuaternionYPlug = PlugDescriptor("rotateQuaternionY")
	rotateQuaternionZ_:RotateQuaternionZPlug = PlugDescriptor("rotateQuaternionZ")
	rqz_:RotateQuaternionZPlug = PlugDescriptor("rotateQuaternionZ")
	node:Transform

	pass


class RotationInterpolationPlug(Plug):
	node:Transform

	pass


class ScaleXPlug(Plug):
	parent:ScalePlug = PlugDescriptor("scale")
	node:Transform

	pass


class ScaleYPlug(Plug):
	parent:ScalePlug = PlugDescriptor("scale")
	node:Transform

	pass


class ScaleZPlug(Plug):
	parent:ScalePlug = PlugDescriptor("scale")
	node:Transform

	pass


class ScalePlug(Plug):
	scaleX_:ScaleXPlug = PlugDescriptor("scaleX")
	sx_:ScaleXPlug = PlugDescriptor("scaleX")
	scaleY_:ScaleYPlug = PlugDescriptor("scaleY")
	sy_:ScaleYPlug = PlugDescriptor("scaleY")
	scaleZ_:ScaleZPlug = PlugDescriptor("scaleZ")
	sz_:ScaleZPlug = PlugDescriptor("scaleZ")
	node:Transform

	pass


class ScalePivotXPlug(Plug):
	parent:ScalePivotPlug = PlugDescriptor("scalePivot")
	node:Transform

	pass


class ScalePivotYPlug(Plug):
	parent:ScalePivotPlug = PlugDescriptor("scalePivot")
	node:Transform

	pass


class ScalePivotZPlug(Plug):
	parent:ScalePivotPlug = PlugDescriptor("scalePivot")
	node:Transform

	pass


class ScalePivotPlug(Plug):
	scalePivotX_:ScalePivotXPlug = PlugDescriptor("scalePivotX")
	spx_:ScalePivotXPlug = PlugDescriptor("scalePivotX")
	scalePivotY_:ScalePivotYPlug = PlugDescriptor("scalePivotY")
	spy_:ScalePivotYPlug = PlugDescriptor("scalePivotY")
	scalePivotZ_:ScalePivotZPlug = PlugDescriptor("scalePivotZ")
	spz_:ScalePivotZPlug = PlugDescriptor("scalePivotZ")
	node:Transform

	pass


class ScalePivotTranslateXPlug(Plug):
	parent:ScalePivotTranslatePlug = PlugDescriptor("scalePivotTranslate")
	node:Transform

	pass


class ScalePivotTranslateYPlug(Plug):
	parent:ScalePivotTranslatePlug = PlugDescriptor("scalePivotTranslate")
	node:Transform

	pass


class ScalePivotTranslateZPlug(Plug):
	parent:ScalePivotTranslatePlug = PlugDescriptor("scalePivotTranslate")
	node:Transform

	pass


class ScalePivotTranslatePlug(Plug):
	scalePivotTranslateX_:ScalePivotTranslateXPlug = PlugDescriptor("scalePivotTranslateX")
	sptx_:ScalePivotTranslateXPlug = PlugDescriptor("scalePivotTranslateX")
	scalePivotTranslateY_:ScalePivotTranslateYPlug = PlugDescriptor("scalePivotTranslateY")
	spty_:ScalePivotTranslateYPlug = PlugDescriptor("scalePivotTranslateY")
	scalePivotTranslateZ_:ScalePivotTranslateZPlug = PlugDescriptor("scalePivotTranslateZ")
	sptz_:ScalePivotTranslateZPlug = PlugDescriptor("scalePivotTranslateZ")
	node:Transform

	pass


class SelectHandleXPlug(Plug):
	parent:SelectHandlePlug = PlugDescriptor("selectHandle")
	node:Transform

	pass


class SelectHandleYPlug(Plug):
	parent:SelectHandlePlug = PlugDescriptor("selectHandle")
	node:Transform

	pass


class SelectHandleZPlug(Plug):
	parent:SelectHandlePlug = PlugDescriptor("selectHandle")
	node:Transform

	pass


class SelectHandlePlug(Plug):
	selectHandleX_:SelectHandleXPlug = PlugDescriptor("selectHandleX")
	hdlx_:SelectHandleXPlug = PlugDescriptor("selectHandleX")
	selectHandleY_:SelectHandleYPlug = PlugDescriptor("selectHandleY")
	hdly_:SelectHandleYPlug = PlugDescriptor("selectHandleY")
	selectHandleZ_:SelectHandleZPlug = PlugDescriptor("selectHandleZ")
	hdlz_:SelectHandleZPlug = PlugDescriptor("selectHandleZ")
	node:Transform

	pass


class SelectionChildHighlightingPlug(Plug):
	node:Transform

	pass


class ShearXYPlug(Plug):
	parent:ShearPlug = PlugDescriptor("shear")
	node:Transform

	pass


class ShearXZPlug(Plug):
	parent:ShearPlug = PlugDescriptor("shear")
	node:Transform

	pass


class ShearYZPlug(Plug):
	parent:ShearPlug = PlugDescriptor("shear")
	node:Transform

	pass


class ShearPlug(Plug):
	shearXY_:ShearXYPlug = PlugDescriptor("shearXY")
	shxy_:ShearXYPlug = PlugDescriptor("shearXY")
	shearXZ_:ShearXZPlug = PlugDescriptor("shearXZ")
	shxz_:ShearXZPlug = PlugDescriptor("shearXZ")
	shearYZ_:ShearYZPlug = PlugDescriptor("shearYZ")
	shyz_:ShearYZPlug = PlugDescriptor("shearYZ")
	node:Transform

	pass


class ShowManipDefaultPlug(Plug):
	node:Transform

	pass


class SpecifiedManipLocationPlug(Plug):
	node:Transform

	pass


class TemplatePlug(Plug):
	node:Transform

	pass


class TemplateNamePlug(Plug):
	node:Transform

	pass


class TemplatePathPlug(Plug):
	node:Transform

	pass


class TemplateVersionPlug(Plug):
	node:Transform

	pass


class TransMinusRotatePivotXPlug(Plug):
	parent:TransMinusRotatePivotPlug = PlugDescriptor("transMinusRotatePivot")
	node:Transform

	pass


class TransMinusRotatePivotYPlug(Plug):
	parent:TransMinusRotatePivotPlug = PlugDescriptor("transMinusRotatePivot")
	node:Transform

	pass


class TransMinusRotatePivotZPlug(Plug):
	parent:TransMinusRotatePivotPlug = PlugDescriptor("transMinusRotatePivot")
	node:Transform

	pass


class TransMinusRotatePivotPlug(Plug):
	transMinusRotatePivotX_:TransMinusRotatePivotXPlug = PlugDescriptor("transMinusRotatePivotX")
	tmrx_:TransMinusRotatePivotXPlug = PlugDescriptor("transMinusRotatePivotX")
	transMinusRotatePivotY_:TransMinusRotatePivotYPlug = PlugDescriptor("transMinusRotatePivotY")
	tmry_:TransMinusRotatePivotYPlug = PlugDescriptor("transMinusRotatePivotY")
	transMinusRotatePivotZ_:TransMinusRotatePivotZPlug = PlugDescriptor("transMinusRotatePivotZ")
	tmrz_:TransMinusRotatePivotZPlug = PlugDescriptor("transMinusRotatePivotZ")
	node:Transform

	pass


class TranslateXPlug(Plug):
	parent:TranslatePlug = PlugDescriptor("translate")
	node:Transform

	pass


class TranslateYPlug(Plug):
	parent:TranslatePlug = PlugDescriptor("translate")
	node:Transform

	pass


class TranslateZPlug(Plug):
	parent:TranslatePlug = PlugDescriptor("translate")
	node:Transform

	pass


class TranslatePlug(Plug):
	translateX_:TranslateXPlug = PlugDescriptor("translateX")
	tx_:TranslateXPlug = PlugDescriptor("translateX")
	translateY_:TranslateYPlug = PlugDescriptor("translateY")
	ty_:TranslateYPlug = PlugDescriptor("translateY")
	translateZ_:TranslateZPlug = PlugDescriptor("translateZ")
	tz_:TranslateZPlug = PlugDescriptor("translateZ")
	node:Transform

	pass


class UiTreatmentPlug(Plug):
	node:Transform

	pass


class UseObjectColorPlug(Plug):
	node:Transform

	pass


class UseOutlinerColorPlug(Plug):
	node:Transform

	pass


class ViewModePlug(Plug):
	node:Transform

	pass


class ViewNamePlug(Plug):
	node:Transform

	pass


class VisibilityPlug(Plug):
	node:Transform

	pass


class WireColorBPlug(Plug):
	parent:WireColorRGBPlug = PlugDescriptor("wireColorRGB")
	node:Transform

	pass


class WireColorGPlug(Plug):
	parent:WireColorRGBPlug = PlugDescriptor("wireColorRGB")
	node:Transform

	pass


class WireColorRPlug(Plug):
	parent:WireColorRGBPlug = PlugDescriptor("wireColorRGB")
	node:Transform

	pass


class WireColorRGBPlug(Plug):
	wireColorB_:WireColorBPlug = PlugDescriptor("wireColorB")
	wfcb_:WireColorBPlug = PlugDescriptor("wireColorB")
	wireColorG_:WireColorGPlug = PlugDescriptor("wireColorG")
	wfcg_:WireColorGPlug = PlugDescriptor("wireColorG")
	wireColorR_:WireColorRPlug = PlugDescriptor("wireColorR")
	wfcr_:WireColorRPlug = PlugDescriptor("wireColorR")
	node:Transform

	pass


class WorldInverseMatrixPlug(Plug):
	node:Transform

	pass


class WorldMatrixPlug(Plug):
	node:Transform

	pass


class XformMatrixPlug(Plug):
	node:Transform

	pass

# endregion


# define node class
class Transform(DagNode):
	binMembership_:BinMembershipPlug = PlugDescriptor("binMembership")
	blackBox_:BlackBoxPlug = PlugDescriptor("blackBox")
	borderConnections_:BorderConnectionsPlug = PlugDescriptor("borderConnections")
	boundingBoxMaxX_:BoundingBoxMaxXPlug = PlugDescriptor("boundingBoxMaxX")
	boundingBoxMaxY_:BoundingBoxMaxYPlug = PlugDescriptor("boundingBoxMaxY")
	boundingBoxMaxZ_:BoundingBoxMaxZPlug = PlugDescriptor("boundingBoxMaxZ")
	boundingBoxMax_:BoundingBoxMaxPlug = PlugDescriptor("boundingBoxMax")
	boundingBoxMinX_:BoundingBoxMinXPlug = PlugDescriptor("boundingBoxMinX")
	boundingBoxMinY_:BoundingBoxMinYPlug = PlugDescriptor("boundingBoxMinY")
	boundingBoxMinZ_:BoundingBoxMinZPlug = PlugDescriptor("boundingBoxMinZ")
	boundingBoxMin_:BoundingBoxMinPlug = PlugDescriptor("boundingBoxMin")
	boundingBoxSizeX_:BoundingBoxSizeXPlug = PlugDescriptor("boundingBoxSizeX")
	boundingBoxSizeY_:BoundingBoxSizeYPlug = PlugDescriptor("boundingBoxSizeY")
	boundingBoxSizeZ_:BoundingBoxSizeZPlug = PlugDescriptor("boundingBoxSizeZ")
	boundingBoxSize_:BoundingBoxSizePlug = PlugDescriptor("boundingBoxSize")
	boundingBox_:BoundingBoxPlug = PlugDescriptor("boundingBox")
	caching_:CachingPlug = PlugDescriptor("caching")
	boundingBoxCenterX_:BoundingBoxCenterXPlug = PlugDescriptor("boundingBoxCenterX")
	boundingBoxCenterY_:BoundingBoxCenterYPlug = PlugDescriptor("boundingBoxCenterY")
	boundingBoxCenterZ_:BoundingBoxCenterZPlug = PlugDescriptor("boundingBoxCenterZ")
	center_:CenterPlug = PlugDescriptor("center")
	containerType_:ContainerTypePlug = PlugDescriptor("containerType")
	creationDate_:CreationDatePlug = PlugDescriptor("creationDate")
	creator_:CreatorPlug = PlugDescriptor("creator")
	customTreatment_:CustomTreatmentPlug = PlugDescriptor("customTreatment")
	dagLocalInverseMatrix_:DagLocalInverseMatrixPlug = PlugDescriptor("dagLocalInverseMatrix")
	dagLocalMatrix_:DagLocalMatrixPlug = PlugDescriptor("dagLocalMatrix")
	displayHandle_:DisplayHandlePlug = PlugDescriptor("displayHandle")
	displayLocalAxis_:DisplayLocalAxisPlug = PlugDescriptor("displayLocalAxis")
	displayRotatePivot_:DisplayRotatePivotPlug = PlugDescriptor("displayRotatePivot")
	displayScalePivot_:DisplayScalePivotPlug = PlugDescriptor("displayScalePivot")
	hideOnPlayback_:HideOnPlaybackPlug = PlugDescriptor("hideOnPlayback")
	overrideColor_:OverrideColorPlug = PlugDescriptor("overrideColor")
	overrideColorA_:OverrideColorAPlug = PlugDescriptor("overrideColorA")
	overrideColorRGB_:OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	overrideDisplayType_:OverrideDisplayTypePlug = PlugDescriptor("overrideDisplayType")
	overrideEnabled_:OverrideEnabledPlug = PlugDescriptor("overrideEnabled")
	overrideLevelOfDetail_:OverrideLevelOfDetailPlug = PlugDescriptor("overrideLevelOfDetail")
	overridePlayback_:OverridePlaybackPlug = PlugDescriptor("overridePlayback")
	overrideRGBColors_:OverrideRGBColorsPlug = PlugDescriptor("overrideRGBColors")
	overrideShading_:OverrideShadingPlug = PlugDescriptor("overrideShading")
	overrideTexturing_:OverrideTexturingPlug = PlugDescriptor("overrideTexturing")
	overrideVisibility_:OverrideVisibilityPlug = PlugDescriptor("overrideVisibility")
	drawOverride_:DrawOverridePlug = PlugDescriptor("drawOverride")
	dynamics_:DynamicsPlug = PlugDescriptor("dynamics")
	frozen_:FrozenPlug = PlugDescriptor("frozen")
	geometry_:GeometryPlug = PlugDescriptor("geometry")
	ghostColorPostB_:GhostColorPostBPlug = PlugDescriptor("ghostColorPostB")
	ghostColorPostG_:GhostColorPostGPlug = PlugDescriptor("ghostColorPostG")
	ghostColorPostR_:GhostColorPostRPlug = PlugDescriptor("ghostColorPostR")
	ghostColorPost_:GhostColorPostPlug = PlugDescriptor("ghostColorPost")
	ghostColorPreB_:GhostColorPreBPlug = PlugDescriptor("ghostColorPreB")
	ghostColorPreG_:GhostColorPreGPlug = PlugDescriptor("ghostColorPreG")
	ghostColorPreR_:GhostColorPreRPlug = PlugDescriptor("ghostColorPreR")
	ghostColorPre_:GhostColorPrePlug = PlugDescriptor("ghostColorPre")
	ghostPostFrames_:GhostPostFramesPlug = PlugDescriptor("ghostPostFrames")
	ghostPreFrames_:GhostPreFramesPlug = PlugDescriptor("ghostPreFrames")
	ghostsStep_:GhostsStepPlug = PlugDescriptor("ghostsStep")
	ghostCustomSteps_:GhostCustomStepsPlug = PlugDescriptor("ghostCustomSteps")
	ghostDriver_:GhostDriverPlug = PlugDescriptor("ghostDriver")
	ghostFrames_:GhostFramesPlug = PlugDescriptor("ghostFrames")
	ghostFarOpacity_:GhostFarOpacityPlug = PlugDescriptor("ghostFarOpacity")
	ghostNearOpacity_:GhostNearOpacityPlug = PlugDescriptor("ghostNearOpacity")
	ghostOpacityRange_:GhostOpacityRangePlug = PlugDescriptor("ghostOpacityRange")
	ghostUseDriver_:GhostUseDriverPlug = PlugDescriptor("ghostUseDriver")
	ghosting_:GhostingPlug = PlugDescriptor("ghosting")
	ghostingMode_:GhostingModePlug = PlugDescriptor("ghostingMode")
	hiddenInOutliner_:HiddenInOutlinerPlug = PlugDescriptor("hiddenInOutliner")
	hyperLayout_:HyperLayoutPlug = PlugDescriptor("hyperLayout")
	iconName_:IconNamePlug = PlugDescriptor("iconName")
	inheritsTransform_:InheritsTransformPlug = PlugDescriptor("inheritsTransform")
	objectGrpColor_:ObjectGrpColorPlug = PlugDescriptor("objectGrpColor")
	objectGrpCompList_:ObjectGrpCompListPlug = PlugDescriptor("objectGrpCompList")
	objectGroups_:ObjectGroupsPlug = PlugDescriptor("objectGroups")
	instObjGroups_:InstObjGroupsPlug = PlugDescriptor("instObjGroups")
	intermediateObject_:IntermediateObjectPlug = PlugDescriptor("intermediateObject")
	inverseMatrix_:InverseMatrixPlug = PlugDescriptor("inverseMatrix")
	isCollapsed_:IsCollapsedPlug = PlugDescriptor("isCollapsed")
	isHierarchicalConnection_:IsHierarchicalConnectionPlug = PlugDescriptor("isHierarchicalConnection")
	isHistoricallyInteresting_:IsHistoricallyInterestingPlug = PlugDescriptor("isHistoricallyInteresting")
	lodVisibility_:LodVisibilityPlug = PlugDescriptor("lodVisibility")
	matrix_:MatrixPlug = PlugDescriptor("matrix")
	maxRotXLimit_:MaxRotXLimitPlug = PlugDescriptor("maxRotXLimit")
	maxRotYLimit_:MaxRotYLimitPlug = PlugDescriptor("maxRotYLimit")
	maxRotZLimit_:MaxRotZLimitPlug = PlugDescriptor("maxRotZLimit")
	maxRotLimit_:MaxRotLimitPlug = PlugDescriptor("maxRotLimit")
	maxRotXLimitEnable_:MaxRotXLimitEnablePlug = PlugDescriptor("maxRotXLimitEnable")
	maxRotYLimitEnable_:MaxRotYLimitEnablePlug = PlugDescriptor("maxRotYLimitEnable")
	maxRotZLimitEnable_:MaxRotZLimitEnablePlug = PlugDescriptor("maxRotZLimitEnable")
	maxRotLimitEnable_:MaxRotLimitEnablePlug = PlugDescriptor("maxRotLimitEnable")
	maxScaleXLimit_:MaxScaleXLimitPlug = PlugDescriptor("maxScaleXLimit")
	maxScaleYLimit_:MaxScaleYLimitPlug = PlugDescriptor("maxScaleYLimit")
	maxScaleZLimit_:MaxScaleZLimitPlug = PlugDescriptor("maxScaleZLimit")
	maxScaleLimit_:MaxScaleLimitPlug = PlugDescriptor("maxScaleLimit")
	maxScaleXLimitEnable_:MaxScaleXLimitEnablePlug = PlugDescriptor("maxScaleXLimitEnable")
	maxScaleYLimitEnable_:MaxScaleYLimitEnablePlug = PlugDescriptor("maxScaleYLimitEnable")
	maxScaleZLimitEnable_:MaxScaleZLimitEnablePlug = PlugDescriptor("maxScaleZLimitEnable")
	maxScaleLimitEnable_:MaxScaleLimitEnablePlug = PlugDescriptor("maxScaleLimitEnable")
	maxTransXLimit_:MaxTransXLimitPlug = PlugDescriptor("maxTransXLimit")
	maxTransYLimit_:MaxTransYLimitPlug = PlugDescriptor("maxTransYLimit")
	maxTransZLimit_:MaxTransZLimitPlug = PlugDescriptor("maxTransZLimit")
	maxTransLimit_:MaxTransLimitPlug = PlugDescriptor("maxTransLimit")
	maxTransXLimitEnable_:MaxTransXLimitEnablePlug = PlugDescriptor("maxTransXLimitEnable")
	maxTransYLimitEnable_:MaxTransYLimitEnablePlug = PlugDescriptor("maxTransYLimitEnable")
	maxTransZLimitEnable_:MaxTransZLimitEnablePlug = PlugDescriptor("maxTransZLimitEnable")
	maxTransLimitEnable_:MaxTransLimitEnablePlug = PlugDescriptor("maxTransLimitEnable")
	message_:MessagePlug = PlugDescriptor("message")
	minRotXLimit_:MinRotXLimitPlug = PlugDescriptor("minRotXLimit")
	minRotYLimit_:MinRotYLimitPlug = PlugDescriptor("minRotYLimit")
	minRotZLimit_:MinRotZLimitPlug = PlugDescriptor("minRotZLimit")
	minRotLimit_:MinRotLimitPlug = PlugDescriptor("minRotLimit")
	minRotXLimitEnable_:MinRotXLimitEnablePlug = PlugDescriptor("minRotXLimitEnable")
	minRotYLimitEnable_:MinRotYLimitEnablePlug = PlugDescriptor("minRotYLimitEnable")
	minRotZLimitEnable_:MinRotZLimitEnablePlug = PlugDescriptor("minRotZLimitEnable")
	minRotLimitEnable_:MinRotLimitEnablePlug = PlugDescriptor("minRotLimitEnable")
	minScaleXLimit_:MinScaleXLimitPlug = PlugDescriptor("minScaleXLimit")
	minScaleYLimit_:MinScaleYLimitPlug = PlugDescriptor("minScaleYLimit")
	minScaleZLimit_:MinScaleZLimitPlug = PlugDescriptor("minScaleZLimit")
	minScaleLimit_:MinScaleLimitPlug = PlugDescriptor("minScaleLimit")
	minScaleXLimitEnable_:MinScaleXLimitEnablePlug = PlugDescriptor("minScaleXLimitEnable")
	minScaleYLimitEnable_:MinScaleYLimitEnablePlug = PlugDescriptor("minScaleYLimitEnable")
	minScaleZLimitEnable_:MinScaleZLimitEnablePlug = PlugDescriptor("minScaleZLimitEnable")
	minScaleLimitEnable_:MinScaleLimitEnablePlug = PlugDescriptor("minScaleLimitEnable")
	minTransXLimit_:MinTransXLimitPlug = PlugDescriptor("minTransXLimit")
	minTransYLimit_:MinTransYLimitPlug = PlugDescriptor("minTransYLimit")
	minTransZLimit_:MinTransZLimitPlug = PlugDescriptor("minTransZLimit")
	minTransLimit_:MinTransLimitPlug = PlugDescriptor("minTransLimit")
	minTransXLimitEnable_:MinTransXLimitEnablePlug = PlugDescriptor("minTransXLimitEnable")
	minTransYLimitEnable_:MinTransYLimitEnablePlug = PlugDescriptor("minTransYLimitEnable")
	minTransZLimitEnable_:MinTransZLimitEnablePlug = PlugDescriptor("minTransZLimitEnable")
	minTransLimitEnable_:MinTransLimitEnablePlug = PlugDescriptor("minTransLimitEnable")
	nodeState_:NodeStatePlug = PlugDescriptor("nodeState")
	objectColor_:ObjectColorPlug = PlugDescriptor("objectColor")
	objectColorB_:ObjectColorBPlug = PlugDescriptor("objectColorB")
	objectColorG_:ObjectColorGPlug = PlugDescriptor("objectColorG")
	objectColorR_:ObjectColorRPlug = PlugDescriptor("objectColorR")
	objectColorRGB_:ObjectColorRGBPlug = PlugDescriptor("objectColorRGB")
	objectGroupId_:ObjectGroupIdPlug = PlugDescriptor("objectGroupId")
	offsetParentMatrix_:OffsetParentMatrixPlug = PlugDescriptor("offsetParentMatrix")
	outlinerColorB_:OutlinerColorBPlug = PlugDescriptor("outlinerColorB")
	outlinerColorG_:OutlinerColorGPlug = PlugDescriptor("outlinerColorG")
	outlinerColorR_:OutlinerColorRPlug = PlugDescriptor("outlinerColorR")
	outlinerColor_:OutlinerColorPlug = PlugDescriptor("outlinerColor")
	overrideColorB_:OverrideColorBPlug = PlugDescriptor("overrideColorB")
	overrideColorG_:OverrideColorGPlug = PlugDescriptor("overrideColorG")
	overrideColorR_:OverrideColorRPlug = PlugDescriptor("overrideColorR")
	parentInverseMatrix_:ParentInverseMatrixPlug = PlugDescriptor("parentInverseMatrix")
	parentMatrix_:ParentMatrixPlug = PlugDescriptor("parentMatrix")
	isHierarchicalNode_:IsHierarchicalNodePlug = PlugDescriptor("isHierarchicalNode")
	publishedNode_:PublishedNodePlug = PlugDescriptor("publishedNode")
	publishedNodeType_:PublishedNodeTypePlug = PlugDescriptor("publishedNodeType")
	publishedNodeInfo_:PublishedNodeInfoPlug = PlugDescriptor("publishedNodeInfo")
	identification_:IdentificationPlug = PlugDescriptor("identification")
	layerOverrideColor_:LayerOverrideColorPlug = PlugDescriptor("layerOverrideColor")
	layerRenderable_:LayerRenderablePlug = PlugDescriptor("layerRenderable")
	renderInfo_:RenderInfoPlug = PlugDescriptor("renderInfo")
	renderLayerColor_:RenderLayerColorPlug = PlugDescriptor("renderLayerColor")
	renderLayerId_:RenderLayerIdPlug = PlugDescriptor("renderLayerId")
	renderLayerRenderable_:RenderLayerRenderablePlug = PlugDescriptor("renderLayerRenderable")
	renderLayerInfo_:RenderLayerInfoPlug = PlugDescriptor("renderLayerInfo")
	rmbCommand_:RmbCommandPlug = PlugDescriptor("rmbCommand")
	rotateX_:RotateXPlug = PlugDescriptor("rotateX")
	rotateY_:RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_:RotateZPlug = PlugDescriptor("rotateZ")
	rotate_:RotatePlug = PlugDescriptor("rotate")
	rotateAxisX_:RotateAxisXPlug = PlugDescriptor("rotateAxisX")
	rotateAxisY_:RotateAxisYPlug = PlugDescriptor("rotateAxisY")
	rotateAxisZ_:RotateAxisZPlug = PlugDescriptor("rotateAxisZ")
	rotateAxis_:RotateAxisPlug = PlugDescriptor("rotateAxis")
	rotateOrder_:RotateOrderPlug = PlugDescriptor("rotateOrder")
	rotatePivotX_:RotatePivotXPlug = PlugDescriptor("rotatePivotX")
	rotatePivotY_:RotatePivotYPlug = PlugDescriptor("rotatePivotY")
	rotatePivotZ_:RotatePivotZPlug = PlugDescriptor("rotatePivotZ")
	rotatePivot_:RotatePivotPlug = PlugDescriptor("rotatePivot")
	rotatePivotTranslateX_:RotatePivotTranslateXPlug = PlugDescriptor("rotatePivotTranslateX")
	rotatePivotTranslateY_:RotatePivotTranslateYPlug = PlugDescriptor("rotatePivotTranslateY")
	rotatePivotTranslateZ_:RotatePivotTranslateZPlug = PlugDescriptor("rotatePivotTranslateZ")
	rotatePivotTranslate_:RotatePivotTranslatePlug = PlugDescriptor("rotatePivotTranslate")
	rotateQuaternionW_:RotateQuaternionWPlug = PlugDescriptor("rotateQuaternionW")
	rotateQuaternionX_:RotateQuaternionXPlug = PlugDescriptor("rotateQuaternionX")
	rotateQuaternionY_:RotateQuaternionYPlug = PlugDescriptor("rotateQuaternionY")
	rotateQuaternionZ_:RotateQuaternionZPlug = PlugDescriptor("rotateQuaternionZ")
	rotateQuaternion_:RotateQuaternionPlug = PlugDescriptor("rotateQuaternion")
	rotationInterpolation_:RotationInterpolationPlug = PlugDescriptor("rotationInterpolation")
	scaleX_:ScaleXPlug = PlugDescriptor("scaleX")
	scaleY_:ScaleYPlug = PlugDescriptor("scaleY")
	scaleZ_:ScaleZPlug = PlugDescriptor("scaleZ")
	scale_:ScalePlug = PlugDescriptor("scale")
	scalePivotX_:ScalePivotXPlug = PlugDescriptor("scalePivotX")
	scalePivotY_:ScalePivotYPlug = PlugDescriptor("scalePivotY")
	scalePivotZ_:ScalePivotZPlug = PlugDescriptor("scalePivotZ")
	scalePivot_:ScalePivotPlug = PlugDescriptor("scalePivot")
	scalePivotTranslateX_:ScalePivotTranslateXPlug = PlugDescriptor("scalePivotTranslateX")
	scalePivotTranslateY_:ScalePivotTranslateYPlug = PlugDescriptor("scalePivotTranslateY")
	scalePivotTranslateZ_:ScalePivotTranslateZPlug = PlugDescriptor("scalePivotTranslateZ")
	scalePivotTranslate_:ScalePivotTranslatePlug = PlugDescriptor("scalePivotTranslate")
	selectHandleX_:SelectHandleXPlug = PlugDescriptor("selectHandleX")
	selectHandleY_:SelectHandleYPlug = PlugDescriptor("selectHandleY")
	selectHandleZ_:SelectHandleZPlug = PlugDescriptor("selectHandleZ")
	selectHandle_:SelectHandlePlug = PlugDescriptor("selectHandle")
	selectionChildHighlighting_:SelectionChildHighlightingPlug = PlugDescriptor("selectionChildHighlighting")
	shearXY_:ShearXYPlug = PlugDescriptor("shearXY")
	shearXZ_:ShearXZPlug = PlugDescriptor("shearXZ")
	shearYZ_:ShearYZPlug = PlugDescriptor("shearYZ")
	shear_:ShearPlug = PlugDescriptor("shear")
	showManipDefault_:ShowManipDefaultPlug = PlugDescriptor("showManipDefault")
	specifiedManipLocation_:SpecifiedManipLocationPlug = PlugDescriptor("specifiedManipLocation")
	template_:TemplatePlug = PlugDescriptor("template")
	templateName_:TemplateNamePlug = PlugDescriptor("templateName")
	templatePath_:TemplatePathPlug = PlugDescriptor("templatePath")
	templateVersion_:TemplateVersionPlug = PlugDescriptor("templateVersion")
	transMinusRotatePivotX_:TransMinusRotatePivotXPlug = PlugDescriptor("transMinusRotatePivotX")
	transMinusRotatePivotY_:TransMinusRotatePivotYPlug = PlugDescriptor("transMinusRotatePivotY")
	transMinusRotatePivotZ_:TransMinusRotatePivotZPlug = PlugDescriptor("transMinusRotatePivotZ")
	transMinusRotatePivot_:TransMinusRotatePivotPlug = PlugDescriptor("transMinusRotatePivot")
	translateX_:TranslateXPlug = PlugDescriptor("translateX")
	translateY_:TranslateYPlug = PlugDescriptor("translateY")
	translateZ_:TranslateZPlug = PlugDescriptor("translateZ")
	translate_:TranslatePlug = PlugDescriptor("translate")
	uiTreatment_:UiTreatmentPlug = PlugDescriptor("uiTreatment")
	useObjectColor_:UseObjectColorPlug = PlugDescriptor("useObjectColor")
	useOutlinerColor_:UseOutlinerColorPlug = PlugDescriptor("useOutlinerColor")
	viewMode_:ViewModePlug = PlugDescriptor("viewMode")
	viewName_:ViewNamePlug = PlugDescriptor("viewName")
	visibility_:VisibilityPlug = PlugDescriptor("visibility")
	wireColorB_:WireColorBPlug = PlugDescriptor("wireColorB")
	wireColorG_:WireColorGPlug = PlugDescriptor("wireColorG")
	wireColorR_:WireColorRPlug = PlugDescriptor("wireColorR")
	wireColorRGB_:WireColorRGBPlug = PlugDescriptor("wireColorRGB")
	worldInverseMatrix_:WorldInverseMatrixPlug = PlugDescriptor("worldInverseMatrix")
	worldMatrix_:WorldMatrixPlug = PlugDescriptor("worldMatrix")
	xformMatrix_:XformMatrixPlug = PlugDescriptor("xformMatrix")

	pass


