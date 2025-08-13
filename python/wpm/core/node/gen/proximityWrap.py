

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
WeightGeometryFilter = retriever.getNodeCls("WeightGeometryFilter")
assert WeightGeometryFilter
if T.TYPE_CHECKING:
	from .. import WeightGeometryFilter

# add node doc



# region plug type defs
class AssociativeGeometryPlug(Plug):
	node : ProximityWrap = None
	pass
class BindTagsFilterPlug(Plug):
	node : ProximityWrap = None
	pass
class CacheSetupPlug(Plug):
	node : ProximityWrap = None
	pass
class DriverWeightFunctionPlug(Plug):
	node : ProximityWrap = None
	pass
class DriverBindGeometryPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverClusterMatrixPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverClusterRestMatrixPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverDropoffRatePlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverFalloffEndPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverFalloffRamp_FloatValuePlug(Plug):
	parent : DriverFalloffRampPlug = PlugDescriptor("driverFalloffRamp")
	node : ProximityWrap = None
	pass
class DriverFalloffRamp_InterpPlug(Plug):
	parent : DriverFalloffRampPlug = PlugDescriptor("driverFalloffRamp")
	node : ProximityWrap = None
	pass
class DriverFalloffRamp_PositionPlug(Plug):
	parent : DriverFalloffRampPlug = PlugDescriptor("driverFalloffRamp")
	node : ProximityWrap = None
	pass
class DriverFalloffRampPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	driverFalloffRamp_FloatValue_ : DriverFalloffRamp_FloatValuePlug = PlugDescriptor("driverFalloffRamp_FloatValue")
	dfrmpfv_ : DriverFalloffRamp_FloatValuePlug = PlugDescriptor("driverFalloffRamp_FloatValue")
	driverFalloffRamp_Interp_ : DriverFalloffRamp_InterpPlug = PlugDescriptor("driverFalloffRamp_Interp")
	dfrmpi_ : DriverFalloffRamp_InterpPlug = PlugDescriptor("driverFalloffRamp_Interp")
	driverFalloffRamp_Position_ : DriverFalloffRamp_PositionPlug = PlugDescriptor("driverFalloffRamp_Position")
	dfrmpp_ : DriverFalloffRamp_PositionPlug = PlugDescriptor("driverFalloffRamp_Position")
	node : ProximityWrap = None
	pass
class DriverFalloffStartPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverGeometryPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverOverrideFalloffRampPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverOverrideSmoothNormalsPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverOverrideSpanSamplesPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverSmoothNormalsPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverSpanSamplesPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverStrengthPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverUseTransformAsDeformationPlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriverWrapModePlug(Plug):
	parent : DriversPlug = PlugDescriptor("drivers")
	node : ProximityWrap = None
	pass
class DriversPlug(Plug):
	driverBindGeometry_ : DriverBindGeometryPlug = PlugDescriptor("driverBindGeometry")
	orgdrv_ : DriverBindGeometryPlug = PlugDescriptor("driverBindGeometry")
	driverClusterMatrix_ : DriverClusterMatrixPlug = PlugDescriptor("driverClusterMatrix")
	dcurcls_ : DriverClusterMatrixPlug = PlugDescriptor("driverClusterMatrix")
	driverClusterRestMatrix_ : DriverClusterRestMatrixPlug = PlugDescriptor("driverClusterRestMatrix")
	dorgcls_ : DriverClusterRestMatrixPlug = PlugDescriptor("driverClusterRestMatrix")
	driverDropoffRate_ : DriverDropoffRatePlug = PlugDescriptor("driverDropoffRate")
	ddpo_ : DriverDropoffRatePlug = PlugDescriptor("driverDropoffRate")
	driverFalloffEnd_ : DriverFalloffEndPlug = PlugDescriptor("driverFalloffEnd")
	dfoe_ : DriverFalloffEndPlug = PlugDescriptor("driverFalloffEnd")
	driverFalloffRamp_ : DriverFalloffRampPlug = PlugDescriptor("driverFalloffRamp")
	dfrmp_ : DriverFalloffRampPlug = PlugDescriptor("driverFalloffRamp")
	driverFalloffStart_ : DriverFalloffStartPlug = PlugDescriptor("driverFalloffStart")
	dfos_ : DriverFalloffStartPlug = PlugDescriptor("driverFalloffStart")
	driverGeometry_ : DriverGeometryPlug = PlugDescriptor("driverGeometry")
	curdrv_ : DriverGeometryPlug = PlugDescriptor("driverGeometry")
	driverOverrideFalloffRamp_ : DriverOverrideFalloffRampPlug = PlugDescriptor("driverOverrideFalloffRamp")
	dofrmp_ : DriverOverrideFalloffRampPlug = PlugDescriptor("driverOverrideFalloffRamp")
	driverOverrideSmoothNormals_ : DriverOverrideSmoothNormalsPlug = PlugDescriptor("driverOverrideSmoothNormals")
	dosnrm_ : DriverOverrideSmoothNormalsPlug = PlugDescriptor("driverOverrideSmoothNormals")
	driverOverrideSpanSamples_ : DriverOverrideSpanSamplesPlug = PlugDescriptor("driverOverrideSpanSamples")
	dospns_ : DriverOverrideSpanSamplesPlug = PlugDescriptor("driverOverrideSpanSamples")
	driverSmoothNormals_ : DriverSmoothNormalsPlug = PlugDescriptor("driverSmoothNormals")
	dsnrm_ : DriverSmoothNormalsPlug = PlugDescriptor("driverSmoothNormals")
	driverSpanSamples_ : DriverSpanSamplesPlug = PlugDescriptor("driverSpanSamples")
	dspns_ : DriverSpanSamplesPlug = PlugDescriptor("driverSpanSamples")
	driverStrength_ : DriverStrengthPlug = PlugDescriptor("driverStrength")
	dstrn_ : DriverStrengthPlug = PlugDescriptor("driverStrength")
	driverUseTransformAsDeformation_ : DriverUseTransformAsDeformationPlug = PlugDescriptor("driverUseTransformAsDeformation")
	dxad_ : DriverUseTransformAsDeformationPlug = PlugDescriptor("driverUseTransformAsDeformation")
	driverWrapMode_ : DriverWrapModePlug = PlugDescriptor("driverWrapMode")
	dwmd_ : DriverWrapModePlug = PlugDescriptor("driverWrapMode")
	node : ProximityWrap = None
	pass
class DropoffRateScalePlug(Plug):
	node : ProximityWrap = None
	pass
class FalloffRamp_FloatValuePlug(Plug):
	parent : FalloffRampPlug = PlugDescriptor("falloffRamp")
	node : ProximityWrap = None
	pass
class FalloffRamp_InterpPlug(Plug):
	parent : FalloffRampPlug = PlugDescriptor("falloffRamp")
	node : ProximityWrap = None
	pass
class FalloffRamp_PositionPlug(Plug):
	parent : FalloffRampPlug = PlugDescriptor("falloffRamp")
	node : ProximityWrap = None
	pass
class FalloffRampPlug(Plug):
	falloffRamp_FloatValue_ : FalloffRamp_FloatValuePlug = PlugDescriptor("falloffRamp_FloatValue")
	frmpfv_ : FalloffRamp_FloatValuePlug = PlugDescriptor("falloffRamp_FloatValue")
	falloffRamp_Interp_ : FalloffRamp_InterpPlug = PlugDescriptor("falloffRamp_Interp")
	frmpi_ : FalloffRamp_InterpPlug = PlugDescriptor("falloffRamp_Interp")
	falloffRamp_Position_ : FalloffRamp_PositionPlug = PlugDescriptor("falloffRamp_Position")
	frmpp_ : FalloffRamp_PositionPlug = PlugDescriptor("falloffRamp_Position")
	node : ProximityWrap = None
	pass
class FalloffScalePlug(Plug):
	node : ProximityWrap = None
	pass
class MaxDriversPlug(Plug):
	node : ProximityWrap = None
	pass
class PerDriverVertexWeightsPlug(Plug):
	parent : PerDriverWeightsPlug = PlugDescriptor("perDriverWeights")
	node : ProximityWrap = None
	pass
class PerDriverWeightsPlug(Plug):
	parent : PerDriverWeightsListPlug = PlugDescriptor("perDriverWeightsList")
	perDriverVertexWeights_ : PerDriverVertexWeightsPlug = PlugDescriptor("perDriverVertexWeights")
	pdvw_ : PerDriverVertexWeightsPlug = PlugDescriptor("perDriverVertexWeights")
	node : ProximityWrap = None
	pass
class PerDriverWeightsListPlug(Plug):
	perDriverWeights_ : PerDriverWeightsPlug = PlugDescriptor("perDriverWeights")
	pdw_ : PerDriverWeightsPlug = PlugDescriptor("perDriverWeights")
	node : ProximityWrap = None
	pass
class PerVertexDriverWeightsPlug(Plug):
	parent : PerVertexWeightsPlug = PlugDescriptor("perVertexWeights")
	node : ProximityWrap = None
	pass
class PerVertexWeightsPlug(Plug):
	parent : PerVertexWeightsListPlug = PlugDescriptor("perVertexWeightsList")
	perVertexDriverWeights_ : PerVertexDriverWeightsPlug = PlugDescriptor("perVertexDriverWeights")
	pvdw_ : PerVertexDriverWeightsPlug = PlugDescriptor("perVertexDriverWeights")
	node : ProximityWrap = None
	pass
class PerVertexWeightsListPlug(Plug):
	perVertexWeights_ : PerVertexWeightsPlug = PlugDescriptor("perVertexWeights")
	pvw_ : PerVertexWeightsPlug = PlugDescriptor("perVertexWeights")
	node : ProximityWrap = None
	pass
class SmoothInfluencesPlug(Plug):
	node : ProximityWrap = None
	pass
class SmoothNormalsPlug(Plug):
	node : ProximityWrap = None
	pass
class SoftNormalizationPlug(Plug):
	node : ProximityWrap = None
	pass
class SpanSamplesPlug(Plug):
	node : ProximityWrap = None
	pass
class UseBindTagsPlug(Plug):
	node : ProximityWrap = None
	pass
class WrapModePlug(Plug):
	node : ProximityWrap = None
	pass
# endregion


# define node class
class ProximityWrap(WeightGeometryFilter):
	associativeGeometry_ : AssociativeGeometryPlug = PlugDescriptor("associativeGeometry")
	bindTagsFilter_ : BindTagsFilterPlug = PlugDescriptor("bindTagsFilter")
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	driverWeightFunction_ : DriverWeightFunctionPlug = PlugDescriptor("driverWeightFunction")
	driverBindGeometry_ : DriverBindGeometryPlug = PlugDescriptor("driverBindGeometry")
	driverClusterMatrix_ : DriverClusterMatrixPlug = PlugDescriptor("driverClusterMatrix")
	driverClusterRestMatrix_ : DriverClusterRestMatrixPlug = PlugDescriptor("driverClusterRestMatrix")
	driverDropoffRate_ : DriverDropoffRatePlug = PlugDescriptor("driverDropoffRate")
	driverFalloffEnd_ : DriverFalloffEndPlug = PlugDescriptor("driverFalloffEnd")
	driverFalloffRamp_FloatValue_ : DriverFalloffRamp_FloatValuePlug = PlugDescriptor("driverFalloffRamp_FloatValue")
	driverFalloffRamp_Interp_ : DriverFalloffRamp_InterpPlug = PlugDescriptor("driverFalloffRamp_Interp")
	driverFalloffRamp_Position_ : DriverFalloffRamp_PositionPlug = PlugDescriptor("driverFalloffRamp_Position")
	driverFalloffRamp_ : DriverFalloffRampPlug = PlugDescriptor("driverFalloffRamp")
	driverFalloffStart_ : DriverFalloffStartPlug = PlugDescriptor("driverFalloffStart")
	driverGeometry_ : DriverGeometryPlug = PlugDescriptor("driverGeometry")
	driverOverrideFalloffRamp_ : DriverOverrideFalloffRampPlug = PlugDescriptor("driverOverrideFalloffRamp")
	driverOverrideSmoothNormals_ : DriverOverrideSmoothNormalsPlug = PlugDescriptor("driverOverrideSmoothNormals")
	driverOverrideSpanSamples_ : DriverOverrideSpanSamplesPlug = PlugDescriptor("driverOverrideSpanSamples")
	driverSmoothNormals_ : DriverSmoothNormalsPlug = PlugDescriptor("driverSmoothNormals")
	driverSpanSamples_ : DriverSpanSamplesPlug = PlugDescriptor("driverSpanSamples")
	driverStrength_ : DriverStrengthPlug = PlugDescriptor("driverStrength")
	driverUseTransformAsDeformation_ : DriverUseTransformAsDeformationPlug = PlugDescriptor("driverUseTransformAsDeformation")
	driverWrapMode_ : DriverWrapModePlug = PlugDescriptor("driverWrapMode")
	drivers_ : DriversPlug = PlugDescriptor("drivers")
	dropoffRateScale_ : DropoffRateScalePlug = PlugDescriptor("dropoffRateScale")
	falloffRamp_FloatValue_ : FalloffRamp_FloatValuePlug = PlugDescriptor("falloffRamp_FloatValue")
	falloffRamp_Interp_ : FalloffRamp_InterpPlug = PlugDescriptor("falloffRamp_Interp")
	falloffRamp_Position_ : FalloffRamp_PositionPlug = PlugDescriptor("falloffRamp_Position")
	falloffRamp_ : FalloffRampPlug = PlugDescriptor("falloffRamp")
	falloffScale_ : FalloffScalePlug = PlugDescriptor("falloffScale")
	maxDrivers_ : MaxDriversPlug = PlugDescriptor("maxDrivers")
	perDriverVertexWeights_ : PerDriverVertexWeightsPlug = PlugDescriptor("perDriverVertexWeights")
	perDriverWeights_ : PerDriverWeightsPlug = PlugDescriptor("perDriverWeights")
	perDriverWeightsList_ : PerDriverWeightsListPlug = PlugDescriptor("perDriverWeightsList")
	perVertexDriverWeights_ : PerVertexDriverWeightsPlug = PlugDescriptor("perVertexDriverWeights")
	perVertexWeights_ : PerVertexWeightsPlug = PlugDescriptor("perVertexWeights")
	perVertexWeightsList_ : PerVertexWeightsListPlug = PlugDescriptor("perVertexWeightsList")
	smoothInfluences_ : SmoothInfluencesPlug = PlugDescriptor("smoothInfluences")
	smoothNormals_ : SmoothNormalsPlug = PlugDescriptor("smoothNormals")
	softNormalization_ : SoftNormalizationPlug = PlugDescriptor("softNormalization")
	spanSamples_ : SpanSamplesPlug = PlugDescriptor("spanSamples")
	useBindTags_ : UseBindTagsPlug = PlugDescriptor("useBindTags")
	wrapMode_ : WrapModePlug = PlugDescriptor("wrapMode")

	# node attributes

	typeName = "proximityWrap"
	apiTypeInt = 354
	apiTypeStr = "kProximityWrap"
	typeIdInt = 1347899984
	MFnCls = om.MFnGeometryFilter
	pass

