

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SurfaceShape = retriever.getNodeCls("SurfaceShape")
assert SurfaceShape
if T.TYPE_CHECKING:
	from .. import SurfaceShape

# add node doc



# region plug type defs
class AirFuelRatioPlug(Plug):
	node : FluidShape = None
	pass
class AmbientBrightnessPlug(Plug):
	node : FluidShape = None
	pass
class AmbientColorBPlug(Plug):
	parent : AmbientColorPlug = PlugDescriptor("ambientColor")
	node : FluidShape = None
	pass
class AmbientColorGPlug(Plug):
	parent : AmbientColorPlug = PlugDescriptor("ambientColor")
	node : FluidShape = None
	pass
class AmbientColorRPlug(Plug):
	parent : AmbientColorPlug = PlugDescriptor("ambientColor")
	node : FluidShape = None
	pass
class AmbientColorPlug(Plug):
	ambientColorB_ : AmbientColorBPlug = PlugDescriptor("ambientColorB")
	ambb_ : AmbientColorBPlug = PlugDescriptor("ambientColorB")
	ambientColorG_ : AmbientColorGPlug = PlugDescriptor("ambientColorG")
	ambg_ : AmbientColorGPlug = PlugDescriptor("ambientColorG")
	ambientColorR_ : AmbientColorRPlug = PlugDescriptor("ambientColorR")
	ambr_ : AmbientColorRPlug = PlugDescriptor("ambientColorR")
	node : FluidShape = None
	pass
class AmbientDiffusionPlug(Plug):
	node : FluidShape = None
	pass
class AmplitudePlug(Plug):
	node : FluidShape = None
	pass
class AutoResizePlug(Plug):
	node : FluidShape = None
	pass
class AutoResizeMarginPlug(Plug):
	node : FluidShape = None
	pass
class AutoResizeThresholdPlug(Plug):
	node : FluidShape = None
	pass
class BaseResolutionPlug(Plug):
	node : FluidShape = None
	pass
class BillowDensityPlug(Plug):
	node : FluidShape = None
	pass
class BoundaryDrawPlug(Plug):
	node : FluidShape = None
	pass
class BoundaryXPlug(Plug):
	node : FluidShape = None
	pass
class BoundaryYPlug(Plug):
	node : FluidShape = None
	pass
class BoundaryZPlug(Plug):
	node : FluidShape = None
	pass
class BoxMinRadiusPlug(Plug):
	node : FluidShape = None
	pass
class BoxRadiusPlug(Plug):
	node : FluidShape = None
	pass
class BoxRatioPlug(Plug):
	node : FluidShape = None
	pass
class BuoyancyPlug(Plug):
	node : FluidShape = None
	pass
class CacheColorPlug(Plug):
	node : FluidShape = None
	pass
class CacheDensityPlug(Plug):
	node : FluidShape = None
	pass
class CacheFalloffPlug(Plug):
	node : FluidShape = None
	pass
class CacheReactionPlug(Plug):
	node : FluidShape = None
	pass
class CacheTemperaturePlug(Plug):
	node : FluidShape = None
	pass
class CacheTextureCoordinatesPlug(Plug):
	node : FluidShape = None
	pass
class CacheVelocityPlug(Plug):
	node : FluidShape = None
	pass
class CheckerPlug(Plug):
	node : FluidShape = None
	pass
class CircleRadiusPlug(Plug):
	node : FluidShape = None
	pass
class CircleSizeRatioPlug(Plug):
	node : FluidShape = None
	pass
class CirclesPlug(Plug):
	node : FluidShape = None
	pass
class CollidePlug(Plug):
	node : FluidShape = None
	pass
class CollisionFrictionPlug(Plug):
	parent : CollisionDataPlug = PlugDescriptor("collisionData")
	node : FluidShape = None
	pass
class CollisionGeometryPlug(Plug):
	parent : CollisionDataPlug = PlugDescriptor("collisionData")
	node : FluidShape = None
	pass
class CollisionResiliencePlug(Plug):
	parent : CollisionDataPlug = PlugDescriptor("collisionData")
	node : FluidShape = None
	pass
class CollisionDataPlug(Plug):
	collisionFriction_ : CollisionFrictionPlug = PlugDescriptor("collisionFriction")
	cfr_ : CollisionFrictionPlug = PlugDescriptor("collisionFriction")
	collisionGeometry_ : CollisionGeometryPlug = PlugDescriptor("collisionGeometry")
	cge_ : CollisionGeometryPlug = PlugDescriptor("collisionGeometry")
	collisionResilience_ : CollisionResiliencePlug = PlugDescriptor("collisionResilience")
	crs_ : CollisionResiliencePlug = PlugDescriptor("collisionResilience")
	node : FluidShape = None
	pass
class Color_ColorBPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : FluidShape = None
	pass
class Color_ColorGPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : FluidShape = None
	pass
class Color_ColorRPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : FluidShape = None
	pass
class Color_ColorPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	color_ColorB_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	clcb_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	color_ColorG_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	clcg_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	color_ColorR_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	clcr_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	node : FluidShape = None
	pass
class Color_InterpPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : FluidShape = None
	pass
class Color_PositionPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : FluidShape = None
	pass
class ColorPlug(Plug):
	color_Color_ : Color_ColorPlug = PlugDescriptor("color_Color")
	clc_ : Color_ColorPlug = PlugDescriptor("color_Color")
	color_Interp_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	cli_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	color_Position_ : Color_PositionPlug = PlugDescriptor("color_Position")
	clp_ : Color_PositionPlug = PlugDescriptor("color_Position")
	node : FluidShape = None
	pass
class ColorDiffusionPlug(Plug):
	node : FluidShape = None
	pass
class ColorDissipationPlug(Plug):
	node : FluidShape = None
	pass
class ColorInputPlug(Plug):
	node : FluidShape = None
	pass
class ColorInputBiasPlug(Plug):
	node : FluidShape = None
	pass
class ColorMethodPlug(Plug):
	node : FluidShape = None
	pass
class ColorPerVertexPlug(Plug):
	node : FluidShape = None
	pass
class ColorTexGainPlug(Plug):
	node : FluidShape = None
	pass
class ColorTexturePlug(Plug):
	node : FluidShape = None
	pass
class ConserveMassPlug(Plug):
	node : FluidShape = None
	pass
class ContrastTolerancePlug(Plug):
	node : FluidShape = None
	pass
class CoordinateMethodPlug(Plug):
	node : FluidShape = None
	pass
class CoordinateSpeedPlug(Plug):
	node : FluidShape = None
	pass
class CosinePowerPlug(Plug):
	node : FluidShape = None
	pass
class CurrentTimePlug(Plug):
	node : FluidShape = None
	pass
class DensityBuoyancyPlug(Plug):
	node : FluidShape = None
	pass
class DensityDiffusionPlug(Plug):
	node : FluidShape = None
	pass
class DensityDissipationPlug(Plug):
	node : FluidShape = None
	pass
class DensityGradientPlug(Plug):
	node : FluidShape = None
	pass
class DensityGradientForcePlug(Plug):
	node : FluidShape = None
	pass
class DensityMethodPlug(Plug):
	node : FluidShape = None
	pass
class DensityNoisePlug(Plug):
	node : FluidShape = None
	pass
class DensityPressurePlug(Plug):
	node : FluidShape = None
	pass
class DensityPressureThresholdPlug(Plug):
	node : FluidShape = None
	pass
class DensityScalePlug(Plug):
	node : FluidShape = None
	pass
class DensityTensionPlug(Plug):
	node : FluidShape = None
	pass
class DepthMaxPlug(Plug):
	node : FluidShape = None
	pass
class DimensionsDPlug(Plug):
	parent : DimensionsPlug = PlugDescriptor("dimensions")
	node : FluidShape = None
	pass
class DimensionsHPlug(Plug):
	parent : DimensionsPlug = PlugDescriptor("dimensions")
	node : FluidShape = None
	pass
class DimensionsWPlug(Plug):
	parent : DimensionsPlug = PlugDescriptor("dimensions")
	node : FluidShape = None
	pass
class DimensionsPlug(Plug):
	dimensionsD_ : DimensionsDPlug = PlugDescriptor("dimensionsD")
	dd_ : DimensionsDPlug = PlugDescriptor("dimensionsD")
	dimensionsH_ : DimensionsHPlug = PlugDescriptor("dimensionsH")
	dh_ : DimensionsHPlug = PlugDescriptor("dimensionsH")
	dimensionsW_ : DimensionsWPlug = PlugDescriptor("dimensionsW")
	dw_ : DimensionsWPlug = PlugDescriptor("dimensionsW")
	node : FluidShape = None
	pass
class DirectionalLightXPlug(Plug):
	parent : DirectionalLightPlug = PlugDescriptor("directionalLight")
	node : FluidShape = None
	pass
class DirectionalLightYPlug(Plug):
	parent : DirectionalLightPlug = PlugDescriptor("directionalLight")
	node : FluidShape = None
	pass
class DirectionalLightZPlug(Plug):
	parent : DirectionalLightPlug = PlugDescriptor("directionalLight")
	node : FluidShape = None
	pass
class DirectionalLightPlug(Plug):
	directionalLightX_ : DirectionalLightXPlug = PlugDescriptor("directionalLightX")
	dlx_ : DirectionalLightXPlug = PlugDescriptor("directionalLightX")
	directionalLightY_ : DirectionalLightYPlug = PlugDescriptor("directionalLightY")
	dly_ : DirectionalLightYPlug = PlugDescriptor("directionalLightY")
	directionalLightZ_ : DirectionalLightZPlug = PlugDescriptor("directionalLightZ")
	dlz_ : DirectionalLightZPlug = PlugDescriptor("directionalLightZ")
	node : FluidShape = None
	pass
class DisableInteractiveEvalPlug(Plug):
	node : FluidShape = None
	pass
class DiskCachePlug(Plug):
	node : FluidShape = None
	pass
class DiskCacheICPlug(Plug):
	node : FluidShape = None
	pass
class DoEmissionPlug(Plug):
	node : FluidShape = None
	pass
class DoFieldsPlug(Plug):
	node : FluidShape = None
	pass
class DrawHeadsPlug(Plug):
	node : FluidShape = None
	pass
class DrawSubVolumePlug(Plug):
	node : FluidShape = None
	pass
class DropoffShapePlug(Plug):
	node : FluidShape = None
	pass
class DynamicOffsetXPlug(Plug):
	parent : DynamicOffsetPlug = PlugDescriptor("dynamicOffset")
	node : FluidShape = None
	pass
class DynamicOffsetYPlug(Plug):
	parent : DynamicOffsetPlug = PlugDescriptor("dynamicOffset")
	node : FluidShape = None
	pass
class DynamicOffsetZPlug(Plug):
	parent : DynamicOffsetPlug = PlugDescriptor("dynamicOffset")
	node : FluidShape = None
	pass
class DynamicOffsetPlug(Plug):
	dynamicOffsetX_ : DynamicOffsetXPlug = PlugDescriptor("dynamicOffsetX")
	dofx_ : DynamicOffsetXPlug = PlugDescriptor("dynamicOffsetX")
	dynamicOffsetY_ : DynamicOffsetYPlug = PlugDescriptor("dynamicOffsetY")
	dofy_ : DynamicOffsetYPlug = PlugDescriptor("dynamicOffsetY")
	dynamicOffsetZ_ : DynamicOffsetZPlug = PlugDescriptor("dynamicOffsetZ")
	dofz_ : DynamicOffsetZPlug = PlugDescriptor("dynamicOffsetZ")
	node : FluidShape = None
	pass
class EdgeDropoffPlug(Plug):
	node : FluidShape = None
	pass
class EmissionFunction_HiddenPlug(Plug):
	parent : EmissionFunctionPlug = PlugDescriptor("emissionFunction")
	node : FluidShape = None
	pass
class EmissionFunction_InmapFromPlug(Plug):
	parent : EmissionFunction_InmapPlug = PlugDescriptor("emissionFunction_Inmap")
	node : FluidShape = None
	pass
class EmissionFunction_InmapToPlug(Plug):
	parent : EmissionFunction_InmapPlug = PlugDescriptor("emissionFunction_Inmap")
	node : FluidShape = None
	pass
class EmissionFunction_InmapPlug(Plug):
	parent : EmissionFunctionPlug = PlugDescriptor("emissionFunction")
	emissionFunction_InmapFrom_ : EmissionFunction_InmapFromPlug = PlugDescriptor("emissionFunction_InmapFrom")
	emfif_ : EmissionFunction_InmapFromPlug = PlugDescriptor("emissionFunction_InmapFrom")
	emissionFunction_InmapTo_ : EmissionFunction_InmapToPlug = PlugDescriptor("emissionFunction_InmapTo")
	emfit_ : EmissionFunction_InmapToPlug = PlugDescriptor("emissionFunction_InmapTo")
	node : FluidShape = None
	pass
class EmissionFunction_OutmapFromPlug(Plug):
	parent : EmissionFunction_OutmapPlug = PlugDescriptor("emissionFunction_Outmap")
	node : FluidShape = None
	pass
class EmissionFunction_OutmapToPlug(Plug):
	parent : EmissionFunction_OutmapPlug = PlugDescriptor("emissionFunction_Outmap")
	node : FluidShape = None
	pass
class EmissionFunction_OutmapPlug(Plug):
	parent : EmissionFunctionPlug = PlugDescriptor("emissionFunction")
	emissionFunction_OutmapFrom_ : EmissionFunction_OutmapFromPlug = PlugDescriptor("emissionFunction_OutmapFrom")
	emfof_ : EmissionFunction_OutmapFromPlug = PlugDescriptor("emissionFunction_OutmapFrom")
	emissionFunction_OutmapTo_ : EmissionFunction_OutmapToPlug = PlugDescriptor("emissionFunction_OutmapTo")
	emfot_ : EmissionFunction_OutmapToPlug = PlugDescriptor("emissionFunction_OutmapTo")
	node : FluidShape = None
	pass
class EmissionFunction_RawPlug(Plug):
	parent : EmissionFunctionPlug = PlugDescriptor("emissionFunction")
	node : FluidShape = None
	pass
class EmissionFunctionPlug(Plug):
	parent : EmissionListPlug = PlugDescriptor("emissionList")
	emissionFunction_Hidden_ : EmissionFunction_HiddenPlug = PlugDescriptor("emissionFunction_Hidden")
	emfh_ : EmissionFunction_HiddenPlug = PlugDescriptor("emissionFunction_Hidden")
	emissionFunction_Inmap_ : EmissionFunction_InmapPlug = PlugDescriptor("emissionFunction_Inmap")
	emfi_ : EmissionFunction_InmapPlug = PlugDescriptor("emissionFunction_Inmap")
	emissionFunction_Outmap_ : EmissionFunction_OutmapPlug = PlugDescriptor("emissionFunction_Outmap")
	emfo_ : EmissionFunction_OutmapPlug = PlugDescriptor("emissionFunction_Outmap")
	emissionFunction_Raw_ : EmissionFunction_RawPlug = PlugDescriptor("emissionFunction_Raw")
	emfr_ : EmissionFunction_RawPlug = PlugDescriptor("emissionFunction_Raw")
	node : FluidShape = None
	pass
class EmissionListPlug(Plug):
	emissionFunction_ : EmissionFunctionPlug = PlugDescriptor("emissionFunction")
	emf_ : EmissionFunctionPlug = PlugDescriptor("emissionFunction")
	node : FluidShape = None
	pass
class EmitInSubstepsPlug(Plug):
	node : FluidShape = None
	pass
class EnableLiquidSimulationPlug(Plug):
	node : FluidShape = None
	pass
class Environment_ColorBPlug(Plug):
	parent : Environment_ColorPlug = PlugDescriptor("environment_Color")
	node : FluidShape = None
	pass
class Environment_ColorGPlug(Plug):
	parent : Environment_ColorPlug = PlugDescriptor("environment_Color")
	node : FluidShape = None
	pass
class Environment_ColorRPlug(Plug):
	parent : Environment_ColorPlug = PlugDescriptor("environment_Color")
	node : FluidShape = None
	pass
class Environment_ColorPlug(Plug):
	parent : EnvironmentPlug = PlugDescriptor("environment")
	environment_ColorB_ : Environment_ColorBPlug = PlugDescriptor("environment_ColorB")
	envcb_ : Environment_ColorBPlug = PlugDescriptor("environment_ColorB")
	environment_ColorG_ : Environment_ColorGPlug = PlugDescriptor("environment_ColorG")
	envcg_ : Environment_ColorGPlug = PlugDescriptor("environment_ColorG")
	environment_ColorR_ : Environment_ColorRPlug = PlugDescriptor("environment_ColorR")
	envcr_ : Environment_ColorRPlug = PlugDescriptor("environment_ColorR")
	node : FluidShape = None
	pass
class Environment_InterpPlug(Plug):
	parent : EnvironmentPlug = PlugDescriptor("environment")
	node : FluidShape = None
	pass
class Environment_PositionPlug(Plug):
	parent : EnvironmentPlug = PlugDescriptor("environment")
	node : FluidShape = None
	pass
class EnvironmentPlug(Plug):
	environment_Color_ : Environment_ColorPlug = PlugDescriptor("environment_Color")
	envc_ : Environment_ColorPlug = PlugDescriptor("environment_Color")
	environment_Interp_ : Environment_InterpPlug = PlugDescriptor("environment_Interp")
	envi_ : Environment_InterpPlug = PlugDescriptor("environment_Interp")
	environment_Position_ : Environment_PositionPlug = PlugDescriptor("environment_Position")
	envp_ : Environment_PositionPlug = PlugDescriptor("environment_Position")
	node : FluidShape = None
	pass
class EquilibriumValuePlug(Plug):
	node : FluidShape = None
	pass
class EscapeRadiusPlug(Plug):
	node : FluidShape = None
	pass
class FalloffPlug(Plug):
	node : FluidShape = None
	pass
class FalloffMethodPlug(Plug):
	node : FluidShape = None
	pass
class FarPointObjectXPlug(Plug):
	parent : FarPointObjPlug = PlugDescriptor("farPointObj")
	node : FluidShape = None
	pass
class FarPointObjectYPlug(Plug):
	parent : FarPointObjPlug = PlugDescriptor("farPointObj")
	node : FluidShape = None
	pass
class FarPointObjectZPlug(Plug):
	parent : FarPointObjPlug = PlugDescriptor("farPointObj")
	node : FluidShape = None
	pass
class FarPointObjPlug(Plug):
	farPointObjectX_ : FarPointObjectXPlug = PlugDescriptor("farPointObjectX")
	fox_ : FarPointObjectXPlug = PlugDescriptor("farPointObjectX")
	farPointObjectY_ : FarPointObjectYPlug = PlugDescriptor("farPointObjectY")
	foy_ : FarPointObjectYPlug = PlugDescriptor("farPointObjectY")
	farPointObjectZ_ : FarPointObjectZPlug = PlugDescriptor("farPointObjectZ")
	foz_ : FarPointObjectZPlug = PlugDescriptor("farPointObjectZ")
	node : FluidShape = None
	pass
class FarPointWorldXPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : FluidShape = None
	pass
class FarPointWorldYPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : FluidShape = None
	pass
class FarPointWorldZPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : FluidShape = None
	pass
class FarPointWorldPlug(Plug):
	farPointWorldX_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	fwx_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	farPointWorldY_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	fwy_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	farPointWorldZ_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	fwz_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	node : FluidShape = None
	pass
class FieldDataDeltaTimePlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : FluidShape = None
	pass
class FieldDataMassPlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : FluidShape = None
	pass
class FieldDataPositionPlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : FluidShape = None
	pass
class FieldDataVelocityPlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : FluidShape = None
	pass
class FieldDataPlug(Plug):
	fieldDataDeltaTime_ : FieldDataDeltaTimePlug = PlugDescriptor("fieldDataDeltaTime")
	fdt_ : FieldDataDeltaTimePlug = PlugDescriptor("fieldDataDeltaTime")
	fieldDataMass_ : FieldDataMassPlug = PlugDescriptor("fieldDataMass")
	fdm_ : FieldDataMassPlug = PlugDescriptor("fieldDataMass")
	fieldDataPosition_ : FieldDataPositionPlug = PlugDescriptor("fieldDataPosition")
	fdp_ : FieldDataPositionPlug = PlugDescriptor("fieldDataPosition")
	fieldDataVelocity_ : FieldDataVelocityPlug = PlugDescriptor("fieldDataVelocity")
	fdv_ : FieldDataVelocityPlug = PlugDescriptor("fieldDataVelocity")
	node : FluidShape = None
	pass
class FieldFunction_HiddenPlug(Plug):
	parent : FieldFunctionPlug = PlugDescriptor("fieldFunction")
	node : FluidShape = None
	pass
class FieldFunction_InmapFromPlug(Plug):
	parent : FieldFunction_InmapPlug = PlugDescriptor("fieldFunction_Inmap")
	node : FluidShape = None
	pass
class FieldFunction_InmapToPlug(Plug):
	parent : FieldFunction_InmapPlug = PlugDescriptor("fieldFunction_Inmap")
	node : FluidShape = None
	pass
class FieldFunction_InmapPlug(Plug):
	parent : FieldFunctionPlug = PlugDescriptor("fieldFunction")
	fieldFunction_InmapFrom_ : FieldFunction_InmapFromPlug = PlugDescriptor("fieldFunction_InmapFrom")
	frfif_ : FieldFunction_InmapFromPlug = PlugDescriptor("fieldFunction_InmapFrom")
	fieldFunction_InmapTo_ : FieldFunction_InmapToPlug = PlugDescriptor("fieldFunction_InmapTo")
	frfit_ : FieldFunction_InmapToPlug = PlugDescriptor("fieldFunction_InmapTo")
	node : FluidShape = None
	pass
class FieldFunction_OutmapFromPlug(Plug):
	parent : FieldFunction_OutmapPlug = PlugDescriptor("fieldFunction_Outmap")
	node : FluidShape = None
	pass
class FieldFunction_OutmapToPlug(Plug):
	parent : FieldFunction_OutmapPlug = PlugDescriptor("fieldFunction_Outmap")
	node : FluidShape = None
	pass
class FieldFunction_OutmapPlug(Plug):
	parent : FieldFunctionPlug = PlugDescriptor("fieldFunction")
	fieldFunction_OutmapFrom_ : FieldFunction_OutmapFromPlug = PlugDescriptor("fieldFunction_OutmapFrom")
	frfof_ : FieldFunction_OutmapFromPlug = PlugDescriptor("fieldFunction_OutmapFrom")
	fieldFunction_OutmapTo_ : FieldFunction_OutmapToPlug = PlugDescriptor("fieldFunction_OutmapTo")
	frfot_ : FieldFunction_OutmapToPlug = PlugDescriptor("fieldFunction_OutmapTo")
	node : FluidShape = None
	pass
class FieldFunction_RawPlug(Plug):
	parent : FieldFunctionPlug = PlugDescriptor("fieldFunction")
	node : FluidShape = None
	pass
class FieldFunctionPlug(Plug):
	parent : FieldListPlug = PlugDescriptor("fieldList")
	fieldFunction_Hidden_ : FieldFunction_HiddenPlug = PlugDescriptor("fieldFunction_Hidden")
	frfh_ : FieldFunction_HiddenPlug = PlugDescriptor("fieldFunction_Hidden")
	fieldFunction_Inmap_ : FieldFunction_InmapPlug = PlugDescriptor("fieldFunction_Inmap")
	frfi_ : FieldFunction_InmapPlug = PlugDescriptor("fieldFunction_Inmap")
	fieldFunction_Outmap_ : FieldFunction_OutmapPlug = PlugDescriptor("fieldFunction_Outmap")
	frfo_ : FieldFunction_OutmapPlug = PlugDescriptor("fieldFunction_Outmap")
	fieldFunction_Raw_ : FieldFunction_RawPlug = PlugDescriptor("fieldFunction_Raw")
	frfr_ : FieldFunction_RawPlug = PlugDescriptor("fieldFunction_Raw")
	node : FluidShape = None
	pass
class FieldListPlug(Plug):
	fieldFunction_ : FieldFunctionPlug = PlugDescriptor("fieldFunction")
	frf_ : FieldFunctionPlug = PlugDescriptor("fieldFunction")
	node : FluidShape = None
	pass
class FilterSizeXPlug(Plug):
	parent : FilterSizePlug = PlugDescriptor("filterSize")
	node : FluidShape = None
	pass
class FilterSizeYPlug(Plug):
	parent : FilterSizePlug = PlugDescriptor("filterSize")
	node : FluidShape = None
	pass
class FilterSizeZPlug(Plug):
	parent : FilterSizePlug = PlugDescriptor("filterSize")
	node : FluidShape = None
	pass
class FilterSizePlug(Plug):
	filterSizeX_ : FilterSizeXPlug = PlugDescriptor("filterSizeX")
	fsx_ : FilterSizeXPlug = PlugDescriptor("filterSizeX")
	filterSizeY_ : FilterSizeYPlug = PlugDescriptor("filterSizeY")
	fsy_ : FilterSizeYPlug = PlugDescriptor("filterSizeY")
	filterSizeZ_ : FilterSizeZPlug = PlugDescriptor("filterSizeZ")
	fsz_ : FilterSizeZPlug = PlugDescriptor("filterSizeZ")
	node : FluidShape = None
	pass
class FluidColorEmissionPlug(Plug):
	node : FluidShape = None
	pass
class FluidLightColorBPlug(Plug):
	parent : FluidLightColorPlug = PlugDescriptor("fluidLightColor")
	node : FluidShape = None
	pass
class FluidLightColorGPlug(Plug):
	parent : FluidLightColorPlug = PlugDescriptor("fluidLightColor")
	node : FluidShape = None
	pass
class FluidLightColorRPlug(Plug):
	parent : FluidLightColorPlug = PlugDescriptor("fluidLightColor")
	node : FluidShape = None
	pass
class FluidLightColorPlug(Plug):
	fluidLightColorB_ : FluidLightColorBPlug = PlugDescriptor("fluidLightColorB")
	flib_ : FluidLightColorBPlug = PlugDescriptor("fluidLightColorB")
	fluidLightColorG_ : FluidLightColorGPlug = PlugDescriptor("fluidLightColorG")
	flig_ : FluidLightColorGPlug = PlugDescriptor("fluidLightColorG")
	fluidLightColorR_ : FluidLightColorRPlug = PlugDescriptor("fluidLightColorR")
	flir_ : FluidLightColorRPlug = PlugDescriptor("fluidLightColorR")
	node : FluidShape = None
	pass
class FluidReactantEmissionPlug(Plug):
	node : FluidShape = None
	pass
class FocusPlug(Plug):
	node : FluidShape = None
	pass
class ForceDynamicsPlug(Plug):
	node : FluidShape = None
	pass
class ForwardAdvectionPlug(Plug):
	node : FluidShape = None
	pass
class FrequencyPlug(Plug):
	node : FluidShape = None
	pass
class FrequencyRatioPlug(Plug):
	node : FluidShape = None
	pass
class FrictionPlug(Plug):
	node : FluidShape = None
	pass
class FuelGradientPlug(Plug):
	node : FluidShape = None
	pass
class FuelIgnitionTempPlug(Plug):
	node : FluidShape = None
	pass
class FuelMethodPlug(Plug):
	node : FluidShape = None
	pass
class FuelScalePlug(Plug):
	node : FluidShape = None
	pass
class GlowIntensityPlug(Plug):
	node : FluidShape = None
	pass
class GravityPlug(Plug):
	node : FluidShape = None
	pass
class GridInterpolatorPlug(Plug):
	node : FluidShape = None
	pass
class HardwareSelfShadowPlug(Plug):
	node : FluidShape = None
	pass
class HeatReleasedPlug(Plug):
	node : FluidShape = None
	pass
class HeightFieldPlug(Plug):
	node : FluidShape = None
	pass
class HighDetailSolvePlug(Plug):
	node : FluidShape = None
	pass
class ImplodePlug(Plug):
	node : FluidShape = None
	pass
class ImplodeCenterXPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : FluidShape = None
	pass
class ImplodeCenterYPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : FluidShape = None
	pass
class ImplodeCenterZPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : FluidShape = None
	pass
class ImplodeCenterPlug(Plug):
	implodeCenterX_ : ImplodeCenterXPlug = PlugDescriptor("implodeCenterX")
	imx_ : ImplodeCenterXPlug = PlugDescriptor("implodeCenterX")
	implodeCenterY_ : ImplodeCenterYPlug = PlugDescriptor("implodeCenterY")
	imy_ : ImplodeCenterYPlug = PlugDescriptor("implodeCenterY")
	implodeCenterZ_ : ImplodeCenterZPlug = PlugDescriptor("implodeCenterZ")
	imz_ : ImplodeCenterZPlug = PlugDescriptor("implodeCenterZ")
	node : FluidShape = None
	pass
class InColorPlug(Plug):
	node : FluidShape = None
	pass
class InDensityPlug(Plug):
	node : FluidShape = None
	pass
class InFalloffPlug(Plug):
	node : FluidShape = None
	pass
class InOffsetPlug(Plug):
	node : FluidShape = None
	pass
class InReactionPlug(Plug):
	node : FluidShape = None
	pass
class InResolutionPlug(Plug):
	node : FluidShape = None
	pass
class InTemperaturePlug(Plug):
	node : FluidShape = None
	pass
class InTextureCoordinatesPlug(Plug):
	node : FluidShape = None
	pass
class InVelocityPlug(Plug):
	node : FluidShape = None
	pass
class IncandTexGainPlug(Plug):
	node : FluidShape = None
	pass
class IncandTexturePlug(Plug):
	node : FluidShape = None
	pass
class Incandescence_ColorBPlug(Plug):
	parent : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	node : FluidShape = None
	pass
class Incandescence_ColorGPlug(Plug):
	parent : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	node : FluidShape = None
	pass
class Incandescence_ColorRPlug(Plug):
	parent : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	node : FluidShape = None
	pass
class Incandescence_ColorPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	incandescence_ColorB_ : Incandescence_ColorBPlug = PlugDescriptor("incandescence_ColorB")
	icb_ : Incandescence_ColorBPlug = PlugDescriptor("incandescence_ColorB")
	incandescence_ColorG_ : Incandescence_ColorGPlug = PlugDescriptor("incandescence_ColorG")
	icg_ : Incandescence_ColorGPlug = PlugDescriptor("incandescence_ColorG")
	incandescence_ColorR_ : Incandescence_ColorRPlug = PlugDescriptor("incandescence_ColorR")
	icr_ : Incandescence_ColorRPlug = PlugDescriptor("incandescence_ColorR")
	node : FluidShape = None
	pass
class Incandescence_InterpPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : FluidShape = None
	pass
class Incandescence_PositionPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : FluidShape = None
	pass
class IncandescencePlug(Plug):
	incandescence_Color_ : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	ic_ : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	incandescence_Interp_ : Incandescence_InterpPlug = PlugDescriptor("incandescence_Interp")
	ii_ : Incandescence_InterpPlug = PlugDescriptor("incandescence_Interp")
	incandescence_Position_ : Incandescence_PositionPlug = PlugDescriptor("incandescence_Position")
	ip_ : Incandescence_PositionPlug = PlugDescriptor("incandescence_Position")
	node : FluidShape = None
	pass
class IncandescenceInputPlug(Plug):
	node : FluidShape = None
	pass
class IncandescenceInputBiasPlug(Plug):
	node : FluidShape = None
	pass
class IncandescencePerVertexPlug(Plug):
	node : FluidShape = None
	pass
class InflectionPlug(Plug):
	node : FluidShape = None
	pass
class InheritFactorPlug(Plug):
	node : FluidShape = None
	pass
class InitialConditionsPlug(Plug):
	node : FluidShape = None
	pass
class DeltaTimePlug(Plug):
	parent : InputDataPlug = PlugDescriptor("inputData")
	node : FluidShape = None
	pass
class InputMassPlug(Plug):
	parent : InputDataPlug = PlugDescriptor("inputData")
	node : FluidShape = None
	pass
class InputPositionsPlug(Plug):
	parent : InputDataPlug = PlugDescriptor("inputData")
	node : FluidShape = None
	pass
class InputVelocitiesPlug(Plug):
	parent : InputDataPlug = PlugDescriptor("inputData")
	node : FluidShape = None
	pass
class InputDataPlug(Plug):
	deltaTime_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	dt_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	inputMass_ : InputMassPlug = PlugDescriptor("inputMass")
	inm_ : InputMassPlug = PlugDescriptor("inputMass")
	inputPositions_ : InputPositionsPlug = PlugDescriptor("inputPositions")
	inp_ : InputPositionsPlug = PlugDescriptor("inputPositions")
	inputVelocities_ : InputVelocitiesPlug = PlugDescriptor("inputVelocities")
	inv_ : InputVelocitiesPlug = PlugDescriptor("inputVelocities")
	node : FluidShape = None
	pass
class InputForcePlug(Plug):
	node : FluidShape = None
	pass
class InputForce2Plug(Plug):
	node : FluidShape = None
	pass
class InvertTexturePlug(Plug):
	node : FluidShape = None
	pass
class Is2dPlug(Plug):
	node : FluidShape = None
	pass
class IsFullPlug(Plug):
	node : FluidShape = None
	pass
class JuliaUPlug(Plug):
	node : FluidShape = None
	pass
class JuliaVPlug(Plug):
	node : FluidShape = None
	pass
class LastEvalTimePlug(Plug):
	node : FluidShape = None
	pass
class LeafEffectPlug(Plug):
	node : FluidShape = None
	pass
class LightBrightnessPlug(Plug):
	node : FluidShape = None
	pass
class LightColorBPlug(Plug):
	parent : LightColorPlug = PlugDescriptor("lightColor")
	node : FluidShape = None
	pass
class LightColorGPlug(Plug):
	parent : LightColorPlug = PlugDescriptor("lightColor")
	node : FluidShape = None
	pass
class LightColorRPlug(Plug):
	parent : LightColorPlug = PlugDescriptor("lightColor")
	node : FluidShape = None
	pass
class LightColorPlug(Plug):
	lightColorB_ : LightColorBPlug = PlugDescriptor("lightColorB")
	lcob_ : LightColorBPlug = PlugDescriptor("lightColorB")
	lightColorG_ : LightColorGPlug = PlugDescriptor("lightColorG")
	lcog_ : LightColorGPlug = PlugDescriptor("lightColorG")
	lightColorR_ : LightColorRPlug = PlugDescriptor("lightColorR")
	lcor_ : LightColorRPlug = PlugDescriptor("lightColorR")
	node : FluidShape = None
	pass
class LightAmbientPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : FluidShape = None
	pass
class LightBlindDataPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : FluidShape = None
	pass
class LightDiffusePlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : FluidShape = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : FluidShape = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : FluidShape = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : FluidShape = None
	pass
class LightDirectionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : FluidShape = None
	pass
class LightIntensityBPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : FluidShape = None
	pass
class LightIntensityGPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : FluidShape = None
	pass
class LightIntensityRPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : FluidShape = None
	pass
class LightIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lib_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lig_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lir_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	node : FluidShape = None
	pass
class LightShadowFractionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : FluidShape = None
	pass
class LightSpecularPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : FluidShape = None
	pass
class PreShadowIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : FluidShape = None
	pass
class LightDataArrayPlug(Plug):
	lightAmbient_ : LightAmbientPlug = PlugDescriptor("lightAmbient")
	la_ : LightAmbientPlug = PlugDescriptor("lightAmbient")
	lightBlindData_ : LightBlindDataPlug = PlugDescriptor("lightBlindData")
	lbd_ : LightBlindDataPlug = PlugDescriptor("lightBlindData")
	lightDiffuse_ : LightDiffusePlug = PlugDescriptor("lightDiffuse")
	ldf_ : LightDiffusePlug = PlugDescriptor("lightDiffuse")
	lightDirection_ : LightDirectionPlug = PlugDescriptor("lightDirection")
	ld_ : LightDirectionPlug = PlugDescriptor("lightDirection")
	lightIntensity_ : LightIntensityPlug = PlugDescriptor("lightIntensity")
	li_ : LightIntensityPlug = PlugDescriptor("lightIntensity")
	lightShadowFraction_ : LightShadowFractionPlug = PlugDescriptor("lightShadowFraction")
	lsf_ : LightShadowFractionPlug = PlugDescriptor("lightShadowFraction")
	lightSpecular_ : LightSpecularPlug = PlugDescriptor("lightSpecular")
	ls_ : LightSpecularPlug = PlugDescriptor("lightSpecular")
	preShadowIntensity_ : PreShadowIntensityPlug = PlugDescriptor("preShadowIntensity")
	psi_ : PreShadowIntensityPlug = PlugDescriptor("preShadowIntensity")
	node : FluidShape = None
	pass
class LightReleasedPlug(Plug):
	node : FluidShape = None
	pass
class LightTypePlug(Plug):
	node : FluidShape = None
	pass
class LineBlendingPlug(Plug):
	node : FluidShape = None
	pass
class LineFocusPlug(Plug):
	node : FluidShape = None
	pass
class LineOffsetRatioPlug(Plug):
	node : FluidShape = None
	pass
class LineOffsetUPlug(Plug):
	node : FluidShape = None
	pass
class LineOffsetVPlug(Plug):
	node : FluidShape = None
	pass
class LiquidMethodPlug(Plug):
	node : FluidShape = None
	pass
class LiquidMinDensityPlug(Plug):
	node : FluidShape = None
	pass
class LiquidMistFallPlug(Plug):
	node : FluidShape = None
	pass
class LoadColorPlug(Plug):
	node : FluidShape = None
	pass
class LoadDensityPlug(Plug):
	node : FluidShape = None
	pass
class LoadFalloffPlug(Plug):
	node : FluidShape = None
	pass
class LoadReactionPlug(Plug):
	node : FluidShape = None
	pass
class LoadTemperaturePlug(Plug):
	node : FluidShape = None
	pass
class LoadTextureCoordinatesPlug(Plug):
	node : FluidShape = None
	pass
class LoadVelocityPlug(Plug):
	node : FluidShape = None
	pass
class LobesPlug(Plug):
	node : FluidShape = None
	pass
class LockDrawAxisPlug(Plug):
	node : FluidShape = None
	pass
class MandelbrotDepthPlug(Plug):
	node : FluidShape = None
	pass
class MandelbrotInsideMethodPlug(Plug):
	node : FluidShape = None
	pass
class MandelbrotShadeMethodPlug(Plug):
	node : FluidShape = None
	pass
class MandelbrotTypePlug(Plug):
	node : FluidShape = None
	pass
class MassConversionPlug(Plug):
	node : FluidShape = None
	pass
class MassRangePlug(Plug):
	node : FluidShape = None
	pass
class MatrixEyeToWorldPlug(Plug):
	node : FluidShape = None
	pass
class MatrixWorldToObjectPlug(Plug):
	node : FluidShape = None
	pass
class MatteOpacityPlug(Plug):
	node : FluidShape = None
	pass
class MatteOpacityModePlug(Plug):
	node : FluidShape = None
	pass
class MaxReactionTempPlug(Plug):
	node : FluidShape = None
	pass
class MaxResolutionPlug(Plug):
	node : FluidShape = None
	pass
class MeshMethodPlug(Plug):
	node : FluidShape = None
	pass
class MeshResolutionPlug(Plug):
	node : FluidShape = None
	pass
class MeshSmoothingIterationsPlug(Plug):
	node : FluidShape = None
	pass
class NumWavesPlug(Plug):
	node : FluidShape = None
	pass
class NumericDisplayPlug(Plug):
	node : FluidShape = None
	pass
class ObjectTypePlug(Plug):
	node : FluidShape = None
	pass
class Opacity_FloatValuePlug(Plug):
	parent : OpacityPlug = PlugDescriptor("opacity")
	node : FluidShape = None
	pass
class Opacity_InterpPlug(Plug):
	parent : OpacityPlug = PlugDescriptor("opacity")
	node : FluidShape = None
	pass
class Opacity_PositionPlug(Plug):
	parent : OpacityPlug = PlugDescriptor("opacity")
	node : FluidShape = None
	pass
class OpacityPlug(Plug):
	opacity_FloatValue_ : Opacity_FloatValuePlug = PlugDescriptor("opacity_FloatValue")
	opafv_ : Opacity_FloatValuePlug = PlugDescriptor("opacity_FloatValue")
	opacity_Interp_ : Opacity_InterpPlug = PlugDescriptor("opacity_Interp")
	opai_ : Opacity_InterpPlug = PlugDescriptor("opacity_Interp")
	opacity_Position_ : Opacity_PositionPlug = PlugDescriptor("opacity_Position")
	opap_ : Opacity_PositionPlug = PlugDescriptor("opacity_Position")
	node : FluidShape = None
	pass
class OpacityInputPlug(Plug):
	node : FluidShape = None
	pass
class OpacityInputBiasPlug(Plug):
	node : FluidShape = None
	pass
class OpacityPerVertexPlug(Plug):
	node : FluidShape = None
	pass
class OpacityPreviewGainPlug(Plug):
	node : FluidShape = None
	pass
class OpacityTexGainPlug(Plug):
	node : FluidShape = None
	pass
class OpacityTexturePlug(Plug):
	node : FluidShape = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : FluidShape = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : FluidShape = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : FluidShape = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : FluidShape = None
	pass
class OutGlowColorBPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : FluidShape = None
	pass
class OutGlowColorGPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : FluidShape = None
	pass
class OutGlowColorRPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : FluidShape = None
	pass
class OutGlowColorPlug(Plug):
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	ogb_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	ogg_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	ogr_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	node : FluidShape = None
	pass
class OutGridPlug(Plug):
	node : FluidShape = None
	pass
class OutMatteOpacityBPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : FluidShape = None
	pass
class OutMatteOpacityGPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : FluidShape = None
	pass
class OutMatteOpacityRPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : FluidShape = None
	pass
class OutMatteOpacityPlug(Plug):
	outMatteOpacityB_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	omob_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	outMatteOpacityG_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	omog_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	outMatteOpacityR_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	omor_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	node : FluidShape = None
	pass
class OutMeshPlug(Plug):
	node : FluidShape = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : FluidShape = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : FluidShape = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : FluidShape = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : FluidShape = None
	pass
class OutputForcePlug(Plug):
	node : FluidShape = None
	pass
class OverrideTimeStepPlug(Plug):
	node : FluidShape = None
	pass
class ParticleWeightPlug(Plug):
	node : FluidShape = None
	pass
class PlayFromCachePlug(Plug):
	node : FluidShape = None
	pass
class PointLightXPlug(Plug):
	parent : PointLightPlug = PlugDescriptor("pointLight")
	node : FluidShape = None
	pass
class PointLightYPlug(Plug):
	parent : PointLightPlug = PlugDescriptor("pointLight")
	node : FluidShape = None
	pass
class PointLightZPlug(Plug):
	parent : PointLightPlug = PlugDescriptor("pointLight")
	node : FluidShape = None
	pass
class PointLightPlug(Plug):
	pointLightX_ : PointLightXPlug = PlugDescriptor("pointLightX")
	polx_ : PointLightXPlug = PlugDescriptor("pointLightX")
	pointLightY_ : PointLightYPlug = PlugDescriptor("pointLightY")
	poly_ : PointLightYPlug = PlugDescriptor("pointLightY")
	pointLightZ_ : PointLightZPlug = PlugDescriptor("pointLightZ")
	polz_ : PointLightZPlug = PlugDescriptor("pointLightZ")
	node : FluidShape = None
	pass
class PointLightDecayPlug(Plug):
	node : FluidShape = None
	pass
class PointObjXPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : FluidShape = None
	pass
class PointObjYPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : FluidShape = None
	pass
class PointObjZPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : FluidShape = None
	pass
class PointObjPlug(Plug):
	pointObjX_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pox_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pointObjY_ : PointObjYPlug = PlugDescriptor("pointObjY")
	poy_ : PointObjYPlug = PlugDescriptor("pointObjY")
	pointObjZ_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	poz_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	node : FluidShape = None
	pass
class PointWorldXPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : FluidShape = None
	pass
class PointWorldYPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : FluidShape = None
	pass
class PointWorldZPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : FluidShape = None
	pass
class PointWorldPlug(Plug):
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pwx_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pwy_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pwz_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	node : FluidShape = None
	pass
class PointsPlug(Plug):
	node : FluidShape = None
	pass
class QualityPlug(Plug):
	node : FluidShape = None
	pass
class RandomnessPlug(Plug):
	node : FluidShape = None
	pass
class RatioPlug(Plug):
	node : FluidShape = None
	pass
class RayInstancePlug(Plug):
	node : FluidShape = None
	pass
class ReactionSpeedPlug(Plug):
	node : FluidShape = None
	pass
class RealLightsPlug(Plug):
	node : FluidShape = None
	pass
class RefractiveIndexPlug(Plug):
	node : FluidShape = None
	pass
class RenderInterpolatorPlug(Plug):
	node : FluidShape = None
	pass
class ResizeClosedBoundariesPlug(Plug):
	node : FluidShape = None
	pass
class ResizeInSubstepsPlug(Plug):
	node : FluidShape = None
	pass
class ResizeToEmitterPlug(Plug):
	node : FluidShape = None
	pass
class ResolutionDPlug(Plug):
	parent : ResolutionPlug = PlugDescriptor("resolution")
	node : FluidShape = None
	pass
class ResolutionHPlug(Plug):
	parent : ResolutionPlug = PlugDescriptor("resolution")
	node : FluidShape = None
	pass
class ResolutionWPlug(Plug):
	parent : ResolutionPlug = PlugDescriptor("resolution")
	node : FluidShape = None
	pass
class ResolutionPlug(Plug):
	resolutionD_ : ResolutionDPlug = PlugDescriptor("resolutionD")
	rd_ : ResolutionDPlug = PlugDescriptor("resolutionD")
	resolutionH_ : ResolutionHPlug = PlugDescriptor("resolutionH")
	rh_ : ResolutionHPlug = PlugDescriptor("resolutionH")
	resolutionW_ : ResolutionWPlug = PlugDescriptor("resolutionW")
	rw_ : ResolutionWPlug = PlugDescriptor("resolutionW")
	node : FluidShape = None
	pass
class SampleMethodPlug(Plug):
	node : FluidShape = None
	pass
class SeedPlug(Plug):
	node : FluidShape = None
	pass
class SelfAttractPlug(Plug):
	node : FluidShape = None
	pass
class SelfForcePlug(Plug):
	node : FluidShape = None
	pass
class SelfForceDistancePlug(Plug):
	node : FluidShape = None
	pass
class SelfRepelPlug(Plug):
	node : FluidShape = None
	pass
class SelfShadowingPlug(Plug):
	node : FluidShape = None
	pass
class ShadedDisplayPlug(Plug):
	node : FluidShape = None
	pass
class ShadowDiffusionPlug(Plug):
	node : FluidShape = None
	pass
class ShadowOpacityPlug(Plug):
	node : FluidShape = None
	pass
class SimulationRateScalePlug(Plug):
	node : FluidShape = None
	pass
class SizeRandPlug(Plug):
	node : FluidShape = None
	pass
class SlicesPlug(Plug):
	node : FluidShape = None
	pass
class SoftSurfacePlug(Plug):
	node : FluidShape = None
	pass
class SolverPlug(Plug):
	node : FluidShape = None
	pass
class SolverQualityPlug(Plug):
	node : FluidShape = None
	pass
class SpecularColorBPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : FluidShape = None
	pass
class SpecularColorGPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : FluidShape = None
	pass
class SpecularColorRPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : FluidShape = None
	pass
class SpecularColorPlug(Plug):
	specularColorB_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	spb_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	specularColorG_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	spg_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	specularColorR_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	spr_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	node : FluidShape = None
	pass
class SpottynessPlug(Plug):
	node : FluidShape = None
	pass
class SquareVoxelsPlug(Plug):
	node : FluidShape = None
	pass
class StalksUPlug(Plug):
	node : FluidShape = None
	pass
class StalksVPlug(Plug):
	node : FluidShape = None
	pass
class StartFramePlug(Plug):
	node : FluidShape = None
	pass
class StartTimePlug(Plug):
	node : FluidShape = None
	pass
class SubVolumeCenterDPlug(Plug):
	parent : SubVolumeCenterPlug = PlugDescriptor("subVolumeCenter")
	node : FluidShape = None
	pass
class SubVolumeCenterHPlug(Plug):
	parent : SubVolumeCenterPlug = PlugDescriptor("subVolumeCenter")
	node : FluidShape = None
	pass
class SubVolumeCenterWPlug(Plug):
	parent : SubVolumeCenterPlug = PlugDescriptor("subVolumeCenter")
	node : FluidShape = None
	pass
class SubVolumeCenterPlug(Plug):
	subVolumeCenterD_ : SubVolumeCenterDPlug = PlugDescriptor("subVolumeCenterD")
	scd_ : SubVolumeCenterDPlug = PlugDescriptor("subVolumeCenterD")
	subVolumeCenterH_ : SubVolumeCenterHPlug = PlugDescriptor("subVolumeCenterH")
	sch_ : SubVolumeCenterHPlug = PlugDescriptor("subVolumeCenterH")
	subVolumeCenterW_ : SubVolumeCenterWPlug = PlugDescriptor("subVolumeCenterW")
	scw_ : SubVolumeCenterWPlug = PlugDescriptor("subVolumeCenterW")
	node : FluidShape = None
	pass
class SubVolumeSizeDPlug(Plug):
	parent : SubVolumeSizePlug = PlugDescriptor("subVolumeSize")
	node : FluidShape = None
	pass
class SubVolumeSizeHPlug(Plug):
	parent : SubVolumeSizePlug = PlugDescriptor("subVolumeSize")
	node : FluidShape = None
	pass
class SubVolumeSizeWPlug(Plug):
	parent : SubVolumeSizePlug = PlugDescriptor("subVolumeSize")
	node : FluidShape = None
	pass
class SubVolumeSizePlug(Plug):
	subVolumeSizeD_ : SubVolumeSizeDPlug = PlugDescriptor("subVolumeSizeD")
	ssd_ : SubVolumeSizeDPlug = PlugDescriptor("subVolumeSizeD")
	subVolumeSizeH_ : SubVolumeSizeHPlug = PlugDescriptor("subVolumeSizeH")
	ssh_ : SubVolumeSizeHPlug = PlugDescriptor("subVolumeSizeH")
	subVolumeSizeW_ : SubVolumeSizeWPlug = PlugDescriptor("subVolumeSizeW")
	ssw_ : SubVolumeSizeWPlug = PlugDescriptor("subVolumeSizeW")
	node : FluidShape = None
	pass
class SubstepsPlug(Plug):
	node : FluidShape = None
	pass
class SurfaceRenderPlug(Plug):
	node : FluidShape = None
	pass
class SurfaceShaderDepthPlug(Plug):
	node : FluidShape = None
	pass
class SurfaceThresholdPlug(Plug):
	node : FluidShape = None
	pass
class SurfaceTolerancePlug(Plug):
	node : FluidShape = None
	pass
class TemperatureDiffusionPlug(Plug):
	node : FluidShape = None
	pass
class TemperatureDissipationPlug(Plug):
	node : FluidShape = None
	pass
class TemperatureGradientPlug(Plug):
	node : FluidShape = None
	pass
class TemperatureMethodPlug(Plug):
	node : FluidShape = None
	pass
class TemperatureNoisePlug(Plug):
	node : FluidShape = None
	pass
class TemperaturePressurePlug(Plug):
	node : FluidShape = None
	pass
class TemperaturePressureThresholdPlug(Plug):
	node : FluidShape = None
	pass
class TemperatureScalePlug(Plug):
	node : FluidShape = None
	pass
class TemperatureTensionPlug(Plug):
	node : FluidShape = None
	pass
class TemperatureTurbulencePlug(Plug):
	node : FluidShape = None
	pass
class TensionForcePlug(Plug):
	node : FluidShape = None
	pass
class TextureOriginXPlug(Plug):
	parent : TextureOriginPlug = PlugDescriptor("textureOrigin")
	node : FluidShape = None
	pass
class TextureOriginYPlug(Plug):
	parent : TextureOriginPlug = PlugDescriptor("textureOrigin")
	node : FluidShape = None
	pass
class TextureOriginZPlug(Plug):
	parent : TextureOriginPlug = PlugDescriptor("textureOrigin")
	node : FluidShape = None
	pass
class TextureOriginPlug(Plug):
	textureOriginX_ : TextureOriginXPlug = PlugDescriptor("textureOriginX")
	torx_ : TextureOriginXPlug = PlugDescriptor("textureOriginX")
	textureOriginY_ : TextureOriginYPlug = PlugDescriptor("textureOriginY")
	tory_ : TextureOriginYPlug = PlugDescriptor("textureOriginY")
	textureOriginZ_ : TextureOriginZPlug = PlugDescriptor("textureOriginZ")
	torz_ : TextureOriginZPlug = PlugDescriptor("textureOriginZ")
	node : FluidShape = None
	pass
class TextureRotateXPlug(Plug):
	parent : TextureRotatePlug = PlugDescriptor("textureRotate")
	node : FluidShape = None
	pass
class TextureRotateYPlug(Plug):
	parent : TextureRotatePlug = PlugDescriptor("textureRotate")
	node : FluidShape = None
	pass
class TextureRotateZPlug(Plug):
	parent : TextureRotatePlug = PlugDescriptor("textureRotate")
	node : FluidShape = None
	pass
class TextureRotatePlug(Plug):
	textureRotateX_ : TextureRotateXPlug = PlugDescriptor("textureRotateX")
	trtx_ : TextureRotateXPlug = PlugDescriptor("textureRotateX")
	textureRotateY_ : TextureRotateYPlug = PlugDescriptor("textureRotateY")
	trty_ : TextureRotateYPlug = PlugDescriptor("textureRotateY")
	textureRotateZ_ : TextureRotateZPlug = PlugDescriptor("textureRotateZ")
	trtz_ : TextureRotateZPlug = PlugDescriptor("textureRotateZ")
	node : FluidShape = None
	pass
class TextureScaleXPlug(Plug):
	parent : TextureScalePlug = PlugDescriptor("textureScale")
	node : FluidShape = None
	pass
class TextureScaleYPlug(Plug):
	parent : TextureScalePlug = PlugDescriptor("textureScale")
	node : FluidShape = None
	pass
class TextureScaleZPlug(Plug):
	parent : TextureScalePlug = PlugDescriptor("textureScale")
	node : FluidShape = None
	pass
class TextureScalePlug(Plug):
	textureScaleX_ : TextureScaleXPlug = PlugDescriptor("textureScaleX")
	tscx_ : TextureScaleXPlug = PlugDescriptor("textureScaleX")
	textureScaleY_ : TextureScaleYPlug = PlugDescriptor("textureScaleY")
	tscy_ : TextureScaleYPlug = PlugDescriptor("textureScaleY")
	textureScaleZ_ : TextureScaleZPlug = PlugDescriptor("textureScaleZ")
	tscz_ : TextureScaleZPlug = PlugDescriptor("textureScaleZ")
	node : FluidShape = None
	pass
class TextureTimePlug(Plug):
	node : FluidShape = None
	pass
class TextureTypePlug(Plug):
	node : FluidShape = None
	pass
class ThresholdPlug(Plug):
	node : FluidShape = None
	pass
class TransparencyBPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : FluidShape = None
	pass
class TransparencyGPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : FluidShape = None
	pass
class TransparencyRPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : FluidShape = None
	pass
class TransparencyPlug(Plug):
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	tb_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	tg_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	tr_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	node : FluidShape = None
	pass
class TurbulenceFrequencyPlug(Plug):
	node : FluidShape = None
	pass
class TurbulenceResPlug(Plug):
	node : FluidShape = None
	pass
class TurbulenceSpeedPlug(Plug):
	node : FluidShape = None
	pass
class TurbulenceStrengthPlug(Plug):
	node : FluidShape = None
	pass
class UseGradientNormalsPlug(Plug):
	node : FluidShape = None
	pass
class UsePre70DynamicsPlug(Plug):
	node : FluidShape = None
	pass
class UvwPerVertexPlug(Plug):
	node : FluidShape = None
	pass
class VelocityAdvectPlug(Plug):
	node : FluidShape = None
	pass
class VelocityDampPlug(Plug):
	node : FluidShape = None
	pass
class VelocityDrawPlug(Plug):
	node : FluidShape = None
	pass
class VelocityDrawLengthPlug(Plug):
	node : FluidShape = None
	pass
class VelocityDrawSkipPlug(Plug):
	node : FluidShape = None
	pass
class VelocityGradientPlug(Plug):
	node : FluidShape = None
	pass
class VelocityMethodPlug(Plug):
	node : FluidShape = None
	pass
class VelocityNoisePlug(Plug):
	node : FluidShape = None
	pass
class VelocityPerVertexPlug(Plug):
	node : FluidShape = None
	pass
class VelocityProjectPlug(Plug):
	node : FluidShape = None
	pass
class VelocityScaleXPlug(Plug):
	parent : VelocityScalePlug = PlugDescriptor("velocityScale")
	node : FluidShape = None
	pass
class VelocityScaleYPlug(Plug):
	parent : VelocityScalePlug = PlugDescriptor("velocityScale")
	node : FluidShape = None
	pass
class VelocityScaleZPlug(Plug):
	parent : VelocityScalePlug = PlugDescriptor("velocityScale")
	node : FluidShape = None
	pass
class VelocityScalePlug(Plug):
	velocityScaleX_ : VelocityScaleXPlug = PlugDescriptor("velocityScaleX")
	vsx_ : VelocityScaleXPlug = PlugDescriptor("velocityScaleX")
	velocityScaleY_ : VelocityScaleYPlug = PlugDescriptor("velocityScaleY")
	vsy_ : VelocityScaleYPlug = PlugDescriptor("velocityScaleY")
	velocityScaleZ_ : VelocityScaleZPlug = PlugDescriptor("velocityScaleZ")
	vsz_ : VelocityScaleZPlug = PlugDescriptor("velocityScaleZ")
	node : FluidShape = None
	pass
class VelocitySwirlPlug(Plug):
	node : FluidShape = None
	pass
class ViscosityPlug(Plug):
	node : FluidShape = None
	pass
class VoxelQualityPlug(Plug):
	node : FluidShape = None
	pass
class WireframeDisplayPlug(Plug):
	node : FluidShape = None
	pass
class ZoomFactorPlug(Plug):
	node : FluidShape = None
	pass
# endregion


# define node class
class FluidShape(SurfaceShape):
	airFuelRatio_ : AirFuelRatioPlug = PlugDescriptor("airFuelRatio")
	ambientBrightness_ : AmbientBrightnessPlug = PlugDescriptor("ambientBrightness")
	ambientColorB_ : AmbientColorBPlug = PlugDescriptor("ambientColorB")
	ambientColorG_ : AmbientColorGPlug = PlugDescriptor("ambientColorG")
	ambientColorR_ : AmbientColorRPlug = PlugDescriptor("ambientColorR")
	ambientColor_ : AmbientColorPlug = PlugDescriptor("ambientColor")
	ambientDiffusion_ : AmbientDiffusionPlug = PlugDescriptor("ambientDiffusion")
	amplitude_ : AmplitudePlug = PlugDescriptor("amplitude")
	autoResize_ : AutoResizePlug = PlugDescriptor("autoResize")
	autoResizeMargin_ : AutoResizeMarginPlug = PlugDescriptor("autoResizeMargin")
	autoResizeThreshold_ : AutoResizeThresholdPlug = PlugDescriptor("autoResizeThreshold")
	baseResolution_ : BaseResolutionPlug = PlugDescriptor("baseResolution")
	billowDensity_ : BillowDensityPlug = PlugDescriptor("billowDensity")
	boundaryDraw_ : BoundaryDrawPlug = PlugDescriptor("boundaryDraw")
	boundaryX_ : BoundaryXPlug = PlugDescriptor("boundaryX")
	boundaryY_ : BoundaryYPlug = PlugDescriptor("boundaryY")
	boundaryZ_ : BoundaryZPlug = PlugDescriptor("boundaryZ")
	boxMinRadius_ : BoxMinRadiusPlug = PlugDescriptor("boxMinRadius")
	boxRadius_ : BoxRadiusPlug = PlugDescriptor("boxRadius")
	boxRatio_ : BoxRatioPlug = PlugDescriptor("boxRatio")
	buoyancy_ : BuoyancyPlug = PlugDescriptor("buoyancy")
	cacheColor_ : CacheColorPlug = PlugDescriptor("cacheColor")
	cacheDensity_ : CacheDensityPlug = PlugDescriptor("cacheDensity")
	cacheFalloff_ : CacheFalloffPlug = PlugDescriptor("cacheFalloff")
	cacheReaction_ : CacheReactionPlug = PlugDescriptor("cacheReaction")
	cacheTemperature_ : CacheTemperaturePlug = PlugDescriptor("cacheTemperature")
	cacheTextureCoordinates_ : CacheTextureCoordinatesPlug = PlugDescriptor("cacheTextureCoordinates")
	cacheVelocity_ : CacheVelocityPlug = PlugDescriptor("cacheVelocity")
	checker_ : CheckerPlug = PlugDescriptor("checker")
	circleRadius_ : CircleRadiusPlug = PlugDescriptor("circleRadius")
	circleSizeRatio_ : CircleSizeRatioPlug = PlugDescriptor("circleSizeRatio")
	circles_ : CirclesPlug = PlugDescriptor("circles")
	collide_ : CollidePlug = PlugDescriptor("collide")
	collisionFriction_ : CollisionFrictionPlug = PlugDescriptor("collisionFriction")
	collisionGeometry_ : CollisionGeometryPlug = PlugDescriptor("collisionGeometry")
	collisionResilience_ : CollisionResiliencePlug = PlugDescriptor("collisionResilience")
	collisionData_ : CollisionDataPlug = PlugDescriptor("collisionData")
	color_ColorB_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	color_ColorG_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	color_ColorR_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	color_Color_ : Color_ColorPlug = PlugDescriptor("color_Color")
	color_Interp_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	color_Position_ : Color_PositionPlug = PlugDescriptor("color_Position")
	color_ : ColorPlug = PlugDescriptor("color")
	colorDiffusion_ : ColorDiffusionPlug = PlugDescriptor("colorDiffusion")
	colorDissipation_ : ColorDissipationPlug = PlugDescriptor("colorDissipation")
	colorInput_ : ColorInputPlug = PlugDescriptor("colorInput")
	colorInputBias_ : ColorInputBiasPlug = PlugDescriptor("colorInputBias")
	colorMethod_ : ColorMethodPlug = PlugDescriptor("colorMethod")
	colorPerVertex_ : ColorPerVertexPlug = PlugDescriptor("colorPerVertex")
	colorTexGain_ : ColorTexGainPlug = PlugDescriptor("colorTexGain")
	colorTexture_ : ColorTexturePlug = PlugDescriptor("colorTexture")
	conserveMass_ : ConserveMassPlug = PlugDescriptor("conserveMass")
	contrastTolerance_ : ContrastTolerancePlug = PlugDescriptor("contrastTolerance")
	coordinateMethod_ : CoordinateMethodPlug = PlugDescriptor("coordinateMethod")
	coordinateSpeed_ : CoordinateSpeedPlug = PlugDescriptor("coordinateSpeed")
	cosinePower_ : CosinePowerPlug = PlugDescriptor("cosinePower")
	currentTime_ : CurrentTimePlug = PlugDescriptor("currentTime")
	densityBuoyancy_ : DensityBuoyancyPlug = PlugDescriptor("densityBuoyancy")
	densityDiffusion_ : DensityDiffusionPlug = PlugDescriptor("densityDiffusion")
	densityDissipation_ : DensityDissipationPlug = PlugDescriptor("densityDissipation")
	densityGradient_ : DensityGradientPlug = PlugDescriptor("densityGradient")
	densityGradientForce_ : DensityGradientForcePlug = PlugDescriptor("densityGradientForce")
	densityMethod_ : DensityMethodPlug = PlugDescriptor("densityMethod")
	densityNoise_ : DensityNoisePlug = PlugDescriptor("densityNoise")
	densityPressure_ : DensityPressurePlug = PlugDescriptor("densityPressure")
	densityPressureThreshold_ : DensityPressureThresholdPlug = PlugDescriptor("densityPressureThreshold")
	densityScale_ : DensityScalePlug = PlugDescriptor("densityScale")
	densityTension_ : DensityTensionPlug = PlugDescriptor("densityTension")
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	dimensionsD_ : DimensionsDPlug = PlugDescriptor("dimensionsD")
	dimensionsH_ : DimensionsHPlug = PlugDescriptor("dimensionsH")
	dimensionsW_ : DimensionsWPlug = PlugDescriptor("dimensionsW")
	dimensions_ : DimensionsPlug = PlugDescriptor("dimensions")
	directionalLightX_ : DirectionalLightXPlug = PlugDescriptor("directionalLightX")
	directionalLightY_ : DirectionalLightYPlug = PlugDescriptor("directionalLightY")
	directionalLightZ_ : DirectionalLightZPlug = PlugDescriptor("directionalLightZ")
	directionalLight_ : DirectionalLightPlug = PlugDescriptor("directionalLight")
	disableInteractiveEval_ : DisableInteractiveEvalPlug = PlugDescriptor("disableInteractiveEval")
	diskCache_ : DiskCachePlug = PlugDescriptor("diskCache")
	diskCacheIC_ : DiskCacheICPlug = PlugDescriptor("diskCacheIC")
	doEmission_ : DoEmissionPlug = PlugDescriptor("doEmission")
	doFields_ : DoFieldsPlug = PlugDescriptor("doFields")
	drawHeads_ : DrawHeadsPlug = PlugDescriptor("drawHeads")
	drawSubVolume_ : DrawSubVolumePlug = PlugDescriptor("drawSubVolume")
	dropoffShape_ : DropoffShapePlug = PlugDescriptor("dropoffShape")
	dynamicOffsetX_ : DynamicOffsetXPlug = PlugDescriptor("dynamicOffsetX")
	dynamicOffsetY_ : DynamicOffsetYPlug = PlugDescriptor("dynamicOffsetY")
	dynamicOffsetZ_ : DynamicOffsetZPlug = PlugDescriptor("dynamicOffsetZ")
	dynamicOffset_ : DynamicOffsetPlug = PlugDescriptor("dynamicOffset")
	edgeDropoff_ : EdgeDropoffPlug = PlugDescriptor("edgeDropoff")
	emissionFunction_Hidden_ : EmissionFunction_HiddenPlug = PlugDescriptor("emissionFunction_Hidden")
	emissionFunction_InmapFrom_ : EmissionFunction_InmapFromPlug = PlugDescriptor("emissionFunction_InmapFrom")
	emissionFunction_InmapTo_ : EmissionFunction_InmapToPlug = PlugDescriptor("emissionFunction_InmapTo")
	emissionFunction_Inmap_ : EmissionFunction_InmapPlug = PlugDescriptor("emissionFunction_Inmap")
	emissionFunction_OutmapFrom_ : EmissionFunction_OutmapFromPlug = PlugDescriptor("emissionFunction_OutmapFrom")
	emissionFunction_OutmapTo_ : EmissionFunction_OutmapToPlug = PlugDescriptor("emissionFunction_OutmapTo")
	emissionFunction_Outmap_ : EmissionFunction_OutmapPlug = PlugDescriptor("emissionFunction_Outmap")
	emissionFunction_Raw_ : EmissionFunction_RawPlug = PlugDescriptor("emissionFunction_Raw")
	emissionFunction_ : EmissionFunctionPlug = PlugDescriptor("emissionFunction")
	emissionList_ : EmissionListPlug = PlugDescriptor("emissionList")
	emitInSubsteps_ : EmitInSubstepsPlug = PlugDescriptor("emitInSubsteps")
	enableLiquidSimulation_ : EnableLiquidSimulationPlug = PlugDescriptor("enableLiquidSimulation")
	environment_ColorB_ : Environment_ColorBPlug = PlugDescriptor("environment_ColorB")
	environment_ColorG_ : Environment_ColorGPlug = PlugDescriptor("environment_ColorG")
	environment_ColorR_ : Environment_ColorRPlug = PlugDescriptor("environment_ColorR")
	environment_Color_ : Environment_ColorPlug = PlugDescriptor("environment_Color")
	environment_Interp_ : Environment_InterpPlug = PlugDescriptor("environment_Interp")
	environment_Position_ : Environment_PositionPlug = PlugDescriptor("environment_Position")
	environment_ : EnvironmentPlug = PlugDescriptor("environment")
	equilibriumValue_ : EquilibriumValuePlug = PlugDescriptor("equilibriumValue")
	escapeRadius_ : EscapeRadiusPlug = PlugDescriptor("escapeRadius")
	falloff_ : FalloffPlug = PlugDescriptor("falloff")
	falloffMethod_ : FalloffMethodPlug = PlugDescriptor("falloffMethod")
	farPointObjectX_ : FarPointObjectXPlug = PlugDescriptor("farPointObjectX")
	farPointObjectY_ : FarPointObjectYPlug = PlugDescriptor("farPointObjectY")
	farPointObjectZ_ : FarPointObjectZPlug = PlugDescriptor("farPointObjectZ")
	farPointObj_ : FarPointObjPlug = PlugDescriptor("farPointObj")
	farPointWorldX_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	farPointWorldY_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	farPointWorldZ_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	farPointWorld_ : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	fieldDataDeltaTime_ : FieldDataDeltaTimePlug = PlugDescriptor("fieldDataDeltaTime")
	fieldDataMass_ : FieldDataMassPlug = PlugDescriptor("fieldDataMass")
	fieldDataPosition_ : FieldDataPositionPlug = PlugDescriptor("fieldDataPosition")
	fieldDataVelocity_ : FieldDataVelocityPlug = PlugDescriptor("fieldDataVelocity")
	fieldData_ : FieldDataPlug = PlugDescriptor("fieldData")
	fieldFunction_Hidden_ : FieldFunction_HiddenPlug = PlugDescriptor("fieldFunction_Hidden")
	fieldFunction_InmapFrom_ : FieldFunction_InmapFromPlug = PlugDescriptor("fieldFunction_InmapFrom")
	fieldFunction_InmapTo_ : FieldFunction_InmapToPlug = PlugDescriptor("fieldFunction_InmapTo")
	fieldFunction_Inmap_ : FieldFunction_InmapPlug = PlugDescriptor("fieldFunction_Inmap")
	fieldFunction_OutmapFrom_ : FieldFunction_OutmapFromPlug = PlugDescriptor("fieldFunction_OutmapFrom")
	fieldFunction_OutmapTo_ : FieldFunction_OutmapToPlug = PlugDescriptor("fieldFunction_OutmapTo")
	fieldFunction_Outmap_ : FieldFunction_OutmapPlug = PlugDescriptor("fieldFunction_Outmap")
	fieldFunction_Raw_ : FieldFunction_RawPlug = PlugDescriptor("fieldFunction_Raw")
	fieldFunction_ : FieldFunctionPlug = PlugDescriptor("fieldFunction")
	fieldList_ : FieldListPlug = PlugDescriptor("fieldList")
	filterSizeX_ : FilterSizeXPlug = PlugDescriptor("filterSizeX")
	filterSizeY_ : FilterSizeYPlug = PlugDescriptor("filterSizeY")
	filterSizeZ_ : FilterSizeZPlug = PlugDescriptor("filterSizeZ")
	filterSize_ : FilterSizePlug = PlugDescriptor("filterSize")
	fluidColorEmission_ : FluidColorEmissionPlug = PlugDescriptor("fluidColorEmission")
	fluidLightColorB_ : FluidLightColorBPlug = PlugDescriptor("fluidLightColorB")
	fluidLightColorG_ : FluidLightColorGPlug = PlugDescriptor("fluidLightColorG")
	fluidLightColorR_ : FluidLightColorRPlug = PlugDescriptor("fluidLightColorR")
	fluidLightColor_ : FluidLightColorPlug = PlugDescriptor("fluidLightColor")
	fluidReactantEmission_ : FluidReactantEmissionPlug = PlugDescriptor("fluidReactantEmission")
	focus_ : FocusPlug = PlugDescriptor("focus")
	forceDynamics_ : ForceDynamicsPlug = PlugDescriptor("forceDynamics")
	forwardAdvection_ : ForwardAdvectionPlug = PlugDescriptor("forwardAdvection")
	frequency_ : FrequencyPlug = PlugDescriptor("frequency")
	frequencyRatio_ : FrequencyRatioPlug = PlugDescriptor("frequencyRatio")
	friction_ : FrictionPlug = PlugDescriptor("friction")
	fuelGradient_ : FuelGradientPlug = PlugDescriptor("fuelGradient")
	fuelIgnitionTemp_ : FuelIgnitionTempPlug = PlugDescriptor("fuelIgnitionTemp")
	fuelMethod_ : FuelMethodPlug = PlugDescriptor("fuelMethod")
	fuelScale_ : FuelScalePlug = PlugDescriptor("fuelScale")
	glowIntensity_ : GlowIntensityPlug = PlugDescriptor("glowIntensity")
	gravity_ : GravityPlug = PlugDescriptor("gravity")
	gridInterpolator_ : GridInterpolatorPlug = PlugDescriptor("gridInterpolator")
	hardwareSelfShadow_ : HardwareSelfShadowPlug = PlugDescriptor("hardwareSelfShadow")
	heatReleased_ : HeatReleasedPlug = PlugDescriptor("heatReleased")
	heightField_ : HeightFieldPlug = PlugDescriptor("heightField")
	highDetailSolve_ : HighDetailSolvePlug = PlugDescriptor("highDetailSolve")
	implode_ : ImplodePlug = PlugDescriptor("implode")
	implodeCenterX_ : ImplodeCenterXPlug = PlugDescriptor("implodeCenterX")
	implodeCenterY_ : ImplodeCenterYPlug = PlugDescriptor("implodeCenterY")
	implodeCenterZ_ : ImplodeCenterZPlug = PlugDescriptor("implodeCenterZ")
	implodeCenter_ : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	inColor_ : InColorPlug = PlugDescriptor("inColor")
	inDensity_ : InDensityPlug = PlugDescriptor("inDensity")
	inFalloff_ : InFalloffPlug = PlugDescriptor("inFalloff")
	inOffset_ : InOffsetPlug = PlugDescriptor("inOffset")
	inReaction_ : InReactionPlug = PlugDescriptor("inReaction")
	inResolution_ : InResolutionPlug = PlugDescriptor("inResolution")
	inTemperature_ : InTemperaturePlug = PlugDescriptor("inTemperature")
	inTextureCoordinates_ : InTextureCoordinatesPlug = PlugDescriptor("inTextureCoordinates")
	inVelocity_ : InVelocityPlug = PlugDescriptor("inVelocity")
	incandTexGain_ : IncandTexGainPlug = PlugDescriptor("incandTexGain")
	incandTexture_ : IncandTexturePlug = PlugDescriptor("incandTexture")
	incandescence_ColorB_ : Incandescence_ColorBPlug = PlugDescriptor("incandescence_ColorB")
	incandescence_ColorG_ : Incandescence_ColorGPlug = PlugDescriptor("incandescence_ColorG")
	incandescence_ColorR_ : Incandescence_ColorRPlug = PlugDescriptor("incandescence_ColorR")
	incandescence_Color_ : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	incandescence_Interp_ : Incandescence_InterpPlug = PlugDescriptor("incandescence_Interp")
	incandescence_Position_ : Incandescence_PositionPlug = PlugDescriptor("incandescence_Position")
	incandescence_ : IncandescencePlug = PlugDescriptor("incandescence")
	incandescenceInput_ : IncandescenceInputPlug = PlugDescriptor("incandescenceInput")
	incandescenceInputBias_ : IncandescenceInputBiasPlug = PlugDescriptor("incandescenceInputBias")
	incandescencePerVertex_ : IncandescencePerVertexPlug = PlugDescriptor("incandescencePerVertex")
	inflection_ : InflectionPlug = PlugDescriptor("inflection")
	inheritFactor_ : InheritFactorPlug = PlugDescriptor("inheritFactor")
	initialConditions_ : InitialConditionsPlug = PlugDescriptor("initialConditions")
	deltaTime_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	inputMass_ : InputMassPlug = PlugDescriptor("inputMass")
	inputPositions_ : InputPositionsPlug = PlugDescriptor("inputPositions")
	inputVelocities_ : InputVelocitiesPlug = PlugDescriptor("inputVelocities")
	inputData_ : InputDataPlug = PlugDescriptor("inputData")
	inputForce_ : InputForcePlug = PlugDescriptor("inputForce")
	inputForce2_ : InputForce2Plug = PlugDescriptor("inputForce2")
	invertTexture_ : InvertTexturePlug = PlugDescriptor("invertTexture")
	is2d_ : Is2dPlug = PlugDescriptor("is2d")
	isFull_ : IsFullPlug = PlugDescriptor("isFull")
	juliaU_ : JuliaUPlug = PlugDescriptor("juliaU")
	juliaV_ : JuliaVPlug = PlugDescriptor("juliaV")
	lastEvalTime_ : LastEvalTimePlug = PlugDescriptor("lastEvalTime")
	leafEffect_ : LeafEffectPlug = PlugDescriptor("leafEffect")
	lightBrightness_ : LightBrightnessPlug = PlugDescriptor("lightBrightness")
	lightColorB_ : LightColorBPlug = PlugDescriptor("lightColorB")
	lightColorG_ : LightColorGPlug = PlugDescriptor("lightColorG")
	lightColorR_ : LightColorRPlug = PlugDescriptor("lightColorR")
	lightColor_ : LightColorPlug = PlugDescriptor("lightColor")
	lightAmbient_ : LightAmbientPlug = PlugDescriptor("lightAmbient")
	lightBlindData_ : LightBlindDataPlug = PlugDescriptor("lightBlindData")
	lightDiffuse_ : LightDiffusePlug = PlugDescriptor("lightDiffuse")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	lightDirection_ : LightDirectionPlug = PlugDescriptor("lightDirection")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lightIntensity_ : LightIntensityPlug = PlugDescriptor("lightIntensity")
	lightShadowFraction_ : LightShadowFractionPlug = PlugDescriptor("lightShadowFraction")
	lightSpecular_ : LightSpecularPlug = PlugDescriptor("lightSpecular")
	preShadowIntensity_ : PreShadowIntensityPlug = PlugDescriptor("preShadowIntensity")
	lightDataArray_ : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightReleased_ : LightReleasedPlug = PlugDescriptor("lightReleased")
	lightType_ : LightTypePlug = PlugDescriptor("lightType")
	lineBlending_ : LineBlendingPlug = PlugDescriptor("lineBlending")
	lineFocus_ : LineFocusPlug = PlugDescriptor("lineFocus")
	lineOffsetRatio_ : LineOffsetRatioPlug = PlugDescriptor("lineOffsetRatio")
	lineOffsetU_ : LineOffsetUPlug = PlugDescriptor("lineOffsetU")
	lineOffsetV_ : LineOffsetVPlug = PlugDescriptor("lineOffsetV")
	liquidMethod_ : LiquidMethodPlug = PlugDescriptor("liquidMethod")
	liquidMinDensity_ : LiquidMinDensityPlug = PlugDescriptor("liquidMinDensity")
	liquidMistFall_ : LiquidMistFallPlug = PlugDescriptor("liquidMistFall")
	loadColor_ : LoadColorPlug = PlugDescriptor("loadColor")
	loadDensity_ : LoadDensityPlug = PlugDescriptor("loadDensity")
	loadFalloff_ : LoadFalloffPlug = PlugDescriptor("loadFalloff")
	loadReaction_ : LoadReactionPlug = PlugDescriptor("loadReaction")
	loadTemperature_ : LoadTemperaturePlug = PlugDescriptor("loadTemperature")
	loadTextureCoordinates_ : LoadTextureCoordinatesPlug = PlugDescriptor("loadTextureCoordinates")
	loadVelocity_ : LoadVelocityPlug = PlugDescriptor("loadVelocity")
	lobes_ : LobesPlug = PlugDescriptor("lobes")
	lockDrawAxis_ : LockDrawAxisPlug = PlugDescriptor("lockDrawAxis")
	mandelbrotDepth_ : MandelbrotDepthPlug = PlugDescriptor("mandelbrotDepth")
	mandelbrotInsideMethod_ : MandelbrotInsideMethodPlug = PlugDescriptor("mandelbrotInsideMethod")
	mandelbrotShadeMethod_ : MandelbrotShadeMethodPlug = PlugDescriptor("mandelbrotShadeMethod")
	mandelbrotType_ : MandelbrotTypePlug = PlugDescriptor("mandelbrotType")
	massConversion_ : MassConversionPlug = PlugDescriptor("massConversion")
	massRange_ : MassRangePlug = PlugDescriptor("massRange")
	matrixEyeToWorld_ : MatrixEyeToWorldPlug = PlugDescriptor("matrixEyeToWorld")
	matrixWorldToObject_ : MatrixWorldToObjectPlug = PlugDescriptor("matrixWorldToObject")
	matteOpacity_ : MatteOpacityPlug = PlugDescriptor("matteOpacity")
	matteOpacityMode_ : MatteOpacityModePlug = PlugDescriptor("matteOpacityMode")
	maxReactionTemp_ : MaxReactionTempPlug = PlugDescriptor("maxReactionTemp")
	maxResolution_ : MaxResolutionPlug = PlugDescriptor("maxResolution")
	meshMethod_ : MeshMethodPlug = PlugDescriptor("meshMethod")
	meshResolution_ : MeshResolutionPlug = PlugDescriptor("meshResolution")
	meshSmoothingIterations_ : MeshSmoothingIterationsPlug = PlugDescriptor("meshSmoothingIterations")
	numWaves_ : NumWavesPlug = PlugDescriptor("numWaves")
	numericDisplay_ : NumericDisplayPlug = PlugDescriptor("numericDisplay")
	objectType_ : ObjectTypePlug = PlugDescriptor("objectType")
	opacity_FloatValue_ : Opacity_FloatValuePlug = PlugDescriptor("opacity_FloatValue")
	opacity_Interp_ : Opacity_InterpPlug = PlugDescriptor("opacity_Interp")
	opacity_Position_ : Opacity_PositionPlug = PlugDescriptor("opacity_Position")
	opacity_ : OpacityPlug = PlugDescriptor("opacity")
	opacityInput_ : OpacityInputPlug = PlugDescriptor("opacityInput")
	opacityInputBias_ : OpacityInputBiasPlug = PlugDescriptor("opacityInputBias")
	opacityPerVertex_ : OpacityPerVertexPlug = PlugDescriptor("opacityPerVertex")
	opacityPreviewGain_ : OpacityPreviewGainPlug = PlugDescriptor("opacityPreviewGain")
	opacityTexGain_ : OpacityTexGainPlug = PlugDescriptor("opacityTexGain")
	opacityTexture_ : OpacityTexturePlug = PlugDescriptor("opacityTexture")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	outGlowColor_ : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	outGrid_ : OutGridPlug = PlugDescriptor("outGrid")
	outMatteOpacityB_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	outMatteOpacityG_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	outMatteOpacityR_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	outMatteOpacity_ : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	outMesh_ : OutMeshPlug = PlugDescriptor("outMesh")
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	outTransparency_ : OutTransparencyPlug = PlugDescriptor("outTransparency")
	outputForce_ : OutputForcePlug = PlugDescriptor("outputForce")
	overrideTimeStep_ : OverrideTimeStepPlug = PlugDescriptor("overrideTimeStep")
	particleWeight_ : ParticleWeightPlug = PlugDescriptor("particleWeight")
	playFromCache_ : PlayFromCachePlug = PlugDescriptor("playFromCache")
	pointLightX_ : PointLightXPlug = PlugDescriptor("pointLightX")
	pointLightY_ : PointLightYPlug = PlugDescriptor("pointLightY")
	pointLightZ_ : PointLightZPlug = PlugDescriptor("pointLightZ")
	pointLight_ : PointLightPlug = PlugDescriptor("pointLight")
	pointLightDecay_ : PointLightDecayPlug = PlugDescriptor("pointLightDecay")
	pointObjX_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pointObjY_ : PointObjYPlug = PlugDescriptor("pointObjY")
	pointObjZ_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	pointObj_ : PointObjPlug = PlugDescriptor("pointObj")
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pointWorld_ : PointWorldPlug = PlugDescriptor("pointWorld")
	points_ : PointsPlug = PlugDescriptor("points")
	quality_ : QualityPlug = PlugDescriptor("quality")
	randomness_ : RandomnessPlug = PlugDescriptor("randomness")
	ratio_ : RatioPlug = PlugDescriptor("ratio")
	rayInstance_ : RayInstancePlug = PlugDescriptor("rayInstance")
	reactionSpeed_ : ReactionSpeedPlug = PlugDescriptor("reactionSpeed")
	realLights_ : RealLightsPlug = PlugDescriptor("realLights")
	refractiveIndex_ : RefractiveIndexPlug = PlugDescriptor("refractiveIndex")
	renderInterpolator_ : RenderInterpolatorPlug = PlugDescriptor("renderInterpolator")
	resizeClosedBoundaries_ : ResizeClosedBoundariesPlug = PlugDescriptor("resizeClosedBoundaries")
	resizeInSubsteps_ : ResizeInSubstepsPlug = PlugDescriptor("resizeInSubsteps")
	resizeToEmitter_ : ResizeToEmitterPlug = PlugDescriptor("resizeToEmitter")
	resolutionD_ : ResolutionDPlug = PlugDescriptor("resolutionD")
	resolutionH_ : ResolutionHPlug = PlugDescriptor("resolutionH")
	resolutionW_ : ResolutionWPlug = PlugDescriptor("resolutionW")
	resolution_ : ResolutionPlug = PlugDescriptor("resolution")
	sampleMethod_ : SampleMethodPlug = PlugDescriptor("sampleMethod")
	seed_ : SeedPlug = PlugDescriptor("seed")
	selfAttract_ : SelfAttractPlug = PlugDescriptor("selfAttract")
	selfForce_ : SelfForcePlug = PlugDescriptor("selfForce")
	selfForceDistance_ : SelfForceDistancePlug = PlugDescriptor("selfForceDistance")
	selfRepel_ : SelfRepelPlug = PlugDescriptor("selfRepel")
	selfShadowing_ : SelfShadowingPlug = PlugDescriptor("selfShadowing")
	shadedDisplay_ : ShadedDisplayPlug = PlugDescriptor("shadedDisplay")
	shadowDiffusion_ : ShadowDiffusionPlug = PlugDescriptor("shadowDiffusion")
	shadowOpacity_ : ShadowOpacityPlug = PlugDescriptor("shadowOpacity")
	simulationRateScale_ : SimulationRateScalePlug = PlugDescriptor("simulationRateScale")
	sizeRand_ : SizeRandPlug = PlugDescriptor("sizeRand")
	slices_ : SlicesPlug = PlugDescriptor("slices")
	softSurface_ : SoftSurfacePlug = PlugDescriptor("softSurface")
	solver_ : SolverPlug = PlugDescriptor("solver")
	solverQuality_ : SolverQualityPlug = PlugDescriptor("solverQuality")
	specularColorB_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	specularColorG_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	specularColorR_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	specularColor_ : SpecularColorPlug = PlugDescriptor("specularColor")
	spottyness_ : SpottynessPlug = PlugDescriptor("spottyness")
	squareVoxels_ : SquareVoxelsPlug = PlugDescriptor("squareVoxels")
	stalksU_ : StalksUPlug = PlugDescriptor("stalksU")
	stalksV_ : StalksVPlug = PlugDescriptor("stalksV")
	startFrame_ : StartFramePlug = PlugDescriptor("startFrame")
	startTime_ : StartTimePlug = PlugDescriptor("startTime")
	subVolumeCenterD_ : SubVolumeCenterDPlug = PlugDescriptor("subVolumeCenterD")
	subVolumeCenterH_ : SubVolumeCenterHPlug = PlugDescriptor("subVolumeCenterH")
	subVolumeCenterW_ : SubVolumeCenterWPlug = PlugDescriptor("subVolumeCenterW")
	subVolumeCenter_ : SubVolumeCenterPlug = PlugDescriptor("subVolumeCenter")
	subVolumeSizeD_ : SubVolumeSizeDPlug = PlugDescriptor("subVolumeSizeD")
	subVolumeSizeH_ : SubVolumeSizeHPlug = PlugDescriptor("subVolumeSizeH")
	subVolumeSizeW_ : SubVolumeSizeWPlug = PlugDescriptor("subVolumeSizeW")
	subVolumeSize_ : SubVolumeSizePlug = PlugDescriptor("subVolumeSize")
	substeps_ : SubstepsPlug = PlugDescriptor("substeps")
	surfaceRender_ : SurfaceRenderPlug = PlugDescriptor("surfaceRender")
	surfaceShaderDepth_ : SurfaceShaderDepthPlug = PlugDescriptor("surfaceShaderDepth")
	surfaceThreshold_ : SurfaceThresholdPlug = PlugDescriptor("surfaceThreshold")
	surfaceTolerance_ : SurfaceTolerancePlug = PlugDescriptor("surfaceTolerance")
	temperatureDiffusion_ : TemperatureDiffusionPlug = PlugDescriptor("temperatureDiffusion")
	temperatureDissipation_ : TemperatureDissipationPlug = PlugDescriptor("temperatureDissipation")
	temperatureGradient_ : TemperatureGradientPlug = PlugDescriptor("temperatureGradient")
	temperatureMethod_ : TemperatureMethodPlug = PlugDescriptor("temperatureMethod")
	temperatureNoise_ : TemperatureNoisePlug = PlugDescriptor("temperatureNoise")
	temperaturePressure_ : TemperaturePressurePlug = PlugDescriptor("temperaturePressure")
	temperaturePressureThreshold_ : TemperaturePressureThresholdPlug = PlugDescriptor("temperaturePressureThreshold")
	temperatureScale_ : TemperatureScalePlug = PlugDescriptor("temperatureScale")
	temperatureTension_ : TemperatureTensionPlug = PlugDescriptor("temperatureTension")
	temperatureTurbulence_ : TemperatureTurbulencePlug = PlugDescriptor("temperatureTurbulence")
	tensionForce_ : TensionForcePlug = PlugDescriptor("tensionForce")
	textureOriginX_ : TextureOriginXPlug = PlugDescriptor("textureOriginX")
	textureOriginY_ : TextureOriginYPlug = PlugDescriptor("textureOriginY")
	textureOriginZ_ : TextureOriginZPlug = PlugDescriptor("textureOriginZ")
	textureOrigin_ : TextureOriginPlug = PlugDescriptor("textureOrigin")
	textureRotateX_ : TextureRotateXPlug = PlugDescriptor("textureRotateX")
	textureRotateY_ : TextureRotateYPlug = PlugDescriptor("textureRotateY")
	textureRotateZ_ : TextureRotateZPlug = PlugDescriptor("textureRotateZ")
	textureRotate_ : TextureRotatePlug = PlugDescriptor("textureRotate")
	textureScaleX_ : TextureScaleXPlug = PlugDescriptor("textureScaleX")
	textureScaleY_ : TextureScaleYPlug = PlugDescriptor("textureScaleY")
	textureScaleZ_ : TextureScaleZPlug = PlugDescriptor("textureScaleZ")
	textureScale_ : TextureScalePlug = PlugDescriptor("textureScale")
	textureTime_ : TextureTimePlug = PlugDescriptor("textureTime")
	textureType_ : TextureTypePlug = PlugDescriptor("textureType")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	transparency_ : TransparencyPlug = PlugDescriptor("transparency")
	turbulenceFrequency_ : TurbulenceFrequencyPlug = PlugDescriptor("turbulenceFrequency")
	turbulenceRes_ : TurbulenceResPlug = PlugDescriptor("turbulenceRes")
	turbulenceSpeed_ : TurbulenceSpeedPlug = PlugDescriptor("turbulenceSpeed")
	turbulenceStrength_ : TurbulenceStrengthPlug = PlugDescriptor("turbulenceStrength")
	useGradientNormals_ : UseGradientNormalsPlug = PlugDescriptor("useGradientNormals")
	usePre70Dynamics_ : UsePre70DynamicsPlug = PlugDescriptor("usePre70Dynamics")
	uvwPerVertex_ : UvwPerVertexPlug = PlugDescriptor("uvwPerVertex")
	velocityAdvect_ : VelocityAdvectPlug = PlugDescriptor("velocityAdvect")
	velocityDamp_ : VelocityDampPlug = PlugDescriptor("velocityDamp")
	velocityDraw_ : VelocityDrawPlug = PlugDescriptor("velocityDraw")
	velocityDrawLength_ : VelocityDrawLengthPlug = PlugDescriptor("velocityDrawLength")
	velocityDrawSkip_ : VelocityDrawSkipPlug = PlugDescriptor("velocityDrawSkip")
	velocityGradient_ : VelocityGradientPlug = PlugDescriptor("velocityGradient")
	velocityMethod_ : VelocityMethodPlug = PlugDescriptor("velocityMethod")
	velocityNoise_ : VelocityNoisePlug = PlugDescriptor("velocityNoise")
	velocityPerVertex_ : VelocityPerVertexPlug = PlugDescriptor("velocityPerVertex")
	velocityProject_ : VelocityProjectPlug = PlugDescriptor("velocityProject")
	velocityScaleX_ : VelocityScaleXPlug = PlugDescriptor("velocityScaleX")
	velocityScaleY_ : VelocityScaleYPlug = PlugDescriptor("velocityScaleY")
	velocityScaleZ_ : VelocityScaleZPlug = PlugDescriptor("velocityScaleZ")
	velocityScale_ : VelocityScalePlug = PlugDescriptor("velocityScale")
	velocitySwirl_ : VelocitySwirlPlug = PlugDescriptor("velocitySwirl")
	viscosity_ : ViscosityPlug = PlugDescriptor("viscosity")
	voxelQuality_ : VoxelQualityPlug = PlugDescriptor("voxelQuality")
	wireframeDisplay_ : WireframeDisplayPlug = PlugDescriptor("wireframeDisplay")
	zoomFactor_ : ZoomFactorPlug = PlugDescriptor("zoomFactor")

	# node attributes

	typeName = "fluidShape"
	typeIdInt = 1179407689
	nodeLeafClassAttrs = ["airFuelRatio", "ambientBrightness", "ambientColorB", "ambientColorG", "ambientColorR", "ambientColor", "ambientDiffusion", "amplitude", "autoResize", "autoResizeMargin", "autoResizeThreshold", "baseResolution", "billowDensity", "boundaryDraw", "boundaryX", "boundaryY", "boundaryZ", "boxMinRadius", "boxRadius", "boxRatio", "buoyancy", "cacheColor", "cacheDensity", "cacheFalloff", "cacheReaction", "cacheTemperature", "cacheTextureCoordinates", "cacheVelocity", "checker", "circleRadius", "circleSizeRatio", "circles", "collide", "collisionFriction", "collisionGeometry", "collisionResilience", "collisionData", "color_ColorB", "color_ColorG", "color_ColorR", "color_Color", "color_Interp", "color_Position", "color", "colorDiffusion", "colorDissipation", "colorInput", "colorInputBias", "colorMethod", "colorPerVertex", "colorTexGain", "colorTexture", "conserveMass", "contrastTolerance", "coordinateMethod", "coordinateSpeed", "cosinePower", "currentTime", "densityBuoyancy", "densityDiffusion", "densityDissipation", "densityGradient", "densityGradientForce", "densityMethod", "densityNoise", "densityPressure", "densityPressureThreshold", "densityScale", "densityTension", "depthMax", "dimensionsD", "dimensionsH", "dimensionsW", "dimensions", "directionalLightX", "directionalLightY", "directionalLightZ", "directionalLight", "disableInteractiveEval", "diskCache", "diskCacheIC", "doEmission", "doFields", "drawHeads", "drawSubVolume", "dropoffShape", "dynamicOffsetX", "dynamicOffsetY", "dynamicOffsetZ", "dynamicOffset", "edgeDropoff", "emissionFunction_Hidden", "emissionFunction_InmapFrom", "emissionFunction_InmapTo", "emissionFunction_Inmap", "emissionFunction_OutmapFrom", "emissionFunction_OutmapTo", "emissionFunction_Outmap", "emissionFunction_Raw", "emissionFunction", "emissionList", "emitInSubsteps", "enableLiquidSimulation", "environment_ColorB", "environment_ColorG", "environment_ColorR", "environment_Color", "environment_Interp", "environment_Position", "environment", "equilibriumValue", "escapeRadius", "falloff", "falloffMethod", "farPointObjectX", "farPointObjectY", "farPointObjectZ", "farPointObj", "farPointWorldX", "farPointWorldY", "farPointWorldZ", "farPointWorld", "fieldDataDeltaTime", "fieldDataMass", "fieldDataPosition", "fieldDataVelocity", "fieldData", "fieldFunction_Hidden", "fieldFunction_InmapFrom", "fieldFunction_InmapTo", "fieldFunction_Inmap", "fieldFunction_OutmapFrom", "fieldFunction_OutmapTo", "fieldFunction_Outmap", "fieldFunction_Raw", "fieldFunction", "fieldList", "filterSizeX", "filterSizeY", "filterSizeZ", "filterSize", "fluidColorEmission", "fluidLightColorB", "fluidLightColorG", "fluidLightColorR", "fluidLightColor", "fluidReactantEmission", "focus", "forceDynamics", "forwardAdvection", "frequency", "frequencyRatio", "friction", "fuelGradient", "fuelIgnitionTemp", "fuelMethod", "fuelScale", "glowIntensity", "gravity", "gridInterpolator", "hardwareSelfShadow", "heatReleased", "heightField", "highDetailSolve", "implode", "implodeCenterX", "implodeCenterY", "implodeCenterZ", "implodeCenter", "inColor", "inDensity", "inFalloff", "inOffset", "inReaction", "inResolution", "inTemperature", "inTextureCoordinates", "inVelocity", "incandTexGain", "incandTexture", "incandescence_ColorB", "incandescence_ColorG", "incandescence_ColorR", "incandescence_Color", "incandescence_Interp", "incandescence_Position", "incandescence", "incandescenceInput", "incandescenceInputBias", "incandescencePerVertex", "inflection", "inheritFactor", "initialConditions", "deltaTime", "inputMass", "inputPositions", "inputVelocities", "inputData", "inputForce", "inputForce2", "invertTexture", "is2d", "isFull", "juliaU", "juliaV", "lastEvalTime", "leafEffect", "lightBrightness", "lightColorB", "lightColorG", "lightColorR", "lightColor", "lightAmbient", "lightBlindData", "lightDiffuse", "lightDirectionX", "lightDirectionY", "lightDirectionZ", "lightDirection", "lightIntensityB", "lightIntensityG", "lightIntensityR", "lightIntensity", "lightShadowFraction", "lightSpecular", "preShadowIntensity", "lightDataArray", "lightReleased", "lightType", "lineBlending", "lineFocus", "lineOffsetRatio", "lineOffsetU", "lineOffsetV", "liquidMethod", "liquidMinDensity", "liquidMistFall", "loadColor", "loadDensity", "loadFalloff", "loadReaction", "loadTemperature", "loadTextureCoordinates", "loadVelocity", "lobes", "lockDrawAxis", "mandelbrotDepth", "mandelbrotInsideMethod", "mandelbrotShadeMethod", "mandelbrotType", "massConversion", "massRange", "matrixEyeToWorld", "matrixWorldToObject", "matteOpacity", "matteOpacityMode", "maxReactionTemp", "maxResolution", "meshMethod", "meshResolution", "meshSmoothingIterations", "numWaves", "numericDisplay", "objectType", "opacity_FloatValue", "opacity_Interp", "opacity_Position", "opacity", "opacityInput", "opacityInputBias", "opacityPerVertex", "opacityPreviewGain", "opacityTexGain", "opacityTexture", "outColorB", "outColorG", "outColorR", "outColor", "outGlowColorB", "outGlowColorG", "outGlowColorR", "outGlowColor", "outGrid", "outMatteOpacityB", "outMatteOpacityG", "outMatteOpacityR", "outMatteOpacity", "outMesh", "outTransparencyB", "outTransparencyG", "outTransparencyR", "outTransparency", "outputForce", "overrideTimeStep", "particleWeight", "playFromCache", "pointLightX", "pointLightY", "pointLightZ", "pointLight", "pointLightDecay", "pointObjX", "pointObjY", "pointObjZ", "pointObj", "pointWorldX", "pointWorldY", "pointWorldZ", "pointWorld", "points", "quality", "randomness", "ratio", "rayInstance", "reactionSpeed", "realLights", "refractiveIndex", "renderInterpolator", "resizeClosedBoundaries", "resizeInSubsteps", "resizeToEmitter", "resolutionD", "resolutionH", "resolutionW", "resolution", "sampleMethod", "seed", "selfAttract", "selfForce", "selfForceDistance", "selfRepel", "selfShadowing", "shadedDisplay", "shadowDiffusion", "shadowOpacity", "simulationRateScale", "sizeRand", "slices", "softSurface", "solver", "solverQuality", "specularColorB", "specularColorG", "specularColorR", "specularColor", "spottyness", "squareVoxels", "stalksU", "stalksV", "startFrame", "startTime", "subVolumeCenterD", "subVolumeCenterH", "subVolumeCenterW", "subVolumeCenter", "subVolumeSizeD", "subVolumeSizeH", "subVolumeSizeW", "subVolumeSize", "substeps", "surfaceRender", "surfaceShaderDepth", "surfaceThreshold", "surfaceTolerance", "temperatureDiffusion", "temperatureDissipation", "temperatureGradient", "temperatureMethod", "temperatureNoise", "temperaturePressure", "temperaturePressureThreshold", "temperatureScale", "temperatureTension", "temperatureTurbulence", "tensionForce", "textureOriginX", "textureOriginY", "textureOriginZ", "textureOrigin", "textureRotateX", "textureRotateY", "textureRotateZ", "textureRotate", "textureScaleX", "textureScaleY", "textureScaleZ", "textureScale", "textureTime", "textureType", "threshold", "transparencyB", "transparencyG", "transparencyR", "transparency", "turbulenceFrequency", "turbulenceRes", "turbulenceSpeed", "turbulenceStrength", "useGradientNormals", "usePre70Dynamics", "uvwPerVertex", "velocityAdvect", "velocityDamp", "velocityDraw", "velocityDrawLength", "velocityDrawSkip", "velocityGradient", "velocityMethod", "velocityNoise", "velocityPerVertex", "velocityProject", "velocityScaleX", "velocityScaleY", "velocityScaleZ", "velocityScale", "velocitySwirl", "viscosity", "voxelQuality", "wireframeDisplay", "zoomFactor"]
	nodeLeafPlugs = ["airFuelRatio", "ambientBrightness", "ambientColor", "ambientDiffusion", "amplitude", "autoResize", "autoResizeMargin", "autoResizeThreshold", "baseResolution", "billowDensity", "boundaryDraw", "boundaryX", "boundaryY", "boundaryZ", "boxMinRadius", "boxRadius", "boxRatio", "buoyancy", "cacheColor", "cacheDensity", "cacheFalloff", "cacheReaction", "cacheTemperature", "cacheTextureCoordinates", "cacheVelocity", "checker", "circleRadius", "circleSizeRatio", "circles", "collide", "collisionData", "color", "colorDiffusion", "colorDissipation", "colorInput", "colorInputBias", "colorMethod", "colorPerVertex", "colorTexGain", "colorTexture", "conserveMass", "contrastTolerance", "coordinateMethod", "coordinateSpeed", "cosinePower", "currentTime", "densityBuoyancy", "densityDiffusion", "densityDissipation", "densityGradient", "densityGradientForce", "densityMethod", "densityNoise", "densityPressure", "densityPressureThreshold", "densityScale", "densityTension", "depthMax", "dimensions", "directionalLight", "disableInteractiveEval", "diskCache", "diskCacheIC", "doEmission", "doFields", "drawHeads", "drawSubVolume", "dropoffShape", "dynamicOffset", "edgeDropoff", "emissionList", "emitInSubsteps", "enableLiquidSimulation", "environment", "equilibriumValue", "escapeRadius", "falloff", "falloffMethod", "farPointObj", "farPointWorld", "fieldData", "fieldList", "filterSize", "fluidColorEmission", "fluidLightColor", "fluidReactantEmission", "focus", "forceDynamics", "forwardAdvection", "frequency", "frequencyRatio", "friction", "fuelGradient", "fuelIgnitionTemp", "fuelMethod", "fuelScale", "glowIntensity", "gravity", "gridInterpolator", "hardwareSelfShadow", "heatReleased", "heightField", "highDetailSolve", "implode", "implodeCenter", "inColor", "inDensity", "inFalloff", "inOffset", "inReaction", "inResolution", "inTemperature", "inTextureCoordinates", "inVelocity", "incandTexGain", "incandTexture", "incandescence", "incandescenceInput", "incandescenceInputBias", "incandescencePerVertex", "inflection", "inheritFactor", "initialConditions", "inputData", "inputForce", "inputForce2", "invertTexture", "is2d", "isFull", "juliaU", "juliaV", "lastEvalTime", "leafEffect", "lightBrightness", "lightColor", "lightDataArray", "lightReleased", "lightType", "lineBlending", "lineFocus", "lineOffsetRatio", "lineOffsetU", "lineOffsetV", "liquidMethod", "liquidMinDensity", "liquidMistFall", "loadColor", "loadDensity", "loadFalloff", "loadReaction", "loadTemperature", "loadTextureCoordinates", "loadVelocity", "lobes", "lockDrawAxis", "mandelbrotDepth", "mandelbrotInsideMethod", "mandelbrotShadeMethod", "mandelbrotType", "massConversion", "massRange", "matrixEyeToWorld", "matrixWorldToObject", "matteOpacity", "matteOpacityMode", "maxReactionTemp", "maxResolution", "meshMethod", "meshResolution", "meshSmoothingIterations", "numWaves", "numericDisplay", "objectType", "opacity", "opacityInput", "opacityInputBias", "opacityPerVertex", "opacityPreviewGain", "opacityTexGain", "opacityTexture", "outColor", "outGlowColor", "outGrid", "outMatteOpacity", "outMesh", "outTransparency", "outputForce", "overrideTimeStep", "particleWeight", "playFromCache", "pointLight", "pointLightDecay", "pointObj", "pointWorld", "points", "quality", "randomness", "ratio", "rayInstance", "reactionSpeed", "realLights", "refractiveIndex", "renderInterpolator", "resizeClosedBoundaries", "resizeInSubsteps", "resizeToEmitter", "resolution", "sampleMethod", "seed", "selfAttract", "selfForce", "selfForceDistance", "selfRepel", "selfShadowing", "shadedDisplay", "shadowDiffusion", "shadowOpacity", "simulationRateScale", "sizeRand", "slices", "softSurface", "solver", "solverQuality", "specularColor", "spottyness", "squareVoxels", "stalksU", "stalksV", "startFrame", "startTime", "subVolumeCenter", "subVolumeSize", "substeps", "surfaceRender", "surfaceShaderDepth", "surfaceThreshold", "surfaceTolerance", "temperatureDiffusion", "temperatureDissipation", "temperatureGradient", "temperatureMethod", "temperatureNoise", "temperaturePressure", "temperaturePressureThreshold", "temperatureScale", "temperatureTension", "temperatureTurbulence", "tensionForce", "textureOrigin", "textureRotate", "textureScale", "textureTime", "textureType", "threshold", "transparency", "turbulenceFrequency", "turbulenceRes", "turbulenceSpeed", "turbulenceStrength", "useGradientNormals", "usePre70Dynamics", "uvwPerVertex", "velocityAdvect", "velocityDamp", "velocityDraw", "velocityDrawLength", "velocityDrawSkip", "velocityGradient", "velocityMethod", "velocityNoise", "velocityPerVertex", "velocityProject", "velocityScale", "velocitySwirl", "viscosity", "voxelQuality", "wireframeDisplay", "zoomFactor"]
	pass

