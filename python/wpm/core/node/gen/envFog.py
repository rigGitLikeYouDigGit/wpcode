

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
LightFog = retriever.getNodeCls("LightFog")
assert LightFog
if T.TYPE_CHECKING:
	from .. import LightFog

# add node doc



# region plug type defs
class AirColorBPlug(Plug):
	parent : AirColorPlug = PlugDescriptor("airColor")
	node : EnvFog = None
	pass
class AirColorGPlug(Plug):
	parent : AirColorPlug = PlugDescriptor("airColor")
	node : EnvFog = None
	pass
class AirColorRPlug(Plug):
	parent : AirColorPlug = PlugDescriptor("airColor")
	node : EnvFog = None
	pass
class AirColorPlug(Plug):
	airColorB_ : AirColorBPlug = PlugDescriptor("airColorB")
	acb_ : AirColorBPlug = PlugDescriptor("airColorB")
	airColorG_ : AirColorGPlug = PlugDescriptor("airColorG")
	acg_ : AirColorGPlug = PlugDescriptor("airColorG")
	airColorR_ : AirColorRPlug = PlugDescriptor("airColorR")
	acr_ : AirColorRPlug = PlugDescriptor("airColorR")
	node : EnvFog = None
	pass
class AirDecayPlug(Plug):
	node : EnvFog = None
	pass
class AirDensityPlug(Plug):
	node : EnvFog = None
	pass
class AirLightScatterPlug(Plug):
	node : EnvFog = None
	pass
class AirMaxHeightPlug(Plug):
	node : EnvFog = None
	pass
class AirMinHeightPlug(Plug):
	node : EnvFog = None
	pass
class AirOpacityBPlug(Plug):
	parent : AirOpacityPlug = PlugDescriptor("airOpacity")
	node : EnvFog = None
	pass
class AirOpacityGPlug(Plug):
	parent : AirOpacityPlug = PlugDescriptor("airOpacity")
	node : EnvFog = None
	pass
class AirOpacityRPlug(Plug):
	parent : AirOpacityPlug = PlugDescriptor("airOpacity")
	node : EnvFog = None
	pass
class AirOpacityPlug(Plug):
	airOpacityB_ : AirOpacityBPlug = PlugDescriptor("airOpacityB")
	aob_ : AirOpacityBPlug = PlugDescriptor("airOpacityB")
	airOpacityG_ : AirOpacityGPlug = PlugDescriptor("airOpacityG")
	aog_ : AirOpacityGPlug = PlugDescriptor("airOpacityG")
	airOpacityR_ : AirOpacityRPlug = PlugDescriptor("airOpacityR")
	aor_ : AirOpacityRPlug = PlugDescriptor("airOpacityR")
	node : EnvFog = None
	pass
class BlendRangePlug(Plug):
	node : EnvFog = None
	pass
class DistanceClipPlanesPlug(Plug):
	node : EnvFog = None
	pass
class EndDistancePlug(Plug):
	node : EnvFog = None
	pass
class FogAxisPlug(Plug):
	node : EnvFog = None
	pass
class FogColorBPlug(Plug):
	parent : FogColorPlug = PlugDescriptor("fogColor")
	node : EnvFog = None
	pass
class FogColorGPlug(Plug):
	parent : FogColorPlug = PlugDescriptor("fogColor")
	node : EnvFog = None
	pass
class FogColorRPlug(Plug):
	parent : FogColorPlug = PlugDescriptor("fogColor")
	node : EnvFog = None
	pass
class FogColorPlug(Plug):
	fogColorB_ : FogColorBPlug = PlugDescriptor("fogColorB")
	fcb_ : FogColorBPlug = PlugDescriptor("fogColorB")
	fogColorG_ : FogColorGPlug = PlugDescriptor("fogColorG")
	fcg_ : FogColorGPlug = PlugDescriptor("fogColorG")
	fogColorR_ : FogColorRPlug = PlugDescriptor("fogColorR")
	fcr_ : FogColorRPlug = PlugDescriptor("fogColorR")
	node : EnvFog = None
	pass
class FogDecayPlug(Plug):
	node : EnvFog = None
	pass
class FogDensityPlug(Plug):
	node : EnvFog = None
	pass
class FogFarDistancePlug(Plug):
	node : EnvFog = None
	pass
class FogLightScatterPlug(Plug):
	node : EnvFog = None
	pass
class FogMaxHeightPlug(Plug):
	node : EnvFog = None
	pass
class FogMinHeightPlug(Plug):
	node : EnvFog = None
	pass
class FogNearDistancePlug(Plug):
	node : EnvFog = None
	pass
class FogOpacityBPlug(Plug):
	parent : FogOpacityPlug = PlugDescriptor("fogOpacity")
	node : EnvFog = None
	pass
class FogOpacityGPlug(Plug):
	parent : FogOpacityPlug = PlugDescriptor("fogOpacity")
	node : EnvFog = None
	pass
class FogOpacityRPlug(Plug):
	parent : FogOpacityPlug = PlugDescriptor("fogOpacity")
	node : EnvFog = None
	pass
class FogOpacityPlug(Plug):
	fogOpacityB_ : FogOpacityBPlug = PlugDescriptor("fogOpacityB")
	fob_ : FogOpacityBPlug = PlugDescriptor("fogOpacityB")
	fogOpacityG_ : FogOpacityGPlug = PlugDescriptor("fogOpacityG")
	fog_ : FogOpacityGPlug = PlugDescriptor("fogOpacityG")
	fogOpacityR_ : FogOpacityRPlug = PlugDescriptor("fogOpacityR")
	for_ : FogOpacityRPlug = PlugDescriptor("fogOpacityR")
	node : EnvFog = None
	pass
class FogTypePlug(Plug):
	node : EnvFog = None
	pass
class LayerPlug(Plug):
	node : EnvFog = None
	pass
class MatrixEyeToWorldPlug(Plug):
	node : EnvFog = None
	pass
class MaxHeightPlug(Plug):
	node : EnvFog = None
	pass
class MinHeightPlug(Plug):
	node : EnvFog = None
	pass
class PhysicalFogPlug(Plug):
	node : EnvFog = None
	pass
class PlanetRadiusPlug(Plug):
	node : EnvFog = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvFog = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvFog = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvFog = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : EnvFog = None
	pass
class PointWorldXPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : EnvFog = None
	pass
class PointWorldYPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : EnvFog = None
	pass
class PointWorldZPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : EnvFog = None
	pass
class PointWorldPlug(Plug):
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pwx_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pwy_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pwz_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	node : EnvFog = None
	pass
class RayDirectionXPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : EnvFog = None
	pass
class RayDirectionYPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : EnvFog = None
	pass
class RayDirectionZPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : EnvFog = None
	pass
class RayDirectionPlug(Plug):
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rx_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	ry_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rz_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	node : EnvFog = None
	pass
class SaturationDistancePlug(Plug):
	node : EnvFog = None
	pass
class StartDistancePlug(Plug):
	node : EnvFog = None
	pass
class SunAzimuthPlug(Plug):
	node : EnvFog = None
	pass
class SunColorBPlug(Plug):
	parent : SunColorPlug = PlugDescriptor("sunColor")
	node : EnvFog = None
	pass
class SunColorGPlug(Plug):
	parent : SunColorPlug = PlugDescriptor("sunColor")
	node : EnvFog = None
	pass
class SunColorRPlug(Plug):
	parent : SunColorPlug = PlugDescriptor("sunColor")
	node : EnvFog = None
	pass
class SunColorPlug(Plug):
	sunColorB_ : SunColorBPlug = PlugDescriptor("sunColorB")
	snb_ : SunColorBPlug = PlugDescriptor("sunColorB")
	sunColorG_ : SunColorGPlug = PlugDescriptor("sunColorG")
	sng_ : SunColorGPlug = PlugDescriptor("sunColorG")
	sunColorR_ : SunColorRPlug = PlugDescriptor("sunColorR")
	snr_ : SunColorRPlug = PlugDescriptor("sunColorR")
	node : EnvFog = None
	pass
class SunElevationPlug(Plug):
	node : EnvFog = None
	pass
class SunIntensityPlug(Plug):
	node : EnvFog = None
	pass
class UseDistancePlug(Plug):
	node : EnvFog = None
	pass
class UseHeightPlug(Plug):
	node : EnvFog = None
	pass
class UseLayerPlug(Plug):
	node : EnvFog = None
	pass
class WaterColorBPlug(Plug):
	parent : WaterColorPlug = PlugDescriptor("waterColor")
	node : EnvFog = None
	pass
class WaterColorGPlug(Plug):
	parent : WaterColorPlug = PlugDescriptor("waterColor")
	node : EnvFog = None
	pass
class WaterColorRPlug(Plug):
	parent : WaterColorPlug = PlugDescriptor("waterColor")
	node : EnvFog = None
	pass
class WaterColorPlug(Plug):
	waterColorB_ : WaterColorBPlug = PlugDescriptor("waterColorB")
	wcb_ : WaterColorBPlug = PlugDescriptor("waterColorB")
	waterColorG_ : WaterColorGPlug = PlugDescriptor("waterColorG")
	wcg_ : WaterColorGPlug = PlugDescriptor("waterColorG")
	waterColorR_ : WaterColorRPlug = PlugDescriptor("waterColorR")
	wcr_ : WaterColorRPlug = PlugDescriptor("waterColorR")
	node : EnvFog = None
	pass
class WaterDensityPlug(Plug):
	node : EnvFog = None
	pass
class WaterDepthPlug(Plug):
	node : EnvFog = None
	pass
class WaterLevelPlug(Plug):
	node : EnvFog = None
	pass
class WaterLightDecayPlug(Plug):
	node : EnvFog = None
	pass
class WaterLightScatterPlug(Plug):
	node : EnvFog = None
	pass
class WaterOpacityBPlug(Plug):
	parent : WaterOpacityPlug = PlugDescriptor("waterOpacity")
	node : EnvFog = None
	pass
class WaterOpacityGPlug(Plug):
	parent : WaterOpacityPlug = PlugDescriptor("waterOpacity")
	node : EnvFog = None
	pass
class WaterOpacityRPlug(Plug):
	parent : WaterOpacityPlug = PlugDescriptor("waterOpacity")
	node : EnvFog = None
	pass
class WaterOpacityPlug(Plug):
	waterOpacityB_ : WaterOpacityBPlug = PlugDescriptor("waterOpacityB")
	wob_ : WaterOpacityBPlug = PlugDescriptor("waterOpacityB")
	waterOpacityG_ : WaterOpacityGPlug = PlugDescriptor("waterOpacityG")
	wog_ : WaterOpacityGPlug = PlugDescriptor("waterOpacityG")
	waterOpacityR_ : WaterOpacityRPlug = PlugDescriptor("waterOpacityR")
	wor_ : WaterOpacityRPlug = PlugDescriptor("waterOpacityR")
	node : EnvFog = None
	pass
# endregion


# define node class
class EnvFog(LightFog):
	airColorB_ : AirColorBPlug = PlugDescriptor("airColorB")
	airColorG_ : AirColorGPlug = PlugDescriptor("airColorG")
	airColorR_ : AirColorRPlug = PlugDescriptor("airColorR")
	airColor_ : AirColorPlug = PlugDescriptor("airColor")
	airDecay_ : AirDecayPlug = PlugDescriptor("airDecay")
	airDensity_ : AirDensityPlug = PlugDescriptor("airDensity")
	airLightScatter_ : AirLightScatterPlug = PlugDescriptor("airLightScatter")
	airMaxHeight_ : AirMaxHeightPlug = PlugDescriptor("airMaxHeight")
	airMinHeight_ : AirMinHeightPlug = PlugDescriptor("airMinHeight")
	airOpacityB_ : AirOpacityBPlug = PlugDescriptor("airOpacityB")
	airOpacityG_ : AirOpacityGPlug = PlugDescriptor("airOpacityG")
	airOpacityR_ : AirOpacityRPlug = PlugDescriptor("airOpacityR")
	airOpacity_ : AirOpacityPlug = PlugDescriptor("airOpacity")
	blendRange_ : BlendRangePlug = PlugDescriptor("blendRange")
	distanceClipPlanes_ : DistanceClipPlanesPlug = PlugDescriptor("distanceClipPlanes")
	endDistance_ : EndDistancePlug = PlugDescriptor("endDistance")
	fogAxis_ : FogAxisPlug = PlugDescriptor("fogAxis")
	fogColorB_ : FogColorBPlug = PlugDescriptor("fogColorB")
	fogColorG_ : FogColorGPlug = PlugDescriptor("fogColorG")
	fogColorR_ : FogColorRPlug = PlugDescriptor("fogColorR")
	fogColor_ : FogColorPlug = PlugDescriptor("fogColor")
	fogDecay_ : FogDecayPlug = PlugDescriptor("fogDecay")
	fogDensity_ : FogDensityPlug = PlugDescriptor("fogDensity")
	fogFarDistance_ : FogFarDistancePlug = PlugDescriptor("fogFarDistance")
	fogLightScatter_ : FogLightScatterPlug = PlugDescriptor("fogLightScatter")
	fogMaxHeight_ : FogMaxHeightPlug = PlugDescriptor("fogMaxHeight")
	fogMinHeight_ : FogMinHeightPlug = PlugDescriptor("fogMinHeight")
	fogNearDistance_ : FogNearDistancePlug = PlugDescriptor("fogNearDistance")
	fogOpacityB_ : FogOpacityBPlug = PlugDescriptor("fogOpacityB")
	fogOpacityG_ : FogOpacityGPlug = PlugDescriptor("fogOpacityG")
	fogOpacityR_ : FogOpacityRPlug = PlugDescriptor("fogOpacityR")
	fogOpacity_ : FogOpacityPlug = PlugDescriptor("fogOpacity")
	fogType_ : FogTypePlug = PlugDescriptor("fogType")
	layer_ : LayerPlug = PlugDescriptor("layer")
	matrixEyeToWorld_ : MatrixEyeToWorldPlug = PlugDescriptor("matrixEyeToWorld")
	maxHeight_ : MaxHeightPlug = PlugDescriptor("maxHeight")
	minHeight_ : MinHeightPlug = PlugDescriptor("minHeight")
	physicalFog_ : PhysicalFogPlug = PlugDescriptor("physicalFog")
	planetRadius_ : PlanetRadiusPlug = PlugDescriptor("planetRadius")
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pointCamera_ : PointCameraPlug = PlugDescriptor("pointCamera")
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pointWorld_ : PointWorldPlug = PlugDescriptor("pointWorld")
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rayDirection_ : RayDirectionPlug = PlugDescriptor("rayDirection")
	saturationDistance_ : SaturationDistancePlug = PlugDescriptor("saturationDistance")
	startDistance_ : StartDistancePlug = PlugDescriptor("startDistance")
	sunAzimuth_ : SunAzimuthPlug = PlugDescriptor("sunAzimuth")
	sunColorB_ : SunColorBPlug = PlugDescriptor("sunColorB")
	sunColorG_ : SunColorGPlug = PlugDescriptor("sunColorG")
	sunColorR_ : SunColorRPlug = PlugDescriptor("sunColorR")
	sunColor_ : SunColorPlug = PlugDescriptor("sunColor")
	sunElevation_ : SunElevationPlug = PlugDescriptor("sunElevation")
	sunIntensity_ : SunIntensityPlug = PlugDescriptor("sunIntensity")
	useDistance_ : UseDistancePlug = PlugDescriptor("useDistance")
	useHeight_ : UseHeightPlug = PlugDescriptor("useHeight")
	useLayer_ : UseLayerPlug = PlugDescriptor("useLayer")
	waterColorB_ : WaterColorBPlug = PlugDescriptor("waterColorB")
	waterColorG_ : WaterColorGPlug = PlugDescriptor("waterColorG")
	waterColorR_ : WaterColorRPlug = PlugDescriptor("waterColorR")
	waterColor_ : WaterColorPlug = PlugDescriptor("waterColor")
	waterDensity_ : WaterDensityPlug = PlugDescriptor("waterDensity")
	waterDepth_ : WaterDepthPlug = PlugDescriptor("waterDepth")
	waterLevel_ : WaterLevelPlug = PlugDescriptor("waterLevel")
	waterLightDecay_ : WaterLightDecayPlug = PlugDescriptor("waterLightDecay")
	waterLightScatter_ : WaterLightScatterPlug = PlugDescriptor("waterLightScatter")
	waterOpacityB_ : WaterOpacityBPlug = PlugDescriptor("waterOpacityB")
	waterOpacityG_ : WaterOpacityGPlug = PlugDescriptor("waterOpacityG")
	waterOpacityR_ : WaterOpacityRPlug = PlugDescriptor("waterOpacityR")
	waterOpacity_ : WaterOpacityPlug = PlugDescriptor("waterOpacity")

	# node attributes

	typeName = "envFog"
	typeIdInt = 1380271687
	pass

