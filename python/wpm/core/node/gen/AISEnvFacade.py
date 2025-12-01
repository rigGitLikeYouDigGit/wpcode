

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
EnvFacade = retriever.getNodeCls("EnvFacade")
assert EnvFacade
if T.TYPE_CHECKING:
	from .. import EnvFacade

# add node doc



# region plug type defs
class AntiAliasingQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class BackgroundColorBPlug(Plug):
	parent : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	node : AISEnvFacade = None
	pass
class BackgroundColorGPlug(Plug):
	parent : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	node : AISEnvFacade = None
	pass
class BackgroundColorRPlug(Plug):
	parent : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	node : AISEnvFacade = None
	pass
class BackgroundColorPlug(Plug):
	backgroundColorB_ : BackgroundColorBPlug = PlugDescriptor("backgroundColorB")
	bcb_ : BackgroundColorBPlug = PlugDescriptor("backgroundColorB")
	backgroundColorG_ : BackgroundColorGPlug = PlugDescriptor("backgroundColorG")
	bcg_ : BackgroundColorGPlug = PlugDescriptor("backgroundColorG")
	backgroundColorR_ : BackgroundColorRPlug = PlugDescriptor("backgroundColorR")
	bcr_ : BackgroundColorRPlug = PlugDescriptor("backgroundColorR")
	node : AISEnvFacade = None
	pass
class ExtraLightColorGPlug(Plug):
	parent : ExtraLightColorPlug = PlugDescriptor("extraLightColor")
	node : AISEnvFacade = None
	pass
class ExtraLightColorRPlug(Plug):
	parent : ExtraLightColorPlug = PlugDescriptor("extraLightColor")
	node : AISEnvFacade = None
	pass
class ExtraLightColorbPlug(Plug):
	parent : ExtraLightColorPlug = PlugDescriptor("extraLightColor")
	node : AISEnvFacade = None
	pass
class ExtraLightColorPlug(Plug):
	parent : ExtraLightInfoPlug = PlugDescriptor("extraLightInfo")
	extraLightColorG_ : ExtraLightColorGPlug = PlugDescriptor("extraLightColorG")
	elcg_ : ExtraLightColorGPlug = PlugDescriptor("extraLightColorG")
	extraLightColorR_ : ExtraLightColorRPlug = PlugDescriptor("extraLightColorR")
	elcr_ : ExtraLightColorRPlug = PlugDescriptor("extraLightColorR")
	extraLightColorb_ : ExtraLightColorbPlug = PlugDescriptor("extraLightColorb")
	elcb_ : ExtraLightColorbPlug = PlugDescriptor("extraLightColorb")
	node : AISEnvFacade = None
	pass
class ExtraLightIntensityPlug(Plug):
	parent : ExtraLightInfoPlug = PlugDescriptor("extraLightInfo")
	node : AISEnvFacade = None
	pass
class ExtraLightOnOffPlug(Plug):
	parent : ExtraLightInfoPlug = PlugDescriptor("extraLightInfo")
	node : AISEnvFacade = None
	pass
class ExtraLightShadowColorBPlug(Plug):
	parent : ExtraLightShadowColorPlug = PlugDescriptor("extraLightShadowColor")
	node : AISEnvFacade = None
	pass
class ExtraLightShadowColorGPlug(Plug):
	parent : ExtraLightShadowColorPlug = PlugDescriptor("extraLightShadowColor")
	node : AISEnvFacade = None
	pass
class ExtraLightShadowColorRPlug(Plug):
	parent : ExtraLightShadowColorPlug = PlugDescriptor("extraLightShadowColor")
	node : AISEnvFacade = None
	pass
class ExtraLightShadowColorPlug(Plug):
	parent : ExtraLightInfoPlug = PlugDescriptor("extraLightInfo")
	extraLightShadowColorB_ : ExtraLightShadowColorBPlug = PlugDescriptor("extraLightShadowColorB")
	elscb_ : ExtraLightShadowColorBPlug = PlugDescriptor("extraLightShadowColorB")
	extraLightShadowColorG_ : ExtraLightShadowColorGPlug = PlugDescriptor("extraLightShadowColorG")
	elscg_ : ExtraLightShadowColorGPlug = PlugDescriptor("extraLightShadowColorG")
	extraLightShadowColorR_ : ExtraLightShadowColorRPlug = PlugDescriptor("extraLightShadowColorR")
	elscr_ : ExtraLightShadowColorRPlug = PlugDescriptor("extraLightShadowColorR")
	node : AISEnvFacade = None
	pass
class ExtraLightShadowSmoothnessPlug(Plug):
	parent : ExtraLightInfoPlug = PlugDescriptor("extraLightInfo")
	node : AISEnvFacade = None
	pass
class ExtraLightShadowWidthPlug(Plug):
	parent : ExtraLightInfoPlug = PlugDescriptor("extraLightInfo")
	node : AISEnvFacade = None
	pass
class ExtraLightShadowsPlug(Plug):
	parent : ExtraLightInfoPlug = PlugDescriptor("extraLightInfo")
	node : AISEnvFacade = None
	pass
class ExtraLightShapePlug(Plug):
	parent : ExtraLightInfoPlug = PlugDescriptor("extraLightInfo")
	node : AISEnvFacade = None
	pass
class ExtraLightInfoPlug(Plug):
	extraLightColor_ : ExtraLightColorPlug = PlugDescriptor("extraLightColor")
	elc_ : ExtraLightColorPlug = PlugDescriptor("extraLightColor")
	extraLightIntensity_ : ExtraLightIntensityPlug = PlugDescriptor("extraLightIntensity")
	elin_ : ExtraLightIntensityPlug = PlugDescriptor("extraLightIntensity")
	extraLightOnOff_ : ExtraLightOnOffPlug = PlugDescriptor("extraLightOnOff")
	eloo_ : ExtraLightOnOffPlug = PlugDescriptor("extraLightOnOff")
	extraLightShadowColor_ : ExtraLightShadowColorPlug = PlugDescriptor("extraLightShadowColor")
	elsc_ : ExtraLightShadowColorPlug = PlugDescriptor("extraLightShadowColor")
	extraLightShadowSmoothness_ : ExtraLightShadowSmoothnessPlug = PlugDescriptor("extraLightShadowSmoothness")
	elss_ : ExtraLightShadowSmoothnessPlug = PlugDescriptor("extraLightShadowSmoothness")
	extraLightShadowWidth_ : ExtraLightShadowWidthPlug = PlugDescriptor("extraLightShadowWidth")
	elsw_ : ExtraLightShadowWidthPlug = PlugDescriptor("extraLightShadowWidth")
	extraLightShadows_ : ExtraLightShadowsPlug = PlugDescriptor("extraLightShadows")
	elsd_ : ExtraLightShadowsPlug = PlugDescriptor("extraLightShadows")
	extraLightShape_ : ExtraLightShapePlug = PlugDescriptor("extraLightShape")
	elsh_ : ExtraLightShapePlug = PlugDescriptor("extraLightShape")
	node : AISEnvFacade = None
	pass
class FactoryAntiAliasingQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class FactoryGlobalIlluminationQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class FactoryReflectionsQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class FactoryRefractionsQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class FactoryTessellationQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class FloorHeightPlug(Plug):
	node : AISEnvFacade = None
	pass
class GlobalIlluminationQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class HasFloorPlug(Plug):
	node : AISEnvFacade = None
	pass
class HasInfiniteStagePlug(Plug):
	node : AISEnvFacade = None
	pass
class HasUserDefinedStageRadiusPlug(Plug):
	node : AISEnvFacade = None
	pass
class MaxMentalRayQualityNodePlug(Plug):
	node : AISEnvFacade = None
	pass
class MinMentalRayQualityNodePlug(Plug):
	node : AISEnvFacade = None
	pass
class OriginalStageRadiusPlug(Plug):
	node : AISEnvFacade = None
	pass
class ReflectionsQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class RefractionsQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class TessellationQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class TestAntiAliasingQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class TestGlobalIlluminationQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class TestReflectionsQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class TestRefractionsQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class TestTessellationQualityPlug(Plug):
	node : AISEnvFacade = None
	pass
class UserDefinedStageRadiusPlug(Plug):
	node : AISEnvFacade = None
	pass
# endregion


# define node class
class AISEnvFacade(EnvFacade):
	antiAliasingQuality_ : AntiAliasingQualityPlug = PlugDescriptor("antiAliasingQuality")
	backgroundColorB_ : BackgroundColorBPlug = PlugDescriptor("backgroundColorB")
	backgroundColorG_ : BackgroundColorGPlug = PlugDescriptor("backgroundColorG")
	backgroundColorR_ : BackgroundColorRPlug = PlugDescriptor("backgroundColorR")
	backgroundColor_ : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	extraLightColorG_ : ExtraLightColorGPlug = PlugDescriptor("extraLightColorG")
	extraLightColorR_ : ExtraLightColorRPlug = PlugDescriptor("extraLightColorR")
	extraLightColorb_ : ExtraLightColorbPlug = PlugDescriptor("extraLightColorb")
	extraLightColor_ : ExtraLightColorPlug = PlugDescriptor("extraLightColor")
	extraLightIntensity_ : ExtraLightIntensityPlug = PlugDescriptor("extraLightIntensity")
	extraLightOnOff_ : ExtraLightOnOffPlug = PlugDescriptor("extraLightOnOff")
	extraLightShadowColorB_ : ExtraLightShadowColorBPlug = PlugDescriptor("extraLightShadowColorB")
	extraLightShadowColorG_ : ExtraLightShadowColorGPlug = PlugDescriptor("extraLightShadowColorG")
	extraLightShadowColorR_ : ExtraLightShadowColorRPlug = PlugDescriptor("extraLightShadowColorR")
	extraLightShadowColor_ : ExtraLightShadowColorPlug = PlugDescriptor("extraLightShadowColor")
	extraLightShadowSmoothness_ : ExtraLightShadowSmoothnessPlug = PlugDescriptor("extraLightShadowSmoothness")
	extraLightShadowWidth_ : ExtraLightShadowWidthPlug = PlugDescriptor("extraLightShadowWidth")
	extraLightShadows_ : ExtraLightShadowsPlug = PlugDescriptor("extraLightShadows")
	extraLightShape_ : ExtraLightShapePlug = PlugDescriptor("extraLightShape")
	extraLightInfo_ : ExtraLightInfoPlug = PlugDescriptor("extraLightInfo")
	factoryAntiAliasingQuality_ : FactoryAntiAliasingQualityPlug = PlugDescriptor("factoryAntiAliasingQuality")
	factoryGlobalIlluminationQuality_ : FactoryGlobalIlluminationQualityPlug = PlugDescriptor("factoryGlobalIlluminationQuality")
	factoryReflectionsQuality_ : FactoryReflectionsQualityPlug = PlugDescriptor("factoryReflectionsQuality")
	factoryRefractionsQuality_ : FactoryRefractionsQualityPlug = PlugDescriptor("factoryRefractionsQuality")
	factoryTessellationQuality_ : FactoryTessellationQualityPlug = PlugDescriptor("factoryTessellationQuality")
	floorHeight_ : FloorHeightPlug = PlugDescriptor("floorHeight")
	globalIlluminationQuality_ : GlobalIlluminationQualityPlug = PlugDescriptor("globalIlluminationQuality")
	hasFloor_ : HasFloorPlug = PlugDescriptor("hasFloor")
	hasInfiniteStage_ : HasInfiniteStagePlug = PlugDescriptor("hasInfiniteStage")
	hasUserDefinedStageRadius_ : HasUserDefinedStageRadiusPlug = PlugDescriptor("hasUserDefinedStageRadius")
	maxMentalRayQualityNode_ : MaxMentalRayQualityNodePlug = PlugDescriptor("maxMentalRayQualityNode")
	minMentalRayQualityNode_ : MinMentalRayQualityNodePlug = PlugDescriptor("minMentalRayQualityNode")
	originalStageRadius_ : OriginalStageRadiusPlug = PlugDescriptor("originalStageRadius")
	reflectionsQuality_ : ReflectionsQualityPlug = PlugDescriptor("reflectionsQuality")
	refractionsQuality_ : RefractionsQualityPlug = PlugDescriptor("refractionsQuality")
	tessellationQuality_ : TessellationQualityPlug = PlugDescriptor("tessellationQuality")
	testAntiAliasingQuality_ : TestAntiAliasingQualityPlug = PlugDescriptor("testAntiAliasingQuality")
	testGlobalIlluminationQuality_ : TestGlobalIlluminationQualityPlug = PlugDescriptor("testGlobalIlluminationQuality")
	testReflectionsQuality_ : TestReflectionsQualityPlug = PlugDescriptor("testReflectionsQuality")
	testRefractionsQuality_ : TestRefractionsQualityPlug = PlugDescriptor("testRefractionsQuality")
	testTessellationQuality_ : TestTessellationQualityPlug = PlugDescriptor("testTessellationQuality")
	userDefinedStageRadius_ : UserDefinedStageRadiusPlug = PlugDescriptor("userDefinedStageRadius")

	# node attributes

	typeName = "AISEnvFacade"
	typeIdInt = 1380271702
	nodeLeafClassAttrs = ["antiAliasingQuality", "backgroundColorB", "backgroundColorG", "backgroundColorR", "backgroundColor", "extraLightColorG", "extraLightColorR", "extraLightColorb", "extraLightColor", "extraLightIntensity", "extraLightOnOff", "extraLightShadowColorB", "extraLightShadowColorG", "extraLightShadowColorR", "extraLightShadowColor", "extraLightShadowSmoothness", "extraLightShadowWidth", "extraLightShadows", "extraLightShape", "extraLightInfo", "factoryAntiAliasingQuality", "factoryGlobalIlluminationQuality", "factoryReflectionsQuality", "factoryRefractionsQuality", "factoryTessellationQuality", "floorHeight", "globalIlluminationQuality", "hasFloor", "hasInfiniteStage", "hasUserDefinedStageRadius", "maxMentalRayQualityNode", "minMentalRayQualityNode", "originalStageRadius", "reflectionsQuality", "refractionsQuality", "tessellationQuality", "testAntiAliasingQuality", "testGlobalIlluminationQuality", "testReflectionsQuality", "testRefractionsQuality", "testTessellationQuality", "userDefinedStageRadius"]
	nodeLeafPlugs = ["antiAliasingQuality", "backgroundColor", "extraLightInfo", "factoryAntiAliasingQuality", "factoryGlobalIlluminationQuality", "factoryReflectionsQuality", "factoryRefractionsQuality", "factoryTessellationQuality", "floorHeight", "globalIlluminationQuality", "hasFloor", "hasInfiniteStage", "hasUserDefinedStageRadius", "maxMentalRayQualityNode", "minMentalRayQualityNode", "originalStageRadius", "reflectionsQuality", "refractionsQuality", "tessellationQuality", "testAntiAliasingQuality", "testGlobalIlluminationQuality", "testReflectionsQuality", "testRefractionsQuality", "testTessellationQuality", "userDefinedStageRadius"]
	pass

