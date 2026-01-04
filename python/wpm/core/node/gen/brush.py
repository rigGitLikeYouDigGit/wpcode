

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class AttractRadiusOffsetPlug(Plug):
	node : Brush = None
	pass
class AttractRadiusScalePlug(Plug):
	node : Brush = None
	pass
class AzimuthMaxPlug(Plug):
	node : Brush = None
	pass
class AzimuthMinPlug(Plug):
	node : Brush = None
	pass
class BackShadowPlug(Plug):
	node : Brush = None
	pass
class BendPlug(Plug):
	node : Brush = None
	pass
class BendBiasPlug(Plug):
	node : Brush = None
	pass
class BinMembershipPlug(Plug):
	node : Brush = None
	pass
class BlurIntensityPlug(Plug):
	node : Brush = None
	pass
class BlurMultPlug(Plug):
	node : Brush = None
	pass
class BranchAfterTwigsPlug(Plug):
	node : Brush = None
	pass
class BranchDropoutPlug(Plug):
	node : Brush = None
	pass
class BranchReflectivityPlug(Plug):
	node : Brush = None
	pass
class BranchThornsPlug(Plug):
	node : Brush = None
	pass
class BranchesPlug(Plug):
	node : Brush = None
	pass
class BrightnessRandPlug(Plug):
	node : Brush = None
	pass
class BrushTypePlug(Plug):
	node : Brush = None
	pass
class BrushWidthPlug(Plug):
	node : Brush = None
	pass
class BudColorBPlug(Plug):
	parent : BudColorPlug = PlugDescriptor("budColor")
	node : Brush = None
	pass
class BudColorGPlug(Plug):
	parent : BudColorPlug = PlugDescriptor("budColor")
	node : Brush = None
	pass
class BudColorRPlug(Plug):
	parent : BudColorPlug = PlugDescriptor("budColor")
	node : Brush = None
	pass
class BudColorPlug(Plug):
	budColorB_ : BudColorBPlug = PlugDescriptor("budColorB")
	bub_ : BudColorBPlug = PlugDescriptor("budColorB")
	budColorG_ : BudColorGPlug = PlugDescriptor("budColorG")
	bug_ : BudColorGPlug = PlugDescriptor("budColorG")
	budColorR_ : BudColorRPlug = PlugDescriptor("budColorR")
	bur_ : BudColorRPlug = PlugDescriptor("budColorR")
	node : Brush = None
	pass
class BudSizePlug(Plug):
	node : Brush = None
	pass
class BudsPlug(Plug):
	node : Brush = None
	pass
class BumpBlurPlug(Plug):
	node : Brush = None
	pass
class BumpIntensityPlug(Plug):
	node : Brush = None
	pass
class CastShadowsPlug(Plug):
	node : Brush = None
	pass
class CenterShadowPlug(Plug):
	node : Brush = None
	pass
class CollideMethodPlug(Plug):
	node : Brush = None
	pass
class Color1BPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Brush = None
	pass
class Color1GPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Brush = None
	pass
class Color1RPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Brush = None
	pass
class Color1Plug(Plug):
	color1B_ : Color1BPlug = PlugDescriptor("color1B")
	c1b_ : Color1BPlug = PlugDescriptor("color1B")
	color1G_ : Color1GPlug = PlugDescriptor("color1G")
	c1g_ : Color1GPlug = PlugDescriptor("color1G")
	color1R_ : Color1RPlug = PlugDescriptor("color1R")
	c1r_ : Color1RPlug = PlugDescriptor("color1R")
	node : Brush = None
	pass
class Color2BPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Brush = None
	pass
class Color2GPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Brush = None
	pass
class Color2RPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Brush = None
	pass
class Color2Plug(Plug):
	color2B_ : Color2BPlug = PlugDescriptor("color2B")
	c2b_ : Color2BPlug = PlugDescriptor("color2B")
	color2G_ : Color2GPlug = PlugDescriptor("color2G")
	c2g_ : Color2GPlug = PlugDescriptor("color2G")
	color2R_ : Color2RPlug = PlugDescriptor("color2R")
	c2r_ : Color2RPlug = PlugDescriptor("color2R")
	node : Brush = None
	pass
class ColorLengthMapPlug(Plug):
	node : Brush = None
	pass
class CreationScriptPlug(Plug):
	node : Brush = None
	pass
class CurlPlug(Plug):
	node : Brush = None
	pass
class CurlFrequencyPlug(Plug):
	node : Brush = None
	pass
class CurlOffsetPlug(Plug):
	node : Brush = None
	pass
class CurveAttractPlug(Plug):
	node : Brush = None
	pass
class CurveFollowPlug(Plug):
	node : Brush = None
	pass
class CurveMaxDistPlug(Plug):
	node : Brush = None
	pass
class DeflectionPlug(Plug):
	node : Brush = None
	pass
class DeflectionMaxPlug(Plug):
	node : Brush = None
	pass
class DeflectionMinPlug(Plug):
	node : Brush = None
	pass
class DepthPlug(Plug):
	node : Brush = None
	pass
class DepthShadowPlug(Plug):
	node : Brush = None
	pass
class DepthShadowDepthPlug(Plug):
	node : Brush = None
	pass
class DepthShadowTypePlug(Plug):
	node : Brush = None
	pass
class DisplacementDelayPlug(Plug):
	node : Brush = None
	pass
class DisplacementOffsetPlug(Plug):
	node : Brush = None
	pass
class DisplacementScalePlug(Plug):
	node : Brush = None
	pass
class DistanceScalingPlug(Plug):
	node : Brush = None
	pass
class EdgeAntialiasPlug(Plug):
	node : Brush = None
	pass
class EdgeClipPlug(Plug):
	node : Brush = None
	pass
class EdgeClipDepthPlug(Plug):
	node : Brush = None
	pass
class ElevationMaxPlug(Plug):
	node : Brush = None
	pass
class ElevationMinPlug(Plug):
	node : Brush = None
	pass
class EndCapsPlug(Plug):
	node : Brush = None
	pass
class EndTimePlug(Plug):
	node : Brush = None
	pass
class Environment_ColorBPlug(Plug):
	parent : Environment_ColorPlug = PlugDescriptor("environment_Color")
	node : Brush = None
	pass
class Environment_ColorGPlug(Plug):
	parent : Environment_ColorPlug = PlugDescriptor("environment_Color")
	node : Brush = None
	pass
class Environment_ColorRPlug(Plug):
	parent : Environment_ColorPlug = PlugDescriptor("environment_Color")
	node : Brush = None
	pass
class Environment_ColorPlug(Plug):
	parent : EnvironmentPlug = PlugDescriptor("environment")
	environment_ColorB_ : Environment_ColorBPlug = PlugDescriptor("environment_ColorB")
	envcb_ : Environment_ColorBPlug = PlugDescriptor("environment_ColorB")
	environment_ColorG_ : Environment_ColorGPlug = PlugDescriptor("environment_ColorG")
	envcg_ : Environment_ColorGPlug = PlugDescriptor("environment_ColorG")
	environment_ColorR_ : Environment_ColorRPlug = PlugDescriptor("environment_ColorR")
	envcr_ : Environment_ColorRPlug = PlugDescriptor("environment_ColorR")
	node : Brush = None
	pass
class Environment_InterpPlug(Plug):
	parent : EnvironmentPlug = PlugDescriptor("environment")
	node : Brush = None
	pass
class Environment_PositionPlug(Plug):
	parent : EnvironmentPlug = PlugDescriptor("environment")
	node : Brush = None
	pass
class EnvironmentPlug(Plug):
	environment_Color_ : Environment_ColorPlug = PlugDescriptor("environment_Color")
	envc_ : Environment_ColorPlug = PlugDescriptor("environment_Color")
	environment_Interp_ : Environment_InterpPlug = PlugDescriptor("environment_Interp")
	envi_ : Environment_InterpPlug = PlugDescriptor("environment_Interp")
	environment_Position_ : Environment_PositionPlug = PlugDescriptor("environment_Position")
	envp_ : Environment_PositionPlug = PlugDescriptor("environment_Position")
	node : Brush = None
	pass
class FakeShadowPlug(Plug):
	node : Brush = None
	pass
class Flatness1Plug(Plug):
	node : Brush = None
	pass
class Flatness2Plug(Plug):
	node : Brush = None
	pass
class FlowSpeedPlug(Plug):
	node : Brush = None
	pass
class FlowerAngle1Plug(Plug):
	node : Brush = None
	pass
class FlowerAngle2Plug(Plug):
	node : Brush = None
	pass
class FlowerFaceSunPlug(Plug):
	node : Brush = None
	pass
class FlowerHueRandPlug(Plug):
	node : Brush = None
	pass
class FlowerImagePlug(Plug):
	node : Brush = None
	pass
class FlowerLocationPlug(Plug):
	node : Brush = None
	pass
class FlowerReflectivityPlug(Plug):
	node : Brush = None
	pass
class FlowerSatRandPlug(Plug):
	node : Brush = None
	pass
class FlowerSizeDecayPlug(Plug):
	node : Brush = None
	pass
class FlowerSizeRandPlug(Plug):
	node : Brush = None
	pass
class FlowerSpecularPlug(Plug):
	node : Brush = None
	pass
class FlowerStartPlug(Plug):
	node : Brush = None
	pass
class FlowerStiffnessPlug(Plug):
	node : Brush = None
	pass
class FlowerThornsPlug(Plug):
	node : Brush = None
	pass
class FlowerTranslucencePlug(Plug):
	node : Brush = None
	pass
class FlowerTwistPlug(Plug):
	node : Brush = None
	pass
class FlowerUseBranchTexPlug(Plug):
	node : Brush = None
	pass
class FlowerValRandPlug(Plug):
	node : Brush = None
	pass
class FlowersPlug(Plug):
	node : Brush = None
	pass
class ForwardTwistPlug(Plug):
	node : Brush = None
	pass
class FractalAmplitudePlug(Plug):
	node : Brush = None
	pass
class FractalRatioPlug(Plug):
	node : Brush = None
	pass
class FractalThresholdPlug(Plug):
	node : Brush = None
	pass
class FrameExtensionPlug(Plug):
	node : Brush = None
	pass
class FringeRemovalPlug(Plug):
	node : Brush = None
	pass
class GapRandPlug(Plug):
	node : Brush = None
	pass
class GapSizePlug(Plug):
	node : Brush = None
	pass
class GapSpacingPlug(Plug):
	node : Brush = None
	pass
class GlobalScalePlug(Plug):
	node : Brush = None
	pass
class GlowPlug(Plug):
	node : Brush = None
	pass
class GlowColorBPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : Brush = None
	pass
class GlowColorGPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : Brush = None
	pass
class GlowColorRPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : Brush = None
	pass
class GlowColorPlug(Plug):
	glowColorB_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	glb_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	glowColorG_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	glg_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	glowColorR_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	glr_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	node : Brush = None
	pass
class GlowSpreadPlug(Plug):
	node : Brush = None
	pass
class GravityPlug(Plug):
	node : Brush = None
	pass
class HardEdgesPlug(Plug):
	node : Brush = None
	pass
class HueRandPlug(Plug):
	node : Brush = None
	pass
class IlluminatedPlug(Plug):
	node : Brush = None
	pass
class ImageNamePlug(Plug):
	node : Brush = None
	pass
class IncandLengthMapPlug(Plug):
	node : Brush = None
	pass
class Incandescence1BPlug(Plug):
	parent : Incandescence1Plug = PlugDescriptor("incandescence1")
	node : Brush = None
	pass
class Incandescence1GPlug(Plug):
	parent : Incandescence1Plug = PlugDescriptor("incandescence1")
	node : Brush = None
	pass
class Incandescence1RPlug(Plug):
	parent : Incandescence1Plug = PlugDescriptor("incandescence1")
	node : Brush = None
	pass
class Incandescence1Plug(Plug):
	incandescence1B_ : Incandescence1BPlug = PlugDescriptor("incandescence1B")
	i1b_ : Incandescence1BPlug = PlugDescriptor("incandescence1B")
	incandescence1G_ : Incandescence1GPlug = PlugDescriptor("incandescence1G")
	i1g_ : Incandescence1GPlug = PlugDescriptor("incandescence1G")
	incandescence1R_ : Incandescence1RPlug = PlugDescriptor("incandescence1R")
	i1r_ : Incandescence1RPlug = PlugDescriptor("incandescence1R")
	node : Brush = None
	pass
class Incandescence2BPlug(Plug):
	parent : Incandescence2Plug = PlugDescriptor("incandescence2")
	node : Brush = None
	pass
class Incandescence2GPlug(Plug):
	parent : Incandescence2Plug = PlugDescriptor("incandescence2")
	node : Brush = None
	pass
class Incandescence2RPlug(Plug):
	parent : Incandescence2Plug = PlugDescriptor("incandescence2")
	node : Brush = None
	pass
class Incandescence2Plug(Plug):
	incandescence2B_ : Incandescence2BPlug = PlugDescriptor("incandescence2B")
	i2b_ : Incandescence2BPlug = PlugDescriptor("incandescence2B")
	incandescence2G_ : Incandescence2GPlug = PlugDescriptor("incandescence2G")
	i2g_ : Incandescence2GPlug = PlugDescriptor("incandescence2G")
	incandescence2R_ : Incandescence2RPlug = PlugDescriptor("incandescence2R")
	i2r_ : Incandescence2RPlug = PlugDescriptor("incandescence2R")
	node : Brush = None
	pass
class LeafAngle1Plug(Plug):
	node : Brush = None
	pass
class LeafAngle2Plug(Plug):
	node : Brush = None
	pass
class LeafBaseWidthPlug(Plug):
	node : Brush = None
	pass
class LeafBendPlug(Plug):
	node : Brush = None
	pass
class LeafColor1BPlug(Plug):
	parent : LeafColor1Plug = PlugDescriptor("leafColor1")
	node : Brush = None
	pass
class LeafColor1GPlug(Plug):
	parent : LeafColor1Plug = PlugDescriptor("leafColor1")
	node : Brush = None
	pass
class LeafColor1RPlug(Plug):
	parent : LeafColor1Plug = PlugDescriptor("leafColor1")
	node : Brush = None
	pass
class LeafColor1Plug(Plug):
	leafColor1B_ : LeafColor1BPlug = PlugDescriptor("leafColor1B")
	lb1_ : LeafColor1BPlug = PlugDescriptor("leafColor1B")
	leafColor1G_ : LeafColor1GPlug = PlugDescriptor("leafColor1G")
	lg1_ : LeafColor1GPlug = PlugDescriptor("leafColor1G")
	leafColor1R_ : LeafColor1RPlug = PlugDescriptor("leafColor1R")
	lr1_ : LeafColor1RPlug = PlugDescriptor("leafColor1R")
	node : Brush = None
	pass
class LeafColor2BPlug(Plug):
	parent : LeafColor2Plug = PlugDescriptor("leafColor2")
	node : Brush = None
	pass
class LeafColor2GPlug(Plug):
	parent : LeafColor2Plug = PlugDescriptor("leafColor2")
	node : Brush = None
	pass
class LeafColor2RPlug(Plug):
	parent : LeafColor2Plug = PlugDescriptor("leafColor2")
	node : Brush = None
	pass
class LeafColor2Plug(Plug):
	leafColor2B_ : LeafColor2BPlug = PlugDescriptor("leafColor2B")
	lb2_ : LeafColor2BPlug = PlugDescriptor("leafColor2B")
	leafColor2G_ : LeafColor2GPlug = PlugDescriptor("leafColor2G")
	lg2_ : LeafColor2GPlug = PlugDescriptor("leafColor2G")
	leafColor2R_ : LeafColor2RPlug = PlugDescriptor("leafColor2R")
	lr2_ : LeafColor2RPlug = PlugDescriptor("leafColor2R")
	node : Brush = None
	pass
class LeafCurl_FloatValuePlug(Plug):
	parent : LeafCurlPlug = PlugDescriptor("leafCurl")
	node : Brush = None
	pass
class LeafCurl_InterpPlug(Plug):
	parent : LeafCurlPlug = PlugDescriptor("leafCurl")
	node : Brush = None
	pass
class LeafCurl_PositionPlug(Plug):
	parent : LeafCurlPlug = PlugDescriptor("leafCurl")
	node : Brush = None
	pass
class LeafCurlPlug(Plug):
	leafCurl_FloatValue_ : LeafCurl_FloatValuePlug = PlugDescriptor("leafCurl_FloatValue")
	lclfv_ : LeafCurl_FloatValuePlug = PlugDescriptor("leafCurl_FloatValue")
	leafCurl_Interp_ : LeafCurl_InterpPlug = PlugDescriptor("leafCurl_Interp")
	lcli_ : LeafCurl_InterpPlug = PlugDescriptor("leafCurl_Interp")
	leafCurl_Position_ : LeafCurl_PositionPlug = PlugDescriptor("leafCurl_Position")
	lclp_ : LeafCurl_PositionPlug = PlugDescriptor("leafCurl_Position")
	node : Brush = None
	pass
class LeafDropoutPlug(Plug):
	node : Brush = None
	pass
class LeafFaceSunPlug(Plug):
	node : Brush = None
	pass
class LeafFlatnessPlug(Plug):
	node : Brush = None
	pass
class LeafForwardTwistPlug(Plug):
	node : Brush = None
	pass
class LeafHueRandPlug(Plug):
	node : Brush = None
	pass
class LeafImagePlug(Plug):
	node : Brush = None
	pass
class LeafLengthPlug(Plug):
	node : Brush = None
	pass
class LeafLocationPlug(Plug):
	node : Brush = None
	pass
class LeafReflectivityPlug(Plug):
	node : Brush = None
	pass
class LeafSatRandPlug(Plug):
	node : Brush = None
	pass
class LeafSegmentsPlug(Plug):
	node : Brush = None
	pass
class LeafSizeDecayPlug(Plug):
	node : Brush = None
	pass
class LeafSizeRandPlug(Plug):
	node : Brush = None
	pass
class LeafSpecularPlug(Plug):
	node : Brush = None
	pass
class LeafStartPlug(Plug):
	node : Brush = None
	pass
class LeafStiffnessPlug(Plug):
	node : Brush = None
	pass
class LeafThornsPlug(Plug):
	node : Brush = None
	pass
class LeafTipWidthPlug(Plug):
	node : Brush = None
	pass
class LeafTranslucencePlug(Plug):
	node : Brush = None
	pass
class LeafTwirlPlug(Plug):
	node : Brush = None
	pass
class LeafTwistPlug(Plug):
	node : Brush = None
	pass
class LeafUseBranchTexPlug(Plug):
	node : Brush = None
	pass
class LeafValRandPlug(Plug):
	node : Brush = None
	pass
class LeafWidthScale_FloatValuePlug(Plug):
	parent : LeafWidthScalePlug = PlugDescriptor("leafWidthScale")
	node : Brush = None
	pass
class LeafWidthScale_InterpPlug(Plug):
	parent : LeafWidthScalePlug = PlugDescriptor("leafWidthScale")
	node : Brush = None
	pass
class LeafWidthScale_PositionPlug(Plug):
	parent : LeafWidthScalePlug = PlugDescriptor("leafWidthScale")
	node : Brush = None
	pass
class LeafWidthScalePlug(Plug):
	leafWidthScale_FloatValue_ : LeafWidthScale_FloatValuePlug = PlugDescriptor("leafWidthScale_FloatValue")
	lwsfv_ : LeafWidthScale_FloatValuePlug = PlugDescriptor("leafWidthScale_FloatValue")
	leafWidthScale_Interp_ : LeafWidthScale_InterpPlug = PlugDescriptor("leafWidthScale_Interp")
	lwsi_ : LeafWidthScale_InterpPlug = PlugDescriptor("leafWidthScale_Interp")
	leafWidthScale_Position_ : LeafWidthScale_PositionPlug = PlugDescriptor("leafWidthScale_Position")
	lwsp_ : LeafWidthScale_PositionPlug = PlugDescriptor("leafWidthScale_Position")
	node : Brush = None
	pass
class LeavesPlug(Plug):
	node : Brush = None
	pass
class LeavesInClusterPlug(Plug):
	node : Brush = None
	pass
class LengthFlexPlug(Plug):
	node : Brush = None
	pass
class LengthMaxPlug(Plug):
	node : Brush = None
	pass
class LengthMinPlug(Plug):
	node : Brush = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : Brush = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : Brush = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : Brush = None
	pass
class LightDirectionPlug(Plug):
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : Brush = None
	pass
class LightingBasedWidthPlug(Plug):
	node : Brush = None
	pass
class LuminanceIsDisplacementPlug(Plug):
	node : Brush = None
	pass
class MapColorPlug(Plug):
	node : Brush = None
	pass
class MapDisplacementPlug(Plug):
	node : Brush = None
	pass
class MapMethodPlug(Plug):
	node : Brush = None
	pass
class MapOpacityPlug(Plug):
	node : Brush = None
	pass
class MaxAttractDistancePlug(Plug):
	node : Brush = None
	pass
class MaxPixelWidthPlug(Plug):
	node : Brush = None
	pass
class MiddleBranchPlug(Plug):
	node : Brush = None
	pass
class MinPixelWidthPlug(Plug):
	node : Brush = None
	pass
class MinSizePlug(Plug):
	node : Brush = None
	pass
class ModifyAlphaPlug(Plug):
	node : Brush = None
	pass
class ModifyColorPlug(Plug):
	node : Brush = None
	pass
class ModifyDepthPlug(Plug):
	node : Brush = None
	pass
class MomentumPlug(Plug):
	node : Brush = None
	pass
class MultiStreakDiffuseRandPlug(Plug):
	node : Brush = None
	pass
class MultiStreakLightAllPlug(Plug):
	node : Brush = None
	pass
class MultiStreakSpecularRandPlug(Plug):
	node : Brush = None
	pass
class MultiStreakSpread1Plug(Plug):
	node : Brush = None
	pass
class MultiStreakSpread2Plug(Plug):
	node : Brush = None
	pass
class MultiStreaksPlug(Plug):
	node : Brush = None
	pass
class NoisePlug(Plug):
	node : Brush = None
	pass
class NoiseFrequencyPlug(Plug):
	node : Brush = None
	pass
class NoiseOffsetPlug(Plug):
	node : Brush = None
	pass
class NumBranchesPlug(Plug):
	node : Brush = None
	pass
class NumFlowersPlug(Plug):
	node : Brush = None
	pass
class NumLeafClustersPlug(Plug):
	node : Brush = None
	pass
class NumTwigClustersPlug(Plug):
	node : Brush = None
	pass
class OcclusionWidthScalePlug(Plug):
	node : Brush = None
	pass
class OccupyAttractionPlug(Plug):
	node : Brush = None
	pass
class OccupyBranchTerminationPlug(Plug):
	node : Brush = None
	pass
class OccupyRadiusOffsetPlug(Plug):
	node : Brush = None
	pass
class OccupyRadiusScalePlug(Plug):
	node : Brush = None
	pass
class OffsetUPlug(Plug):
	node : Brush = None
	pass
class OffsetVPlug(Plug):
	node : Brush = None
	pass
class OutBrushPlug(Plug):
	node : Brush = None
	pass
class PathAttractPlug(Plug):
	node : Brush = None
	pass
class PathFollowPlug(Plug):
	node : Brush = None
	pass
class PerPixelLightingPlug(Plug):
	node : Brush = None
	pass
class PetalBaseWidthPlug(Plug):
	node : Brush = None
	pass
class PetalBendPlug(Plug):
	node : Brush = None
	pass
class PetalColor1BPlug(Plug):
	parent : PetalColor1Plug = PlugDescriptor("petalColor1")
	node : Brush = None
	pass
class PetalColor1GPlug(Plug):
	parent : PetalColor1Plug = PlugDescriptor("petalColor1")
	node : Brush = None
	pass
class PetalColor1RPlug(Plug):
	parent : PetalColor1Plug = PlugDescriptor("petalColor1")
	node : Brush = None
	pass
class PetalColor1Plug(Plug):
	petalColor1B_ : PetalColor1BPlug = PlugDescriptor("petalColor1B")
	pb1_ : PetalColor1BPlug = PlugDescriptor("petalColor1B")
	petalColor1G_ : PetalColor1GPlug = PlugDescriptor("petalColor1G")
	pg1_ : PetalColor1GPlug = PlugDescriptor("petalColor1G")
	petalColor1R_ : PetalColor1RPlug = PlugDescriptor("petalColor1R")
	pr1_ : PetalColor1RPlug = PlugDescriptor("petalColor1R")
	node : Brush = None
	pass
class PetalColor2BPlug(Plug):
	parent : PetalColor2Plug = PlugDescriptor("petalColor2")
	node : Brush = None
	pass
class PetalColor2GPlug(Plug):
	parent : PetalColor2Plug = PlugDescriptor("petalColor2")
	node : Brush = None
	pass
class PetalColor2RPlug(Plug):
	parent : PetalColor2Plug = PlugDescriptor("petalColor2")
	node : Brush = None
	pass
class PetalColor2Plug(Plug):
	petalColor2B_ : PetalColor2BPlug = PlugDescriptor("petalColor2B")
	pb2_ : PetalColor2BPlug = PlugDescriptor("petalColor2B")
	petalColor2G_ : PetalColor2GPlug = PlugDescriptor("petalColor2G")
	pg2_ : PetalColor2GPlug = PlugDescriptor("petalColor2G")
	petalColor2R_ : PetalColor2RPlug = PlugDescriptor("petalColor2R")
	pr2_ : PetalColor2RPlug = PlugDescriptor("petalColor2R")
	node : Brush = None
	pass
class PetalCurl_FloatValuePlug(Plug):
	parent : PetalCurlPlug = PlugDescriptor("petalCurl")
	node : Brush = None
	pass
class PetalCurl_InterpPlug(Plug):
	parent : PetalCurlPlug = PlugDescriptor("petalCurl")
	node : Brush = None
	pass
class PetalCurl_PositionPlug(Plug):
	parent : PetalCurlPlug = PlugDescriptor("petalCurl")
	node : Brush = None
	pass
class PetalCurlPlug(Plug):
	petalCurl_FloatValue_ : PetalCurl_FloatValuePlug = PlugDescriptor("petalCurl_FloatValue")
	pclfv_ : PetalCurl_FloatValuePlug = PlugDescriptor("petalCurl_FloatValue")
	petalCurl_Interp_ : PetalCurl_InterpPlug = PlugDescriptor("petalCurl_Interp")
	pcli_ : PetalCurl_InterpPlug = PlugDescriptor("petalCurl_Interp")
	petalCurl_Position_ : PetalCurl_PositionPlug = PlugDescriptor("petalCurl_Position")
	pclp_ : PetalCurl_PositionPlug = PlugDescriptor("petalCurl_Position")
	node : Brush = None
	pass
class PetalDropoutPlug(Plug):
	node : Brush = None
	pass
class PetalFlatnessPlug(Plug):
	node : Brush = None
	pass
class PetalForwardTwistPlug(Plug):
	node : Brush = None
	pass
class PetalLengthPlug(Plug):
	node : Brush = None
	pass
class PetalSegmentsPlug(Plug):
	node : Brush = None
	pass
class PetalTipWidthPlug(Plug):
	node : Brush = None
	pass
class PetalTwirlPlug(Plug):
	node : Brush = None
	pass
class PetalWidthScale_FloatValuePlug(Plug):
	parent : PetalWidthScalePlug = PlugDescriptor("petalWidthScale")
	node : Brush = None
	pass
class PetalWidthScale_InterpPlug(Plug):
	parent : PetalWidthScalePlug = PlugDescriptor("petalWidthScale")
	node : Brush = None
	pass
class PetalWidthScale_PositionPlug(Plug):
	parent : PetalWidthScalePlug = PlugDescriptor("petalWidthScale")
	node : Brush = None
	pass
class PetalWidthScalePlug(Plug):
	petalWidthScale_FloatValue_ : PetalWidthScale_FloatValuePlug = PlugDescriptor("petalWidthScale_FloatValue")
	pwsfv_ : PetalWidthScale_FloatValuePlug = PlugDescriptor("petalWidthScale_FloatValue")
	petalWidthScale_Interp_ : PetalWidthScale_InterpPlug = PlugDescriptor("petalWidthScale_Interp")
	pwsi_ : PetalWidthScale_InterpPlug = PlugDescriptor("petalWidthScale_Interp")
	petalWidthScale_Position_ : PetalWidthScale_PositionPlug = PlugDescriptor("petalWidthScale_Position")
	pwsp_ : PetalWidthScale_PositionPlug = PlugDescriptor("petalWidthScale_Position")
	node : Brush = None
	pass
class PetalsInFlowerPlug(Plug):
	node : Brush = None
	pass
class RandomPlug(Plug):
	node : Brush = None
	pass
class RealLightsPlug(Plug):
	node : Brush = None
	pass
class ReflectionRolloff_FloatValuePlug(Plug):
	parent : ReflectionRolloffPlug = PlugDescriptor("reflectionRolloff")
	node : Brush = None
	pass
class ReflectionRolloff_InterpPlug(Plug):
	parent : ReflectionRolloffPlug = PlugDescriptor("reflectionRolloff")
	node : Brush = None
	pass
class ReflectionRolloff_PositionPlug(Plug):
	parent : ReflectionRolloffPlug = PlugDescriptor("reflectionRolloff")
	node : Brush = None
	pass
class ReflectionRolloffPlug(Plug):
	reflectionRolloff_FloatValue_ : ReflectionRolloff_FloatValuePlug = PlugDescriptor("reflectionRolloff_FloatValue")
	rrofv_ : ReflectionRolloff_FloatValuePlug = PlugDescriptor("reflectionRolloff_FloatValue")
	reflectionRolloff_Interp_ : ReflectionRolloff_InterpPlug = PlugDescriptor("reflectionRolloff_Interp")
	rroi_ : ReflectionRolloff_InterpPlug = PlugDescriptor("reflectionRolloff_Interp")
	reflectionRolloff_Position_ : ReflectionRolloff_PositionPlug = PlugDescriptor("reflectionRolloff_Position")
	rrop_ : ReflectionRolloff_PositionPlug = PlugDescriptor("reflectionRolloff_Position")
	node : Brush = None
	pass
class RepeatUPlug(Plug):
	node : Brush = None
	pass
class RepeatVPlug(Plug):
	node : Brush = None
	pass
class RootFadePlug(Plug):
	node : Brush = None
	pass
class RuntimeScriptPlug(Plug):
	node : Brush = None
	pass
class SatRandPlug(Plug):
	node : Brush = None
	pass
class ScreenspaceWidthPlug(Plug):
	node : Brush = None
	pass
class SegmentLengthBiasPlug(Plug):
	node : Brush = None
	pass
class SegmentWidthBiasPlug(Plug):
	node : Brush = None
	pass
class SegmentsPlug(Plug):
	node : Brush = None
	pass
class ShaderGlowPlug(Plug):
	node : Brush = None
	pass
class ShadowDiffusionPlug(Plug):
	node : Brush = None
	pass
class ShadowOffsetPlug(Plug):
	node : Brush = None
	pass
class ShadowTransparencyPlug(Plug):
	node : Brush = None
	pass
class SimplifyMethodPlug(Plug):
	node : Brush = None
	pass
class SingleSidedPlug(Plug):
	node : Brush = None
	pass
class SmearPlug(Plug):
	node : Brush = None
	pass
class SmearUPlug(Plug):
	node : Brush = None
	pass
class SmearVPlug(Plug):
	node : Brush = None
	pass
class SoftnessPlug(Plug):
	node : Brush = None
	pass
class SpecularPlug(Plug):
	node : Brush = None
	pass
class SpecularColorBPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : Brush = None
	pass
class SpecularColorGPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : Brush = None
	pass
class SpecularColorRPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : Brush = None
	pass
class SpecularColorPlug(Plug):
	specularColorB_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	spb_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	specularColorG_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	spg_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	specularColorR_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	spr_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	node : Brush = None
	pass
class SpecularPowerPlug(Plug):
	node : Brush = None
	pass
class SpiralDecayPlug(Plug):
	node : Brush = None
	pass
class SpiralMaxPlug(Plug):
	node : Brush = None
	pass
class SpiralMinPlug(Plug):
	node : Brush = None
	pass
class SplitAnglePlug(Plug):
	node : Brush = None
	pass
class SplitBiasPlug(Plug):
	node : Brush = None
	pass
class SplitLengthMapPlug(Plug):
	node : Brush = None
	pass
class SplitMaxDepthPlug(Plug):
	node : Brush = None
	pass
class SplitRandPlug(Plug):
	node : Brush = None
	pass
class SplitSizeDecayPlug(Plug):
	node : Brush = None
	pass
class SplitTwistPlug(Plug):
	node : Brush = None
	pass
class StampDensityPlug(Plug):
	node : Brush = None
	pass
class StartBranchesPlug(Plug):
	node : Brush = None
	pass
class StartTimePlug(Plug):
	node : Brush = None
	pass
class StartTubesPlug(Plug):
	node : Brush = None
	pass
class StrokeTimePlug(Plug):
	node : Brush = None
	pass
class SubSegmentsPlug(Plug):
	node : Brush = None
	pass
class SunDirectionXPlug(Plug):
	parent : SunDirectionPlug = PlugDescriptor("sunDirection")
	node : Brush = None
	pass
class SunDirectionYPlug(Plug):
	parent : SunDirectionPlug = PlugDescriptor("sunDirection")
	node : Brush = None
	pass
class SunDirectionZPlug(Plug):
	parent : SunDirectionPlug = PlugDescriptor("sunDirection")
	node : Brush = None
	pass
class SunDirectionPlug(Plug):
	sunDirectionX_ : SunDirectionXPlug = PlugDescriptor("sunDirectionX")
	sndx_ : SunDirectionXPlug = PlugDescriptor("sunDirectionX")
	sunDirectionY_ : SunDirectionYPlug = PlugDescriptor("sunDirectionY")
	sndy_ : SunDirectionYPlug = PlugDescriptor("sunDirectionY")
	sunDirectionZ_ : SunDirectionZPlug = PlugDescriptor("sunDirectionZ")
	sndz_ : SunDirectionZPlug = PlugDescriptor("sunDirectionZ")
	node : Brush = None
	pass
class SurfaceAttractPlug(Plug):
	node : Brush = None
	pass
class SurfaceCollidePlug(Plug):
	node : Brush = None
	pass
class SurfaceSampleDensityPlug(Plug):
	node : Brush = None
	pass
class SurfaceSnapPlug(Plug):
	node : Brush = None
	pass
class TerminalLeafPlug(Plug):
	node : Brush = None
	pass
class TexAlpha1Plug(Plug):
	node : Brush = None
	pass
class TexAlpha2Plug(Plug):
	node : Brush = None
	pass
class TexColor1BPlug(Plug):
	parent : TexColor1Plug = PlugDescriptor("texColor1")
	node : Brush = None
	pass
class TexColor1GPlug(Plug):
	parent : TexColor1Plug = PlugDescriptor("texColor1")
	node : Brush = None
	pass
class TexColor1RPlug(Plug):
	parent : TexColor1Plug = PlugDescriptor("texColor1")
	node : Brush = None
	pass
class TexColor1Plug(Plug):
	texColor1B_ : TexColor1BPlug = PlugDescriptor("texColor1B")
	x1b_ : TexColor1BPlug = PlugDescriptor("texColor1B")
	texColor1G_ : TexColor1GPlug = PlugDescriptor("texColor1G")
	x1g_ : TexColor1GPlug = PlugDescriptor("texColor1G")
	texColor1R_ : TexColor1RPlug = PlugDescriptor("texColor1R")
	x1r_ : TexColor1RPlug = PlugDescriptor("texColor1R")
	node : Brush = None
	pass
class TexColor2BPlug(Plug):
	parent : TexColor2Plug = PlugDescriptor("texColor2")
	node : Brush = None
	pass
class TexColor2GPlug(Plug):
	parent : TexColor2Plug = PlugDescriptor("texColor2")
	node : Brush = None
	pass
class TexColor2RPlug(Plug):
	parent : TexColor2Plug = PlugDescriptor("texColor2")
	node : Brush = None
	pass
class TexColor2Plug(Plug):
	texColor2B_ : TexColor2BPlug = PlugDescriptor("texColor2B")
	x2b_ : TexColor2BPlug = PlugDescriptor("texColor2B")
	texColor2G_ : TexColor2GPlug = PlugDescriptor("texColor2G")
	x2g_ : TexColor2GPlug = PlugDescriptor("texColor2G")
	texColor2R_ : TexColor2RPlug = PlugDescriptor("texColor2R")
	x2r_ : TexColor2RPlug = PlugDescriptor("texColor2R")
	node : Brush = None
	pass
class TexColorOffsetPlug(Plug):
	node : Brush = None
	pass
class TexColorScalePlug(Plug):
	node : Brush = None
	pass
class TexOpacityOffsetPlug(Plug):
	node : Brush = None
	pass
class TexOpacityScalePlug(Plug):
	node : Brush = None
	pass
class TexUniformityPlug(Plug):
	node : Brush = None
	pass
class TextureFlowPlug(Plug):
	node : Brush = None
	pass
class TextureTypePlug(Plug):
	node : Brush = None
	pass
class ThornBaseColorBPlug(Plug):
	parent : ThornBaseColorPlug = PlugDescriptor("thornBaseColor")
	node : Brush = None
	pass
class ThornBaseColorGPlug(Plug):
	parent : ThornBaseColorPlug = PlugDescriptor("thornBaseColor")
	node : Brush = None
	pass
class ThornBaseColorRPlug(Plug):
	parent : ThornBaseColorPlug = PlugDescriptor("thornBaseColor")
	node : Brush = None
	pass
class ThornBaseColorPlug(Plug):
	thornBaseColorB_ : ThornBaseColorBPlug = PlugDescriptor("thornBaseColorB")
	tbcb_ : ThornBaseColorBPlug = PlugDescriptor("thornBaseColorB")
	thornBaseColorG_ : ThornBaseColorGPlug = PlugDescriptor("thornBaseColorG")
	tbcg_ : ThornBaseColorGPlug = PlugDescriptor("thornBaseColorG")
	thornBaseColorR_ : ThornBaseColorRPlug = PlugDescriptor("thornBaseColorR")
	tbcr_ : ThornBaseColorRPlug = PlugDescriptor("thornBaseColorR")
	node : Brush = None
	pass
class ThornBaseWidthPlug(Plug):
	node : Brush = None
	pass
class ThornDensityPlug(Plug):
	node : Brush = None
	pass
class ThornElevationPlug(Plug):
	node : Brush = None
	pass
class ThornLengthPlug(Plug):
	node : Brush = None
	pass
class ThornSpecularPlug(Plug):
	node : Brush = None
	pass
class ThornTipColorBPlug(Plug):
	parent : ThornTipColorPlug = PlugDescriptor("thornTipColor")
	node : Brush = None
	pass
class ThornTipColorGPlug(Plug):
	parent : ThornTipColorPlug = PlugDescriptor("thornTipColor")
	node : Brush = None
	pass
class ThornTipColorRPlug(Plug):
	parent : ThornTipColorPlug = PlugDescriptor("thornTipColor")
	node : Brush = None
	pass
class ThornTipColorPlug(Plug):
	thornTipColorB_ : ThornTipColorBPlug = PlugDescriptor("thornTipColorB")
	ttcb_ : ThornTipColorBPlug = PlugDescriptor("thornTipColorB")
	thornTipColorG_ : ThornTipColorGPlug = PlugDescriptor("thornTipColorG")
	ttcg_ : ThornTipColorGPlug = PlugDescriptor("thornTipColorG")
	thornTipColorR_ : ThornTipColorRPlug = PlugDescriptor("thornTipColorR")
	ttcr_ : ThornTipColorRPlug = PlugDescriptor("thornTipColorR")
	node : Brush = None
	pass
class ThornTipWidthPlug(Plug):
	node : Brush = None
	pass
class TimePlug(Plug):
	node : Brush = None
	pass
class TimeClipPlug(Plug):
	node : Brush = None
	pass
class TipFadePlug(Plug):
	node : Brush = None
	pass
class TranslucencePlug(Plug):
	node : Brush = None
	pass
class TranspLengthMapPlug(Plug):
	node : Brush = None
	pass
class Transparency1BPlug(Plug):
	parent : Transparency1Plug = PlugDescriptor("transparency1")
	node : Brush = None
	pass
class Transparency1GPlug(Plug):
	parent : Transparency1Plug = PlugDescriptor("transparency1")
	node : Brush = None
	pass
class Transparency1RPlug(Plug):
	parent : Transparency1Plug = PlugDescriptor("transparency1")
	node : Brush = None
	pass
class Transparency1Plug(Plug):
	transparency1B_ : Transparency1BPlug = PlugDescriptor("transparency1B")
	t1b_ : Transparency1BPlug = PlugDescriptor("transparency1B")
	transparency1G_ : Transparency1GPlug = PlugDescriptor("transparency1G")
	t1g_ : Transparency1GPlug = PlugDescriptor("transparency1G")
	transparency1R_ : Transparency1RPlug = PlugDescriptor("transparency1R")
	t1r_ : Transparency1RPlug = PlugDescriptor("transparency1R")
	node : Brush = None
	pass
class Transparency2BPlug(Plug):
	parent : Transparency2Plug = PlugDescriptor("transparency2")
	node : Brush = None
	pass
class Transparency2GPlug(Plug):
	parent : Transparency2Plug = PlugDescriptor("transparency2")
	node : Brush = None
	pass
class Transparency2RPlug(Plug):
	parent : Transparency2Plug = PlugDescriptor("transparency2")
	node : Brush = None
	pass
class Transparency2Plug(Plug):
	transparency2B_ : Transparency2BPlug = PlugDescriptor("transparency2B")
	t2b_ : Transparency2BPlug = PlugDescriptor("transparency2B")
	transparency2G_ : Transparency2GPlug = PlugDescriptor("transparency2G")
	t2g_ : Transparency2GPlug = PlugDescriptor("transparency2G")
	transparency2R_ : Transparency2RPlug = PlugDescriptor("transparency2R")
	t2r_ : Transparency2RPlug = PlugDescriptor("transparency2R")
	node : Brush = None
	pass
class TubeCompletionPlug(Plug):
	node : Brush = None
	pass
class TubeDirectionPlug(Plug):
	node : Brush = None
	pass
class TubeRandPlug(Plug):
	node : Brush = None
	pass
class TubeSectionsPlug(Plug):
	node : Brush = None
	pass
class TubeWidth1Plug(Plug):
	node : Brush = None
	pass
class TubeWidth2Plug(Plug):
	node : Brush = None
	pass
class TubesPlug(Plug):
	node : Brush = None
	pass
class TubesPerStepPlug(Plug):
	node : Brush = None
	pass
class TurbulencePlug(Plug):
	node : Brush = None
	pass
class TurbulenceFrequencyPlug(Plug):
	node : Brush = None
	pass
class TurbulenceInterpolationPlug(Plug):
	node : Brush = None
	pass
class TurbulenceOffsetXPlug(Plug):
	parent : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	node : Brush = None
	pass
class TurbulenceOffsetYPlug(Plug):
	parent : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	node : Brush = None
	pass
class TurbulenceOffsetZPlug(Plug):
	parent : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	node : Brush = None
	pass
class TurbulenceOffsetPlug(Plug):
	turbulenceOffsetX_ : TurbulenceOffsetXPlug = PlugDescriptor("turbulenceOffsetX")
	trx_ : TurbulenceOffsetXPlug = PlugDescriptor("turbulenceOffsetX")
	turbulenceOffsetY_ : TurbulenceOffsetYPlug = PlugDescriptor("turbulenceOffsetY")
	try_ : TurbulenceOffsetYPlug = PlugDescriptor("turbulenceOffsetY")
	turbulenceOffsetZ_ : TurbulenceOffsetZPlug = PlugDescriptor("turbulenceOffsetZ")
	trz_ : TurbulenceOffsetZPlug = PlugDescriptor("turbulenceOffsetZ")
	node : Brush = None
	pass
class TurbulenceSpeedPlug(Plug):
	node : Brush = None
	pass
class TurbulenceTypePlug(Plug):
	node : Brush = None
	pass
class TwigAngle1Plug(Plug):
	node : Brush = None
	pass
class TwigAngle2Plug(Plug):
	node : Brush = None
	pass
class TwigBaseWidthPlug(Plug):
	node : Brush = None
	pass
class TwigDropoutPlug(Plug):
	node : Brush = None
	pass
class TwigLengthPlug(Plug):
	node : Brush = None
	pass
class TwigLengthScale_FloatValuePlug(Plug):
	parent : TwigLengthScalePlug = PlugDescriptor("twigLengthScale")
	node : Brush = None
	pass
class TwigLengthScale_InterpPlug(Plug):
	parent : TwigLengthScalePlug = PlugDescriptor("twigLengthScale")
	node : Brush = None
	pass
class TwigLengthScale_PositionPlug(Plug):
	parent : TwigLengthScalePlug = PlugDescriptor("twigLengthScale")
	node : Brush = None
	pass
class TwigLengthScalePlug(Plug):
	twigLengthScale_FloatValue_ : TwigLengthScale_FloatValuePlug = PlugDescriptor("twigLengthScale_FloatValue")
	tlsfv_ : TwigLengthScale_FloatValuePlug = PlugDescriptor("twigLengthScale_FloatValue")
	twigLengthScale_Interp_ : TwigLengthScale_InterpPlug = PlugDescriptor("twigLengthScale_Interp")
	tlsi_ : TwigLengthScale_InterpPlug = PlugDescriptor("twigLengthScale_Interp")
	twigLengthScale_Position_ : TwigLengthScale_PositionPlug = PlugDescriptor("twigLengthScale_Position")
	tlsp_ : TwigLengthScale_PositionPlug = PlugDescriptor("twigLengthScale_Position")
	node : Brush = None
	pass
class TwigStartPlug(Plug):
	node : Brush = None
	pass
class TwigStiffnessPlug(Plug):
	node : Brush = None
	pass
class TwigThornsPlug(Plug):
	node : Brush = None
	pass
class TwigTipWidthPlug(Plug):
	node : Brush = None
	pass
class TwigTwistPlug(Plug):
	node : Brush = None
	pass
class TwigsPlug(Plug):
	node : Brush = None
	pass
class TwigsInClusterPlug(Plug):
	node : Brush = None
	pass
class TwistPlug(Plug):
	node : Brush = None
	pass
class TwistRandPlug(Plug):
	node : Brush = None
	pass
class TwistRatePlug(Plug):
	node : Brush = None
	pass
class UniformForceXPlug(Plug):
	parent : UniformForcePlug = PlugDescriptor("uniformForce")
	node : Brush = None
	pass
class UniformForceYPlug(Plug):
	parent : UniformForcePlug = PlugDescriptor("uniformForce")
	node : Brush = None
	pass
class UniformForceZPlug(Plug):
	parent : UniformForcePlug = PlugDescriptor("uniformForce")
	node : Brush = None
	pass
class UniformForcePlug(Plug):
	uniformForceX_ : UniformForceXPlug = PlugDescriptor("uniformForceX")
	ufx_ : UniformForceXPlug = PlugDescriptor("uniformForceX")
	uniformForceY_ : UniformForceYPlug = PlugDescriptor("uniformForceY")
	ufy_ : UniformForceYPlug = PlugDescriptor("uniformForceY")
	uniformForceZ_ : UniformForceZPlug = PlugDescriptor("uniformForceZ")
	ufz_ : UniformForceZPlug = PlugDescriptor("uniformForceZ")
	node : Brush = None
	pass
class UseFrameExtensionPlug(Plug):
	node : Brush = None
	pass
class ValRandPlug(Plug):
	node : Brush = None
	pass
class WidthBiasPlug(Plug):
	node : Brush = None
	pass
class WidthLengthMapPlug(Plug):
	node : Brush = None
	pass
class WidthRandPlug(Plug):
	node : Brush = None
	pass
class WidthScale_FloatValuePlug(Plug):
	parent : WidthScalePlug = PlugDescriptor("widthScale")
	node : Brush = None
	pass
class WidthScale_InterpPlug(Plug):
	parent : WidthScalePlug = PlugDescriptor("widthScale")
	node : Brush = None
	pass
class WidthScale_PositionPlug(Plug):
	parent : WidthScalePlug = PlugDescriptor("widthScale")
	node : Brush = None
	pass
class WidthScalePlug(Plug):
	widthScale_FloatValue_ : WidthScale_FloatValuePlug = PlugDescriptor("widthScale_FloatValue")
	wscfv_ : WidthScale_FloatValuePlug = PlugDescriptor("widthScale_FloatValue")
	widthScale_Interp_ : WidthScale_InterpPlug = PlugDescriptor("widthScale_Interp")
	wsci_ : WidthScale_InterpPlug = PlugDescriptor("widthScale_Interp")
	widthScale_Position_ : WidthScale_PositionPlug = PlugDescriptor("widthScale_Position")
	wscp_ : WidthScale_PositionPlug = PlugDescriptor("widthScale_Position")
	node : Brush = None
	pass
class WigglePlug(Plug):
	node : Brush = None
	pass
class WiggleFrequencyPlug(Plug):
	node : Brush = None
	pass
class WiggleOffsetPlug(Plug):
	node : Brush = None
	pass
# endregion


# define node class
class Brush(_BASE_):
	attractRadiusOffset_ : AttractRadiusOffsetPlug = PlugDescriptor("attractRadiusOffset")
	attractRadiusScale_ : AttractRadiusScalePlug = PlugDescriptor("attractRadiusScale")
	azimuthMax_ : AzimuthMaxPlug = PlugDescriptor("azimuthMax")
	azimuthMin_ : AzimuthMinPlug = PlugDescriptor("azimuthMin")
	backShadow_ : BackShadowPlug = PlugDescriptor("backShadow")
	bend_ : BendPlug = PlugDescriptor("bend")
	bendBias_ : BendBiasPlug = PlugDescriptor("bendBias")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	blurIntensity_ : BlurIntensityPlug = PlugDescriptor("blurIntensity")
	blurMult_ : BlurMultPlug = PlugDescriptor("blurMult")
	branchAfterTwigs_ : BranchAfterTwigsPlug = PlugDescriptor("branchAfterTwigs")
	branchDropout_ : BranchDropoutPlug = PlugDescriptor("branchDropout")
	branchReflectivity_ : BranchReflectivityPlug = PlugDescriptor("branchReflectivity")
	branchThorns_ : BranchThornsPlug = PlugDescriptor("branchThorns")
	branches_ : BranchesPlug = PlugDescriptor("branches")
	brightnessRand_ : BrightnessRandPlug = PlugDescriptor("brightnessRand")
	brushType_ : BrushTypePlug = PlugDescriptor("brushType")
	brushWidth_ : BrushWidthPlug = PlugDescriptor("brushWidth")
	budColorB_ : BudColorBPlug = PlugDescriptor("budColorB")
	budColorG_ : BudColorGPlug = PlugDescriptor("budColorG")
	budColorR_ : BudColorRPlug = PlugDescriptor("budColorR")
	budColor_ : BudColorPlug = PlugDescriptor("budColor")
	budSize_ : BudSizePlug = PlugDescriptor("budSize")
	buds_ : BudsPlug = PlugDescriptor("buds")
	bumpBlur_ : BumpBlurPlug = PlugDescriptor("bumpBlur")
	bumpIntensity_ : BumpIntensityPlug = PlugDescriptor("bumpIntensity")
	castShadows_ : CastShadowsPlug = PlugDescriptor("castShadows")
	centerShadow_ : CenterShadowPlug = PlugDescriptor("centerShadow")
	collideMethod_ : CollideMethodPlug = PlugDescriptor("collideMethod")
	color1B_ : Color1BPlug = PlugDescriptor("color1B")
	color1G_ : Color1GPlug = PlugDescriptor("color1G")
	color1R_ : Color1RPlug = PlugDescriptor("color1R")
	color1_ : Color1Plug = PlugDescriptor("color1")
	color2B_ : Color2BPlug = PlugDescriptor("color2B")
	color2G_ : Color2GPlug = PlugDescriptor("color2G")
	color2R_ : Color2RPlug = PlugDescriptor("color2R")
	color2_ : Color2Plug = PlugDescriptor("color2")
	colorLengthMap_ : ColorLengthMapPlug = PlugDescriptor("colorLengthMap")
	creationScript_ : CreationScriptPlug = PlugDescriptor("creationScript")
	curl_ : CurlPlug = PlugDescriptor("curl")
	curlFrequency_ : CurlFrequencyPlug = PlugDescriptor("curlFrequency")
	curlOffset_ : CurlOffsetPlug = PlugDescriptor("curlOffset")
	curveAttract_ : CurveAttractPlug = PlugDescriptor("curveAttract")
	curveFollow_ : CurveFollowPlug = PlugDescriptor("curveFollow")
	curveMaxDist_ : CurveMaxDistPlug = PlugDescriptor("curveMaxDist")
	deflection_ : DeflectionPlug = PlugDescriptor("deflection")
	deflectionMax_ : DeflectionMaxPlug = PlugDescriptor("deflectionMax")
	deflectionMin_ : DeflectionMinPlug = PlugDescriptor("deflectionMin")
	depth_ : DepthPlug = PlugDescriptor("depth")
	depthShadow_ : DepthShadowPlug = PlugDescriptor("depthShadow")
	depthShadowDepth_ : DepthShadowDepthPlug = PlugDescriptor("depthShadowDepth")
	depthShadowType_ : DepthShadowTypePlug = PlugDescriptor("depthShadowType")
	displacementDelay_ : DisplacementDelayPlug = PlugDescriptor("displacementDelay")
	displacementOffset_ : DisplacementOffsetPlug = PlugDescriptor("displacementOffset")
	displacementScale_ : DisplacementScalePlug = PlugDescriptor("displacementScale")
	distanceScaling_ : DistanceScalingPlug = PlugDescriptor("distanceScaling")
	edgeAntialias_ : EdgeAntialiasPlug = PlugDescriptor("edgeAntialias")
	edgeClip_ : EdgeClipPlug = PlugDescriptor("edgeClip")
	edgeClipDepth_ : EdgeClipDepthPlug = PlugDescriptor("edgeClipDepth")
	elevationMax_ : ElevationMaxPlug = PlugDescriptor("elevationMax")
	elevationMin_ : ElevationMinPlug = PlugDescriptor("elevationMin")
	endCaps_ : EndCapsPlug = PlugDescriptor("endCaps")
	endTime_ : EndTimePlug = PlugDescriptor("endTime")
	environment_ColorB_ : Environment_ColorBPlug = PlugDescriptor("environment_ColorB")
	environment_ColorG_ : Environment_ColorGPlug = PlugDescriptor("environment_ColorG")
	environment_ColorR_ : Environment_ColorRPlug = PlugDescriptor("environment_ColorR")
	environment_Color_ : Environment_ColorPlug = PlugDescriptor("environment_Color")
	environment_Interp_ : Environment_InterpPlug = PlugDescriptor("environment_Interp")
	environment_Position_ : Environment_PositionPlug = PlugDescriptor("environment_Position")
	environment_ : EnvironmentPlug = PlugDescriptor("environment")
	fakeShadow_ : FakeShadowPlug = PlugDescriptor("fakeShadow")
	flatness1_ : Flatness1Plug = PlugDescriptor("flatness1")
	flatness2_ : Flatness2Plug = PlugDescriptor("flatness2")
	flowSpeed_ : FlowSpeedPlug = PlugDescriptor("flowSpeed")
	flowerAngle1_ : FlowerAngle1Plug = PlugDescriptor("flowerAngle1")
	flowerAngle2_ : FlowerAngle2Plug = PlugDescriptor("flowerAngle2")
	flowerFaceSun_ : FlowerFaceSunPlug = PlugDescriptor("flowerFaceSun")
	flowerHueRand_ : FlowerHueRandPlug = PlugDescriptor("flowerHueRand")
	flowerImage_ : FlowerImagePlug = PlugDescriptor("flowerImage")
	flowerLocation_ : FlowerLocationPlug = PlugDescriptor("flowerLocation")
	flowerReflectivity_ : FlowerReflectivityPlug = PlugDescriptor("flowerReflectivity")
	flowerSatRand_ : FlowerSatRandPlug = PlugDescriptor("flowerSatRand")
	flowerSizeDecay_ : FlowerSizeDecayPlug = PlugDescriptor("flowerSizeDecay")
	flowerSizeRand_ : FlowerSizeRandPlug = PlugDescriptor("flowerSizeRand")
	flowerSpecular_ : FlowerSpecularPlug = PlugDescriptor("flowerSpecular")
	flowerStart_ : FlowerStartPlug = PlugDescriptor("flowerStart")
	flowerStiffness_ : FlowerStiffnessPlug = PlugDescriptor("flowerStiffness")
	flowerThorns_ : FlowerThornsPlug = PlugDescriptor("flowerThorns")
	flowerTranslucence_ : FlowerTranslucencePlug = PlugDescriptor("flowerTranslucence")
	flowerTwist_ : FlowerTwistPlug = PlugDescriptor("flowerTwist")
	flowerUseBranchTex_ : FlowerUseBranchTexPlug = PlugDescriptor("flowerUseBranchTex")
	flowerValRand_ : FlowerValRandPlug = PlugDescriptor("flowerValRand")
	flowers_ : FlowersPlug = PlugDescriptor("flowers")
	forwardTwist_ : ForwardTwistPlug = PlugDescriptor("forwardTwist")
	fractalAmplitude_ : FractalAmplitudePlug = PlugDescriptor("fractalAmplitude")
	fractalRatio_ : FractalRatioPlug = PlugDescriptor("fractalRatio")
	fractalThreshold_ : FractalThresholdPlug = PlugDescriptor("fractalThreshold")
	frameExtension_ : FrameExtensionPlug = PlugDescriptor("frameExtension")
	fringeRemoval_ : FringeRemovalPlug = PlugDescriptor("fringeRemoval")
	gapRand_ : GapRandPlug = PlugDescriptor("gapRand")
	gapSize_ : GapSizePlug = PlugDescriptor("gapSize")
	gapSpacing_ : GapSpacingPlug = PlugDescriptor("gapSpacing")
	globalScale_ : GlobalScalePlug = PlugDescriptor("globalScale")
	glow_ : GlowPlug = PlugDescriptor("glow")
	glowColorB_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	glowColorG_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	glowColorR_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	glowColor_ : GlowColorPlug = PlugDescriptor("glowColor")
	glowSpread_ : GlowSpreadPlug = PlugDescriptor("glowSpread")
	gravity_ : GravityPlug = PlugDescriptor("gravity")
	hardEdges_ : HardEdgesPlug = PlugDescriptor("hardEdges")
	hueRand_ : HueRandPlug = PlugDescriptor("hueRand")
	illuminated_ : IlluminatedPlug = PlugDescriptor("illuminated")
	imageName_ : ImageNamePlug = PlugDescriptor("imageName")
	incandLengthMap_ : IncandLengthMapPlug = PlugDescriptor("incandLengthMap")
	incandescence1B_ : Incandescence1BPlug = PlugDescriptor("incandescence1B")
	incandescence1G_ : Incandescence1GPlug = PlugDescriptor("incandescence1G")
	incandescence1R_ : Incandescence1RPlug = PlugDescriptor("incandescence1R")
	incandescence1_ : Incandescence1Plug = PlugDescriptor("incandescence1")
	incandescence2B_ : Incandescence2BPlug = PlugDescriptor("incandescence2B")
	incandescence2G_ : Incandescence2GPlug = PlugDescriptor("incandescence2G")
	incandescence2R_ : Incandescence2RPlug = PlugDescriptor("incandescence2R")
	incandescence2_ : Incandescence2Plug = PlugDescriptor("incandescence2")
	leafAngle1_ : LeafAngle1Plug = PlugDescriptor("leafAngle1")
	leafAngle2_ : LeafAngle2Plug = PlugDescriptor("leafAngle2")
	leafBaseWidth_ : LeafBaseWidthPlug = PlugDescriptor("leafBaseWidth")
	leafBend_ : LeafBendPlug = PlugDescriptor("leafBend")
	leafColor1B_ : LeafColor1BPlug = PlugDescriptor("leafColor1B")
	leafColor1G_ : LeafColor1GPlug = PlugDescriptor("leafColor1G")
	leafColor1R_ : LeafColor1RPlug = PlugDescriptor("leafColor1R")
	leafColor1_ : LeafColor1Plug = PlugDescriptor("leafColor1")
	leafColor2B_ : LeafColor2BPlug = PlugDescriptor("leafColor2B")
	leafColor2G_ : LeafColor2GPlug = PlugDescriptor("leafColor2G")
	leafColor2R_ : LeafColor2RPlug = PlugDescriptor("leafColor2R")
	leafColor2_ : LeafColor2Plug = PlugDescriptor("leafColor2")
	leafCurl_FloatValue_ : LeafCurl_FloatValuePlug = PlugDescriptor("leafCurl_FloatValue")
	leafCurl_Interp_ : LeafCurl_InterpPlug = PlugDescriptor("leafCurl_Interp")
	leafCurl_Position_ : LeafCurl_PositionPlug = PlugDescriptor("leafCurl_Position")
	leafCurl_ : LeafCurlPlug = PlugDescriptor("leafCurl")
	leafDropout_ : LeafDropoutPlug = PlugDescriptor("leafDropout")
	leafFaceSun_ : LeafFaceSunPlug = PlugDescriptor("leafFaceSun")
	leafFlatness_ : LeafFlatnessPlug = PlugDescriptor("leafFlatness")
	leafForwardTwist_ : LeafForwardTwistPlug = PlugDescriptor("leafForwardTwist")
	leafHueRand_ : LeafHueRandPlug = PlugDescriptor("leafHueRand")
	leafImage_ : LeafImagePlug = PlugDescriptor("leafImage")
	leafLength_ : LeafLengthPlug = PlugDescriptor("leafLength")
	leafLocation_ : LeafLocationPlug = PlugDescriptor("leafLocation")
	leafReflectivity_ : LeafReflectivityPlug = PlugDescriptor("leafReflectivity")
	leafSatRand_ : LeafSatRandPlug = PlugDescriptor("leafSatRand")
	leafSegments_ : LeafSegmentsPlug = PlugDescriptor("leafSegments")
	leafSizeDecay_ : LeafSizeDecayPlug = PlugDescriptor("leafSizeDecay")
	leafSizeRand_ : LeafSizeRandPlug = PlugDescriptor("leafSizeRand")
	leafSpecular_ : LeafSpecularPlug = PlugDescriptor("leafSpecular")
	leafStart_ : LeafStartPlug = PlugDescriptor("leafStart")
	leafStiffness_ : LeafStiffnessPlug = PlugDescriptor("leafStiffness")
	leafThorns_ : LeafThornsPlug = PlugDescriptor("leafThorns")
	leafTipWidth_ : LeafTipWidthPlug = PlugDescriptor("leafTipWidth")
	leafTranslucence_ : LeafTranslucencePlug = PlugDescriptor("leafTranslucence")
	leafTwirl_ : LeafTwirlPlug = PlugDescriptor("leafTwirl")
	leafTwist_ : LeafTwistPlug = PlugDescriptor("leafTwist")
	leafUseBranchTex_ : LeafUseBranchTexPlug = PlugDescriptor("leafUseBranchTex")
	leafValRand_ : LeafValRandPlug = PlugDescriptor("leafValRand")
	leafWidthScale_FloatValue_ : LeafWidthScale_FloatValuePlug = PlugDescriptor("leafWidthScale_FloatValue")
	leafWidthScale_Interp_ : LeafWidthScale_InterpPlug = PlugDescriptor("leafWidthScale_Interp")
	leafWidthScale_Position_ : LeafWidthScale_PositionPlug = PlugDescriptor("leafWidthScale_Position")
	leafWidthScale_ : LeafWidthScalePlug = PlugDescriptor("leafWidthScale")
	leaves_ : LeavesPlug = PlugDescriptor("leaves")
	leavesInCluster_ : LeavesInClusterPlug = PlugDescriptor("leavesInCluster")
	lengthFlex_ : LengthFlexPlug = PlugDescriptor("lengthFlex")
	lengthMax_ : LengthMaxPlug = PlugDescriptor("lengthMax")
	lengthMin_ : LengthMinPlug = PlugDescriptor("lengthMin")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	lightDirection_ : LightDirectionPlug = PlugDescriptor("lightDirection")
	lightingBasedWidth_ : LightingBasedWidthPlug = PlugDescriptor("lightingBasedWidth")
	luminanceIsDisplacement_ : LuminanceIsDisplacementPlug = PlugDescriptor("luminanceIsDisplacement")
	mapColor_ : MapColorPlug = PlugDescriptor("mapColor")
	mapDisplacement_ : MapDisplacementPlug = PlugDescriptor("mapDisplacement")
	mapMethod_ : MapMethodPlug = PlugDescriptor("mapMethod")
	mapOpacity_ : MapOpacityPlug = PlugDescriptor("mapOpacity")
	maxAttractDistance_ : MaxAttractDistancePlug = PlugDescriptor("maxAttractDistance")
	maxPixelWidth_ : MaxPixelWidthPlug = PlugDescriptor("maxPixelWidth")
	middleBranch_ : MiddleBranchPlug = PlugDescriptor("middleBranch")
	minPixelWidth_ : MinPixelWidthPlug = PlugDescriptor("minPixelWidth")
	minSize_ : MinSizePlug = PlugDescriptor("minSize")
	modifyAlpha_ : ModifyAlphaPlug = PlugDescriptor("modifyAlpha")
	modifyColor_ : ModifyColorPlug = PlugDescriptor("modifyColor")
	modifyDepth_ : ModifyDepthPlug = PlugDescriptor("modifyDepth")
	momentum_ : MomentumPlug = PlugDescriptor("momentum")
	multiStreakDiffuseRand_ : MultiStreakDiffuseRandPlug = PlugDescriptor("multiStreakDiffuseRand")
	multiStreakLightAll_ : MultiStreakLightAllPlug = PlugDescriptor("multiStreakLightAll")
	multiStreakSpecularRand_ : MultiStreakSpecularRandPlug = PlugDescriptor("multiStreakSpecularRand")
	multiStreakSpread1_ : MultiStreakSpread1Plug = PlugDescriptor("multiStreakSpread1")
	multiStreakSpread2_ : MultiStreakSpread2Plug = PlugDescriptor("multiStreakSpread2")
	multiStreaks_ : MultiStreaksPlug = PlugDescriptor("multiStreaks")
	noise_ : NoisePlug = PlugDescriptor("noise")
	noiseFrequency_ : NoiseFrequencyPlug = PlugDescriptor("noiseFrequency")
	noiseOffset_ : NoiseOffsetPlug = PlugDescriptor("noiseOffset")
	numBranches_ : NumBranchesPlug = PlugDescriptor("numBranches")
	numFlowers_ : NumFlowersPlug = PlugDescriptor("numFlowers")
	numLeafClusters_ : NumLeafClustersPlug = PlugDescriptor("numLeafClusters")
	numTwigClusters_ : NumTwigClustersPlug = PlugDescriptor("numTwigClusters")
	occlusionWidthScale_ : OcclusionWidthScalePlug = PlugDescriptor("occlusionWidthScale")
	occupyAttraction_ : OccupyAttractionPlug = PlugDescriptor("occupyAttraction")
	occupyBranchTermination_ : OccupyBranchTerminationPlug = PlugDescriptor("occupyBranchTermination")
	occupyRadiusOffset_ : OccupyRadiusOffsetPlug = PlugDescriptor("occupyRadiusOffset")
	occupyRadiusScale_ : OccupyRadiusScalePlug = PlugDescriptor("occupyRadiusScale")
	offsetU_ : OffsetUPlug = PlugDescriptor("offsetU")
	offsetV_ : OffsetVPlug = PlugDescriptor("offsetV")
	outBrush_ : OutBrushPlug = PlugDescriptor("outBrush")
	pathAttract_ : PathAttractPlug = PlugDescriptor("pathAttract")
	pathFollow_ : PathFollowPlug = PlugDescriptor("pathFollow")
	perPixelLighting_ : PerPixelLightingPlug = PlugDescriptor("perPixelLighting")
	petalBaseWidth_ : PetalBaseWidthPlug = PlugDescriptor("petalBaseWidth")
	petalBend_ : PetalBendPlug = PlugDescriptor("petalBend")
	petalColor1B_ : PetalColor1BPlug = PlugDescriptor("petalColor1B")
	petalColor1G_ : PetalColor1GPlug = PlugDescriptor("petalColor1G")
	petalColor1R_ : PetalColor1RPlug = PlugDescriptor("petalColor1R")
	petalColor1_ : PetalColor1Plug = PlugDescriptor("petalColor1")
	petalColor2B_ : PetalColor2BPlug = PlugDescriptor("petalColor2B")
	petalColor2G_ : PetalColor2GPlug = PlugDescriptor("petalColor2G")
	petalColor2R_ : PetalColor2RPlug = PlugDescriptor("petalColor2R")
	petalColor2_ : PetalColor2Plug = PlugDescriptor("petalColor2")
	petalCurl_FloatValue_ : PetalCurl_FloatValuePlug = PlugDescriptor("petalCurl_FloatValue")
	petalCurl_Interp_ : PetalCurl_InterpPlug = PlugDescriptor("petalCurl_Interp")
	petalCurl_Position_ : PetalCurl_PositionPlug = PlugDescriptor("petalCurl_Position")
	petalCurl_ : PetalCurlPlug = PlugDescriptor("petalCurl")
	petalDropout_ : PetalDropoutPlug = PlugDescriptor("petalDropout")
	petalFlatness_ : PetalFlatnessPlug = PlugDescriptor("petalFlatness")
	petalForwardTwist_ : PetalForwardTwistPlug = PlugDescriptor("petalForwardTwist")
	petalLength_ : PetalLengthPlug = PlugDescriptor("petalLength")
	petalSegments_ : PetalSegmentsPlug = PlugDescriptor("petalSegments")
	petalTipWidth_ : PetalTipWidthPlug = PlugDescriptor("petalTipWidth")
	petalTwirl_ : PetalTwirlPlug = PlugDescriptor("petalTwirl")
	petalWidthScale_FloatValue_ : PetalWidthScale_FloatValuePlug = PlugDescriptor("petalWidthScale_FloatValue")
	petalWidthScale_Interp_ : PetalWidthScale_InterpPlug = PlugDescriptor("petalWidthScale_Interp")
	petalWidthScale_Position_ : PetalWidthScale_PositionPlug = PlugDescriptor("petalWidthScale_Position")
	petalWidthScale_ : PetalWidthScalePlug = PlugDescriptor("petalWidthScale")
	petalsInFlower_ : PetalsInFlowerPlug = PlugDescriptor("petalsInFlower")
	random_ : RandomPlug = PlugDescriptor("random")
	realLights_ : RealLightsPlug = PlugDescriptor("realLights")
	reflectionRolloff_FloatValue_ : ReflectionRolloff_FloatValuePlug = PlugDescriptor("reflectionRolloff_FloatValue")
	reflectionRolloff_Interp_ : ReflectionRolloff_InterpPlug = PlugDescriptor("reflectionRolloff_Interp")
	reflectionRolloff_Position_ : ReflectionRolloff_PositionPlug = PlugDescriptor("reflectionRolloff_Position")
	reflectionRolloff_ : ReflectionRolloffPlug = PlugDescriptor("reflectionRolloff")
	repeatU_ : RepeatUPlug = PlugDescriptor("repeatU")
	repeatV_ : RepeatVPlug = PlugDescriptor("repeatV")
	rootFade_ : RootFadePlug = PlugDescriptor("rootFade")
	runtimeScript_ : RuntimeScriptPlug = PlugDescriptor("runtimeScript")
	satRand_ : SatRandPlug = PlugDescriptor("satRand")
	screenspaceWidth_ : ScreenspaceWidthPlug = PlugDescriptor("screenspaceWidth")
	segmentLengthBias_ : SegmentLengthBiasPlug = PlugDescriptor("segmentLengthBias")
	segmentWidthBias_ : SegmentWidthBiasPlug = PlugDescriptor("segmentWidthBias")
	segments_ : SegmentsPlug = PlugDescriptor("segments")
	shaderGlow_ : ShaderGlowPlug = PlugDescriptor("shaderGlow")
	shadowDiffusion_ : ShadowDiffusionPlug = PlugDescriptor("shadowDiffusion")
	shadowOffset_ : ShadowOffsetPlug = PlugDescriptor("shadowOffset")
	shadowTransparency_ : ShadowTransparencyPlug = PlugDescriptor("shadowTransparency")
	simplifyMethod_ : SimplifyMethodPlug = PlugDescriptor("simplifyMethod")
	singleSided_ : SingleSidedPlug = PlugDescriptor("singleSided")
	smear_ : SmearPlug = PlugDescriptor("smear")
	smearU_ : SmearUPlug = PlugDescriptor("smearU")
	smearV_ : SmearVPlug = PlugDescriptor("smearV")
	softness_ : SoftnessPlug = PlugDescriptor("softness")
	specular_ : SpecularPlug = PlugDescriptor("specular")
	specularColorB_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	specularColorG_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	specularColorR_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	specularColor_ : SpecularColorPlug = PlugDescriptor("specularColor")
	specularPower_ : SpecularPowerPlug = PlugDescriptor("specularPower")
	spiralDecay_ : SpiralDecayPlug = PlugDescriptor("spiralDecay")
	spiralMax_ : SpiralMaxPlug = PlugDescriptor("spiralMax")
	spiralMin_ : SpiralMinPlug = PlugDescriptor("spiralMin")
	splitAngle_ : SplitAnglePlug = PlugDescriptor("splitAngle")
	splitBias_ : SplitBiasPlug = PlugDescriptor("splitBias")
	splitLengthMap_ : SplitLengthMapPlug = PlugDescriptor("splitLengthMap")
	splitMaxDepth_ : SplitMaxDepthPlug = PlugDescriptor("splitMaxDepth")
	splitRand_ : SplitRandPlug = PlugDescriptor("splitRand")
	splitSizeDecay_ : SplitSizeDecayPlug = PlugDescriptor("splitSizeDecay")
	splitTwist_ : SplitTwistPlug = PlugDescriptor("splitTwist")
	stampDensity_ : StampDensityPlug = PlugDescriptor("stampDensity")
	startBranches_ : StartBranchesPlug = PlugDescriptor("startBranches")
	startTime_ : StartTimePlug = PlugDescriptor("startTime")
	startTubes_ : StartTubesPlug = PlugDescriptor("startTubes")
	strokeTime_ : StrokeTimePlug = PlugDescriptor("strokeTime")
	subSegments_ : SubSegmentsPlug = PlugDescriptor("subSegments")
	sunDirectionX_ : SunDirectionXPlug = PlugDescriptor("sunDirectionX")
	sunDirectionY_ : SunDirectionYPlug = PlugDescriptor("sunDirectionY")
	sunDirectionZ_ : SunDirectionZPlug = PlugDescriptor("sunDirectionZ")
	sunDirection_ : SunDirectionPlug = PlugDescriptor("sunDirection")
	surfaceAttract_ : SurfaceAttractPlug = PlugDescriptor("surfaceAttract")
	surfaceCollide_ : SurfaceCollidePlug = PlugDescriptor("surfaceCollide")
	surfaceSampleDensity_ : SurfaceSampleDensityPlug = PlugDescriptor("surfaceSampleDensity")
	surfaceSnap_ : SurfaceSnapPlug = PlugDescriptor("surfaceSnap")
	terminalLeaf_ : TerminalLeafPlug = PlugDescriptor("terminalLeaf")
	texAlpha1_ : TexAlpha1Plug = PlugDescriptor("texAlpha1")
	texAlpha2_ : TexAlpha2Plug = PlugDescriptor("texAlpha2")
	texColor1B_ : TexColor1BPlug = PlugDescriptor("texColor1B")
	texColor1G_ : TexColor1GPlug = PlugDescriptor("texColor1G")
	texColor1R_ : TexColor1RPlug = PlugDescriptor("texColor1R")
	texColor1_ : TexColor1Plug = PlugDescriptor("texColor1")
	texColor2B_ : TexColor2BPlug = PlugDescriptor("texColor2B")
	texColor2G_ : TexColor2GPlug = PlugDescriptor("texColor2G")
	texColor2R_ : TexColor2RPlug = PlugDescriptor("texColor2R")
	texColor2_ : TexColor2Plug = PlugDescriptor("texColor2")
	texColorOffset_ : TexColorOffsetPlug = PlugDescriptor("texColorOffset")
	texColorScale_ : TexColorScalePlug = PlugDescriptor("texColorScale")
	texOpacityOffset_ : TexOpacityOffsetPlug = PlugDescriptor("texOpacityOffset")
	texOpacityScale_ : TexOpacityScalePlug = PlugDescriptor("texOpacityScale")
	texUniformity_ : TexUniformityPlug = PlugDescriptor("texUniformity")
	textureFlow_ : TextureFlowPlug = PlugDescriptor("textureFlow")
	textureType_ : TextureTypePlug = PlugDescriptor("textureType")
	thornBaseColorB_ : ThornBaseColorBPlug = PlugDescriptor("thornBaseColorB")
	thornBaseColorG_ : ThornBaseColorGPlug = PlugDescriptor("thornBaseColorG")
	thornBaseColorR_ : ThornBaseColorRPlug = PlugDescriptor("thornBaseColorR")
	thornBaseColor_ : ThornBaseColorPlug = PlugDescriptor("thornBaseColor")
	thornBaseWidth_ : ThornBaseWidthPlug = PlugDescriptor("thornBaseWidth")
	thornDensity_ : ThornDensityPlug = PlugDescriptor("thornDensity")
	thornElevation_ : ThornElevationPlug = PlugDescriptor("thornElevation")
	thornLength_ : ThornLengthPlug = PlugDescriptor("thornLength")
	thornSpecular_ : ThornSpecularPlug = PlugDescriptor("thornSpecular")
	thornTipColorB_ : ThornTipColorBPlug = PlugDescriptor("thornTipColorB")
	thornTipColorG_ : ThornTipColorGPlug = PlugDescriptor("thornTipColorG")
	thornTipColorR_ : ThornTipColorRPlug = PlugDescriptor("thornTipColorR")
	thornTipColor_ : ThornTipColorPlug = PlugDescriptor("thornTipColor")
	thornTipWidth_ : ThornTipWidthPlug = PlugDescriptor("thornTipWidth")
	time_ : TimePlug = PlugDescriptor("time")
	timeClip_ : TimeClipPlug = PlugDescriptor("timeClip")
	tipFade_ : TipFadePlug = PlugDescriptor("tipFade")
	translucence_ : TranslucencePlug = PlugDescriptor("translucence")
	transpLengthMap_ : TranspLengthMapPlug = PlugDescriptor("transpLengthMap")
	transparency1B_ : Transparency1BPlug = PlugDescriptor("transparency1B")
	transparency1G_ : Transparency1GPlug = PlugDescriptor("transparency1G")
	transparency1R_ : Transparency1RPlug = PlugDescriptor("transparency1R")
	transparency1_ : Transparency1Plug = PlugDescriptor("transparency1")
	transparency2B_ : Transparency2BPlug = PlugDescriptor("transparency2B")
	transparency2G_ : Transparency2GPlug = PlugDescriptor("transparency2G")
	transparency2R_ : Transparency2RPlug = PlugDescriptor("transparency2R")
	transparency2_ : Transparency2Plug = PlugDescriptor("transparency2")
	tubeCompletion_ : TubeCompletionPlug = PlugDescriptor("tubeCompletion")
	tubeDirection_ : TubeDirectionPlug = PlugDescriptor("tubeDirection")
	tubeRand_ : TubeRandPlug = PlugDescriptor("tubeRand")
	tubeSections_ : TubeSectionsPlug = PlugDescriptor("tubeSections")
	tubeWidth1_ : TubeWidth1Plug = PlugDescriptor("tubeWidth1")
	tubeWidth2_ : TubeWidth2Plug = PlugDescriptor("tubeWidth2")
	tubes_ : TubesPlug = PlugDescriptor("tubes")
	tubesPerStep_ : TubesPerStepPlug = PlugDescriptor("tubesPerStep")
	turbulence_ : TurbulencePlug = PlugDescriptor("turbulence")
	turbulenceFrequency_ : TurbulenceFrequencyPlug = PlugDescriptor("turbulenceFrequency")
	turbulenceInterpolation_ : TurbulenceInterpolationPlug = PlugDescriptor("turbulenceInterpolation")
	turbulenceOffsetX_ : TurbulenceOffsetXPlug = PlugDescriptor("turbulenceOffsetX")
	turbulenceOffsetY_ : TurbulenceOffsetYPlug = PlugDescriptor("turbulenceOffsetY")
	turbulenceOffsetZ_ : TurbulenceOffsetZPlug = PlugDescriptor("turbulenceOffsetZ")
	turbulenceOffset_ : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	turbulenceSpeed_ : TurbulenceSpeedPlug = PlugDescriptor("turbulenceSpeed")
	turbulenceType_ : TurbulenceTypePlug = PlugDescriptor("turbulenceType")
	twigAngle1_ : TwigAngle1Plug = PlugDescriptor("twigAngle1")
	twigAngle2_ : TwigAngle2Plug = PlugDescriptor("twigAngle2")
	twigBaseWidth_ : TwigBaseWidthPlug = PlugDescriptor("twigBaseWidth")
	twigDropout_ : TwigDropoutPlug = PlugDescriptor("twigDropout")
	twigLength_ : TwigLengthPlug = PlugDescriptor("twigLength")
	twigLengthScale_FloatValue_ : TwigLengthScale_FloatValuePlug = PlugDescriptor("twigLengthScale_FloatValue")
	twigLengthScale_Interp_ : TwigLengthScale_InterpPlug = PlugDescriptor("twigLengthScale_Interp")
	twigLengthScale_Position_ : TwigLengthScale_PositionPlug = PlugDescriptor("twigLengthScale_Position")
	twigLengthScale_ : TwigLengthScalePlug = PlugDescriptor("twigLengthScale")
	twigStart_ : TwigStartPlug = PlugDescriptor("twigStart")
	twigStiffness_ : TwigStiffnessPlug = PlugDescriptor("twigStiffness")
	twigThorns_ : TwigThornsPlug = PlugDescriptor("twigThorns")
	twigTipWidth_ : TwigTipWidthPlug = PlugDescriptor("twigTipWidth")
	twigTwist_ : TwigTwistPlug = PlugDescriptor("twigTwist")
	twigs_ : TwigsPlug = PlugDescriptor("twigs")
	twigsInCluster_ : TwigsInClusterPlug = PlugDescriptor("twigsInCluster")
	twist_ : TwistPlug = PlugDescriptor("twist")
	twistRand_ : TwistRandPlug = PlugDescriptor("twistRand")
	twistRate_ : TwistRatePlug = PlugDescriptor("twistRate")
	uniformForceX_ : UniformForceXPlug = PlugDescriptor("uniformForceX")
	uniformForceY_ : UniformForceYPlug = PlugDescriptor("uniformForceY")
	uniformForceZ_ : UniformForceZPlug = PlugDescriptor("uniformForceZ")
	uniformForce_ : UniformForcePlug = PlugDescriptor("uniformForce")
	useFrameExtension_ : UseFrameExtensionPlug = PlugDescriptor("useFrameExtension")
	valRand_ : ValRandPlug = PlugDescriptor("valRand")
	widthBias_ : WidthBiasPlug = PlugDescriptor("widthBias")
	widthLengthMap_ : WidthLengthMapPlug = PlugDescriptor("widthLengthMap")
	widthRand_ : WidthRandPlug = PlugDescriptor("widthRand")
	widthScale_FloatValue_ : WidthScale_FloatValuePlug = PlugDescriptor("widthScale_FloatValue")
	widthScale_Interp_ : WidthScale_InterpPlug = PlugDescriptor("widthScale_Interp")
	widthScale_Position_ : WidthScale_PositionPlug = PlugDescriptor("widthScale_Position")
	widthScale_ : WidthScalePlug = PlugDescriptor("widthScale")
	wiggle_ : WigglePlug = PlugDescriptor("wiggle")
	wiggleFrequency_ : WiggleFrequencyPlug = PlugDescriptor("wiggleFrequency")
	wiggleOffset_ : WiggleOffsetPlug = PlugDescriptor("wiggleOffset")

	# node attributes

	typeName = "brush"
	apiTypeInt = 765
	apiTypeStr = "kBrush"
	typeIdInt = 1112691528
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["attractRadiusOffset", "attractRadiusScale", "azimuthMax", "azimuthMin", "backShadow", "bend", "bendBias", "binMembership", "blurIntensity", "blurMult", "branchAfterTwigs", "branchDropout", "branchReflectivity", "branchThorns", "branches", "brightnessRand", "brushType", "brushWidth", "budColorB", "budColorG", "budColorR", "budColor", "budSize", "buds", "bumpBlur", "bumpIntensity", "castShadows", "centerShadow", "collideMethod", "color1B", "color1G", "color1R", "color1", "color2B", "color2G", "color2R", "color2", "colorLengthMap", "creationScript", "curl", "curlFrequency", "curlOffset", "curveAttract", "curveFollow", "curveMaxDist", "deflection", "deflectionMax", "deflectionMin", "depth", "depthShadow", "depthShadowDepth", "depthShadowType", "displacementDelay", "displacementOffset", "displacementScale", "distanceScaling", "edgeAntialias", "edgeClip", "edgeClipDepth", "elevationMax", "elevationMin", "endCaps", "endTime", "environment_ColorB", "environment_ColorG", "environment_ColorR", "environment_Color", "environment_Interp", "environment_Position", "environment", "fakeShadow", "flatness1", "flatness2", "flowSpeed", "flowerAngle1", "flowerAngle2", "flowerFaceSun", "flowerHueRand", "flowerImage", "flowerLocation", "flowerReflectivity", "flowerSatRand", "flowerSizeDecay", "flowerSizeRand", "flowerSpecular", "flowerStart", "flowerStiffness", "flowerThorns", "flowerTranslucence", "flowerTwist", "flowerUseBranchTex", "flowerValRand", "flowers", "forwardTwist", "fractalAmplitude", "fractalRatio", "fractalThreshold", "frameExtension", "fringeRemoval", "gapRand", "gapSize", "gapSpacing", "globalScale", "glow", "glowColorB", "glowColorG", "glowColorR", "glowColor", "glowSpread", "gravity", "hardEdges", "hueRand", "illuminated", "imageName", "incandLengthMap", "incandescence1B", "incandescence1G", "incandescence1R", "incandescence1", "incandescence2B", "incandescence2G", "incandescence2R", "incandescence2", "leafAngle1", "leafAngle2", "leafBaseWidth", "leafBend", "leafColor1B", "leafColor1G", "leafColor1R", "leafColor1", "leafColor2B", "leafColor2G", "leafColor2R", "leafColor2", "leafCurl_FloatValue", "leafCurl_Interp", "leafCurl_Position", "leafCurl", "leafDropout", "leafFaceSun", "leafFlatness", "leafForwardTwist", "leafHueRand", "leafImage", "leafLength", "leafLocation", "leafReflectivity", "leafSatRand", "leafSegments", "leafSizeDecay", "leafSizeRand", "leafSpecular", "leafStart", "leafStiffness", "leafThorns", "leafTipWidth", "leafTranslucence", "leafTwirl", "leafTwist", "leafUseBranchTex", "leafValRand", "leafWidthScale_FloatValue", "leafWidthScale_Interp", "leafWidthScale_Position", "leafWidthScale", "leaves", "leavesInCluster", "lengthFlex", "lengthMax", "lengthMin", "lightDirectionX", "lightDirectionY", "lightDirectionZ", "lightDirection", "lightingBasedWidth", "luminanceIsDisplacement", "mapColor", "mapDisplacement", "mapMethod", "mapOpacity", "maxAttractDistance", "maxPixelWidth", "middleBranch", "minPixelWidth", "minSize", "modifyAlpha", "modifyColor", "modifyDepth", "momentum", "multiStreakDiffuseRand", "multiStreakLightAll", "multiStreakSpecularRand", "multiStreakSpread1", "multiStreakSpread2", "multiStreaks", "noise", "noiseFrequency", "noiseOffset", "numBranches", "numFlowers", "numLeafClusters", "numTwigClusters", "occlusionWidthScale", "occupyAttraction", "occupyBranchTermination", "occupyRadiusOffset", "occupyRadiusScale", "offsetU", "offsetV", "outBrush", "pathAttract", "pathFollow", "perPixelLighting", "petalBaseWidth", "petalBend", "petalColor1B", "petalColor1G", "petalColor1R", "petalColor1", "petalColor2B", "petalColor2G", "petalColor2R", "petalColor2", "petalCurl_FloatValue", "petalCurl_Interp", "petalCurl_Position", "petalCurl", "petalDropout", "petalFlatness", "petalForwardTwist", "petalLength", "petalSegments", "petalTipWidth", "petalTwirl", "petalWidthScale_FloatValue", "petalWidthScale_Interp", "petalWidthScale_Position", "petalWidthScale", "petalsInFlower", "random", "realLights", "reflectionRolloff_FloatValue", "reflectionRolloff_Interp", "reflectionRolloff_Position", "reflectionRolloff", "repeatU", "repeatV", "rootFade", "runtimeScript", "satRand", "screenspaceWidth", "segmentLengthBias", "segmentWidthBias", "segments", "shaderGlow", "shadowDiffusion", "shadowOffset", "shadowTransparency", "simplifyMethod", "singleSided", "smear", "smearU", "smearV", "softness", "specular", "specularColorB", "specularColorG", "specularColorR", "specularColor", "specularPower", "spiralDecay", "spiralMax", "spiralMin", "splitAngle", "splitBias", "splitLengthMap", "splitMaxDepth", "splitRand", "splitSizeDecay", "splitTwist", "stampDensity", "startBranches", "startTime", "startTubes", "strokeTime", "subSegments", "sunDirectionX", "sunDirectionY", "sunDirectionZ", "sunDirection", "surfaceAttract", "surfaceCollide", "surfaceSampleDensity", "surfaceSnap", "terminalLeaf", "texAlpha1", "texAlpha2", "texColor1B", "texColor1G", "texColor1R", "texColor1", "texColor2B", "texColor2G", "texColor2R", "texColor2", "texColorOffset", "texColorScale", "texOpacityOffset", "texOpacityScale", "texUniformity", "textureFlow", "textureType", "thornBaseColorB", "thornBaseColorG", "thornBaseColorR", "thornBaseColor", "thornBaseWidth", "thornDensity", "thornElevation", "thornLength", "thornSpecular", "thornTipColorB", "thornTipColorG", "thornTipColorR", "thornTipColor", "thornTipWidth", "time", "timeClip", "tipFade", "translucence", "transpLengthMap", "transparency1B", "transparency1G", "transparency1R", "transparency1", "transparency2B", "transparency2G", "transparency2R", "transparency2", "tubeCompletion", "tubeDirection", "tubeRand", "tubeSections", "tubeWidth1", "tubeWidth2", "tubes", "tubesPerStep", "turbulence", "turbulenceFrequency", "turbulenceInterpolation", "turbulenceOffsetX", "turbulenceOffsetY", "turbulenceOffsetZ", "turbulenceOffset", "turbulenceSpeed", "turbulenceType", "twigAngle1", "twigAngle2", "twigBaseWidth", "twigDropout", "twigLength", "twigLengthScale_FloatValue", "twigLengthScale_Interp", "twigLengthScale_Position", "twigLengthScale", "twigStart", "twigStiffness", "twigThorns", "twigTipWidth", "twigTwist", "twigs", "twigsInCluster", "twist", "twistRand", "twistRate", "uniformForceX", "uniformForceY", "uniformForceZ", "uniformForce", "useFrameExtension", "valRand", "widthBias", "widthLengthMap", "widthRand", "widthScale_FloatValue", "widthScale_Interp", "widthScale_Position", "widthScale", "wiggle", "wiggleFrequency", "wiggleOffset"]
	nodeLeafPlugs = ["attractRadiusOffset", "attractRadiusScale", "azimuthMax", "azimuthMin", "backShadow", "bend", "bendBias", "binMembership", "blurIntensity", "blurMult", "branchAfterTwigs", "branchDropout", "branchReflectivity", "branchThorns", "branches", "brightnessRand", "brushType", "brushWidth", "budColor", "budSize", "buds", "bumpBlur", "bumpIntensity", "castShadows", "centerShadow", "collideMethod", "color1", "color2", "colorLengthMap", "creationScript", "curl", "curlFrequency", "curlOffset", "curveAttract", "curveFollow", "curveMaxDist", "deflection", "deflectionMax", "deflectionMin", "depth", "depthShadow", "depthShadowDepth", "depthShadowType", "displacementDelay", "displacementOffset", "displacementScale", "distanceScaling", "edgeAntialias", "edgeClip", "edgeClipDepth", "elevationMax", "elevationMin", "endCaps", "endTime", "environment", "fakeShadow", "flatness1", "flatness2", "flowSpeed", "flowerAngle1", "flowerAngle2", "flowerFaceSun", "flowerHueRand", "flowerImage", "flowerLocation", "flowerReflectivity", "flowerSatRand", "flowerSizeDecay", "flowerSizeRand", "flowerSpecular", "flowerStart", "flowerStiffness", "flowerThorns", "flowerTranslucence", "flowerTwist", "flowerUseBranchTex", "flowerValRand", "flowers", "forwardTwist", "fractalAmplitude", "fractalRatio", "fractalThreshold", "frameExtension", "fringeRemoval", "gapRand", "gapSize", "gapSpacing", "globalScale", "glow", "glowColor", "glowSpread", "gravity", "hardEdges", "hueRand", "illuminated", "imageName", "incandLengthMap", "incandescence1", "incandescence2", "leafAngle1", "leafAngle2", "leafBaseWidth", "leafBend", "leafColor1", "leafColor2", "leafCurl", "leafDropout", "leafFaceSun", "leafFlatness", "leafForwardTwist", "leafHueRand", "leafImage", "leafLength", "leafLocation", "leafReflectivity", "leafSatRand", "leafSegments", "leafSizeDecay", "leafSizeRand", "leafSpecular", "leafStart", "leafStiffness", "leafThorns", "leafTipWidth", "leafTranslucence", "leafTwirl", "leafTwist", "leafUseBranchTex", "leafValRand", "leafWidthScale", "leaves", "leavesInCluster", "lengthFlex", "lengthMax", "lengthMin", "lightDirection", "lightingBasedWidth", "luminanceIsDisplacement", "mapColor", "mapDisplacement", "mapMethod", "mapOpacity", "maxAttractDistance", "maxPixelWidth", "middleBranch", "minPixelWidth", "minSize", "modifyAlpha", "modifyColor", "modifyDepth", "momentum", "multiStreakDiffuseRand", "multiStreakLightAll", "multiStreakSpecularRand", "multiStreakSpread1", "multiStreakSpread2", "multiStreaks", "noise", "noiseFrequency", "noiseOffset", "numBranches", "numFlowers", "numLeafClusters", "numTwigClusters", "occlusionWidthScale", "occupyAttraction", "occupyBranchTermination", "occupyRadiusOffset", "occupyRadiusScale", "offsetU", "offsetV", "outBrush", "pathAttract", "pathFollow", "perPixelLighting", "petalBaseWidth", "petalBend", "petalColor1", "petalColor2", "petalCurl", "petalDropout", "petalFlatness", "petalForwardTwist", "petalLength", "petalSegments", "petalTipWidth", "petalTwirl", "petalWidthScale", "petalsInFlower", "random", "realLights", "reflectionRolloff", "repeatU", "repeatV", "rootFade", "runtimeScript", "satRand", "screenspaceWidth", "segmentLengthBias", "segmentWidthBias", "segments", "shaderGlow", "shadowDiffusion", "shadowOffset", "shadowTransparency", "simplifyMethod", "singleSided", "smear", "smearU", "smearV", "softness", "specular", "specularColor", "specularPower", "spiralDecay", "spiralMax", "spiralMin", "splitAngle", "splitBias", "splitLengthMap", "splitMaxDepth", "splitRand", "splitSizeDecay", "splitTwist", "stampDensity", "startBranches", "startTime", "startTubes", "strokeTime", "subSegments", "sunDirection", "surfaceAttract", "surfaceCollide", "surfaceSampleDensity", "surfaceSnap", "terminalLeaf", "texAlpha1", "texAlpha2", "texColor1", "texColor2", "texColorOffset", "texColorScale", "texOpacityOffset", "texOpacityScale", "texUniformity", "textureFlow", "textureType", "thornBaseColor", "thornBaseWidth", "thornDensity", "thornElevation", "thornLength", "thornSpecular", "thornTipColor", "thornTipWidth", "time", "timeClip", "tipFade", "translucence", "transpLengthMap", "transparency1", "transparency2", "tubeCompletion", "tubeDirection", "tubeRand", "tubeSections", "tubeWidth1", "tubeWidth2", "tubes", "tubesPerStep", "turbulence", "turbulenceFrequency", "turbulenceInterpolation", "turbulenceOffset", "turbulenceSpeed", "turbulenceType", "twigAngle1", "twigAngle2", "twigBaseWidth", "twigDropout", "twigLength", "twigLengthScale", "twigStart", "twigStiffness", "twigThorns", "twigTipWidth", "twigTwist", "twigs", "twigsInCluster", "twist", "twistRand", "twistRate", "uniformForce", "useFrameExtension", "valRand", "widthBias", "widthLengthMap", "widthRand", "widthScale", "wiggle", "wiggleFrequency", "wiggleOffset"]
	pass

