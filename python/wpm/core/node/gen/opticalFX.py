

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class ActivePlug(Plug):
	node : OpticalFX = None
	pass
class BinMembershipPlug(Plug):
	node : OpticalFX = None
	pass
class FlareColSpreadPlug(Plug):
	node : OpticalFX = None
	pass
class FlareColorBPlug(Plug):
	parent : FlareColorPlug = PlugDescriptor("flareColor")
	node : OpticalFX = None
	pass
class FlareColorGPlug(Plug):
	parent : FlareColorPlug = PlugDescriptor("flareColor")
	node : OpticalFX = None
	pass
class FlareColorRPlug(Plug):
	parent : FlareColorPlug = PlugDescriptor("flareColor")
	node : OpticalFX = None
	pass
class FlareColorPlug(Plug):
	flareColorB_ : FlareColorBPlug = PlugDescriptor("flareColorB")
	rb_ : FlareColorBPlug = PlugDescriptor("flareColorB")
	flareColorG_ : FlareColorGPlug = PlugDescriptor("flareColorG")
	rg_ : FlareColorGPlug = PlugDescriptor("flareColorG")
	flareColorR_ : FlareColorRPlug = PlugDescriptor("flareColorR")
	rr_ : FlareColorRPlug = PlugDescriptor("flareColorR")
	node : OpticalFX = None
	pass
class FlareFocusPlug(Plug):
	node : OpticalFX = None
	pass
class FlareHorizontalPlug(Plug):
	node : OpticalFX = None
	pass
class FlareIntensityPlug(Plug):
	node : OpticalFX = None
	pass
class FlareLengthPlug(Plug):
	node : OpticalFX = None
	pass
class FlareMaxSizePlug(Plug):
	node : OpticalFX = None
	pass
class FlareMinSizePlug(Plug):
	node : OpticalFX = None
	pass
class FlareNumCirclesPlug(Plug):
	node : OpticalFX = None
	pass
class FlareVerticalPlug(Plug):
	node : OpticalFX = None
	pass
class FogColorBPlug(Plug):
	parent : FogColorPlug = PlugDescriptor("fogColor")
	node : OpticalFX = None
	pass
class FogColorGPlug(Plug):
	parent : FogColorPlug = PlugDescriptor("fogColor")
	node : OpticalFX = None
	pass
class FogColorRPlug(Plug):
	parent : FogColorPlug = PlugDescriptor("fogColor")
	node : OpticalFX = None
	pass
class FogColorPlug(Plug):
	fogColorB_ : FogColorBPlug = PlugDescriptor("fogColorB")
	fb_ : FogColorBPlug = PlugDescriptor("fogColorB")
	fogColorG_ : FogColorGPlug = PlugDescriptor("fogColorG")
	fg_ : FogColorGPlug = PlugDescriptor("fogColorG")
	fogColorR_ : FogColorRPlug = PlugDescriptor("fogColorR")
	fr_ : FogColorRPlug = PlugDescriptor("fogColorR")
	node : OpticalFX = None
	pass
class FogIntensityPlug(Plug):
	node : OpticalFX = None
	pass
class FogNoisePlug(Plug):
	node : OpticalFX = None
	pass
class FogOpacityPlug(Plug):
	node : OpticalFX = None
	pass
class FogRadialNoisePlug(Plug):
	node : OpticalFX = None
	pass
class FogSpreadPlug(Plug):
	node : OpticalFX = None
	pass
class FogStarlevelPlug(Plug):
	node : OpticalFX = None
	pass
class FogTypePlug(Plug):
	node : OpticalFX = None
	pass
class GlowColorBPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : OpticalFX = None
	pass
class GlowColorGPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : OpticalFX = None
	pass
class GlowColorRPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : OpticalFX = None
	pass
class GlowColorPlug(Plug):
	glowColorB_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	gb_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	glowColorG_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	gg_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	glowColorR_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	gr_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	node : OpticalFX = None
	pass
class GlowIntensityPlug(Plug):
	node : OpticalFX = None
	pass
class GlowNoisePlug(Plug):
	node : OpticalFX = None
	pass
class GlowOpacityPlug(Plug):
	node : OpticalFX = None
	pass
class GlowRadialNoisePlug(Plug):
	node : OpticalFX = None
	pass
class GlowSpreadPlug(Plug):
	node : OpticalFX = None
	pass
class GlowStarLevelPlug(Plug):
	node : OpticalFX = None
	pass
class GlowTypePlug(Plug):
	node : OpticalFX = None
	pass
class GlowVisibilityPlug(Plug):
	node : OpticalFX = None
	pass
class HaloColorBPlug(Plug):
	parent : HaloColorPlug = PlugDescriptor("haloColor")
	node : OpticalFX = None
	pass
class HaloColorGPlug(Plug):
	parent : HaloColorPlug = PlugDescriptor("haloColor")
	node : OpticalFX = None
	pass
class HaloColorRPlug(Plug):
	parent : HaloColorPlug = PlugDescriptor("haloColor")
	node : OpticalFX = None
	pass
class HaloColorPlug(Plug):
	haloColorB_ : HaloColorBPlug = PlugDescriptor("haloColorB")
	hb_ : HaloColorBPlug = PlugDescriptor("haloColorB")
	haloColorG_ : HaloColorGPlug = PlugDescriptor("haloColorG")
	hg_ : HaloColorGPlug = PlugDescriptor("haloColorG")
	haloColorR_ : HaloColorRPlug = PlugDescriptor("haloColorR")
	hr_ : HaloColorRPlug = PlugDescriptor("haloColorR")
	node : OpticalFX = None
	pass
class HaloIntensityPlug(Plug):
	node : OpticalFX = None
	pass
class HaloSpreadPlug(Plug):
	node : OpticalFX = None
	pass
class HaloTypePlug(Plug):
	node : OpticalFX = None
	pass
class HexagonFlarePlug(Plug):
	node : OpticalFX = None
	pass
class IgnoreLightPlug(Plug):
	node : OpticalFX = None
	pass
class LensFlarePlug(Plug):
	node : OpticalFX = None
	pass
class LightColorBPlug(Plug):
	parent : LightColorPlug = PlugDescriptor("lightColor")
	node : OpticalFX = None
	pass
class LightColorGPlug(Plug):
	parent : LightColorPlug = PlugDescriptor("lightColor")
	node : OpticalFX = None
	pass
class LightColorRPlug(Plug):
	parent : LightColorPlug = PlugDescriptor("lightColor")
	node : OpticalFX = None
	pass
class LightColorPlug(Plug):
	lightColorB_ : LightColorBPlug = PlugDescriptor("lightColorB")
	lgb_ : LightColorBPlug = PlugDescriptor("lightColorB")
	lightColorG_ : LightColorGPlug = PlugDescriptor("lightColorG")
	lcg_ : LightColorGPlug = PlugDescriptor("lightColorG")
	lightColorR_ : LightColorRPlug = PlugDescriptor("lightColorR")
	lcr_ : LightColorRPlug = PlugDescriptor("lightColorR")
	node : OpticalFX = None
	pass
class LightConnectionPlug(Plug):
	node : OpticalFX = None
	pass
class LightWorldMatPlug(Plug):
	node : OpticalFX = None
	pass
class NoiseThresholdPlug(Plug):
	node : OpticalFX = None
	pass
class NoiseUoffsetPlug(Plug):
	node : OpticalFX = None
	pass
class NoiseUscalePlug(Plug):
	node : OpticalFX = None
	pass
class NoiseVoffsetPlug(Plug):
	node : OpticalFX = None
	pass
class NoiseVscalePlug(Plug):
	node : OpticalFX = None
	pass
class RadialFrequencyPlug(Plug):
	node : OpticalFX = None
	pass
class RotationPlug(Plug):
	node : OpticalFX = None
	pass
class StarPointsPlug(Plug):
	node : OpticalFX = None
	pass
class VisibilityBPlug(Plug):
	parent : VisibilityPlug = PlugDescriptor("visibility")
	node : OpticalFX = None
	pass
class VisibilityGPlug(Plug):
	parent : VisibilityPlug = PlugDescriptor("visibility")
	node : OpticalFX = None
	pass
class VisibilityRPlug(Plug):
	parent : VisibilityPlug = PlugDescriptor("visibility")
	node : OpticalFX = None
	pass
class VisibilityPlug(Plug):
	visibilityB_ : VisibilityBPlug = PlugDescriptor("visibilityB")
	vbb_ : VisibilityBPlug = PlugDescriptor("visibilityB")
	visibilityG_ : VisibilityGPlug = PlugDescriptor("visibilityG")
	vbg_ : VisibilityGPlug = PlugDescriptor("visibilityG")
	visibilityR_ : VisibilityRPlug = PlugDescriptor("visibilityR")
	vbr_ : VisibilityRPlug = PlugDescriptor("visibilityR")
	node : OpticalFX = None
	pass
# endregion


# define node class
class OpticalFX(_BASE_):
	active_ : ActivePlug = PlugDescriptor("active")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	flareColSpread_ : FlareColSpreadPlug = PlugDescriptor("flareColSpread")
	flareColorB_ : FlareColorBPlug = PlugDescriptor("flareColorB")
	flareColorG_ : FlareColorGPlug = PlugDescriptor("flareColorG")
	flareColorR_ : FlareColorRPlug = PlugDescriptor("flareColorR")
	flareColor_ : FlareColorPlug = PlugDescriptor("flareColor")
	flareFocus_ : FlareFocusPlug = PlugDescriptor("flareFocus")
	flareHorizontal_ : FlareHorizontalPlug = PlugDescriptor("flareHorizontal")
	flareIntensity_ : FlareIntensityPlug = PlugDescriptor("flareIntensity")
	flareLength_ : FlareLengthPlug = PlugDescriptor("flareLength")
	flareMaxSize_ : FlareMaxSizePlug = PlugDescriptor("flareMaxSize")
	flareMinSize_ : FlareMinSizePlug = PlugDescriptor("flareMinSize")
	flareNumCircles_ : FlareNumCirclesPlug = PlugDescriptor("flareNumCircles")
	flareVertical_ : FlareVerticalPlug = PlugDescriptor("flareVertical")
	fogColorB_ : FogColorBPlug = PlugDescriptor("fogColorB")
	fogColorG_ : FogColorGPlug = PlugDescriptor("fogColorG")
	fogColorR_ : FogColorRPlug = PlugDescriptor("fogColorR")
	fogColor_ : FogColorPlug = PlugDescriptor("fogColor")
	fogIntensity_ : FogIntensityPlug = PlugDescriptor("fogIntensity")
	fogNoise_ : FogNoisePlug = PlugDescriptor("fogNoise")
	fogOpacity_ : FogOpacityPlug = PlugDescriptor("fogOpacity")
	fogRadialNoise_ : FogRadialNoisePlug = PlugDescriptor("fogRadialNoise")
	fogSpread_ : FogSpreadPlug = PlugDescriptor("fogSpread")
	fogStarlevel_ : FogStarlevelPlug = PlugDescriptor("fogStarlevel")
	fogType_ : FogTypePlug = PlugDescriptor("fogType")
	glowColorB_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	glowColorG_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	glowColorR_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	glowColor_ : GlowColorPlug = PlugDescriptor("glowColor")
	glowIntensity_ : GlowIntensityPlug = PlugDescriptor("glowIntensity")
	glowNoise_ : GlowNoisePlug = PlugDescriptor("glowNoise")
	glowOpacity_ : GlowOpacityPlug = PlugDescriptor("glowOpacity")
	glowRadialNoise_ : GlowRadialNoisePlug = PlugDescriptor("glowRadialNoise")
	glowSpread_ : GlowSpreadPlug = PlugDescriptor("glowSpread")
	glowStarLevel_ : GlowStarLevelPlug = PlugDescriptor("glowStarLevel")
	glowType_ : GlowTypePlug = PlugDescriptor("glowType")
	glowVisibility_ : GlowVisibilityPlug = PlugDescriptor("glowVisibility")
	haloColorB_ : HaloColorBPlug = PlugDescriptor("haloColorB")
	haloColorG_ : HaloColorGPlug = PlugDescriptor("haloColorG")
	haloColorR_ : HaloColorRPlug = PlugDescriptor("haloColorR")
	haloColor_ : HaloColorPlug = PlugDescriptor("haloColor")
	haloIntensity_ : HaloIntensityPlug = PlugDescriptor("haloIntensity")
	haloSpread_ : HaloSpreadPlug = PlugDescriptor("haloSpread")
	haloType_ : HaloTypePlug = PlugDescriptor("haloType")
	hexagonFlare_ : HexagonFlarePlug = PlugDescriptor("hexagonFlare")
	ignoreLight_ : IgnoreLightPlug = PlugDescriptor("ignoreLight")
	lensFlare_ : LensFlarePlug = PlugDescriptor("lensFlare")
	lightColorB_ : LightColorBPlug = PlugDescriptor("lightColorB")
	lightColorG_ : LightColorGPlug = PlugDescriptor("lightColorG")
	lightColorR_ : LightColorRPlug = PlugDescriptor("lightColorR")
	lightColor_ : LightColorPlug = PlugDescriptor("lightColor")
	lightConnection_ : LightConnectionPlug = PlugDescriptor("lightConnection")
	lightWorldMat_ : LightWorldMatPlug = PlugDescriptor("lightWorldMat")
	noiseThreshold_ : NoiseThresholdPlug = PlugDescriptor("noiseThreshold")
	noiseUoffset_ : NoiseUoffsetPlug = PlugDescriptor("noiseUoffset")
	noiseUscale_ : NoiseUscalePlug = PlugDescriptor("noiseUscale")
	noiseVoffset_ : NoiseVoffsetPlug = PlugDescriptor("noiseVoffset")
	noiseVscale_ : NoiseVscalePlug = PlugDescriptor("noiseVscale")
	radialFrequency_ : RadialFrequencyPlug = PlugDescriptor("radialFrequency")
	rotation_ : RotationPlug = PlugDescriptor("rotation")
	starPoints_ : StarPointsPlug = PlugDescriptor("starPoints")
	visibilityB_ : VisibilityBPlug = PlugDescriptor("visibilityB")
	visibilityG_ : VisibilityGPlug = PlugDescriptor("visibilityG")
	visibilityR_ : VisibilityRPlug = PlugDescriptor("visibilityR")
	visibility_ : VisibilityPlug = PlugDescriptor("visibility")

	# node attributes

	typeName = "opticalFX"
	apiTypeInt = 450
	apiTypeStr = "kOpticalFX"
	typeIdInt = 1330660952
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["active", "binMembership", "flareColSpread", "flareColorB", "flareColorG", "flareColorR", "flareColor", "flareFocus", "flareHorizontal", "flareIntensity", "flareLength", "flareMaxSize", "flareMinSize", "flareNumCircles", "flareVertical", "fogColorB", "fogColorG", "fogColorR", "fogColor", "fogIntensity", "fogNoise", "fogOpacity", "fogRadialNoise", "fogSpread", "fogStarlevel", "fogType", "glowColorB", "glowColorG", "glowColorR", "glowColor", "glowIntensity", "glowNoise", "glowOpacity", "glowRadialNoise", "glowSpread", "glowStarLevel", "glowType", "glowVisibility", "haloColorB", "haloColorG", "haloColorR", "haloColor", "haloIntensity", "haloSpread", "haloType", "hexagonFlare", "ignoreLight", "lensFlare", "lightColorB", "lightColorG", "lightColorR", "lightColor", "lightConnection", "lightWorldMat", "noiseThreshold", "noiseUoffset", "noiseUscale", "noiseVoffset", "noiseVscale", "radialFrequency", "rotation", "starPoints", "visibilityB", "visibilityG", "visibilityR", "visibility"]
	nodeLeafPlugs = ["active", "binMembership", "flareColSpread", "flareColor", "flareFocus", "flareHorizontal", "flareIntensity", "flareLength", "flareMaxSize", "flareMinSize", "flareNumCircles", "flareVertical", "fogColor", "fogIntensity", "fogNoise", "fogOpacity", "fogRadialNoise", "fogSpread", "fogStarlevel", "fogType", "glowColor", "glowIntensity", "glowNoise", "glowOpacity", "glowRadialNoise", "glowSpread", "glowStarLevel", "glowType", "glowVisibility", "haloColor", "haloIntensity", "haloSpread", "haloType", "hexagonFlare", "ignoreLight", "lensFlare", "lightColor", "lightConnection", "lightWorldMat", "noiseThreshold", "noiseUoffset", "noiseUscale", "noiseVoffset", "noiseVscale", "radialFrequency", "rotation", "starPoints", "visibility"]
	pass

