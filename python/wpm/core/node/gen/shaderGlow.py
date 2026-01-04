

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
class AutoExposurePlug(Plug):
	node : ShaderGlow = None
	pass
class BinMembershipPlug(Plug):
	node : ShaderGlow = None
	pass
class GlowColorBPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : ShaderGlow = None
	pass
class GlowColorGPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : ShaderGlow = None
	pass
class GlowColorRPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : ShaderGlow = None
	pass
class GlowColorPlug(Plug):
	glowColorB_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	gb_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	glowColorG_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	gg_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	glowColorR_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	gr_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	node : ShaderGlow = None
	pass
class GlowEccentricityPlug(Plug):
	node : ShaderGlow = None
	pass
class GlowFilterWidthPlug(Plug):
	node : ShaderGlow = None
	pass
class GlowIntensityPlug(Plug):
	node : ShaderGlow = None
	pass
class GlowOpacityPlug(Plug):
	node : ShaderGlow = None
	pass
class GlowRadialNoisePlug(Plug):
	node : ShaderGlow = None
	pass
class GlowRingFrequencyPlug(Plug):
	node : ShaderGlow = None
	pass
class GlowRingIntensityPlug(Plug):
	node : ShaderGlow = None
	pass
class GlowSpreadPlug(Plug):
	node : ShaderGlow = None
	pass
class GlowStarLevelPlug(Plug):
	node : ShaderGlow = None
	pass
class GlowTypePlug(Plug):
	node : ShaderGlow = None
	pass
class HaloColorBPlug(Plug):
	parent : HaloColorPlug = PlugDescriptor("haloColor")
	node : ShaderGlow = None
	pass
class HaloColorGPlug(Plug):
	parent : HaloColorPlug = PlugDescriptor("haloColor")
	node : ShaderGlow = None
	pass
class HaloColorRPlug(Plug):
	parent : HaloColorPlug = PlugDescriptor("haloColor")
	node : ShaderGlow = None
	pass
class HaloColorPlug(Plug):
	haloColorB_ : HaloColorBPlug = PlugDescriptor("haloColorB")
	hb_ : HaloColorBPlug = PlugDescriptor("haloColorB")
	haloColorG_ : HaloColorGPlug = PlugDescriptor("haloColorG")
	hg_ : HaloColorGPlug = PlugDescriptor("haloColorG")
	haloColorR_ : HaloColorRPlug = PlugDescriptor("haloColorR")
	hr_ : HaloColorRPlug = PlugDescriptor("haloColorR")
	node : ShaderGlow = None
	pass
class HaloEccentricityPlug(Plug):
	node : ShaderGlow = None
	pass
class HaloFilterWidthPlug(Plug):
	node : ShaderGlow = None
	pass
class HaloIntensityPlug(Plug):
	node : ShaderGlow = None
	pass
class HaloOpacityPlug(Plug):
	node : ShaderGlow = None
	pass
class HaloRadialNoisePlug(Plug):
	node : ShaderGlow = None
	pass
class HaloRingFrequencyPlug(Plug):
	node : ShaderGlow = None
	pass
class HaloRingIntensityPlug(Plug):
	node : ShaderGlow = None
	pass
class HaloSpreadPlug(Plug):
	node : ShaderGlow = None
	pass
class HaloStarLevelPlug(Plug):
	node : ShaderGlow = None
	pass
class HaloTypePlug(Plug):
	node : ShaderGlow = None
	pass
class QualityPlug(Plug):
	node : ShaderGlow = None
	pass
class RadialFrequencyPlug(Plug):
	node : ShaderGlow = None
	pass
class RotationPlug(Plug):
	node : ShaderGlow = None
	pass
class StarPointsPlug(Plug):
	node : ShaderGlow = None
	pass
class ThresholdPlug(Plug):
	node : ShaderGlow = None
	pass
# endregion


# define node class
class ShaderGlow(_BASE_):
	autoExposure_ : AutoExposurePlug = PlugDescriptor("autoExposure")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	glowColorB_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	glowColorG_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	glowColorR_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	glowColor_ : GlowColorPlug = PlugDescriptor("glowColor")
	glowEccentricity_ : GlowEccentricityPlug = PlugDescriptor("glowEccentricity")
	glowFilterWidth_ : GlowFilterWidthPlug = PlugDescriptor("glowFilterWidth")
	glowIntensity_ : GlowIntensityPlug = PlugDescriptor("glowIntensity")
	glowOpacity_ : GlowOpacityPlug = PlugDescriptor("glowOpacity")
	glowRadialNoise_ : GlowRadialNoisePlug = PlugDescriptor("glowRadialNoise")
	glowRingFrequency_ : GlowRingFrequencyPlug = PlugDescriptor("glowRingFrequency")
	glowRingIntensity_ : GlowRingIntensityPlug = PlugDescriptor("glowRingIntensity")
	glowSpread_ : GlowSpreadPlug = PlugDescriptor("glowSpread")
	glowStarLevel_ : GlowStarLevelPlug = PlugDescriptor("glowStarLevel")
	glowType_ : GlowTypePlug = PlugDescriptor("glowType")
	haloColorB_ : HaloColorBPlug = PlugDescriptor("haloColorB")
	haloColorG_ : HaloColorGPlug = PlugDescriptor("haloColorG")
	haloColorR_ : HaloColorRPlug = PlugDescriptor("haloColorR")
	haloColor_ : HaloColorPlug = PlugDescriptor("haloColor")
	haloEccentricity_ : HaloEccentricityPlug = PlugDescriptor("haloEccentricity")
	haloFilterWidth_ : HaloFilterWidthPlug = PlugDescriptor("haloFilterWidth")
	haloIntensity_ : HaloIntensityPlug = PlugDescriptor("haloIntensity")
	haloOpacity_ : HaloOpacityPlug = PlugDescriptor("haloOpacity")
	haloRadialNoise_ : HaloRadialNoisePlug = PlugDescriptor("haloRadialNoise")
	haloRingFrequency_ : HaloRingFrequencyPlug = PlugDescriptor("haloRingFrequency")
	haloRingIntensity_ : HaloRingIntensityPlug = PlugDescriptor("haloRingIntensity")
	haloSpread_ : HaloSpreadPlug = PlugDescriptor("haloSpread")
	haloStarLevel_ : HaloStarLevelPlug = PlugDescriptor("haloStarLevel")
	haloType_ : HaloTypePlug = PlugDescriptor("haloType")
	quality_ : QualityPlug = PlugDescriptor("quality")
	radialFrequency_ : RadialFrequencyPlug = PlugDescriptor("radialFrequency")
	rotation_ : RotationPlug = PlugDescriptor("rotation")
	starPoints_ : StarPointsPlug = PlugDescriptor("starPoints")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")

	# node attributes

	typeName = "shaderGlow"
	apiTypeInt = 475
	apiTypeStr = "kShaderGlow"
	typeIdInt = 1397245772
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["autoExposure", "binMembership", "glowColorB", "glowColorG", "glowColorR", "glowColor", "glowEccentricity", "glowFilterWidth", "glowIntensity", "glowOpacity", "glowRadialNoise", "glowRingFrequency", "glowRingIntensity", "glowSpread", "glowStarLevel", "glowType", "haloColorB", "haloColorG", "haloColorR", "haloColor", "haloEccentricity", "haloFilterWidth", "haloIntensity", "haloOpacity", "haloRadialNoise", "haloRingFrequency", "haloRingIntensity", "haloSpread", "haloStarLevel", "haloType", "quality", "radialFrequency", "rotation", "starPoints", "threshold"]
	nodeLeafPlugs = ["autoExposure", "binMembership", "glowColor", "glowEccentricity", "glowFilterWidth", "glowIntensity", "glowOpacity", "glowRadialNoise", "glowRingFrequency", "glowRingIntensity", "glowSpread", "glowStarLevel", "glowType", "haloColor", "haloEccentricity", "haloFilterWidth", "haloIntensity", "haloOpacity", "haloRadialNoise", "haloRingFrequency", "haloRingIntensity", "haloSpread", "haloStarLevel", "haloType", "quality", "radialFrequency", "rotation", "starPoints", "threshold"]
	pass

