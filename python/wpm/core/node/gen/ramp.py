

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Texture2d = retriever.getNodeCls("Texture2d")
assert Texture2d
if T.TYPE_CHECKING:
	from .. import Texture2d

# add node doc



# region plug type defs
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Ramp = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Ramp = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Ramp = None
	pass
class ColorPlug(Plug):
	parent : ColorEntryListPlug = PlugDescriptor("colorEntryList")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	ecb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	ecg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	ecr_ : ColorRPlug = PlugDescriptor("colorR")
	node : Ramp = None
	pass
class PositionPlug(Plug):
	parent : ColorEntryListPlug = PlugDescriptor("colorEntryList")
	node : Ramp = None
	pass
class ColorEntryListPlug(Plug):
	color_ : ColorPlug = PlugDescriptor("color")
	ec_ : ColorPlug = PlugDescriptor("color")
	position_ : PositionPlug = PlugDescriptor("position")
	ep_ : PositionPlug = PlugDescriptor("position")
	node : Ramp = None
	pass
class HueNoisePlug(Plug):
	node : Ramp = None
	pass
class HueNoiseFreqPlug(Plug):
	node : Ramp = None
	pass
class InterpolationPlug(Plug):
	node : Ramp = None
	pass
class NoisePlug(Plug):
	node : Ramp = None
	pass
class NoiseFreqPlug(Plug):
	node : Ramp = None
	pass
class SatNoisePlug(Plug):
	node : Ramp = None
	pass
class SatNoiseFreqPlug(Plug):
	node : Ramp = None
	pass
class TypePlug(Plug):
	node : Ramp = None
	pass
class UWavePlug(Plug):
	node : Ramp = None
	pass
class VWavePlug(Plug):
	node : Ramp = None
	pass
class ValNoisePlug(Plug):
	node : Ramp = None
	pass
class ValNoiseFreqPlug(Plug):
	node : Ramp = None
	pass
# endregion


# define node class
class Ramp(Texture2d):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	position_ : PositionPlug = PlugDescriptor("position")
	colorEntryList_ : ColorEntryListPlug = PlugDescriptor("colorEntryList")
	hueNoise_ : HueNoisePlug = PlugDescriptor("hueNoise")
	hueNoiseFreq_ : HueNoiseFreqPlug = PlugDescriptor("hueNoiseFreq")
	interpolation_ : InterpolationPlug = PlugDescriptor("interpolation")
	noise_ : NoisePlug = PlugDescriptor("noise")
	noiseFreq_ : NoiseFreqPlug = PlugDescriptor("noiseFreq")
	satNoise_ : SatNoisePlug = PlugDescriptor("satNoise")
	satNoiseFreq_ : SatNoiseFreqPlug = PlugDescriptor("satNoiseFreq")
	type_ : TypePlug = PlugDescriptor("type")
	uWave_ : UWavePlug = PlugDescriptor("uWave")
	vWave_ : VWavePlug = PlugDescriptor("vWave")
	valNoise_ : ValNoisePlug = PlugDescriptor("valNoise")
	valNoiseFreq_ : ValNoiseFreqPlug = PlugDescriptor("valNoiseFreq")

	# node attributes

	typeName = "ramp"
	apiTypeInt = 504
	apiTypeStr = "kRamp"
	typeIdInt = 1381257793
	MFnCls = om.MFnDependencyNode
	pass

