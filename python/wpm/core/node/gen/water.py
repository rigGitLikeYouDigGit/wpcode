

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
class BoxMaxUPlug(Plug):
	parent : BoxMaxPlug = PlugDescriptor("boxMax")
	node : Water = None
	pass
class BoxMaxVPlug(Plug):
	parent : BoxMaxPlug = PlugDescriptor("boxMax")
	node : Water = None
	pass
class BoxMaxPlug(Plug):
	boxMaxU_ : BoxMaxUPlug = PlugDescriptor("boxMaxU")
	bu2_ : BoxMaxUPlug = PlugDescriptor("boxMaxU")
	boxMaxV_ : BoxMaxVPlug = PlugDescriptor("boxMaxV")
	bv2_ : BoxMaxVPlug = PlugDescriptor("boxMaxV")
	node : Water = None
	pass
class BoxMinUPlug(Plug):
	parent : BoxMinPlug = PlugDescriptor("boxMin")
	node : Water = None
	pass
class BoxMinVPlug(Plug):
	parent : BoxMinPlug = PlugDescriptor("boxMin")
	node : Water = None
	pass
class BoxMinPlug(Plug):
	boxMinU_ : BoxMinUPlug = PlugDescriptor("boxMinU")
	bu1_ : BoxMinUPlug = PlugDescriptor("boxMinU")
	boxMinV_ : BoxMinVPlug = PlugDescriptor("boxMinV")
	bv1_ : BoxMinVPlug = PlugDescriptor("boxMinV")
	node : Water = None
	pass
class DropSizePlug(Plug):
	node : Water = None
	pass
class FastPlug(Plug):
	node : Water = None
	pass
class GroupVelocityPlug(Plug):
	node : Water = None
	pass
class NumberOfWavesPlug(Plug):
	node : Water = None
	pass
class PhaseVelocityPlug(Plug):
	node : Water = None
	pass
class ReflectionBoxPlug(Plug):
	node : Water = None
	pass
class RippleAmplitudePlug(Plug):
	node : Water = None
	pass
class RippleFrequencyPlug(Plug):
	node : Water = None
	pass
class RippleOriginUPlug(Plug):
	parent : RippleOriginPlug = PlugDescriptor("rippleOrigin")
	node : Water = None
	pass
class RippleOriginVPlug(Plug):
	parent : RippleOriginPlug = PlugDescriptor("rippleOrigin")
	node : Water = None
	pass
class RippleOriginPlug(Plug):
	rippleOriginU_ : RippleOriginUPlug = PlugDescriptor("rippleOriginU")
	rcu_ : RippleOriginUPlug = PlugDescriptor("rippleOriginU")
	rippleOriginV_ : RippleOriginVPlug = PlugDescriptor("rippleOriginV")
	rcv_ : RippleOriginVPlug = PlugDescriptor("rippleOriginV")
	node : Water = None
	pass
class RippleTimePlug(Plug):
	node : Water = None
	pass
class SmoothnessPlug(Plug):
	node : Water = None
	pass
class SpreadRatePlug(Plug):
	node : Water = None
	pass
class SpreadStartPlug(Plug):
	node : Water = None
	pass
class SubWaveFrequencyPlug(Plug):
	node : Water = None
	pass
class WaveAmplitudePlug(Plug):
	node : Water = None
	pass
class WaveFrequencyPlug(Plug):
	node : Water = None
	pass
class WaveTimePlug(Plug):
	node : Water = None
	pass
class WaveVelocityPlug(Plug):
	node : Water = None
	pass
class WindUPlug(Plug):
	parent : WindUVPlug = PlugDescriptor("windUV")
	node : Water = None
	pass
class WindVPlug(Plug):
	parent : WindUVPlug = PlugDescriptor("windUV")
	node : Water = None
	pass
class WindUVPlug(Plug):
	windU_ : WindUPlug = PlugDescriptor("windU")
	wiu_ : WindUPlug = PlugDescriptor("windU")
	windV_ : WindVPlug = PlugDescriptor("windV")
	wiv_ : WindVPlug = PlugDescriptor("windV")
	node : Water = None
	pass
# endregion


# define node class
class Water(Texture2d):
	boxMaxU_ : BoxMaxUPlug = PlugDescriptor("boxMaxU")
	boxMaxV_ : BoxMaxVPlug = PlugDescriptor("boxMaxV")
	boxMax_ : BoxMaxPlug = PlugDescriptor("boxMax")
	boxMinU_ : BoxMinUPlug = PlugDescriptor("boxMinU")
	boxMinV_ : BoxMinVPlug = PlugDescriptor("boxMinV")
	boxMin_ : BoxMinPlug = PlugDescriptor("boxMin")
	dropSize_ : DropSizePlug = PlugDescriptor("dropSize")
	fast_ : FastPlug = PlugDescriptor("fast")
	groupVelocity_ : GroupVelocityPlug = PlugDescriptor("groupVelocity")
	numberOfWaves_ : NumberOfWavesPlug = PlugDescriptor("numberOfWaves")
	phaseVelocity_ : PhaseVelocityPlug = PlugDescriptor("phaseVelocity")
	reflectionBox_ : ReflectionBoxPlug = PlugDescriptor("reflectionBox")
	rippleAmplitude_ : RippleAmplitudePlug = PlugDescriptor("rippleAmplitude")
	rippleFrequency_ : RippleFrequencyPlug = PlugDescriptor("rippleFrequency")
	rippleOriginU_ : RippleOriginUPlug = PlugDescriptor("rippleOriginU")
	rippleOriginV_ : RippleOriginVPlug = PlugDescriptor("rippleOriginV")
	rippleOrigin_ : RippleOriginPlug = PlugDescriptor("rippleOrigin")
	rippleTime_ : RippleTimePlug = PlugDescriptor("rippleTime")
	smoothness_ : SmoothnessPlug = PlugDescriptor("smoothness")
	spreadRate_ : SpreadRatePlug = PlugDescriptor("spreadRate")
	spreadStart_ : SpreadStartPlug = PlugDescriptor("spreadStart")
	subWaveFrequency_ : SubWaveFrequencyPlug = PlugDescriptor("subWaveFrequency")
	waveAmplitude_ : WaveAmplitudePlug = PlugDescriptor("waveAmplitude")
	waveFrequency_ : WaveFrequencyPlug = PlugDescriptor("waveFrequency")
	waveTime_ : WaveTimePlug = PlugDescriptor("waveTime")
	waveVelocity_ : WaveVelocityPlug = PlugDescriptor("waveVelocity")
	windU_ : WindUPlug = PlugDescriptor("windU")
	windV_ : WindVPlug = PlugDescriptor("windV")
	windUV_ : WindUVPlug = PlugDescriptor("windUV")

	# node attributes

	typeName = "water"
	apiTypeInt = 506
	apiTypeStr = "kWater"
	typeIdInt = 1381259073
	MFnCls = om.MFnDependencyNode
	pass

