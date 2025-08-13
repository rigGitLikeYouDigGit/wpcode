

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
class AmplitudePlug(Plug):
	node : Noise = None
	pass
class DensityPlug(Plug):
	node : Noise = None
	pass
class DepthMaxPlug(Plug):
	node : Noise = None
	pass
class FalloffPlug(Plug):
	node : Noise = None
	pass
class FrequencyPlug(Plug):
	node : Noise = None
	pass
class FrequencyRatioPlug(Plug):
	node : Noise = None
	pass
class ImplodePlug(Plug):
	node : Noise = None
	pass
class ImplodeCenterUPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : Noise = None
	pass
class ImplodeCenterVPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : Noise = None
	pass
class ImplodeCenterPlug(Plug):
	implodeCenterU_ : ImplodeCenterUPlug = PlugDescriptor("implodeCenterU")
	imu_ : ImplodeCenterUPlug = PlugDescriptor("implodeCenterU")
	implodeCenterV_ : ImplodeCenterVPlug = PlugDescriptor("implodeCenterV")
	imv_ : ImplodeCenterVPlug = PlugDescriptor("implodeCenterV")
	node : Noise = None
	pass
class InflectionPlug(Plug):
	node : Noise = None
	pass
class NoiseTypePlug(Plug):
	node : Noise = None
	pass
class NumWavesPlug(Plug):
	node : Noise = None
	pass
class RandomnessPlug(Plug):
	node : Noise = None
	pass
class RatioPlug(Plug):
	node : Noise = None
	pass
class SizeRandPlug(Plug):
	node : Noise = None
	pass
class SpottynessPlug(Plug):
	node : Noise = None
	pass
class ThresholdPlug(Plug):
	node : Noise = None
	pass
class TimePlug(Plug):
	node : Noise = None
	pass
# endregion


# define node class
class Noise(Texture2d):
	amplitude_ : AmplitudePlug = PlugDescriptor("amplitude")
	density_ : DensityPlug = PlugDescriptor("density")
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	falloff_ : FalloffPlug = PlugDescriptor("falloff")
	frequency_ : FrequencyPlug = PlugDescriptor("frequency")
	frequencyRatio_ : FrequencyRatioPlug = PlugDescriptor("frequencyRatio")
	implode_ : ImplodePlug = PlugDescriptor("implode")
	implodeCenterU_ : ImplodeCenterUPlug = PlugDescriptor("implodeCenterU")
	implodeCenterV_ : ImplodeCenterVPlug = PlugDescriptor("implodeCenterV")
	implodeCenter_ : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	inflection_ : InflectionPlug = PlugDescriptor("inflection")
	noiseType_ : NoiseTypePlug = PlugDescriptor("noiseType")
	numWaves_ : NumWavesPlug = PlugDescriptor("numWaves")
	randomness_ : RandomnessPlug = PlugDescriptor("randomness")
	ratio_ : RatioPlug = PlugDescriptor("ratio")
	sizeRand_ : SizeRandPlug = PlugDescriptor("sizeRand")
	spottyness_ : SpottynessPlug = PlugDescriptor("spottyness")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")
	time_ : TimePlug = PlugDescriptor("time")

	# node attributes

	typeName = "noise"
	apiTypeInt = 879
	apiTypeStr = "kNoise"
	typeIdInt = 1381256755
	MFnCls = om.MFnDependencyNode
	pass

