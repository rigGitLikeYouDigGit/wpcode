

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
	node : Fractal = None
	pass
class AnimatedPlug(Plug):
	node : Fractal = None
	pass
class BiasPlug(Plug):
	node : Fractal = None
	pass
class FrequencyRatioPlug(Plug):
	node : Fractal = None
	pass
class InflectionPlug(Plug):
	node : Fractal = None
	pass
class LevelMaxPlug(Plug):
	node : Fractal = None
	pass
class LevelMinPlug(Plug):
	node : Fractal = None
	pass
class RatioPlug(Plug):
	node : Fractal = None
	pass
class ThresholdPlug(Plug):
	node : Fractal = None
	pass
class TimePlug(Plug):
	node : Fractal = None
	pass
class TimeRatioPlug(Plug):
	node : Fractal = None
	pass
# endregion


# define node class
class Fractal(Texture2d):
	amplitude_ : AmplitudePlug = PlugDescriptor("amplitude")
	animated_ : AnimatedPlug = PlugDescriptor("animated")
	bias_ : BiasPlug = PlugDescriptor("bias")
	frequencyRatio_ : FrequencyRatioPlug = PlugDescriptor("frequencyRatio")
	inflection_ : InflectionPlug = PlugDescriptor("inflection")
	levelMax_ : LevelMaxPlug = PlugDescriptor("levelMax")
	levelMin_ : LevelMinPlug = PlugDescriptor("levelMin")
	ratio_ : RatioPlug = PlugDescriptor("ratio")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")
	time_ : TimePlug = PlugDescriptor("time")
	timeRatio_ : TimeRatioPlug = PlugDescriptor("timeRatio")

	# node attributes

	typeName = "fractal"
	apiTypeInt = 501
	apiTypeStr = "kFractal"
	typeIdInt = 1381249606
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["amplitude", "animated", "bias", "frequencyRatio", "inflection", "levelMax", "levelMin", "ratio", "threshold", "time", "timeRatio"]
	nodeLeafPlugs = ["amplitude", "animated", "bias", "frequencyRatio", "inflection", "levelMax", "levelMin", "ratio", "threshold", "time", "timeRatio"]
	pass

