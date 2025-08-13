

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
class BrightSpreadPlug(Plug):
	node : Cloth = None
	pass
class GapColorBPlug(Plug):
	parent : GapColorPlug = PlugDescriptor("gapColor")
	node : Cloth = None
	pass
class GapColorGPlug(Plug):
	parent : GapColorPlug = PlugDescriptor("gapColor")
	node : Cloth = None
	pass
class GapColorRPlug(Plug):
	parent : GapColorPlug = PlugDescriptor("gapColor")
	node : Cloth = None
	pass
class GapColorPlug(Plug):
	gapColorB_ : GapColorBPlug = PlugDescriptor("gapColorB")
	gcb_ : GapColorBPlug = PlugDescriptor("gapColorB")
	gapColorG_ : GapColorGPlug = PlugDescriptor("gapColorG")
	gcg_ : GapColorGPlug = PlugDescriptor("gapColorG")
	gapColorR_ : GapColorRPlug = PlugDescriptor("gapColorR")
	gcr_ : GapColorRPlug = PlugDescriptor("gapColorR")
	node : Cloth = None
	pass
class RandomnessPlug(Plug):
	node : Cloth = None
	pass
class UColorBPlug(Plug):
	parent : UColorPlug = PlugDescriptor("uColor")
	node : Cloth = None
	pass
class UColorGPlug(Plug):
	parent : UColorPlug = PlugDescriptor("uColor")
	node : Cloth = None
	pass
class UColorRPlug(Plug):
	parent : UColorPlug = PlugDescriptor("uColor")
	node : Cloth = None
	pass
class UColorPlug(Plug):
	uColorB_ : UColorBPlug = PlugDescriptor("uColorB")
	ucb_ : UColorBPlug = PlugDescriptor("uColorB")
	uColorG_ : UColorGPlug = PlugDescriptor("uColorG")
	ucg_ : UColorGPlug = PlugDescriptor("uColorG")
	uColorR_ : UColorRPlug = PlugDescriptor("uColorR")
	ucr_ : UColorRPlug = PlugDescriptor("uColorR")
	node : Cloth = None
	pass
class UWavePlug(Plug):
	node : Cloth = None
	pass
class UWidthPlug(Plug):
	node : Cloth = None
	pass
class VColorBPlug(Plug):
	parent : VColorPlug = PlugDescriptor("vColor")
	node : Cloth = None
	pass
class VColorGPlug(Plug):
	parent : VColorPlug = PlugDescriptor("vColor")
	node : Cloth = None
	pass
class VColorRPlug(Plug):
	parent : VColorPlug = PlugDescriptor("vColor")
	node : Cloth = None
	pass
class VColorPlug(Plug):
	vColorB_ : VColorBPlug = PlugDescriptor("vColorB")
	vcb_ : VColorBPlug = PlugDescriptor("vColorB")
	vColorG_ : VColorGPlug = PlugDescriptor("vColorG")
	vcg_ : VColorGPlug = PlugDescriptor("vColorG")
	vColorR_ : VColorRPlug = PlugDescriptor("vColorR")
	vcr_ : VColorRPlug = PlugDescriptor("vColorR")
	node : Cloth = None
	pass
class VWavePlug(Plug):
	node : Cloth = None
	pass
class VWidthPlug(Plug):
	node : Cloth = None
	pass
class WidthSpreadPlug(Plug):
	node : Cloth = None
	pass
# endregion


# define node class
class Cloth(Texture2d):
	brightSpread_ : BrightSpreadPlug = PlugDescriptor("brightSpread")
	gapColorB_ : GapColorBPlug = PlugDescriptor("gapColorB")
	gapColorG_ : GapColorGPlug = PlugDescriptor("gapColorG")
	gapColorR_ : GapColorRPlug = PlugDescriptor("gapColorR")
	gapColor_ : GapColorPlug = PlugDescriptor("gapColor")
	randomness_ : RandomnessPlug = PlugDescriptor("randomness")
	uColorB_ : UColorBPlug = PlugDescriptor("uColorB")
	uColorG_ : UColorGPlug = PlugDescriptor("uColorG")
	uColorR_ : UColorRPlug = PlugDescriptor("uColorR")
	uColor_ : UColorPlug = PlugDescriptor("uColor")
	uWave_ : UWavePlug = PlugDescriptor("uWave")
	uWidth_ : UWidthPlug = PlugDescriptor("uWidth")
	vColorB_ : VColorBPlug = PlugDescriptor("vColorB")
	vColorG_ : VColorGPlug = PlugDescriptor("vColorG")
	vColorR_ : VColorRPlug = PlugDescriptor("vColorR")
	vColor_ : VColorPlug = PlugDescriptor("vColor")
	vWave_ : VWavePlug = PlugDescriptor("vWave")
	vWidth_ : VWidthPlug = PlugDescriptor("vWidth")
	widthSpread_ : WidthSpreadPlug = PlugDescriptor("widthSpread")

	# node attributes

	typeName = "cloth"
	apiTypeInt = 499
	apiTypeStr = "kCloth"
	typeIdInt = 1381253964
	MFnCls = om.MFnDependencyNode
	pass

