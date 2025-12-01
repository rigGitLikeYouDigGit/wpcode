

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
class BinMembershipPlug(Plug):
	node : SurfaceShader = None
	pass
class MaterialAlphaGainPlug(Plug):
	node : SurfaceShader = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : SurfaceShader = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : SurfaceShader = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : SurfaceShader = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : SurfaceShader = None
	pass
class OutGlowColorBPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : SurfaceShader = None
	pass
class OutGlowColorGPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : SurfaceShader = None
	pass
class OutGlowColorRPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : SurfaceShader = None
	pass
class OutGlowColorPlug(Plug):
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	ogb_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	ogg_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	ogr_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	node : SurfaceShader = None
	pass
class OutMatteOpacityBPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : SurfaceShader = None
	pass
class OutMatteOpacityGPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : SurfaceShader = None
	pass
class OutMatteOpacityRPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : SurfaceShader = None
	pass
class OutMatteOpacityPlug(Plug):
	outMatteOpacityB_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	omob_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	outMatteOpacityG_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	omog_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	outMatteOpacityR_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	omor_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	node : SurfaceShader = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : SurfaceShader = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : SurfaceShader = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : SurfaceShader = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : SurfaceShader = None
	pass
# endregion


# define node class
class SurfaceShader(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	materialAlphaGain_ : MaterialAlphaGainPlug = PlugDescriptor("materialAlphaGain")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	outGlowColor_ : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	outMatteOpacityB_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	outMatteOpacityG_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	outMatteOpacityR_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	outMatteOpacity_ : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	outTransparency_ : OutTransparencyPlug = PlugDescriptor("outTransparency")

	# node attributes

	typeName = "surfaceShader"
	apiTypeInt = 488
	apiTypeStr = "kSurfaceShader"
	typeIdInt = 1381192520
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "materialAlphaGain", "outColorB", "outColorG", "outColorR", "outColor", "outGlowColorB", "outGlowColorG", "outGlowColorR", "outGlowColor", "outMatteOpacityB", "outMatteOpacityG", "outMatteOpacityR", "outMatteOpacity", "outTransparencyB", "outTransparencyG", "outTransparencyR", "outTransparency"]
	nodeLeafPlugs = ["binMembership", "materialAlphaGain", "outColor", "outGlowColor", "outMatteOpacity", "outTransparency"]
	pass

