

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ShadingDependNode = Catalogue.ShadingDependNode
else:
	from .. import retriever
	ShadingDependNode = retriever.getNodeCls("ShadingDependNode")
	assert ShadingDependNode

# add node doc



# region plug type defs
class AlphaIsLuminancePlug(Plug):
	node : LayeredTexture = None
	pass
class HardwareColorBPlug(Plug):
	parent : HardwareColorPlug = PlugDescriptor("hardwareColor")
	node : LayeredTexture = None
	pass
class HardwareColorGPlug(Plug):
	parent : HardwareColorPlug = PlugDescriptor("hardwareColor")
	node : LayeredTexture = None
	pass
class HardwareColorRPlug(Plug):
	parent : HardwareColorPlug = PlugDescriptor("hardwareColor")
	node : LayeredTexture = None
	pass
class HardwareColorPlug(Plug):
	hardwareColorB_ : HardwareColorBPlug = PlugDescriptor("hardwareColorB")
	hcb_ : HardwareColorBPlug = PlugDescriptor("hardwareColorB")
	hardwareColorG_ : HardwareColorGPlug = PlugDescriptor("hardwareColorG")
	hcg_ : HardwareColorGPlug = PlugDescriptor("hardwareColorG")
	hardwareColorR_ : HardwareColorRPlug = PlugDescriptor("hardwareColorR")
	hcr_ : HardwareColorRPlug = PlugDescriptor("hardwareColorR")
	node : LayeredTexture = None
	pass
class AlphaPlug(Plug):
	parent : InputsPlug = PlugDescriptor("inputs")
	node : LayeredTexture = None
	pass
class BlendModePlug(Plug):
	parent : InputsPlug = PlugDescriptor("inputs")
	node : LayeredTexture = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : LayeredTexture = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : LayeredTexture = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : LayeredTexture = None
	pass
class ColorPlug(Plug):
	parent : InputsPlug = PlugDescriptor("inputs")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : LayeredTexture = None
	pass
class IsVisiblePlug(Plug):
	parent : InputsPlug = PlugDescriptor("inputs")
	node : LayeredTexture = None
	pass
class InputsPlug(Plug):
	alpha_ : AlphaPlug = PlugDescriptor("alpha")
	a_ : AlphaPlug = PlugDescriptor("alpha")
	blendMode_ : BlendModePlug = PlugDescriptor("blendMode")
	bm_ : BlendModePlug = PlugDescriptor("blendMode")
	color_ : ColorPlug = PlugDescriptor("color")
	c_ : ColorPlug = PlugDescriptor("color")
	isVisible_ : IsVisiblePlug = PlugDescriptor("isVisible")
	iv_ : IsVisiblePlug = PlugDescriptor("isVisible")
	node : LayeredTexture = None
	pass
class OutAlphaPlug(Plug):
	node : LayeredTexture = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : LayeredTexture = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : LayeredTexture = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : LayeredTexture = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : LayeredTexture = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : LayeredTexture = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : LayeredTexture = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : LayeredTexture = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : LayeredTexture = None
	pass
# endregion


# define node class
class LayeredTexture(ShadingDependNode):
	alphaIsLuminance_ : AlphaIsLuminancePlug = PlugDescriptor("alphaIsLuminance")
	hardwareColorB_ : HardwareColorBPlug = PlugDescriptor("hardwareColorB")
	hardwareColorG_ : HardwareColorGPlug = PlugDescriptor("hardwareColorG")
	hardwareColorR_ : HardwareColorRPlug = PlugDescriptor("hardwareColorR")
	hardwareColor_ : HardwareColorPlug = PlugDescriptor("hardwareColor")
	alpha_ : AlphaPlug = PlugDescriptor("alpha")
	blendMode_ : BlendModePlug = PlugDescriptor("blendMode")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	isVisible_ : IsVisiblePlug = PlugDescriptor("isVisible")
	inputs_ : InputsPlug = PlugDescriptor("inputs")
	outAlpha_ : OutAlphaPlug = PlugDescriptor("outAlpha")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	outTransparency_ : OutTransparencyPlug = PlugDescriptor("outTransparency")

	# node attributes

	typeName = "layeredTexture"
	apiTypeInt = 804
	apiTypeStr = "kLayeredTexture"
	typeIdInt = 1280922196
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["alphaIsLuminance", "hardwareColorB", "hardwareColorG", "hardwareColorR", "hardwareColor", "alpha", "blendMode", "colorB", "colorG", "colorR", "color", "isVisible", "inputs", "outAlpha", "outColorB", "outColorG", "outColorR", "outColor", "outTransparencyB", "outTransparencyG", "outTransparencyR", "outTransparency"]
	nodeLeafPlugs = ["alphaIsLuminance", "hardwareColor", "inputs", "outAlpha", "outColor", "outTransparency"]
	pass

