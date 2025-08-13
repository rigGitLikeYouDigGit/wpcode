

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ShadingDependNode = retriever.getNodeCls("ShadingDependNode")
assert ShadingDependNode
if T.TYPE_CHECKING:
	from .. import ShadingDependNode

# add node doc



# region plug type defs
class CompositingFlagPlug(Plug):
	node : LayeredShader = None
	pass
class HardwareColorBPlug(Plug):
	parent : HardwareColorPlug = PlugDescriptor("hardwareColor")
	node : LayeredShader = None
	pass
class HardwareColorGPlug(Plug):
	parent : HardwareColorPlug = PlugDescriptor("hardwareColor")
	node : LayeredShader = None
	pass
class HardwareColorRPlug(Plug):
	parent : HardwareColorPlug = PlugDescriptor("hardwareColor")
	node : LayeredShader = None
	pass
class HardwareColorPlug(Plug):
	hardwareColorB_ : HardwareColorBPlug = PlugDescriptor("hardwareColorB")
	hcb_ : HardwareColorBPlug = PlugDescriptor("hardwareColorB")
	hardwareColorG_ : HardwareColorGPlug = PlugDescriptor("hardwareColorG")
	hcg_ : HardwareColorGPlug = PlugDescriptor("hardwareColorG")
	hardwareColorR_ : HardwareColorRPlug = PlugDescriptor("hardwareColorR")
	hcr_ : HardwareColorRPlug = PlugDescriptor("hardwareColorR")
	node : LayeredShader = None
	pass
class HardwareShaderBPlug(Plug):
	parent : HardwareShaderPlug = PlugDescriptor("hardwareShader")
	node : LayeredShader = None
	pass
class HardwareShaderGPlug(Plug):
	parent : HardwareShaderPlug = PlugDescriptor("hardwareShader")
	node : LayeredShader = None
	pass
class HardwareShaderRPlug(Plug):
	parent : HardwareShaderPlug = PlugDescriptor("hardwareShader")
	node : LayeredShader = None
	pass
class HardwareShaderPlug(Plug):
	hardwareShaderB_ : HardwareShaderBPlug = PlugDescriptor("hardwareShaderB")
	hwb_ : HardwareShaderBPlug = PlugDescriptor("hardwareShaderB")
	hardwareShaderG_ : HardwareShaderGPlug = PlugDescriptor("hardwareShaderG")
	hwg_ : HardwareShaderGPlug = PlugDescriptor("hardwareShaderG")
	hardwareShaderR_ : HardwareShaderRPlug = PlugDescriptor("hardwareShaderR")
	hwr_ : HardwareShaderRPlug = PlugDescriptor("hardwareShaderR")
	node : LayeredShader = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : LayeredShader = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : LayeredShader = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : LayeredShader = None
	pass
class ColorPlug(Plug):
	parent : InputsPlug = PlugDescriptor("inputs")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : LayeredShader = None
	pass
class GlowColorBPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : LayeredShader = None
	pass
class GlowColorGPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : LayeredShader = None
	pass
class GlowColorRPlug(Plug):
	parent : GlowColorPlug = PlugDescriptor("glowColor")
	node : LayeredShader = None
	pass
class GlowColorPlug(Plug):
	parent : InputsPlug = PlugDescriptor("inputs")
	glowColorB_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	gb_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	glowColorG_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	gg_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	glowColorR_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	gr_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	node : LayeredShader = None
	pass
class TransparencyBPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : LayeredShader = None
	pass
class TransparencyGPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : LayeredShader = None
	pass
class TransparencyRPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : LayeredShader = None
	pass
class TransparencyPlug(Plug):
	parent : InputsPlug = PlugDescriptor("inputs")
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	tb_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	tg_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	tr_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	node : LayeredShader = None
	pass
class InputsPlug(Plug):
	color_ : ColorPlug = PlugDescriptor("color")
	c_ : ColorPlug = PlugDescriptor("color")
	glowColor_ : GlowColorPlug = PlugDescriptor("glowColor")
	g_ : GlowColorPlug = PlugDescriptor("glowColor")
	transparency_ : TransparencyPlug = PlugDescriptor("transparency")
	t_ : TransparencyPlug = PlugDescriptor("transparency")
	node : LayeredShader = None
	pass
class MatteOpacityPlug(Plug):
	node : LayeredShader = None
	pass
class MatteOpacityModePlug(Plug):
	node : LayeredShader = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : LayeredShader = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : LayeredShader = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : LayeredShader = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : LayeredShader = None
	pass
class OutGlowColorBPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : LayeredShader = None
	pass
class OutGlowColorGPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : LayeredShader = None
	pass
class OutGlowColorRPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : LayeredShader = None
	pass
class OutGlowColorPlug(Plug):
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	ogb_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	ogg_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	ogr_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	node : LayeredShader = None
	pass
class OutMatteOpacityBPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : LayeredShader = None
	pass
class OutMatteOpacityGPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : LayeredShader = None
	pass
class OutMatteOpacityRPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : LayeredShader = None
	pass
class OutMatteOpacityPlug(Plug):
	outMatteOpacityB_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	omob_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	outMatteOpacityG_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	omog_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	outMatteOpacityR_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	omor_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	node : LayeredShader = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : LayeredShader = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : LayeredShader = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : LayeredShader = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : LayeredShader = None
	pass
class RenderPassModePlug(Plug):
	node : LayeredShader = None
	pass
# endregion


# define node class
class LayeredShader(ShadingDependNode):
	compositingFlag_ : CompositingFlagPlug = PlugDescriptor("compositingFlag")
	hardwareColorB_ : HardwareColorBPlug = PlugDescriptor("hardwareColorB")
	hardwareColorG_ : HardwareColorGPlug = PlugDescriptor("hardwareColorG")
	hardwareColorR_ : HardwareColorRPlug = PlugDescriptor("hardwareColorR")
	hardwareColor_ : HardwareColorPlug = PlugDescriptor("hardwareColor")
	hardwareShaderB_ : HardwareShaderBPlug = PlugDescriptor("hardwareShaderB")
	hardwareShaderG_ : HardwareShaderGPlug = PlugDescriptor("hardwareShaderG")
	hardwareShaderR_ : HardwareShaderRPlug = PlugDescriptor("hardwareShaderR")
	hardwareShader_ : HardwareShaderPlug = PlugDescriptor("hardwareShader")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	glowColorB_ : GlowColorBPlug = PlugDescriptor("glowColorB")
	glowColorG_ : GlowColorGPlug = PlugDescriptor("glowColorG")
	glowColorR_ : GlowColorRPlug = PlugDescriptor("glowColorR")
	glowColor_ : GlowColorPlug = PlugDescriptor("glowColor")
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	transparency_ : TransparencyPlug = PlugDescriptor("transparency")
	inputs_ : InputsPlug = PlugDescriptor("inputs")
	matteOpacity_ : MatteOpacityPlug = PlugDescriptor("matteOpacity")
	matteOpacityMode_ : MatteOpacityModePlug = PlugDescriptor("matteOpacityMode")
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
	renderPassMode_ : RenderPassModePlug = PlugDescriptor("renderPassMode")

	# node attributes

	typeName = "layeredShader"
	apiTypeInt = 376
	apiTypeStr = "kLayeredShader"
	typeIdInt = 1280922195
	MFnCls = om.MFnDependencyNode
	pass

