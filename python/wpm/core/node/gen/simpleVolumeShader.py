

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
class BinMembershipPlug(Plug):
	node : SimpleVolumeShader = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : SimpleVolumeShader = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : SimpleVolumeShader = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : SimpleVolumeShader = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : SimpleVolumeShader = None
	pass
class FarPointWorldXPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : SimpleVolumeShader = None
	pass
class FarPointWorldYPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : SimpleVolumeShader = None
	pass
class FarPointWorldZPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : SimpleVolumeShader = None
	pass
class FarPointWorldPlug(Plug):
	farPointWorldX_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	fpx_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	farPointWorldY_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	fpy_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	farPointWorldZ_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	fpz_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	node : SimpleVolumeShader = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : SimpleVolumeShader = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : SimpleVolumeShader = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : SimpleVolumeShader = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : SimpleVolumeShader = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : SimpleVolumeShader = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : SimpleVolumeShader = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : SimpleVolumeShader = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : SimpleVolumeShader = None
	pass
class Parameter1Plug(Plug):
	node : SimpleVolumeShader = None
	pass
class PointWorldXPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : SimpleVolumeShader = None
	pass
class PointWorldYPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : SimpleVolumeShader = None
	pass
class PointWorldZPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : SimpleVolumeShader = None
	pass
class PointWorldPlug(Plug):
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	px_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	py_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pz_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	node : SimpleVolumeShader = None
	pass
# endregion


# define node class
class SimpleVolumeShader(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	farPointWorldX_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	farPointWorldY_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	farPointWorldZ_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	farPointWorld_ : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	outTransparency_ : OutTransparencyPlug = PlugDescriptor("outTransparency")
	parameter1_ : Parameter1Plug = PlugDescriptor("parameter1")
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pointWorld_ : PointWorldPlug = PlugDescriptor("pointWorld")

	# node attributes

	typeName = "simpleVolumeShader"
	apiTypeInt = 480
	apiTypeStr = "kSimpleVolumeShader"
	typeIdInt = 1398166344
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "colorB", "colorG", "colorR", "color", "farPointWorldX", "farPointWorldY", "farPointWorldZ", "farPointWorld", "outColorB", "outColorG", "outColorR", "outColor", "outTransparencyB", "outTransparencyG", "outTransparencyR", "outTransparency", "parameter1", "pointWorldX", "pointWorldY", "pointWorldZ", "pointWorld"]
	nodeLeafPlugs = ["binMembership", "color", "farPointWorld", "outColor", "outTransparency", "parameter1", "pointWorld"]
	pass

