

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	FluidShape = Catalogue.FluidShape
else:
	from .. import retriever
	FluidShape = retriever.getNodeCls("FluidShape")
	assert FluidShape

# add node doc



# region plug type defs
class AlphaGainPlug(Plug):
	node : FluidTexture2D = None
	pass
class AlphaOffsetPlug(Plug):
	node : FluidTexture2D = None
	pass
class DefaultColorBPlug(Plug):
	parent : DefaultColorPlug = PlugDescriptor("defaultColor")
	node : FluidTexture2D = None
	pass
class DefaultColorGPlug(Plug):
	parent : DefaultColorPlug = PlugDescriptor("defaultColor")
	node : FluidTexture2D = None
	pass
class DefaultColorRPlug(Plug):
	parent : DefaultColorPlug = PlugDescriptor("defaultColor")
	node : FluidTexture2D = None
	pass
class DefaultColorPlug(Plug):
	defaultColorB_ : DefaultColorBPlug = PlugDescriptor("defaultColorB")
	dcb_ : DefaultColorBPlug = PlugDescriptor("defaultColorB")
	defaultColorG_ : DefaultColorGPlug = PlugDescriptor("defaultColorG")
	dcg_ : DefaultColorGPlug = PlugDescriptor("defaultColorG")
	defaultColorR_ : DefaultColorRPlug = PlugDescriptor("defaultColorR")
	dcr_ : DefaultColorRPlug = PlugDescriptor("defaultColorR")
	node : FluidTexture2D = None
	pass
class OutAlphaPlug(Plug):
	node : FluidTexture2D = None
	pass
class OutUPlug(Plug):
	parent : OutUVPlug = PlugDescriptor("outUV")
	node : FluidTexture2D = None
	pass
class OutVPlug(Plug):
	parent : OutUVPlug = PlugDescriptor("outUV")
	node : FluidTexture2D = None
	pass
class OutUVPlug(Plug):
	outU_ : OutUPlug = PlugDescriptor("outU")
	ou_ : OutUPlug = PlugDescriptor("outU")
	outV_ : OutVPlug = PlugDescriptor("outV")
	ov_ : OutVPlug = PlugDescriptor("outV")
	node : FluidTexture2D = None
	pass
class UCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : FluidTexture2D = None
	pass
class VCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : FluidTexture2D = None
	pass
class UvCoordPlug(Plug):
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	uvu_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	uvv_ : VCoordPlug = PlugDescriptor("vCoord")
	node : FluidTexture2D = None
	pass
class UvFilterSizeXPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : FluidTexture2D = None
	pass
class UvFilterSizeYPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : FluidTexture2D = None
	pass
class UvFilterSizePlug(Plug):
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvfsx_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	uvfsy_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	node : FluidTexture2D = None
	pass
# endregion


# define node class
class FluidTexture2D(FluidShape):
	alphaGain_ : AlphaGainPlug = PlugDescriptor("alphaGain")
	alphaOffset_ : AlphaOffsetPlug = PlugDescriptor("alphaOffset")
	defaultColorB_ : DefaultColorBPlug = PlugDescriptor("defaultColorB")
	defaultColorG_ : DefaultColorGPlug = PlugDescriptor("defaultColorG")
	defaultColorR_ : DefaultColorRPlug = PlugDescriptor("defaultColorR")
	defaultColor_ : DefaultColorPlug = PlugDescriptor("defaultColor")
	outAlpha_ : OutAlphaPlug = PlugDescriptor("outAlpha")
	outU_ : OutUPlug = PlugDescriptor("outU")
	outV_ : OutVPlug = PlugDescriptor("outV")
	outUV_ : OutUVPlug = PlugDescriptor("outUV")
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	uvCoord_ : UvCoordPlug = PlugDescriptor("uvCoord")
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	uvFilterSize_ : UvFilterSizePlug = PlugDescriptor("uvFilterSize")

	# node attributes

	typeName = "fluidTexture2D"
	apiTypeInt = 909
	apiTypeStr = "kFluidTexture2D"
	typeIdInt = 1179407444
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["alphaGain", "alphaOffset", "defaultColorB", "defaultColorG", "defaultColorR", "defaultColor", "outAlpha", "outU", "outV", "outUV", "uCoord", "vCoord", "uvCoord", "uvFilterSizeX", "uvFilterSizeY", "uvFilterSize"]
	nodeLeafPlugs = ["alphaGain", "alphaOffset", "defaultColor", "outAlpha", "outUV", "uvCoord", "uvFilterSize"]
	pass

