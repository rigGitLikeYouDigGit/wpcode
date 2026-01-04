

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Texture2d = Catalogue.Texture2d
else:
	from .. import retriever
	Texture2d = retriever.getNodeCls("Texture2d")
	assert Texture2d

# add node doc



# region plug type defs
class ContrastPlug(Plug):
	node : Grid = None
	pass
class FillerColorBPlug(Plug):
	parent : FillerColorPlug = PlugDescriptor("fillerColor")
	node : Grid = None
	pass
class FillerColorGPlug(Plug):
	parent : FillerColorPlug = PlugDescriptor("fillerColor")
	node : Grid = None
	pass
class FillerColorRPlug(Plug):
	parent : FillerColorPlug = PlugDescriptor("fillerColor")
	node : Grid = None
	pass
class FillerColorPlug(Plug):
	fillerColorB_ : FillerColorBPlug = PlugDescriptor("fillerColorB")
	fcb_ : FillerColorBPlug = PlugDescriptor("fillerColorB")
	fillerColorG_ : FillerColorGPlug = PlugDescriptor("fillerColorG")
	fcg_ : FillerColorGPlug = PlugDescriptor("fillerColorG")
	fillerColorR_ : FillerColorRPlug = PlugDescriptor("fillerColorR")
	fcr_ : FillerColorRPlug = PlugDescriptor("fillerColorR")
	node : Grid = None
	pass
class LineColorBPlug(Plug):
	parent : LineColorPlug = PlugDescriptor("lineColor")
	node : Grid = None
	pass
class LineColorGPlug(Plug):
	parent : LineColorPlug = PlugDescriptor("lineColor")
	node : Grid = None
	pass
class LineColorRPlug(Plug):
	parent : LineColorPlug = PlugDescriptor("lineColor")
	node : Grid = None
	pass
class LineColorPlug(Plug):
	lineColorB_ : LineColorBPlug = PlugDescriptor("lineColorB")
	lcb_ : LineColorBPlug = PlugDescriptor("lineColorB")
	lineColorG_ : LineColorGPlug = PlugDescriptor("lineColorG")
	lcg_ : LineColorGPlug = PlugDescriptor("lineColorG")
	lineColorR_ : LineColorRPlug = PlugDescriptor("lineColorR")
	lcr_ : LineColorRPlug = PlugDescriptor("lineColorR")
	node : Grid = None
	pass
class UWidthPlug(Plug):
	node : Grid = None
	pass
class VWidthPlug(Plug):
	node : Grid = None
	pass
# endregion


# define node class
class Grid(Texture2d):
	contrast_ : ContrastPlug = PlugDescriptor("contrast")
	fillerColorB_ : FillerColorBPlug = PlugDescriptor("fillerColorB")
	fillerColorG_ : FillerColorGPlug = PlugDescriptor("fillerColorG")
	fillerColorR_ : FillerColorRPlug = PlugDescriptor("fillerColorR")
	fillerColor_ : FillerColorPlug = PlugDescriptor("fillerColor")
	lineColorB_ : LineColorBPlug = PlugDescriptor("lineColorB")
	lineColorG_ : LineColorGPlug = PlugDescriptor("lineColorG")
	lineColorR_ : LineColorRPlug = PlugDescriptor("lineColorR")
	lineColor_ : LineColorPlug = PlugDescriptor("lineColor")
	uWidth_ : UWidthPlug = PlugDescriptor("uWidth")
	vWidth_ : VWidthPlug = PlugDescriptor("vWidth")

	# node attributes

	typeName = "grid"
	apiTypeInt = 502
	apiTypeStr = "kGrid"
	typeIdInt = 1381254980
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["contrast", "fillerColorB", "fillerColorG", "fillerColorR", "fillerColor", "lineColorB", "lineColorG", "lineColorR", "lineColor", "uWidth", "vWidth"]
	nodeLeafPlugs = ["contrast", "fillerColor", "lineColor", "uWidth", "vWidth"]
	pass

