

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
BakeSet = retriever.getNodeCls("BakeSet")
assert BakeSet
if T.TYPE_CHECKING:
	from .. import BakeSet

# add node doc



# region plug type defs
class BackgroundColorBPlug(Plug):
	parent : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	node : TextureBakeSet = None
	pass
class BackgroundColorGPlug(Plug):
	parent : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	node : TextureBakeSet = None
	pass
class BackgroundColorRPlug(Plug):
	parent : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	node : TextureBakeSet = None
	pass
class BackgroundColorPlug(Plug):
	backgroundColorB_ : BackgroundColorBPlug = PlugDescriptor("backgroundColorB")
	bgb_ : BackgroundColorBPlug = PlugDescriptor("backgroundColorB")
	backgroundColorG_ : BackgroundColorGPlug = PlugDescriptor("backgroundColorG")
	bgg_ : BackgroundColorGPlug = PlugDescriptor("backgroundColorG")
	backgroundColorR_ : BackgroundColorRPlug = PlugDescriptor("backgroundColorR")
	bgr_ : BackgroundColorRPlug = PlugDescriptor("backgroundColorR")
	node : TextureBakeSet = None
	pass
class BackgroundModePlug(Plug):
	node : TextureBakeSet = None
	pass
class BakeToOneMapPlug(Plug):
	node : TextureBakeSet = None
	pass
class BitsPerChannelPlug(Plug):
	node : TextureBakeSet = None
	pass
class FileFormatPlug(Plug):
	node : TextureBakeSet = None
	pass
class FillScalePlug(Plug):
	node : TextureBakeSet = None
	pass
class FillTextureSeamsPlug(Plug):
	node : TextureBakeSet = None
	pass
class FinalGatherQualityPlug(Plug):
	node : TextureBakeSet = None
	pass
class FinalGatherReflectPlug(Plug):
	node : TextureBakeSet = None
	pass
class OverrideUvSetPlug(Plug):
	node : TextureBakeSet = None
	pass
class PrefixPlug(Plug):
	node : TextureBakeSet = None
	pass
class SamplesPlug(Plug):
	node : TextureBakeSet = None
	pass
class SeparationPlug(Plug):
	node : TextureBakeSet = None
	pass
class UMaxPlug(Plug):
	node : TextureBakeSet = None
	pass
class UMinPlug(Plug):
	node : TextureBakeSet = None
	pass
class UvRangePlug(Plug):
	node : TextureBakeSet = None
	pass
class UvSetNamePlug(Plug):
	node : TextureBakeSet = None
	pass
class VMaxPlug(Plug):
	node : TextureBakeSet = None
	pass
class VMinPlug(Plug):
	node : TextureBakeSet = None
	pass
class XResolutionPlug(Plug):
	node : TextureBakeSet = None
	pass
class YResolutionPlug(Plug):
	node : TextureBakeSet = None
	pass
# endregion


# define node class
class TextureBakeSet(BakeSet):
	backgroundColorB_ : BackgroundColorBPlug = PlugDescriptor("backgroundColorB")
	backgroundColorG_ : BackgroundColorGPlug = PlugDescriptor("backgroundColorG")
	backgroundColorR_ : BackgroundColorRPlug = PlugDescriptor("backgroundColorR")
	backgroundColor_ : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	backgroundMode_ : BackgroundModePlug = PlugDescriptor("backgroundMode")
	bakeToOneMap_ : BakeToOneMapPlug = PlugDescriptor("bakeToOneMap")
	bitsPerChannel_ : BitsPerChannelPlug = PlugDescriptor("bitsPerChannel")
	fileFormat_ : FileFormatPlug = PlugDescriptor("fileFormat")
	fillScale_ : FillScalePlug = PlugDescriptor("fillScale")
	fillTextureSeams_ : FillTextureSeamsPlug = PlugDescriptor("fillTextureSeams")
	finalGatherQuality_ : FinalGatherQualityPlug = PlugDescriptor("finalGatherQuality")
	finalGatherReflect_ : FinalGatherReflectPlug = PlugDescriptor("finalGatherReflect")
	overrideUvSet_ : OverrideUvSetPlug = PlugDescriptor("overrideUvSet")
	prefix_ : PrefixPlug = PlugDescriptor("prefix")
	samples_ : SamplesPlug = PlugDescriptor("samples")
	separation_ : SeparationPlug = PlugDescriptor("separation")
	uMax_ : UMaxPlug = PlugDescriptor("uMax")
	uMin_ : UMinPlug = PlugDescriptor("uMin")
	uvRange_ : UvRangePlug = PlugDescriptor("uvRange")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")
	vMax_ : VMaxPlug = PlugDescriptor("vMax")
	vMin_ : VMinPlug = PlugDescriptor("vMin")
	xResolution_ : XResolutionPlug = PlugDescriptor("xResolution")
	yResolution_ : YResolutionPlug = PlugDescriptor("yResolution")

	# node attributes

	typeName = "textureBakeSet"
	apiTypeInt = 472
	apiTypeStr = "kTextureBakeSet"
	typeIdInt = 1413628235
	MFnCls = om.MFnSet
	pass

