

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
	node : DisplayLayer = None
	pass
class DisplayOrderPlug(Plug):
	node : DisplayLayer = None
	pass
class ColorPlug(Plug):
	parent : DrawInfoPlug = PlugDescriptor("drawInfo")
	node : DisplayLayer = None
	pass
class DisplayTypePlug(Plug):
	parent : DrawInfoPlug = PlugDescriptor("drawInfo")
	node : DisplayLayer = None
	pass
class EnabledPlug(Plug):
	parent : DrawInfoPlug = PlugDescriptor("drawInfo")
	node : DisplayLayer = None
	pass
class HideOnPlaybackPlug(Plug):
	parent : DrawInfoPlug = PlugDescriptor("drawInfo")
	node : DisplayLayer = None
	pass
class LevelOfDetailPlug(Plug):
	parent : DrawInfoPlug = PlugDescriptor("drawInfo")
	node : DisplayLayer = None
	pass
class OverrideColorAPlug(Plug):
	parent : DrawInfoPlug = PlugDescriptor("drawInfo")
	node : DisplayLayer = None
	pass
class OverrideColorRGBPlug(Plug):
	parent : DrawInfoPlug = PlugDescriptor("drawInfo")
	overrideColorB_ : OverrideColorBPlug = PlugDescriptor("overrideColorB")
	ovcb_ : OverrideColorBPlug = PlugDescriptor("overrideColorB")
	overrideColorG_ : OverrideColorGPlug = PlugDescriptor("overrideColorG")
	ovcg_ : OverrideColorGPlug = PlugDescriptor("overrideColorG")
	overrideColorR_ : OverrideColorRPlug = PlugDescriptor("overrideColorR")
	ovcr_ : OverrideColorRPlug = PlugDescriptor("overrideColorR")
	node : DisplayLayer = None
	pass
class OverrideRGBColorsPlug(Plug):
	parent : DrawInfoPlug = PlugDescriptor("drawInfo")
	node : DisplayLayer = None
	pass
class PlaybackPlug(Plug):
	parent : DrawInfoPlug = PlugDescriptor("drawInfo")
	node : DisplayLayer = None
	pass
class ShadingPlug(Plug):
	parent : DrawInfoPlug = PlugDescriptor("drawInfo")
	node : DisplayLayer = None
	pass
class TexturingPlug(Plug):
	parent : DrawInfoPlug = PlugDescriptor("drawInfo")
	node : DisplayLayer = None
	pass
class VisibilityPlug(Plug):
	parent : DrawInfoPlug = PlugDescriptor("drawInfo")
	node : DisplayLayer = None
	pass
class DrawInfoPlug(Plug):
	color_ : ColorPlug = PlugDescriptor("color")
	c_ : ColorPlug = PlugDescriptor("color")
	displayType_ : DisplayTypePlug = PlugDescriptor("displayType")
	dt_ : DisplayTypePlug = PlugDescriptor("displayType")
	enabled_ : EnabledPlug = PlugDescriptor("enabled")
	e_ : EnabledPlug = PlugDescriptor("enabled")
	hideOnPlayback_ : HideOnPlaybackPlug = PlugDescriptor("hideOnPlayback")
	hpb_ : HideOnPlaybackPlug = PlugDescriptor("hideOnPlayback")
	levelOfDetail_ : LevelOfDetailPlug = PlugDescriptor("levelOfDetail")
	lod_ : LevelOfDetailPlug = PlugDescriptor("levelOfDetail")
	overrideColorA_ : OverrideColorAPlug = PlugDescriptor("overrideColorA")
	ovca_ : OverrideColorAPlug = PlugDescriptor("overrideColorA")
	overrideColorRGB_ : OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	ovrgb_ : OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	overrideRGBColors_ : OverrideRGBColorsPlug = PlugDescriptor("overrideRGBColors")
	ovrgbf_ : OverrideRGBColorsPlug = PlugDescriptor("overrideRGBColors")
	playback_ : PlaybackPlug = PlugDescriptor("playback")
	p_ : PlaybackPlug = PlugDescriptor("playback")
	shading_ : ShadingPlug = PlugDescriptor("shading")
	s_ : ShadingPlug = PlugDescriptor("shading")
	texturing_ : TexturingPlug = PlugDescriptor("texturing")
	t_ : TexturingPlug = PlugDescriptor("texturing")
	visibility_ : VisibilityPlug = PlugDescriptor("visibility")
	v_ : VisibilityPlug = PlugDescriptor("visibility")
	node : DisplayLayer = None
	pass
class IdentificationPlug(Plug):
	node : DisplayLayer = None
	pass
class OverrideColorBPlug(Plug):
	parent : OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	node : DisplayLayer = None
	pass
class OverrideColorGPlug(Plug):
	parent : OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	node : DisplayLayer = None
	pass
class OverrideColorRPlug(Plug):
	parent : OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	node : DisplayLayer = None
	pass
class UfeMembersPlug(Plug):
	node : DisplayLayer = None
	pass
# endregion


# define node class
class DisplayLayer(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	displayOrder_ : DisplayOrderPlug = PlugDescriptor("displayOrder")
	color_ : ColorPlug = PlugDescriptor("color")
	displayType_ : DisplayTypePlug = PlugDescriptor("displayType")
	enabled_ : EnabledPlug = PlugDescriptor("enabled")
	hideOnPlayback_ : HideOnPlaybackPlug = PlugDescriptor("hideOnPlayback")
	levelOfDetail_ : LevelOfDetailPlug = PlugDescriptor("levelOfDetail")
	overrideColorA_ : OverrideColorAPlug = PlugDescriptor("overrideColorA")
	overrideColorRGB_ : OverrideColorRGBPlug = PlugDescriptor("overrideColorRGB")
	overrideRGBColors_ : OverrideRGBColorsPlug = PlugDescriptor("overrideRGBColors")
	playback_ : PlaybackPlug = PlugDescriptor("playback")
	shading_ : ShadingPlug = PlugDescriptor("shading")
	texturing_ : TexturingPlug = PlugDescriptor("texturing")
	visibility_ : VisibilityPlug = PlugDescriptor("visibility")
	drawInfo_ : DrawInfoPlug = PlugDescriptor("drawInfo")
	identification_ : IdentificationPlug = PlugDescriptor("identification")
	overrideColorB_ : OverrideColorBPlug = PlugDescriptor("overrideColorB")
	overrideColorG_ : OverrideColorGPlug = PlugDescriptor("overrideColorG")
	overrideColorR_ : OverrideColorRPlug = PlugDescriptor("overrideColorR")
	ufeMembers_ : UfeMembersPlug = PlugDescriptor("ufeMembers")

	# node attributes

	typeName = "displayLayer"
	apiTypeInt = 733
	apiTypeStr = "kDisplayLayer"
	typeIdInt = 1146310732
	MFnCls = om.MFnDisplayLayer
	nodeLeafClassAttrs = ["binMembership", "displayOrder", "color", "displayType", "enabled", "hideOnPlayback", "levelOfDetail", "overrideColorA", "overrideColorRGB", "overrideRGBColors", "playback", "shading", "texturing", "visibility", "drawInfo", "identification", "overrideColorB", "overrideColorG", "overrideColorR", "ufeMembers"]
	nodeLeafPlugs = ["binMembership", "displayOrder", "drawInfo", "identification", "ufeMembers"]
	pass

