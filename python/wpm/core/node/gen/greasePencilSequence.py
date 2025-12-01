

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
class ActiveFrameIndexPlug(Plug):
	node : GreasePencilSequence = None
	pass
class AlphaMultiplierPlug(Plug):
	node : GreasePencilSequence = None
	pass
class BinMembershipPlug(Plug):
	node : GreasePencilSequence = None
	pass
class BlendLenPlug(Plug):
	node : GreasePencilSequence = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : GreasePencilSequence = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : GreasePencilSequence = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : GreasePencilSequence = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	clb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	clg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	clr_ : ColorRPlug = PlugDescriptor("colorR")
	node : GreasePencilSequence = None
	pass
class FrameAlphaPlug(Plug):
	parent : FramePlug = PlugDescriptor("frame")
	node : GreasePencilSequence = None
	pass
class FrameEnablePlug(Plug):
	parent : FramePlug = PlugDescriptor("frame")
	node : GreasePencilSequence = None
	pass
class FrameImagePlug(Plug):
	parent : FramePlug = PlugDescriptor("frame")
	node : GreasePencilSequence = None
	pass
class FrameLabelPlug(Plug):
	parent : FramePlug = PlugDescriptor("frame")
	node : GreasePencilSequence = None
	pass
class FrameTimePlug(Plug):
	parent : FramePlug = PlugDescriptor("frame")
	node : GreasePencilSequence = None
	pass
class FramePlug(Plug):
	frameAlpha_ : FrameAlphaPlug = PlugDescriptor("frameAlpha")
	fal_ : FrameAlphaPlug = PlugDescriptor("frameAlpha")
	frameEnable_ : FrameEnablePlug = PlugDescriptor("frameEnable")
	fen_ : FrameEnablePlug = PlugDescriptor("frameEnable")
	frameImage_ : FrameImagePlug = PlugDescriptor("frameImage")
	fim_ : FrameImagePlug = PlugDescriptor("frameImage")
	frameLabel_ : FrameLabelPlug = PlugDescriptor("frameLabel")
	fl_ : FrameLabelPlug = PlugDescriptor("frameLabel")
	frameTime_ : FrameTimePlug = PlugDescriptor("frameTime")
	ftv_ : FrameTimePlug = PlugDescriptor("frameTime")
	node : GreasePencilSequence = None
	pass
class PostFramesPlug(Plug):
	node : GreasePencilSequence = None
	pass
class PostGhostPlug(Plug):
	node : GreasePencilSequence = None
	pass
class PreFramesPlug(Plug):
	node : GreasePencilSequence = None
	pass
class PreGhostPlug(Plug):
	node : GreasePencilSequence = None
	pass
class TimeInputPlug(Plug):
	node : GreasePencilSequence = None
	pass
# endregion


# define node class
class GreasePencilSequence(_BASE_):
	activeFrameIndex_ : ActiveFrameIndexPlug = PlugDescriptor("activeFrameIndex")
	alphaMultiplier_ : AlphaMultiplierPlug = PlugDescriptor("alphaMultiplier")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	blendLen_ : BlendLenPlug = PlugDescriptor("blendLen")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	frameAlpha_ : FrameAlphaPlug = PlugDescriptor("frameAlpha")
	frameEnable_ : FrameEnablePlug = PlugDescriptor("frameEnable")
	frameImage_ : FrameImagePlug = PlugDescriptor("frameImage")
	frameLabel_ : FrameLabelPlug = PlugDescriptor("frameLabel")
	frameTime_ : FrameTimePlug = PlugDescriptor("frameTime")
	frame_ : FramePlug = PlugDescriptor("frame")
	postFrames_ : PostFramesPlug = PlugDescriptor("postFrames")
	postGhost_ : PostGhostPlug = PlugDescriptor("postGhost")
	preFrames_ : PreFramesPlug = PlugDescriptor("preFrames")
	preGhost_ : PreGhostPlug = PlugDescriptor("preGhost")
	timeInput_ : TimeInputPlug = PlugDescriptor("timeInput")

	# node attributes

	typeName = "greasePencilSequence"
	apiTypeInt = 1088
	apiTypeStr = "kGreasePencilSequence"
	typeIdInt = 1196446545
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["activeFrameIndex", "alphaMultiplier", "binMembership", "blendLen", "colorB", "colorG", "colorR", "color", "frameAlpha", "frameEnable", "frameImage", "frameLabel", "frameTime", "frame", "postFrames", "postGhost", "preFrames", "preGhost", "timeInput"]
	nodeLeafPlugs = ["activeFrameIndex", "alphaMultiplier", "binMembership", "blendLen", "color", "frame", "postFrames", "postGhost", "preFrames", "preGhost", "timeInput"]
	pass

