

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Transform = Catalogue.Transform
else:
	from .. import retriever
	Transform = retriever.getNodeCls("Transform")
	assert Transform

# add node doc



# region plug type defs
class ClipDataPlug(Plug):
	node : ClipGhostShape = None
	pass
class ClipDirectionScalePlug(Plug):
	node : ClipGhostShape = None
	pass
class ClipEnabledPlug(Plug):
	node : ClipGhostShape = None
	pass
class ClipGhostDataPlug(Plug):
	node : ClipGhostShape = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : ClipGhostShape = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : ClipGhostShape = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : ClipGhostShape = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	colr_ : ColorRPlug = PlugDescriptor("colorR")
	node : ClipGhostShape = None
	pass
class IntermediatePosesPlug(Plug):
	node : ClipGhostShape = None
	pass
class ShowClipPathPlug(Plug):
	node : ClipGhostShape = None
	pass
class ShowEndPosePlug(Plug):
	node : ClipGhostShape = None
	pass
class ShowIntermediatePosesPlug(Plug):
	node : ClipGhostShape = None
	pass
class ShowStartPosePlug(Plug):
	node : ClipGhostShape = None
	pass
class TrackMutedPlug(Plug):
	node : ClipGhostShape = None
	pass
# endregion


# define node class
class ClipGhostShape(Transform):
	clipData_ : ClipDataPlug = PlugDescriptor("clipData")
	clipDirectionScale_ : ClipDirectionScalePlug = PlugDescriptor("clipDirectionScale")
	clipEnabled_ : ClipEnabledPlug = PlugDescriptor("clipEnabled")
	clipGhostData_ : ClipGhostDataPlug = PlugDescriptor("clipGhostData")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	intermediatePoses_ : IntermediatePosesPlug = PlugDescriptor("intermediatePoses")
	showClipPath_ : ShowClipPathPlug = PlugDescriptor("showClipPath")
	showEndPose_ : ShowEndPosePlug = PlugDescriptor("showEndPose")
	showIntermediatePoses_ : ShowIntermediatePosesPlug = PlugDescriptor("showIntermediatePoses")
	showStartPose_ : ShowStartPosePlug = PlugDescriptor("showStartPose")
	trackMuted_ : TrackMutedPlug = PlugDescriptor("trackMuted")

	# node attributes

	typeName = "clipGhostShape"
	apiTypeInt = 1082
	apiTypeStr = "kClipGhostShape"
	typeIdInt = 1128747848
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["clipData", "clipDirectionScale", "clipEnabled", "clipGhostData", "colorB", "colorG", "colorR", "color", "intermediatePoses", "showClipPath", "showEndPose", "showIntermediatePoses", "showStartPose", "trackMuted"]
	nodeLeafPlugs = ["clipData", "clipDirectionScale", "clipEnabled", "clipGhostData", "color", "intermediatePoses", "showClipPath", "showEndPose", "showIntermediatePoses", "showStartPose", "trackMuted"]
	pass

