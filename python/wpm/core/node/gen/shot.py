

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
class AudioPlug(Plug):
	node : Shot = None
	pass
class BinMembershipPlug(Plug):
	node : Shot = None
	pass
class CamerasPlug(Plug):
	node : Shot = None
	pass
class ClipPlug(Plug):
	node : Shot = None
	pass
class ClipDurationPlug(Plug):
	node : Shot = None
	pass
class ClipPostHoldPlug(Plug):
	node : Shot = None
	pass
class ClipPreHoldPlug(Plug):
	node : Shot = None
	pass
class ClipScalePlug(Plug):
	node : Shot = None
	pass
class ClipValidPlug(Plug):
	node : Shot = None
	pass
class ClipZeroOffsetPlug(Plug):
	node : Shot = None
	pass
class CurrentCameraPlug(Plug):
	node : Shot = None
	pass
class CustomAnimPlug(Plug):
	node : Shot = None
	pass
class EndFramePlug(Plug):
	node : Shot = None
	pass
class FavoritePlug(Plug):
	node : Shot = None
	pass
class FlagsPlug(Plug):
	node : Shot = None
	pass
class HResolutionPlug(Plug):
	node : Shot = None
	pass
class HasIncomingSttPlug(Plug):
	node : Shot = None
	pass
class HasOutgoingSttPlug(Plug):
	node : Shot = None
	pass
class MembersPlug(Plug):
	node : Shot = None
	pass
class PostHoldPlug(Plug):
	node : Shot = None
	pass
class PreHoldPlug(Plug):
	node : Shot = None
	pass
class ScalePlug(Plug):
	node : Shot = None
	pass
class SequenceEndFramePlug(Plug):
	node : Shot = None
	pass
class SequenceStartFramePlug(Plug):
	node : Shot = None
	pass
class ShotNamePlug(Plug):
	node : Shot = None
	pass
class StartFramePlug(Plug):
	node : Shot = None
	pass
class TrackPlug(Plug):
	node : Shot = None
	pass
class TrackStatePlug(Plug):
	node : Shot = None
	pass
class TransitionInLengthPlug(Plug):
	node : Shot = None
	pass
class TransitionInTypePlug(Plug):
	node : Shot = None
	pass
class TransitionOutLengthPlug(Plug):
	node : Shot = None
	pass
class TransitionOutTypePlug(Plug):
	node : Shot = None
	pass
class UserStatus1Plug(Plug):
	node : Shot = None
	pass
class UserStatus2Plug(Plug):
	node : Shot = None
	pass
class WResolutionPlug(Plug):
	node : Shot = None
	pass
# endregion


# define node class
class Shot(_BASE_):
	audio_ : AudioPlug = PlugDescriptor("audio")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	cameras_ : CamerasPlug = PlugDescriptor("cameras")
	clip_ : ClipPlug = PlugDescriptor("clip")
	clipDuration_ : ClipDurationPlug = PlugDescriptor("clipDuration")
	clipPostHold_ : ClipPostHoldPlug = PlugDescriptor("clipPostHold")
	clipPreHold_ : ClipPreHoldPlug = PlugDescriptor("clipPreHold")
	clipScale_ : ClipScalePlug = PlugDescriptor("clipScale")
	clipValid_ : ClipValidPlug = PlugDescriptor("clipValid")
	clipZeroOffset_ : ClipZeroOffsetPlug = PlugDescriptor("clipZeroOffset")
	currentCamera_ : CurrentCameraPlug = PlugDescriptor("currentCamera")
	customAnim_ : CustomAnimPlug = PlugDescriptor("customAnim")
	endFrame_ : EndFramePlug = PlugDescriptor("endFrame")
	favorite_ : FavoritePlug = PlugDescriptor("favorite")
	flags_ : FlagsPlug = PlugDescriptor("flags")
	hResolution_ : HResolutionPlug = PlugDescriptor("hResolution")
	hasIncomingStt_ : HasIncomingSttPlug = PlugDescriptor("hasIncomingStt")
	hasOutgoingStt_ : HasOutgoingSttPlug = PlugDescriptor("hasOutgoingStt")
	members_ : MembersPlug = PlugDescriptor("members")
	postHold_ : PostHoldPlug = PlugDescriptor("postHold")
	preHold_ : PreHoldPlug = PlugDescriptor("preHold")
	scale_ : ScalePlug = PlugDescriptor("scale")
	sequenceEndFrame_ : SequenceEndFramePlug = PlugDescriptor("sequenceEndFrame")
	sequenceStartFrame_ : SequenceStartFramePlug = PlugDescriptor("sequenceStartFrame")
	shotName_ : ShotNamePlug = PlugDescriptor("shotName")
	startFrame_ : StartFramePlug = PlugDescriptor("startFrame")
	track_ : TrackPlug = PlugDescriptor("track")
	trackState_ : TrackStatePlug = PlugDescriptor("trackState")
	transitionInLength_ : TransitionInLengthPlug = PlugDescriptor("transitionInLength")
	transitionInType_ : TransitionInTypePlug = PlugDescriptor("transitionInType")
	transitionOutLength_ : TransitionOutLengthPlug = PlugDescriptor("transitionOutLength")
	transitionOutType_ : TransitionOutTypePlug = PlugDescriptor("transitionOutType")
	userStatus1_ : UserStatus1Plug = PlugDescriptor("userStatus1")
	userStatus2_ : UserStatus2Plug = PlugDescriptor("userStatus2")
	wResolution_ : WResolutionPlug = PlugDescriptor("wResolution")

	# node attributes

	typeName = "shot"
	apiTypeInt = 1051
	apiTypeStr = "kShot"
	typeIdInt = 1397247828
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["audio", "binMembership", "cameras", "clip", "clipDuration", "clipPostHold", "clipPreHold", "clipScale", "clipValid", "clipZeroOffset", "currentCamera", "customAnim", "endFrame", "favorite", "flags", "hResolution", "hasIncomingStt", "hasOutgoingStt", "members", "postHold", "preHold", "scale", "sequenceEndFrame", "sequenceStartFrame", "shotName", "startFrame", "track", "trackState", "transitionInLength", "transitionInType", "transitionOutLength", "transitionOutType", "userStatus1", "userStatus2", "wResolution"]
	nodeLeafPlugs = ["audio", "binMembership", "cameras", "clip", "clipDuration", "clipPostHold", "clipPreHold", "clipScale", "clipValid", "clipZeroOffset", "currentCamera", "customAnim", "endFrame", "favorite", "flags", "hResolution", "hasIncomingStt", "hasOutgoingStt", "members", "postHold", "preHold", "scale", "sequenceEndFrame", "sequenceStartFrame", "shotName", "startFrame", "track", "trackState", "transitionInLength", "transitionInType", "transitionOutLength", "transitionOutType", "userStatus1", "userStatus2", "wResolution"]
	pass

