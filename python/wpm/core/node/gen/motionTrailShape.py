

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SnapshotShape = retriever.getNodeCls("SnapshotShape")
assert SnapshotShape
if T.TYPE_CHECKING:
	from .. import SnapshotShape

# add node doc



# region plug type defs
class ActiveKeyframeColorBPlug(Plug):
	parent : ActiveKeyframeColorPlug = PlugDescriptor("activeKeyframeColor")
	node : MotionTrailShape = None
	pass
class ActiveKeyframeColorGPlug(Plug):
	parent : ActiveKeyframeColorPlug = PlugDescriptor("activeKeyframeColor")
	node : MotionTrailShape = None
	pass
class ActiveKeyframeColorRPlug(Plug):
	parent : ActiveKeyframeColorPlug = PlugDescriptor("activeKeyframeColor")
	node : MotionTrailShape = None
	pass
class ActiveKeyframeColorPlug(Plug):
	activeKeyframeColorB_ : ActiveKeyframeColorBPlug = PlugDescriptor("activeKeyframeColorB")
	akb_ : ActiveKeyframeColorBPlug = PlugDescriptor("activeKeyframeColorB")
	activeKeyframeColorG_ : ActiveKeyframeColorGPlug = PlugDescriptor("activeKeyframeColorG")
	akg_ : ActiveKeyframeColorGPlug = PlugDescriptor("activeKeyframeColorG")
	activeKeyframeColorR_ : ActiveKeyframeColorRPlug = PlugDescriptor("activeKeyframeColorR")
	akr_ : ActiveKeyframeColorRPlug = PlugDescriptor("activeKeyframeColorR")
	node : MotionTrailShape = None
	pass
class BeadColorBPlug(Plug):
	parent : BeadColorPlug = PlugDescriptor("beadColor")
	node : MotionTrailShape = None
	pass
class BeadColorGPlug(Plug):
	parent : BeadColorPlug = PlugDescriptor("beadColor")
	node : MotionTrailShape = None
	pass
class BeadColorRPlug(Plug):
	parent : BeadColorPlug = PlugDescriptor("beadColor")
	node : MotionTrailShape = None
	pass
class BeadColorPlug(Plug):
	beadColorB_ : BeadColorBPlug = PlugDescriptor("beadColorB")
	bcb_ : BeadColorBPlug = PlugDescriptor("beadColorB")
	beadColorG_ : BeadColorGPlug = PlugDescriptor("beadColorG")
	bcg_ : BeadColorGPlug = PlugDescriptor("beadColorG")
	beadColorR_ : BeadColorRPlug = PlugDescriptor("beadColorR")
	bcr_ : BeadColorRPlug = PlugDescriptor("beadColorR")
	node : MotionTrailShape = None
	pass
class ExtraKeyframeColorBPlug(Plug):
	parent : ExtraKeyframeColorPlug = PlugDescriptor("extraKeyframeColor")
	node : MotionTrailShape = None
	pass
class ExtraKeyframeColorGPlug(Plug):
	parent : ExtraKeyframeColorPlug = PlugDescriptor("extraKeyframeColor")
	node : MotionTrailShape = None
	pass
class ExtraKeyframeColorRPlug(Plug):
	parent : ExtraKeyframeColorPlug = PlugDescriptor("extraKeyframeColor")
	node : MotionTrailShape = None
	pass
class ExtraKeyframeColorPlug(Plug):
	extraKeyframeColorB_ : ExtraKeyframeColorBPlug = PlugDescriptor("extraKeyframeColorB")
	ecb_ : ExtraKeyframeColorBPlug = PlugDescriptor("extraKeyframeColorB")
	extraKeyframeColorG_ : ExtraKeyframeColorGPlug = PlugDescriptor("extraKeyframeColorG")
	ecg_ : ExtraKeyframeColorGPlug = PlugDescriptor("extraKeyframeColorG")
	extraKeyframeColorR_ : ExtraKeyframeColorRPlug = PlugDescriptor("extraKeyframeColorR")
	ecr_ : ExtraKeyframeColorRPlug = PlugDescriptor("extraKeyframeColorR")
	node : MotionTrailShape = None
	pass
class ExtraKeyframeTimesPlug(Plug):
	node : MotionTrailShape = None
	pass
class ExtraTrailColorBPlug(Plug):
	parent : ExtraTrailColorPlug = PlugDescriptor("extraTrailColor")
	node : MotionTrailShape = None
	pass
class ExtraTrailColorGPlug(Plug):
	parent : ExtraTrailColorPlug = PlugDescriptor("extraTrailColor")
	node : MotionTrailShape = None
	pass
class ExtraTrailColorRPlug(Plug):
	parent : ExtraTrailColorPlug = PlugDescriptor("extraTrailColor")
	node : MotionTrailShape = None
	pass
class ExtraTrailColorPlug(Plug):
	extraTrailColorB_ : ExtraTrailColorBPlug = PlugDescriptor("extraTrailColorB")
	etcb_ : ExtraTrailColorBPlug = PlugDescriptor("extraTrailColorB")
	extraTrailColorG_ : ExtraTrailColorGPlug = PlugDescriptor("extraTrailColorG")
	etcg_ : ExtraTrailColorGPlug = PlugDescriptor("extraTrailColorG")
	extraTrailColorR_ : ExtraTrailColorRPlug = PlugDescriptor("extraTrailColorR")
	etcr_ : ExtraTrailColorRPlug = PlugDescriptor("extraTrailColorR")
	node : MotionTrailShape = None
	pass
class FadeInoutFramesPlug(Plug):
	node : MotionTrailShape = None
	pass
class FrameMarkerColorBPlug(Plug):
	parent : FrameMarkerColorPlug = PlugDescriptor("frameMarkerColor")
	node : MotionTrailShape = None
	pass
class FrameMarkerColorGPlug(Plug):
	parent : FrameMarkerColorPlug = PlugDescriptor("frameMarkerColor")
	node : MotionTrailShape = None
	pass
class FrameMarkerColorRPlug(Plug):
	parent : FrameMarkerColorPlug = PlugDescriptor("frameMarkerColor")
	node : MotionTrailShape = None
	pass
class FrameMarkerColorPlug(Plug):
	frameMarkerColorB_ : FrameMarkerColorBPlug = PlugDescriptor("frameMarkerColorB")
	fcb_ : FrameMarkerColorBPlug = PlugDescriptor("frameMarkerColorB")
	frameMarkerColorG_ : FrameMarkerColorGPlug = PlugDescriptor("frameMarkerColorG")
	fcg_ : FrameMarkerColorGPlug = PlugDescriptor("frameMarkerColorG")
	frameMarkerColorR_ : FrameMarkerColorRPlug = PlugDescriptor("frameMarkerColorR")
	fcr_ : FrameMarkerColorRPlug = PlugDescriptor("frameMarkerColorR")
	node : MotionTrailShape = None
	pass
class FrameMarkerSizePlug(Plug):
	node : MotionTrailShape = None
	pass
class IncrementPlug(Plug):
	node : MotionTrailShape = None
	pass
class KeyframeColorBPlug(Plug):
	parent : KeyframeColorPlug = PlugDescriptor("keyframeColor")
	node : MotionTrailShape = None
	pass
class KeyframeColorGPlug(Plug):
	parent : KeyframeColorPlug = PlugDescriptor("keyframeColor")
	node : MotionTrailShape = None
	pass
class KeyframeColorRPlug(Plug):
	parent : KeyframeColorPlug = PlugDescriptor("keyframeColor")
	node : MotionTrailShape = None
	pass
class KeyframeColorPlug(Plug):
	keyframeColorB_ : KeyframeColorBPlug = PlugDescriptor("keyframeColorB")
	kcb_ : KeyframeColorBPlug = PlugDescriptor("keyframeColorB")
	keyframeColorG_ : KeyframeColorGPlug = PlugDescriptor("keyframeColorG")
	kcg_ : KeyframeColorGPlug = PlugDescriptor("keyframeColorG")
	keyframeColorR_ : KeyframeColorRPlug = PlugDescriptor("keyframeColorR")
	kcr_ : KeyframeColorRPlug = PlugDescriptor("keyframeColorR")
	node : MotionTrailShape = None
	pass
class KeyframeFlagsPlug(Plug):
	node : MotionTrailShape = None
	pass
class KeyframeSizePlug(Plug):
	node : MotionTrailShape = None
	pass
class KeyframeTimesPlug(Plug):
	node : MotionTrailShape = None
	pass
class LocalPositionXPlug(Plug):
	parent : LocalPositionPlug = PlugDescriptor("localPosition")
	node : MotionTrailShape = None
	pass
class LocalPositionYPlug(Plug):
	parent : LocalPositionPlug = PlugDescriptor("localPosition")
	node : MotionTrailShape = None
	pass
class LocalPositionZPlug(Plug):
	parent : LocalPositionPlug = PlugDescriptor("localPosition")
	node : MotionTrailShape = None
	pass
class LocalPositionPlug(Plug):
	localPositionX_ : LocalPositionXPlug = PlugDescriptor("localPositionX")
	lpx_ : LocalPositionXPlug = PlugDescriptor("localPositionX")
	localPositionY_ : LocalPositionYPlug = PlugDescriptor("localPositionY")
	lpy_ : LocalPositionYPlug = PlugDescriptor("localPositionY")
	localPositionZ_ : LocalPositionZPlug = PlugDescriptor("localPositionZ")
	lpz_ : LocalPositionZPlug = PlugDescriptor("localPositionZ")
	node : MotionTrailShape = None
	pass
class ModifyKeysPlug(Plug):
	node : MotionTrailShape = None
	pass
class PinnedPlug(Plug):
	node : MotionTrailShape = None
	pass
class PostFramePlug(Plug):
	node : MotionTrailShape = None
	pass
class PreFramePlug(Plug):
	node : MotionTrailShape = None
	pass
class ShowExtraKeysPlug(Plug):
	node : MotionTrailShape = None
	pass
class ShowFrameMarkerFramesPlug(Plug):
	node : MotionTrailShape = None
	pass
class ShowFrameMarkersPlug(Plug):
	node : MotionTrailShape = None
	pass
class ShowInBeadPlug(Plug):
	node : MotionTrailShape = None
	pass
class ShowInTangentPlug(Plug):
	node : MotionTrailShape = None
	pass
class ShowOutBeadPlug(Plug):
	node : MotionTrailShape = None
	pass
class ShowOutTangentPlug(Plug):
	node : MotionTrailShape = None
	pass
class StartTimePlug(Plug):
	node : MotionTrailShape = None
	pass
class TxValuePlug(Plug):
	parent : TangentPointsPlug = PlugDescriptor("tangentPoints")
	node : MotionTrailShape = None
	pass
class TyValuePlug(Plug):
	parent : TangentPointsPlug = PlugDescriptor("tangentPoints")
	node : MotionTrailShape = None
	pass
class TzValuePlug(Plug):
	parent : TangentPointsPlug = PlugDescriptor("tangentPoints")
	node : MotionTrailShape = None
	pass
class TangentPointsPlug(Plug):
	txValue_ : TxValuePlug = PlugDescriptor("txValue")
	txv_ : TxValuePlug = PlugDescriptor("txValue")
	tyValue_ : TyValuePlug = PlugDescriptor("tyValue")
	tyv_ : TyValuePlug = PlugDescriptor("tyValue")
	tzValue_ : TzValuePlug = PlugDescriptor("tzValue")
	tzv_ : TzValuePlug = PlugDescriptor("tzValue")
	node : MotionTrailShape = None
	pass
class TrailColorBPlug(Plug):
	parent : TrailColorPlug = PlugDescriptor("trailColor")
	node : MotionTrailShape = None
	pass
class TrailColorGPlug(Plug):
	parent : TrailColorPlug = PlugDescriptor("trailColor")
	node : MotionTrailShape = None
	pass
class TrailColorRPlug(Plug):
	parent : TrailColorPlug = PlugDescriptor("trailColor")
	node : MotionTrailShape = None
	pass
class TrailColorPlug(Plug):
	trailColorB_ : TrailColorBPlug = PlugDescriptor("trailColorB")
	tcb_ : TrailColorBPlug = PlugDescriptor("trailColorB")
	trailColorG_ : TrailColorGPlug = PlugDescriptor("trailColorG")
	tcg_ : TrailColorGPlug = PlugDescriptor("trailColorG")
	trailColorR_ : TrailColorRPlug = PlugDescriptor("trailColorR")
	tcr_ : TrailColorRPlug = PlugDescriptor("trailColorR")
	node : MotionTrailShape = None
	pass
class TrailDrawModePlug(Plug):
	node : MotionTrailShape = None
	pass
class TrailThicknessPlug(Plug):
	node : MotionTrailShape = None
	pass
class TransformToMovePlug(Plug):
	node : MotionTrailShape = None
	pass
class XrayDrawPlug(Plug):
	node : MotionTrailShape = None
	pass
# endregion


# define node class
class MotionTrailShape(SnapshotShape):
	activeKeyframeColorB_ : ActiveKeyframeColorBPlug = PlugDescriptor("activeKeyframeColorB")
	activeKeyframeColorG_ : ActiveKeyframeColorGPlug = PlugDescriptor("activeKeyframeColorG")
	activeKeyframeColorR_ : ActiveKeyframeColorRPlug = PlugDescriptor("activeKeyframeColorR")
	activeKeyframeColor_ : ActiveKeyframeColorPlug = PlugDescriptor("activeKeyframeColor")
	beadColorB_ : BeadColorBPlug = PlugDescriptor("beadColorB")
	beadColorG_ : BeadColorGPlug = PlugDescriptor("beadColorG")
	beadColorR_ : BeadColorRPlug = PlugDescriptor("beadColorR")
	beadColor_ : BeadColorPlug = PlugDescriptor("beadColor")
	extraKeyframeColorB_ : ExtraKeyframeColorBPlug = PlugDescriptor("extraKeyframeColorB")
	extraKeyframeColorG_ : ExtraKeyframeColorGPlug = PlugDescriptor("extraKeyframeColorG")
	extraKeyframeColorR_ : ExtraKeyframeColorRPlug = PlugDescriptor("extraKeyframeColorR")
	extraKeyframeColor_ : ExtraKeyframeColorPlug = PlugDescriptor("extraKeyframeColor")
	extraKeyframeTimes_ : ExtraKeyframeTimesPlug = PlugDescriptor("extraKeyframeTimes")
	extraTrailColorB_ : ExtraTrailColorBPlug = PlugDescriptor("extraTrailColorB")
	extraTrailColorG_ : ExtraTrailColorGPlug = PlugDescriptor("extraTrailColorG")
	extraTrailColorR_ : ExtraTrailColorRPlug = PlugDescriptor("extraTrailColorR")
	extraTrailColor_ : ExtraTrailColorPlug = PlugDescriptor("extraTrailColor")
	fadeInoutFrames_ : FadeInoutFramesPlug = PlugDescriptor("fadeInoutFrames")
	frameMarkerColorB_ : FrameMarkerColorBPlug = PlugDescriptor("frameMarkerColorB")
	frameMarkerColorG_ : FrameMarkerColorGPlug = PlugDescriptor("frameMarkerColorG")
	frameMarkerColorR_ : FrameMarkerColorRPlug = PlugDescriptor("frameMarkerColorR")
	frameMarkerColor_ : FrameMarkerColorPlug = PlugDescriptor("frameMarkerColor")
	frameMarkerSize_ : FrameMarkerSizePlug = PlugDescriptor("frameMarkerSize")
	increment_ : IncrementPlug = PlugDescriptor("increment")
	keyframeColorB_ : KeyframeColorBPlug = PlugDescriptor("keyframeColorB")
	keyframeColorG_ : KeyframeColorGPlug = PlugDescriptor("keyframeColorG")
	keyframeColorR_ : KeyframeColorRPlug = PlugDescriptor("keyframeColorR")
	keyframeColor_ : KeyframeColorPlug = PlugDescriptor("keyframeColor")
	keyframeFlags_ : KeyframeFlagsPlug = PlugDescriptor("keyframeFlags")
	keyframeSize_ : KeyframeSizePlug = PlugDescriptor("keyframeSize")
	keyframeTimes_ : KeyframeTimesPlug = PlugDescriptor("keyframeTimes")
	localPositionX_ : LocalPositionXPlug = PlugDescriptor("localPositionX")
	localPositionY_ : LocalPositionYPlug = PlugDescriptor("localPositionY")
	localPositionZ_ : LocalPositionZPlug = PlugDescriptor("localPositionZ")
	localPosition_ : LocalPositionPlug = PlugDescriptor("localPosition")
	modifyKeys_ : ModifyKeysPlug = PlugDescriptor("modifyKeys")
	pinned_ : PinnedPlug = PlugDescriptor("pinned")
	postFrame_ : PostFramePlug = PlugDescriptor("postFrame")
	preFrame_ : PreFramePlug = PlugDescriptor("preFrame")
	showExtraKeys_ : ShowExtraKeysPlug = PlugDescriptor("showExtraKeys")
	showFrameMarkerFrames_ : ShowFrameMarkerFramesPlug = PlugDescriptor("showFrameMarkerFrames")
	showFrameMarkers_ : ShowFrameMarkersPlug = PlugDescriptor("showFrameMarkers")
	showInBead_ : ShowInBeadPlug = PlugDescriptor("showInBead")
	showInTangent_ : ShowInTangentPlug = PlugDescriptor("showInTangent")
	showOutBead_ : ShowOutBeadPlug = PlugDescriptor("showOutBead")
	showOutTangent_ : ShowOutTangentPlug = PlugDescriptor("showOutTangent")
	startTime_ : StartTimePlug = PlugDescriptor("startTime")
	txValue_ : TxValuePlug = PlugDescriptor("txValue")
	tyValue_ : TyValuePlug = PlugDescriptor("tyValue")
	tzValue_ : TzValuePlug = PlugDescriptor("tzValue")
	tangentPoints_ : TangentPointsPlug = PlugDescriptor("tangentPoints")
	trailColorB_ : TrailColorBPlug = PlugDescriptor("trailColorB")
	trailColorG_ : TrailColorGPlug = PlugDescriptor("trailColorG")
	trailColorR_ : TrailColorRPlug = PlugDescriptor("trailColorR")
	trailColor_ : TrailColorPlug = PlugDescriptor("trailColor")
	trailDrawMode_ : TrailDrawModePlug = PlugDescriptor("trailDrawMode")
	trailThickness_ : TrailThicknessPlug = PlugDescriptor("trailThickness")
	transformToMove_ : TransformToMovePlug = PlugDescriptor("transformToMove")
	xrayDraw_ : XrayDrawPlug = PlugDescriptor("xrayDraw")

	# node attributes

	typeName = "motionTrailShape"
	typeIdInt = 1297044296
	pass

