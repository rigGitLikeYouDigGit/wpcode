

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
class BinMembershipPlug(Plug):
	node : TimeEditorClipBase = None
	pass
class ClipBlendModePlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipColorBPlug(Plug):
	parent : ClipColorPlug = PlugDescriptor("clipColor")
	node : TimeEditorClipBase = None
	pass
class ClipColorGPlug(Plug):
	parent : ClipColorPlug = PlugDescriptor("clipColor")
	node : TimeEditorClipBase = None
	pass
class ClipColorRPlug(Plug):
	parent : ClipColorPlug = PlugDescriptor("clipColor")
	node : TimeEditorClipBase = None
	pass
class ClipColorPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	clipColorB_ : ClipColorBPlug = PlugDescriptor("clipColorB")
	ccb_ : ClipColorBPlug = PlugDescriptor("clipColorB")
	clipColorG_ : ClipColorGPlug = PlugDescriptor("clipColorG")
	ccg_ : ClipColorGPlug = PlugDescriptor("clipColorG")
	clipColorR_ : ClipColorRPlug = PlugDescriptor("clipColorR")
	ccr_ : ClipColorRPlug = PlugDescriptor("clipColorR")
	node : TimeEditorClipBase = None
	pass
class ClipDurationPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipEvaluationDataPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipHoldAfterPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipHoldBeforePlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipLoopAfterPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipLoopAfterModePlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipLoopBeforePlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipLoopBeforeModePlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipMutedPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipNamePlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipParentPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipScalePlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipStartPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipTypePlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipidPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class CurveStartPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class LocalTimePlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ParentTimePlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class SpeedInputPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class TimeWarpTypePlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class TimeWarpedPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class UseClipColorPlug(Plug):
	parent : ClipPlug = PlugDescriptor("clip")
	node : TimeEditorClipBase = None
	pass
class ClipPlug(Plug):
	clipBlendMode_ : ClipBlendModePlug = PlugDescriptor("clipBlendMode")
	cbm_ : ClipBlendModePlug = PlugDescriptor("clipBlendMode")
	clipColor_ : ClipColorPlug = PlugDescriptor("clipColor")
	cc_ : ClipColorPlug = PlugDescriptor("clipColor")
	clipDuration_ : ClipDurationPlug = PlugDescriptor("clipDuration")
	cpd_ : ClipDurationPlug = PlugDescriptor("clipDuration")
	clipEvaluationData_ : ClipEvaluationDataPlug = PlugDescriptor("clipEvaluationData")
	ced_ : ClipEvaluationDataPlug = PlugDescriptor("clipEvaluationData")
	clipHoldAfter_ : ClipHoldAfterPlug = PlugDescriptor("clipHoldAfter")
	cha_ : ClipHoldAfterPlug = PlugDescriptor("clipHoldAfter")
	clipHoldBefore_ : ClipHoldBeforePlug = PlugDescriptor("clipHoldBefore")
	chb_ : ClipHoldBeforePlug = PlugDescriptor("clipHoldBefore")
	clipLoopAfter_ : ClipLoopAfterPlug = PlugDescriptor("clipLoopAfter")
	cla_ : ClipLoopAfterPlug = PlugDescriptor("clipLoopAfter")
	clipLoopAfterMode_ : ClipLoopAfterModePlug = PlugDescriptor("clipLoopAfterMode")
	clam_ : ClipLoopAfterModePlug = PlugDescriptor("clipLoopAfterMode")
	clipLoopBefore_ : ClipLoopBeforePlug = PlugDescriptor("clipLoopBefore")
	clb_ : ClipLoopBeforePlug = PlugDescriptor("clipLoopBefore")
	clipLoopBeforeMode_ : ClipLoopBeforeModePlug = PlugDescriptor("clipLoopBeforeMode")
	clbm_ : ClipLoopBeforeModePlug = PlugDescriptor("clipLoopBeforeMode")
	clipMuted_ : ClipMutedPlug = PlugDescriptor("clipMuted")
	cm_ : ClipMutedPlug = PlugDescriptor("clipMuted")
	clipName_ : ClipNamePlug = PlugDescriptor("clipName")
	cn_ : ClipNamePlug = PlugDescriptor("clipName")
	clipParent_ : ClipParentPlug = PlugDescriptor("clipParent")
	cprn_ : ClipParentPlug = PlugDescriptor("clipParent")
	clipScale_ : ClipScalePlug = PlugDescriptor("clipScale")
	cscl_ : ClipScalePlug = PlugDescriptor("clipScale")
	clipStart_ : ClipStartPlug = PlugDescriptor("clipStart")
	cst_ : ClipStartPlug = PlugDescriptor("clipStart")
	clipType_ : ClipTypePlug = PlugDescriptor("clipType")
	ct_ : ClipTypePlug = PlugDescriptor("clipType")
	clipid_ : ClipidPlug = PlugDescriptor("clipid")
	cid_ : ClipidPlug = PlugDescriptor("clipid")
	curveStart_ : CurveStartPlug = PlugDescriptor("curveStart")
	cvst_ : CurveStartPlug = PlugDescriptor("curveStart")
	localTime_ : LocalTimePlug = PlugDescriptor("localTime")
	clt_ : LocalTimePlug = PlugDescriptor("localTime")
	parentTime_ : ParentTimePlug = PlugDescriptor("parentTime")
	cpt_ : ParentTimePlug = PlugDescriptor("parentTime")
	speedInput_ : SpeedInputPlug = PlugDescriptor("speedInput")
	sin_ : SpeedInputPlug = PlugDescriptor("speedInput")
	timeWarpType_ : TimeWarpTypePlug = PlugDescriptor("timeWarpType")
	twt_ : TimeWarpTypePlug = PlugDescriptor("timeWarpType")
	timeWarped_ : TimeWarpedPlug = PlugDescriptor("timeWarped")
	tw_ : TimeWarpedPlug = PlugDescriptor("timeWarped")
	useClipColor_ : UseClipColorPlug = PlugDescriptor("useClipColor")
	ucc_ : UseClipColorPlug = PlugDescriptor("useClipColor")
	node : TimeEditorClipBase = None
	pass
class LastEvaluationTimePlug(Plug):
	node : TimeEditorClipBase = None
	pass
class MatchObjPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : TimeEditorClipBase = None
	pass
class MatchTimePlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : TimeEditorClipBase = None
	pass
class MatchclipPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : TimeEditorClipBase = None
	pass
class OffsetModePlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : TimeEditorClipBase = None
	pass
class OffsetMtxPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : TimeEditorClipBase = None
	pass
class PivotMtxPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : TimeEditorClipBase = None
	pass
class RootsPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	rootObj_ : RootObjPlug = PlugDescriptor("rootObj")
	rob_ : RootObjPlug = PlugDescriptor("rootObj")
	rootObjLocalXform_ : RootObjLocalXformPlug = PlugDescriptor("rootObjLocalXform")
	rolx_ : RootObjLocalXformPlug = PlugDescriptor("rootObjLocalXform")
	rootObjParentXform_ : RootObjParentXformPlug = PlugDescriptor("rootObjParentXform")
	ropx_ : RootObjParentXformPlug = PlugDescriptor("rootObjParentXform")
	node : TimeEditorClipBase = None
	pass
class OffsetPlug(Plug):
	matchObj_ : MatchObjPlug = PlugDescriptor("matchObj")
	mob_ : MatchObjPlug = PlugDescriptor("matchObj")
	matchTime_ : MatchTimePlug = PlugDescriptor("matchTime")
	mtm_ : MatchTimePlug = PlugDescriptor("matchTime")
	matchclip_ : MatchclipPlug = PlugDescriptor("matchclip")
	mcl_ : MatchclipPlug = PlugDescriptor("matchclip")
	offsetMode_ : OffsetModePlug = PlugDescriptor("offsetMode")
	ofm_ : OffsetModePlug = PlugDescriptor("offsetMode")
	offsetMtx_ : OffsetMtxPlug = PlugDescriptor("offsetMtx")
	omt_ : OffsetMtxPlug = PlugDescriptor("offsetMtx")
	pivotMtx_ : PivotMtxPlug = PlugDescriptor("pivotMtx")
	pmt_ : PivotMtxPlug = PlugDescriptor("pivotMtx")
	roots_ : RootsPlug = PlugDescriptor("roots")
	rts_ : RootsPlug = PlugDescriptor("roots")
	node : TimeEditorClipBase = None
	pass
class RootObjPlug(Plug):
	parent : RootsPlug = PlugDescriptor("roots")
	node : TimeEditorClipBase = None
	pass
class RootObjLocalXformPlug(Plug):
	parent : RootsPlug = PlugDescriptor("roots")
	node : TimeEditorClipBase = None
	pass
class RootObjParentXformPlug(Plug):
	parent : RootsPlug = PlugDescriptor("roots")
	node : TimeEditorClipBase = None
	pass
# endregion


# define node class
class TimeEditorClipBase(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	clipBlendMode_ : ClipBlendModePlug = PlugDescriptor("clipBlendMode")
	clipColorB_ : ClipColorBPlug = PlugDescriptor("clipColorB")
	clipColorG_ : ClipColorGPlug = PlugDescriptor("clipColorG")
	clipColorR_ : ClipColorRPlug = PlugDescriptor("clipColorR")
	clipColor_ : ClipColorPlug = PlugDescriptor("clipColor")
	clipDuration_ : ClipDurationPlug = PlugDescriptor("clipDuration")
	clipEvaluationData_ : ClipEvaluationDataPlug = PlugDescriptor("clipEvaluationData")
	clipHoldAfter_ : ClipHoldAfterPlug = PlugDescriptor("clipHoldAfter")
	clipHoldBefore_ : ClipHoldBeforePlug = PlugDescriptor("clipHoldBefore")
	clipLoopAfter_ : ClipLoopAfterPlug = PlugDescriptor("clipLoopAfter")
	clipLoopAfterMode_ : ClipLoopAfterModePlug = PlugDescriptor("clipLoopAfterMode")
	clipLoopBefore_ : ClipLoopBeforePlug = PlugDescriptor("clipLoopBefore")
	clipLoopBeforeMode_ : ClipLoopBeforeModePlug = PlugDescriptor("clipLoopBeforeMode")
	clipMuted_ : ClipMutedPlug = PlugDescriptor("clipMuted")
	clipName_ : ClipNamePlug = PlugDescriptor("clipName")
	clipParent_ : ClipParentPlug = PlugDescriptor("clipParent")
	clipScale_ : ClipScalePlug = PlugDescriptor("clipScale")
	clipStart_ : ClipStartPlug = PlugDescriptor("clipStart")
	clipType_ : ClipTypePlug = PlugDescriptor("clipType")
	clipid_ : ClipidPlug = PlugDescriptor("clipid")
	curveStart_ : CurveStartPlug = PlugDescriptor("curveStart")
	localTime_ : LocalTimePlug = PlugDescriptor("localTime")
	parentTime_ : ParentTimePlug = PlugDescriptor("parentTime")
	speedInput_ : SpeedInputPlug = PlugDescriptor("speedInput")
	timeWarpType_ : TimeWarpTypePlug = PlugDescriptor("timeWarpType")
	timeWarped_ : TimeWarpedPlug = PlugDescriptor("timeWarped")
	useClipColor_ : UseClipColorPlug = PlugDescriptor("useClipColor")
	clip_ : ClipPlug = PlugDescriptor("clip")
	lastEvaluationTime_ : LastEvaluationTimePlug = PlugDescriptor("lastEvaluationTime")
	matchObj_ : MatchObjPlug = PlugDescriptor("matchObj")
	matchTime_ : MatchTimePlug = PlugDescriptor("matchTime")
	matchclip_ : MatchclipPlug = PlugDescriptor("matchclip")
	offsetMode_ : OffsetModePlug = PlugDescriptor("offsetMode")
	offsetMtx_ : OffsetMtxPlug = PlugDescriptor("offsetMtx")
	pivotMtx_ : PivotMtxPlug = PlugDescriptor("pivotMtx")
	roots_ : RootsPlug = PlugDescriptor("roots")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	rootObj_ : RootObjPlug = PlugDescriptor("rootObj")
	rootObjLocalXform_ : RootObjLocalXformPlug = PlugDescriptor("rootObjLocalXform")
	rootObjParentXform_ : RootObjParentXformPlug = PlugDescriptor("rootObjParentXform")

	# node attributes

	typeName = "timeEditorClipBase"
	apiTypeInt = 1103
	apiTypeStr = "kTimeEditorClipBase"
	typeIdInt = 1095517004
	MFnCls = om.MFnDependencyNode
	pass

