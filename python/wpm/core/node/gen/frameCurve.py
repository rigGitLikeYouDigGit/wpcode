

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
class ActiveUpMatrixPlug(Plug):
	parent : ActiveUpPlug = PlugDescriptor("activeUp")
	node : FrameCurve = None
	pass
class ActiveUpPlug(Plug):
	activeUpMatrix_ : ActiveUpMatrixPlug = PlugDescriptor("activeUpMatrix")
	activeUpMatrix_ : ActiveUpMatrixPlug = PlugDescriptor("activeUpMatrix")
	node : FrameCurve = None
	pass
class BinMembershipPlug(Plug):
	node : FrameCurve = None
	pass
class CachingPlug(Plug):
	node : FrameCurve = None
	pass
class CurveInPlug(Plug):
	node : FrameCurve = None
	pass
class FrozenPlug(Plug):
	node : FrameCurve = None
	pass
class IsHistoricallyInterestingPlug(Plug):
	node : FrameCurve = None
	pass
class MessagePlug(Plug):
	node : FrameCurve = None
	pass
class NodeStatePlug(Plug):
	node : FrameCurve = None
	pass
class RefCurveInPlug(Plug):
	node : FrameCurve = None
	pass
class RefUpMatrixPlug(Plug):
	parent : RefUpPlug = PlugDescriptor("refUp")
	node : FrameCurve = None
	pass
class RefUpPlug(Plug):
	refUpMatrix_ : RefUpMatrixPlug = PlugDescriptor("refUpMatrix")
	refUpMatrix_ : RefUpMatrixPlug = PlugDescriptor("refUpMatrix")
	node : FrameCurve = None
	pass
class ActiveSampleMatrixInPlug(Plug):
	parent : SampleInPlug = PlugDescriptor("sampleIn")
	node : FrameCurve = None
	pass
class RefSampleMatrixInPlug(Plug):
	parent : SampleInPlug = PlugDescriptor("sampleIn")
	node : FrameCurve = None
	pass
class SampleUInPlug(Plug):
	parent : SampleInPlug = PlugDescriptor("sampleIn")
	node : FrameCurve = None
	pass
class SampleInPlug(Plug):
	activeSampleMatrixIn_ : ActiveSampleMatrixInPlug = PlugDescriptor("activeSampleMatrixIn")
	activeSampleMatrixIn_ : ActiveSampleMatrixInPlug = PlugDescriptor("activeSampleMatrixIn")
	refSampleMatrixIn_ : RefSampleMatrixInPlug = PlugDescriptor("refSampleMatrixIn")
	refSampleMatrixIn_ : RefSampleMatrixInPlug = PlugDescriptor("refSampleMatrixIn")
	sampleUIn_ : SampleUInPlug = PlugDescriptor("sampleUIn")
	sampleUIn_ : SampleUInPlug = PlugDescriptor("sampleUIn")
	node : FrameCurve = None
	pass
class SampleMatrixOnCurveOutPlug(Plug):
	parent : SampleOutPlug = PlugDescriptor("sampleOut")
	node : FrameCurve = None
	pass
class SampleMatrixOutPlug(Plug):
	parent : SampleOutPlug = PlugDescriptor("sampleOut")
	node : FrameCurve = None
	pass
class SampleUOutPlug(Plug):
	parent : SampleOutPlug = PlugDescriptor("sampleOut")
	node : FrameCurve = None
	pass
class SampleOutPlug(Plug):
	sampleMatrixOnCurveOut_ : SampleMatrixOnCurveOutPlug = PlugDescriptor("sampleMatrixOnCurveOut")
	sampleMatrixOnCurveOut_ : SampleMatrixOnCurveOutPlug = PlugDescriptor("sampleMatrixOnCurveOut")
	sampleMatrixOut_ : SampleMatrixOutPlug = PlugDescriptor("sampleMatrixOut")
	sampleMatrixOut_ : SampleMatrixOutPlug = PlugDescriptor("sampleMatrixOut")
	sampleUOut_ : SampleUOutPlug = PlugDescriptor("sampleUOut")
	sampleUOut_ : SampleUOutPlug = PlugDescriptor("sampleUOut")
	node : FrameCurve = None
	pass
class StepsPlug(Plug):
	node : FrameCurve = None
	pass
# endregion


# define node class
class FrameCurve(_BASE_):
	activeUpMatrix_ : ActiveUpMatrixPlug = PlugDescriptor("activeUpMatrix")
	activeUp_ : ActiveUpPlug = PlugDescriptor("activeUp")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	caching_ : CachingPlug = PlugDescriptor("caching")
	curveIn_ : CurveInPlug = PlugDescriptor("curveIn")
	frozen_ : FrozenPlug = PlugDescriptor("frozen")
	isHistoricallyInteresting_ : IsHistoricallyInterestingPlug = PlugDescriptor("isHistoricallyInteresting")
	message_ : MessagePlug = PlugDescriptor("message")
	nodeState_ : NodeStatePlug = PlugDescriptor("nodeState")
	refCurveIn_ : RefCurveInPlug = PlugDescriptor("refCurveIn")
	refUpMatrix_ : RefUpMatrixPlug = PlugDescriptor("refUpMatrix")
	refUp_ : RefUpPlug = PlugDescriptor("refUp")
	activeSampleMatrixIn_ : ActiveSampleMatrixInPlug = PlugDescriptor("activeSampleMatrixIn")
	refSampleMatrixIn_ : RefSampleMatrixInPlug = PlugDescriptor("refSampleMatrixIn")
	sampleUIn_ : SampleUInPlug = PlugDescriptor("sampleUIn")
	sampleIn_ : SampleInPlug = PlugDescriptor("sampleIn")
	sampleMatrixOnCurveOut_ : SampleMatrixOnCurveOutPlug = PlugDescriptor("sampleMatrixOnCurveOut")
	sampleMatrixOut_ : SampleMatrixOutPlug = PlugDescriptor("sampleMatrixOut")
	sampleUOut_ : SampleUOutPlug = PlugDescriptor("sampleUOut")
	sampleOut_ : SampleOutPlug = PlugDescriptor("sampleOut")
	steps_ : StepsPlug = PlugDescriptor("steps")

	# node attributes

	typeName = "frameCurve"
	typeIdInt = 1191586
	pass

