

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
	node : StrataElementOp = None
	pass
class CachingPlug(Plug):
	node : StrataElementOp = None
	pass
class FrozenPlug(Plug):
	node : StrataElementOp = None
	pass
class IsHistoricallyInterestingPlug(Plug):
	node : StrataElementOp = None
	pass
class MessagePlug(Plug):
	node : StrataElementOp = None
	pass
class NodeStatePlug(Plug):
	node : StrataElementOp = None
	pass
class StDriverExpPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StDriverWeightInPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StEdgeCurveInPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StMatchWorldSpaceInPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StNamePlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StPointDriverLocalMatrixInPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StPointWorldMatrixInPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StSpaceExpPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StElementPlug(Plug):
	stDriverExp_ : StDriverExpPlug = PlugDescriptor("stDriverExp")
	stDriverExp_ : StDriverExpPlug = PlugDescriptor("stDriverExp")
	stDriverWeightIn_ : StDriverWeightInPlug = PlugDescriptor("stDriverWeightIn")
	stInDriverWeightIn_ : StDriverWeightInPlug = PlugDescriptor("stDriverWeightIn")
	stEdgeCurveIn_ : StEdgeCurveInPlug = PlugDescriptor("stEdgeCurveIn")
	stEdgeCurveIn_ : StEdgeCurveInPlug = PlugDescriptor("stEdgeCurveIn")
	stMatchWorldSpaceIn_ : StMatchWorldSpaceInPlug = PlugDescriptor("stMatchWorldSpaceIn")
	stMatchWorldSpaceIn_ : StMatchWorldSpaceInPlug = PlugDescriptor("stMatchWorldSpaceIn")
	stName_ : StNamePlug = PlugDescriptor("stName")
	stName_ : StNamePlug = PlugDescriptor("stName")
	stPointDriverLocalMatrixIn_ : StPointDriverLocalMatrixInPlug = PlugDescriptor("stPointDriverLocalMatrixIn")
	stPointDriverLocalMatrixIn_ : StPointDriverLocalMatrixInPlug = PlugDescriptor("stPointDriverLocalMatrixIn")
	stPointWorldMatrixIn_ : StPointWorldMatrixInPlug = PlugDescriptor("stPointWorldMatrixIn")
	stPointWorldMatrixIn_ : StPointWorldMatrixInPlug = PlugDescriptor("stPointWorldMatrixIn")
	stSpaceExp_ : StSpaceExpPlug = PlugDescriptor("stSpaceExp")
	stSpaceExp_ : StSpaceExpPlug = PlugDescriptor("stSpaceExp")
	node : StrataElementOp = None
	pass
class StEdgeCurveOutPlug(Plug):
	parent : StElementOutPlug = PlugDescriptor("stElementOut")
	node : StrataElementOp = None
	pass
class StElTypeIndexPlug(Plug):
	parent : StElementOutPlug = PlugDescriptor("stElementOut")
	node : StrataElementOp = None
	pass
class StGlobalIndexPlug(Plug):
	parent : StElementOutPlug = PlugDescriptor("stElementOut")
	node : StrataElementOp = None
	pass
class StNameOutPlug(Plug):
	parent : StElementOutPlug = PlugDescriptor("stElementOut")
	node : StrataElementOp = None
	pass
class StPointDriverMatrixOutPlug(Plug):
	parent : StElementOutPlug = PlugDescriptor("stElementOut")
	node : StrataElementOp = None
	pass
class StPointFinalWorldMatrixOutPlug(Plug):
	parent : StElementOutPlug = PlugDescriptor("stElementOut")
	node : StrataElementOp = None
	pass
class StPointWeightedDriverMatrixOutPlug(Plug):
	parent : StElementOutPlug = PlugDescriptor("stElementOut")
	node : StrataElementOp = None
	pass
class StPointWeightedLocalOffsetMatrixOutPlug(Plug):
	parent : StElementOutPlug = PlugDescriptor("stElementOut")
	node : StrataElementOp = None
	pass
class StTypeOutPlug(Plug):
	parent : StElementOutPlug = PlugDescriptor("stElementOut")
	node : StrataElementOp = None
	pass
class StElementOutPlug(Plug):
	stEdgeCurveOut_ : StEdgeCurveOutPlug = PlugDescriptor("stEdgeCurveOut")
	stEdgeCurveOut_ : StEdgeCurveOutPlug = PlugDescriptor("stEdgeCurveOut")
	stElTypeIndex_ : StElTypeIndexPlug = PlugDescriptor("stElTypeIndex")
	stElTypeIndex_ : StElTypeIndexPlug = PlugDescriptor("stElTypeIndex")
	stGlobalIndex_ : StGlobalIndexPlug = PlugDescriptor("stGlobalIndex")
	stGlobalIndex_ : StGlobalIndexPlug = PlugDescriptor("stGlobalIndex")
	stNameOut_ : StNameOutPlug = PlugDescriptor("stNameOut")
	stNameOut_ : StNameOutPlug = PlugDescriptor("stNameOut")
	stPointDriverMatrixOut_ : StPointDriverMatrixOutPlug = PlugDescriptor("stPointDriverMatrixOut")
	stPointDriverMatrixOut_ : StPointDriverMatrixOutPlug = PlugDescriptor("stPointDriverMatrixOut")
	stPointFinalWorldMatrixOut_ : StPointFinalWorldMatrixOutPlug = PlugDescriptor("stPointFinalWorldMatrixOut")
	stPointFinalWorldMatrixOut_ : StPointFinalWorldMatrixOutPlug = PlugDescriptor("stPointFinalWorldMatrixOut")
	stPointWeightedDriverMatrixOut_ : StPointWeightedDriverMatrixOutPlug = PlugDescriptor("stPointWeightedDriverMatrixOut")
	stPointWeightedDriverMatrixOut_ : StPointWeightedDriverMatrixOutPlug = PlugDescriptor("stPointWeightedDriverMatrixOut")
	stPointWeightedLocalOffsetMatrixOut_ : StPointWeightedLocalOffsetMatrixOutPlug = PlugDescriptor("stPointWeightedLocalOffsetMatrixOut")
	stPointWeightedLocalOffsetMatrixOut_ : StPointWeightedLocalOffsetMatrixOutPlug = PlugDescriptor("stPointWeightedLocalOffsetMatrixOut")
	stTypeOut_ : StTypeOutPlug = PlugDescriptor("stTypeOut")
	stTypeOut_ : StTypeOutPlug = PlugDescriptor("stTypeOut")
	node : StrataElementOp = None
	pass
class StInputPlug(Plug):
	node : StrataElementOp = None
	pass
class StOpNamePlug(Plug):
	node : StrataElementOp = None
	pass
class StOpNameOutPlug(Plug):
	node : StrataElementOp = None
	pass
class StOutputPlug(Plug):
	node : StrataElementOp = None
	pass
# endregion


# define node class
class StrataElementOp(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	caching_ : CachingPlug = PlugDescriptor("caching")
	frozen_ : FrozenPlug = PlugDescriptor("frozen")
	isHistoricallyInteresting_ : IsHistoricallyInterestingPlug = PlugDescriptor("isHistoricallyInteresting")
	message_ : MessagePlug = PlugDescriptor("message")
	nodeState_ : NodeStatePlug = PlugDescriptor("nodeState")
	stDriverExp_ : StDriverExpPlug = PlugDescriptor("stDriverExp")
	stDriverWeightIn_ : StDriverWeightInPlug = PlugDescriptor("stDriverWeightIn")
	stEdgeCurveIn_ : StEdgeCurveInPlug = PlugDescriptor("stEdgeCurveIn")
	stMatchWorldSpaceIn_ : StMatchWorldSpaceInPlug = PlugDescriptor("stMatchWorldSpaceIn")
	stName_ : StNamePlug = PlugDescriptor("stName")
	stPointDriverLocalMatrixIn_ : StPointDriverLocalMatrixInPlug = PlugDescriptor("stPointDriverLocalMatrixIn")
	stPointWorldMatrixIn_ : StPointWorldMatrixInPlug = PlugDescriptor("stPointWorldMatrixIn")
	stSpaceExp_ : StSpaceExpPlug = PlugDescriptor("stSpaceExp")
	stElement_ : StElementPlug = PlugDescriptor("stElement")
	stEdgeCurveOut_ : StEdgeCurveOutPlug = PlugDescriptor("stEdgeCurveOut")
	stElTypeIndex_ : StElTypeIndexPlug = PlugDescriptor("stElTypeIndex")
	stGlobalIndex_ : StGlobalIndexPlug = PlugDescriptor("stGlobalIndex")
	stNameOut_ : StNameOutPlug = PlugDescriptor("stNameOut")
	stPointDriverMatrixOut_ : StPointDriverMatrixOutPlug = PlugDescriptor("stPointDriverMatrixOut")
	stPointFinalWorldMatrixOut_ : StPointFinalWorldMatrixOutPlug = PlugDescriptor("stPointFinalWorldMatrixOut")
	stPointWeightedDriverMatrixOut_ : StPointWeightedDriverMatrixOutPlug = PlugDescriptor("stPointWeightedDriverMatrixOut")
	stPointWeightedLocalOffsetMatrixOut_ : StPointWeightedLocalOffsetMatrixOutPlug = PlugDescriptor("stPointWeightedLocalOffsetMatrixOut")
	stTypeOut_ : StTypeOutPlug = PlugDescriptor("stTypeOut")
	stElementOut_ : StElementOutPlug = PlugDescriptor("stElementOut")
	stInput_ : StInputPlug = PlugDescriptor("stInput")
	stOpName_ : StOpNamePlug = PlugDescriptor("stOpName")
	stOpNameOut_ : StOpNameOutPlug = PlugDescriptor("stOpNameOut")
	stOutput_ : StOutputPlug = PlugDescriptor("stOutput")

	# node attributes

	typeName = "strataElementOp"
	typeIdInt = 1191074
	pass

