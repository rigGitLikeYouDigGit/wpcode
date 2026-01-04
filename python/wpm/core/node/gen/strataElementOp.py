

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
class StEdgeCurveOutPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StElTypeIndexPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StGlobalIndexPlug(Plug):
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
class StParentExpPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StPointDriverLocalMatrixInPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StPointDriverMatrixOutPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StPointFinalWorldMatrixOutPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StPointWeightedDriverMatrixOutPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StPointWeightedLocalOffsetMatrixOutPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StPointWorldMatrixInPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StTypeOutPlug(Plug):
	parent : StElementPlug = PlugDescriptor("stElement")
	node : StrataElementOp = None
	pass
class StElementPlug(Plug):
	stDriverExp_ : StDriverExpPlug = PlugDescriptor("stDriverExp")
	stDriverExp_ : StDriverExpPlug = PlugDescriptor("stDriverExp")
	stDriverWeightIn_ : StDriverWeightInPlug = PlugDescriptor("stDriverWeightIn")
	stInDriverWeightIn_ : StDriverWeightInPlug = PlugDescriptor("stDriverWeightIn")
	stEdgeCurveOut_ : StEdgeCurveOutPlug = PlugDescriptor("stEdgeCurveOut")
	stEdgeCurveOut_ : StEdgeCurveOutPlug = PlugDescriptor("stEdgeCurveOut")
	stElTypeIndex_ : StElTypeIndexPlug = PlugDescriptor("stElTypeIndex")
	stElTypeIndex_ : StElTypeIndexPlug = PlugDescriptor("stElTypeIndex")
	stGlobalIndex_ : StGlobalIndexPlug = PlugDescriptor("stGlobalIndex")
	stGlobalIndex_ : StGlobalIndexPlug = PlugDescriptor("stGlobalIndex")
	stMatchWorldSpaceIn_ : StMatchWorldSpaceInPlug = PlugDescriptor("stMatchWorldSpaceIn")
	stMatchWorldSpaceIn_ : StMatchWorldSpaceInPlug = PlugDescriptor("stMatchWorldSpaceIn")
	stName_ : StNamePlug = PlugDescriptor("stName")
	stName_ : StNamePlug = PlugDescriptor("stName")
	stParentExp_ : StParentExpPlug = PlugDescriptor("stParentExp")
	stParentExp_ : StParentExpPlug = PlugDescriptor("stParentExp")
	stPointDriverLocalMatrixIn_ : StPointDriverLocalMatrixInPlug = PlugDescriptor("stPointDriverLocalMatrixIn")
	stPointDriverLocalMatrixIn_ : StPointDriverLocalMatrixInPlug = PlugDescriptor("stPointDriverLocalMatrixIn")
	stPointDriverMatrixOut_ : StPointDriverMatrixOutPlug = PlugDescriptor("stPointDriverMatrixOut")
	stPointDriverMatrixOut_ : StPointDriverMatrixOutPlug = PlugDescriptor("stPointDriverMatrixOut")
	stPointFinalWorldMatrixOut_ : StPointFinalWorldMatrixOutPlug = PlugDescriptor("stPointFinalWorldMatrixOut")
	stPointFinalWorldMatrixOut_ : StPointFinalWorldMatrixOutPlug = PlugDescriptor("stPointFinalWorldMatrixOut")
	stPointWeightedDriverMatrixOut_ : StPointWeightedDriverMatrixOutPlug = PlugDescriptor("stPointWeightedDriverMatrixOut")
	stPointWeightedDriverMatrixOut_ : StPointWeightedDriverMatrixOutPlug = PlugDescriptor("stPointWeightedDriverMatrixOut")
	stPointWeightedLocalOffsetMatrixOut_ : StPointWeightedLocalOffsetMatrixOutPlug = PlugDescriptor("stPointWeightedLocalOffsetMatrixOut")
	stPointWeightedLocalOffsetMatrixOut_ : StPointWeightedLocalOffsetMatrixOutPlug = PlugDescriptor("stPointWeightedLocalOffsetMatrixOut")
	stPointWorldMatrixIn_ : StPointWorldMatrixInPlug = PlugDescriptor("stPointWorldMatrixIn")
	stPointWorldMatrixIn_ : StPointWorldMatrixInPlug = PlugDescriptor("stPointWorldMatrixIn")
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
	stDriverExp_ : StDriverExpPlug = PlugDescriptor("stDriverExp")
	stDriverWeightIn_ : StDriverWeightInPlug = PlugDescriptor("stDriverWeightIn")
	stEdgeCurveOut_ : StEdgeCurveOutPlug = PlugDescriptor("stEdgeCurveOut")
	stElTypeIndex_ : StElTypeIndexPlug = PlugDescriptor("stElTypeIndex")
	stGlobalIndex_ : StGlobalIndexPlug = PlugDescriptor("stGlobalIndex")
	stMatchWorldSpaceIn_ : StMatchWorldSpaceInPlug = PlugDescriptor("stMatchWorldSpaceIn")
	stName_ : StNamePlug = PlugDescriptor("stName")
	stParentExp_ : StParentExpPlug = PlugDescriptor("stParentExp")
	stPointDriverLocalMatrixIn_ : StPointDriverLocalMatrixInPlug = PlugDescriptor("stPointDriverLocalMatrixIn")
	stPointDriverMatrixOut_ : StPointDriverMatrixOutPlug = PlugDescriptor("stPointDriverMatrixOut")
	stPointFinalWorldMatrixOut_ : StPointFinalWorldMatrixOutPlug = PlugDescriptor("stPointFinalWorldMatrixOut")
	stPointWeightedDriverMatrixOut_ : StPointWeightedDriverMatrixOutPlug = PlugDescriptor("stPointWeightedDriverMatrixOut")
	stPointWeightedLocalOffsetMatrixOut_ : StPointWeightedLocalOffsetMatrixOutPlug = PlugDescriptor("stPointWeightedLocalOffsetMatrixOut")
	stPointWorldMatrixIn_ : StPointWorldMatrixInPlug = PlugDescriptor("stPointWorldMatrixIn")
	stTypeOut_ : StTypeOutPlug = PlugDescriptor("stTypeOut")
	stElement_ : StElementPlug = PlugDescriptor("stElement")
	stInput_ : StInputPlug = PlugDescriptor("stInput")
	stOpName_ : StOpNamePlug = PlugDescriptor("stOpName")
	stOpNameOut_ : StOpNameOutPlug = PlugDescriptor("stOpNameOut")
	stOutput_ : StOutputPlug = PlugDescriptor("stOutput")

	# node attributes

	typeName = "strataElementOp"
	typeIdInt = 1191074
	nodeLeafClassAttrs = ["binMembership", "stDriverExp", "stDriverWeightIn", "stEdgeCurveOut", "stElTypeIndex", "stGlobalIndex", "stMatchWorldSpaceIn", "stName", "stParentExp", "stPointDriverLocalMatrixIn", "stPointDriverMatrixOut", "stPointFinalWorldMatrixOut", "stPointWeightedDriverMatrixOut", "stPointWeightedLocalOffsetMatrixOut", "stPointWorldMatrixIn", "stTypeOut", "stElement", "stInput", "stOpName", "stOpNameOut", "stOutput"]
	nodeLeafPlugs = ["binMembership", "stElement", "stInput", "stOpName", "stOpNameOut", "stOutput"]
	pass

