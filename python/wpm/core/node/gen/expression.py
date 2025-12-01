

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
class AnimatedPlug(Plug):
	node : Expression = None
	pass
class AttributePlug(Plug):
	node : Expression = None
	pass
class BinMembershipPlug(Plug):
	node : Expression = None
	pass
class EvaluateNowPlug(Plug):
	node : Expression = None
	pass
class ExprConnCountPlug(Plug):
	node : Expression = None
	pass
class ExpressionPlug(Plug):
	node : Expression = None
	pass
class FramePlug(Plug):
	node : Expression = None
	pass
class InputPlug(Plug):
	node : Expression = None
	pass
class InternalExpressionPlug(Plug):
	node : Expression = None
	pass
class LastTimeEvaluatedPlug(Plug):
	node : Expression = None
	pass
class NewFileFormatPlug(Plug):
	node : Expression = None
	pass
class ObjectPlug(Plug):
	node : Expression = None
	pass
class ObjectMsgPlug(Plug):
	node : Expression = None
	pass
class OutputPlug(Plug):
	node : Expression = None
	pass
class TimePlug(Plug):
	node : Expression = None
	pass
class UnitOptionPlug(Plug):
	node : Expression = None
	pass
# endregion


# define node class
class Expression(_BASE_):
	animated_ : AnimatedPlug = PlugDescriptor("animated")
	attribute_ : AttributePlug = PlugDescriptor("attribute")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	evaluateNow_ : EvaluateNowPlug = PlugDescriptor("evaluateNow")
	exprConnCount_ : ExprConnCountPlug = PlugDescriptor("exprConnCount")
	expression_ : ExpressionPlug = PlugDescriptor("expression")
	frame_ : FramePlug = PlugDescriptor("frame")
	input_ : InputPlug = PlugDescriptor("input")
	internalExpression_ : InternalExpressionPlug = PlugDescriptor("internalExpression")
	lastTimeEvaluated_ : LastTimeEvaluatedPlug = PlugDescriptor("lastTimeEvaluated")
	newFileFormat_ : NewFileFormatPlug = PlugDescriptor("newFileFormat")
	object_ : ObjectPlug = PlugDescriptor("object")
	objectMsg_ : ObjectMsgPlug = PlugDescriptor("objectMsg")
	output_ : OutputPlug = PlugDescriptor("output")
	time_ : TimePlug = PlugDescriptor("time")
	unitOption_ : UnitOptionPlug = PlugDescriptor("unitOption")

	# node attributes

	typeName = "expression"
	apiTypeInt = 327
	apiTypeStr = "kExpression"
	typeIdInt = 1145395280
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["animated", "attribute", "binMembership", "evaluateNow", "exprConnCount", "expression", "frame", "input", "internalExpression", "lastTimeEvaluated", "newFileFormat", "object", "objectMsg", "output", "time", "unitOption"]
	nodeLeafPlugs = ["animated", "attribute", "binMembership", "evaluateNow", "exprConnCount", "expression", "frame", "input", "internalExpression", "lastTimeEvaluated", "newFileFormat", "object", "objectMsg", "output", "time", "unitOption"]
	pass

