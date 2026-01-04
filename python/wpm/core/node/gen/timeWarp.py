

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
class ApplyPlug(Plug):
	node : TimeWarp = None
	pass
class BinMembershipPlug(Plug):
	node : TimeWarp = None
	pass
class EndFramesPlug(Plug):
	node : TimeWarp = None
	pass
class InputPlug(Plug):
	node : TimeWarp = None
	pass
class InterpTypePlug(Plug):
	node : TimeWarp = None
	pass
class OrigFramesPlug(Plug):
	node : TimeWarp = None
	pass
class OutputPlug(Plug):
	node : TimeWarp = None
	pass
# endregion


# define node class
class TimeWarp(_BASE_):
	apply_ : ApplyPlug = PlugDescriptor("apply")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	endFrames_ : EndFramesPlug = PlugDescriptor("endFrames")
	input_ : InputPlug = PlugDescriptor("input")
	interpType_ : InterpTypePlug = PlugDescriptor("interpType")
	origFrames_ : OrigFramesPlug = PlugDescriptor("origFrames")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "timeWarp"
	apiTypeInt = 1080
	apiTypeStr = "kTimeWarp"
	typeIdInt = 1414092609
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["apply", "binMembership", "endFrames", "input", "interpType", "origFrames", "output"]
	nodeLeafPlugs = ["apply", "binMembership", "endFrames", "input", "interpType", "origFrames", "output"]
	pass

