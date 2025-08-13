

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
class AfterPlug(Plug):
	node : Script = None
	pass
class BeforePlug(Plug):
	node : Script = None
	pass
class BinMembershipPlug(Plug):
	node : Script = None
	pass
class IgnoreReferenceEditsPlug(Plug):
	node : Script = None
	pass
class ScriptTypePlug(Plug):
	node : Script = None
	pass
class SourceTypePlug(Plug):
	node : Script = None
	pass
# endregion


# define node class
class Script(_BASE_):
	after_ : AfterPlug = PlugDescriptor("after")
	before_ : BeforePlug = PlugDescriptor("before")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	ignoreReferenceEdits_ : IgnoreReferenceEditsPlug = PlugDescriptor("ignoreReferenceEdits")
	scriptType_ : ScriptTypePlug = PlugDescriptor("scriptType")
	sourceType_ : SourceTypePlug = PlugDescriptor("sourceType")

	# node attributes

	typeName = "script"
	apiTypeInt = 639
	apiTypeStr = "kScript"
	typeIdInt = 1396920912
	MFnCls = om.MFnDependencyNode
	pass

