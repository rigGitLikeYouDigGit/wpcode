

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
	node : CharacterMap = None
	pass
class MemberPlug(Plug):
	node : CharacterMap = None
	pass
class MemberIndexPlug(Plug):
	node : CharacterMap = None
	pass
# endregion


# define node class
class CharacterMap(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	member_ : MemberPlug = PlugDescriptor("member")
	memberIndex_ : MemberIndexPlug = PlugDescriptor("memberIndex")

	# node attributes

	typeName = "characterMap"
	apiTypeInt = 803
	apiTypeStr = "kCharacterMap"
	typeIdInt = 1129136464
	MFnCls = om.MFnDependencyNode
	pass

