

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
	node : ClipToGhostData = None
	pass
class CharacterPlug(Plug):
	node : ClipToGhostData = None
	pass
class ClipGhostDataPlug(Plug):
	node : ClipToGhostData = None
	pass
class ClipIntermediatePosesPlug(Plug):
	node : ClipToGhostData = None
	pass
class ClipPostCyclePlug(Plug):
	node : ClipToGhostData = None
	pass
class ClipPreCyclePlug(Plug):
	node : ClipToGhostData = None
	pass
class ClipSourceEndPlug(Plug):
	node : ClipToGhostData = None
	pass
class ClipSourceStartPlug(Plug):
	node : ClipToGhostData = None
	pass
class MembersPlug(Plug):
	node : ClipToGhostData = None
	pass
# endregion


# define node class
class ClipToGhostData(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	character_ : CharacterPlug = PlugDescriptor("character")
	clipGhostData_ : ClipGhostDataPlug = PlugDescriptor("clipGhostData")
	clipIntermediatePoses_ : ClipIntermediatePosesPlug = PlugDescriptor("clipIntermediatePoses")
	clipPostCycle_ : ClipPostCyclePlug = PlugDescriptor("clipPostCycle")
	clipPreCycle_ : ClipPreCyclePlug = PlugDescriptor("clipPreCycle")
	clipSourceEnd_ : ClipSourceEndPlug = PlugDescriptor("clipSourceEnd")
	clipSourceStart_ : ClipSourceStartPlug = PlugDescriptor("clipSourceStart")
	members_ : MembersPlug = PlugDescriptor("members")

	# node attributes

	typeName = "clipToGhostData"
	apiTypeInt = 1083
	apiTypeStr = "kClipToGhostData"
	typeIdInt = 1127368516
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "character", "clipGhostData", "clipIntermediatePoses", "clipPostCycle", "clipPreCycle", "clipSourceEnd", "clipSourceStart", "members"]
	nodeLeafPlugs = ["binMembership", "character", "clipGhostData", "clipIntermediatePoses", "clipPostCycle", "clipPreCycle", "clipSourceEnd", "clipSourceStart", "members"]
	pass

