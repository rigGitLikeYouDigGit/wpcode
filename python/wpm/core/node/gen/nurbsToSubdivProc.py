

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
NurbsToSubdiv = retriever.getNodeCls("NurbsToSubdiv")
assert NurbsToSubdiv
if T.TYPE_CHECKING:
	from .. import NurbsToSubdiv

# add node doc



# region plug type defs
class BridgePlug(Plug):
	node : NurbsToSubdivProc = None
	pass
class BridgeInUPlug(Plug):
	node : NurbsToSubdivProc = None
	pass
class BridgeInVPlug(Plug):
	node : NurbsToSubdivProc = None
	pass
class CapTypePlug(Plug):
	node : NurbsToSubdivProc = None
	pass
class OffsetPlug(Plug):
	node : NurbsToSubdivProc = None
	pass
class SolidTypePlug(Plug):
	node : NurbsToSubdivProc = None
	pass
class TransformPlug(Plug):
	node : NurbsToSubdivProc = None
	pass
# endregion


# define node class
class NurbsToSubdivProc(NurbsToSubdiv):
	bridge_ : BridgePlug = PlugDescriptor("bridge")
	bridgeInU_ : BridgeInUPlug = PlugDescriptor("bridgeInU")
	bridgeInV_ : BridgeInVPlug = PlugDescriptor("bridgeInV")
	capType_ : CapTypePlug = PlugDescriptor("capType")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	solidType_ : SolidTypePlug = PlugDescriptor("solidType")
	transform_ : TransformPlug = PlugDescriptor("transform")

	# node attributes

	typeName = "nurbsToSubdivProc"
	typeIdInt = 1397642320
	pass

