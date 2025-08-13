

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
	node : SubdivComponentId = None
	pass
class InBasePlug(Plug):
	node : SubdivComponentId = None
	pass
class InEdgePlug(Plug):
	node : SubdivComponentId = None
	pass
class InFinalPlug(Plug):
	node : SubdivComponentId = None
	pass
class InLeftPlug(Plug):
	node : SubdivComponentId = None
	pass
class InLevelPlug(Plug):
	node : SubdivComponentId = None
	pass
class InPathPlug(Plug):
	node : SubdivComponentId = None
	pass
class InRightPlug(Plug):
	node : SubdivComponentId = None
	pass
class OutBasePlug(Plug):
	node : SubdivComponentId = None
	pass
class OutEdgePlug(Plug):
	node : SubdivComponentId = None
	pass
class OutFinalPlug(Plug):
	node : SubdivComponentId = None
	pass
class OutLeftPlug(Plug):
	node : SubdivComponentId = None
	pass
class OutLevelPlug(Plug):
	node : SubdivComponentId = None
	pass
class OutPathPlug(Plug):
	node : SubdivComponentId = None
	pass
class OutRightPlug(Plug):
	node : SubdivComponentId = None
	pass
# endregion


# define node class
class SubdivComponentId(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inBase_ : InBasePlug = PlugDescriptor("inBase")
	inEdge_ : InEdgePlug = PlugDescriptor("inEdge")
	inFinal_ : InFinalPlug = PlugDescriptor("inFinal")
	inLeft_ : InLeftPlug = PlugDescriptor("inLeft")
	inLevel_ : InLevelPlug = PlugDescriptor("inLevel")
	inPath_ : InPathPlug = PlugDescriptor("inPath")
	inRight_ : InRightPlug = PlugDescriptor("inRight")
	outBase_ : OutBasePlug = PlugDescriptor("outBase")
	outEdge_ : OutEdgePlug = PlugDescriptor("outEdge")
	outFinal_ : OutFinalPlug = PlugDescriptor("outFinal")
	outLeft_ : OutLeftPlug = PlugDescriptor("outLeft")
	outLevel_ : OutLevelPlug = PlugDescriptor("outLevel")
	outPath_ : OutPathPlug = PlugDescriptor("outPath")
	outRight_ : OutRightPlug = PlugDescriptor("outRight")

	# node attributes

	typeName = "subdivComponentId"
	typeIdInt = 1397967172
	pass

