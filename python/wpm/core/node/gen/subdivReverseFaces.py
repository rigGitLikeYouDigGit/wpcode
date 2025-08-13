

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
	node : SubdivReverseFaces = None
	pass
class InSubdivPlug(Plug):
	node : SubdivReverseFaces = None
	pass
class OutSubdivPlug(Plug):
	node : SubdivReverseFaces = None
	pass
class XMirrorPlug(Plug):
	node : SubdivReverseFaces = None
	pass
class YMirrorPlug(Plug):
	node : SubdivReverseFaces = None
	pass
class ZMirrorPlug(Plug):
	node : SubdivReverseFaces = None
	pass
# endregion


# define node class
class SubdivReverseFaces(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inSubdiv_ : InSubdivPlug = PlugDescriptor("inSubdiv")
	outSubdiv_ : OutSubdivPlug = PlugDescriptor("outSubdiv")
	xMirror_ : XMirrorPlug = PlugDescriptor("xMirror")
	yMirror_ : YMirrorPlug = PlugDescriptor("yMirror")
	zMirror_ : ZMirrorPlug = PlugDescriptor("zMirror")

	# node attributes

	typeName = "subdivReverseFaces"
	apiTypeInt = 816
	apiTypeStr = "kSubdivReverseFaces"
	typeIdInt = 1397904966
	MFnCls = om.MFnDependencyNode
	pass

