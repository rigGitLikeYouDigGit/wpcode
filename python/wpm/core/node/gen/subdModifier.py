

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	SubdBase = Catalogue.SubdBase
else:
	from .. import retriever
	SubdBase = retriever.getNodeCls("SubdBase")
	assert SubdBase

# add node doc



# region plug type defs
class CachedSubdivPlug(Plug):
	node : SubdModifier = None
	pass
class InSubdivPlug(Plug):
	node : SubdModifier = None
	pass
class InputComponentsPlug(Plug):
	node : SubdModifier = None
	pass
# endregion


# define node class
class SubdModifier(SubdBase):
	cachedSubdiv_ : CachedSubdivPlug = PlugDescriptor("cachedSubdiv")
	inSubdiv_ : InSubdivPlug = PlugDescriptor("inSubdiv")
	inputComponents_ : InputComponentsPlug = PlugDescriptor("inputComponents")

	# node attributes

	typeName = "subdModifier"
	apiTypeInt = 854
	apiTypeStr = "kSubdModifier"
	typeIdInt = 1397575492
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["cachedSubdiv", "inSubdiv", "inputComponents"]
	nodeLeafPlugs = ["cachedSubdiv", "inSubdiv", "inputComponents"]
	pass

