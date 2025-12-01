

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ScriptManip = retriever.getNodeCls("ScriptManip")
assert ScriptManip
if T.TYPE_CHECKING:
	from .. import ScriptManip

# add node doc



# region plug type defs

# endregion


# define node class
class TextButtonManip(ScriptManip):

	# node attributes

	typeName = "textButtonManip"
	apiTypeInt = 651
	apiTypeStr = "kTextButtonManip"
	typeIdInt = 1431131202
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

