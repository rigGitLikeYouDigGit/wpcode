

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
TranslateManip = retriever.getNodeCls("TranslateManip")
assert TranslateManip
if T.TYPE_CHECKING:
	from .. import TranslateManip

# add node doc



# region plug type defs

# endregion


# define node class
class JointTranslateManip(TranslateManip):

	# node attributes

	typeName = "jointTranslateManip"
	apiTypeInt = 229
	apiTypeStr = "kJointTranslateManip"
	typeIdInt = 1431128660
	MFnCls = om.MFnManip3D
	pass

