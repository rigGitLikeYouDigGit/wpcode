

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	TranslateManip = Catalogue.TranslateManip
else:
	from .. import retriever
	TranslateManip = retriever.getNodeCls("TranslateManip")
	assert TranslateManip

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
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

