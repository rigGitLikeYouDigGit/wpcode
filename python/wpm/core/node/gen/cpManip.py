

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ManipContainer = Catalogue.ManipContainer
else:
	from .. import retriever
	ManipContainer = retriever.getNodeCls("ManipContainer")
	assert ManipContainer

# add node doc



# region plug type defs

# endregion


# define node class
class CpManip(ManipContainer):

	# node attributes

	typeName = "cpManip"
	apiTypeInt = 156
	apiTypeStr = "kCpManip"
	typeIdInt = 1430471504
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

