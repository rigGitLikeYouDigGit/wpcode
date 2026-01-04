

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	FreePointTriadManip = Catalogue.FreePointTriadManip
else:
	from .. import retriever
	FreePointTriadManip = retriever.getNodeCls("FreePointTriadManip")
	assert FreePointTriadManip

# add node doc



# region plug type defs

# endregion


# define node class
class PolySplitToolManip1(FreePointTriadManip):

	# node attributes

	typeName = "polySplitToolManip1"
	typeIdInt = 1431327537
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

