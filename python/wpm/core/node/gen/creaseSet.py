

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ObjectSet = Catalogue.ObjectSet
else:
	from .. import retriever
	ObjectSet = retriever.getNodeCls("ObjectSet")
	assert ObjectSet

# add node doc



# region plug type defs
class CreaseLevelPlug(Plug):
	node : CreaseSet = None
	pass
# endregion


# define node class
class CreaseSet(ObjectSet):
	creaseLevel_ : CreaseLevelPlug = PlugDescriptor("creaseLevel")

	# node attributes

	typeName = "creaseSet"
	apiTypeInt = 1090
	apiTypeStr = "kCreaseSet"
	typeIdInt = 1129465153
	MFnCls = om.MFnSet
	nodeLeafClassAttrs = ["creaseLevel"]
	nodeLeafPlugs = ["creaseLevel"]
	pass

