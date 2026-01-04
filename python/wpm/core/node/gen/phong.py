

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Reflect = Catalogue.Reflect
else:
	from .. import retriever
	Reflect = retriever.getNodeCls("Reflect")
	assert Reflect

# add node doc



# region plug type defs
class CosinePowerPlug(Plug):
	node : Phong = None
	pass
# endregion


# define node class
class Phong(Reflect):
	cosinePower_ : CosinePowerPlug = PlugDescriptor("cosinePower")

	# node attributes

	typeName = "phong"
	apiTypeInt = 374
	apiTypeStr = "kPhong"
	typeIdInt = 1380993103
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["cosinePower"]
	nodeLeafPlugs = ["cosinePower"]
	pass

