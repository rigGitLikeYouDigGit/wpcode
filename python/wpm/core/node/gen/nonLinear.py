

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
WeightGeometryFilter = retriever.getNodeCls("WeightGeometryFilter")
assert WeightGeometryFilter
if T.TYPE_CHECKING:
	from .. import WeightGeometryFilter

# add node doc



# region plug type defs
class CacheSetupPlug(Plug):
	node : NonLinear = None
	pass
class DeformerDataPlug(Plug):
	node : NonLinear = None
	pass
class MatrixPlug(Plug):
	node : NonLinear = None
	pass
# endregion


# define node class
class NonLinear(WeightGeometryFilter):
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	deformerData_ : DeformerDataPlug = PlugDescriptor("deformerData")
	matrix_ : MatrixPlug = PlugDescriptor("matrix")

	# node attributes

	typeName = "nonLinear"
	apiTypeInt = 623
	apiTypeStr = "kNonLinear"
	typeIdInt = 1179536452
	MFnCls = om.MFnGeometryFilter
	pass

