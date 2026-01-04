

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbstractBaseCreate = Catalogue.AbstractBaseCreate
else:
	from .. import retriever
	AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
	assert AbstractBaseCreate

# add node doc



# region plug type defs
class IndexPlug(Plug):
	node : SmoothCurve = None
	pass
class InputCurvePlug(Plug):
	node : SmoothCurve = None
	pass
class OutputCurvePlug(Plug):
	node : SmoothCurve = None
	pass
class SmoothnessPlug(Plug):
	node : SmoothCurve = None
	pass
# endregion


# define node class
class SmoothCurve(AbstractBaseCreate):
	index_ : IndexPlug = PlugDescriptor("index")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	smoothness_ : SmoothnessPlug = PlugDescriptor("smoothness")

	# node attributes

	typeName = "smoothCurve"
	apiTypeInt = 700
	apiTypeStr = "kSmoothCurve"
	typeIdInt = 1314082115
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["index", "inputCurve", "outputCurve", "smoothness"]
	nodeLeafPlugs = ["index", "inputCurve", "outputCurve", "smoothness"]
	pass

