

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class InputCurvePlug(Plug):
	node : FitBspline = None
	pass
class KeepRangePlug(Plug):
	node : FitBspline = None
	pass
class OutputCurvePlug(Plug):
	node : FitBspline = None
	pass
class TolerancePlug(Plug):
	node : FitBspline = None
	pass
# endregion


# define node class
class FitBspline(AbstractBaseCreate):
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	keepRange_ : KeepRangePlug = PlugDescriptor("keepRange")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "fitBspline"
	apiTypeInt = 71
	apiTypeStr = "kFitBspline"
	typeIdInt = 1313231939
	MFnCls = om.MFnDependencyNode
	pass

