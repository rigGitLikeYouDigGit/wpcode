

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
	nodeLeafClassAttrs = ["inputCurve", "keepRange", "outputCurve", "tolerance"]
	nodeLeafPlugs = ["inputCurve", "keepRange", "outputCurve", "tolerance"]
	pass

