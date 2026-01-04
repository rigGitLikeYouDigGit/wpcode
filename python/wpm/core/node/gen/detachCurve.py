

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
	node : DetachCurve = None
	pass
class KeepPlug(Plug):
	node : DetachCurve = None
	pass
class OutputCurvePlug(Plug):
	node : DetachCurve = None
	pass
class ParameterPlug(Plug):
	node : DetachCurve = None
	pass
# endregion


# define node class
class DetachCurve(AbstractBaseCreate):
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	keep_ : KeepPlug = PlugDescriptor("keep")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	parameter_ : ParameterPlug = PlugDescriptor("parameter")

	# node attributes

	typeName = "detachCurve"
	apiTypeInt = 63
	apiTypeStr = "kDetachCurve"
	typeIdInt = 1313100867
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["inputCurve", "keep", "outputCurve", "parameter"]
	nodeLeafPlugs = ["inputCurve", "keep", "outputCurve", "parameter"]
	pass

