

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
class AutomaticWeightPlug(Plug):
	node : AvgCurves = None
	pass
class InputCurve1Plug(Plug):
	node : AvgCurves = None
	pass
class InputCurve2Plug(Plug):
	node : AvgCurves = None
	pass
class NormalizeWeightsPlug(Plug):
	node : AvgCurves = None
	pass
class OutputCurvePlug(Plug):
	node : AvgCurves = None
	pass
class Weight1Plug(Plug):
	node : AvgCurves = None
	pass
class Weight2Plug(Plug):
	node : AvgCurves = None
	pass
# endregion


# define node class
class AvgCurves(AbstractBaseCreate):
	automaticWeight_ : AutomaticWeightPlug = PlugDescriptor("automaticWeight")
	inputCurve1_ : InputCurve1Plug = PlugDescriptor("inputCurve1")
	inputCurve2_ : InputCurve2Plug = PlugDescriptor("inputCurve2")
	normalizeWeights_ : NormalizeWeightsPlug = PlugDescriptor("normalizeWeights")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	weight1_ : Weight1Plug = PlugDescriptor("weight1")
	weight2_ : Weight2Plug = PlugDescriptor("weight2")

	# node attributes

	typeName = "avgCurves"
	apiTypeInt = 45
	apiTypeStr = "kAvgCurves"
	typeIdInt = 1312899922
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["automaticWeight", "inputCurve1", "inputCurve2", "normalizeWeights", "outputCurve", "weight1", "weight2"]
	nodeLeafPlugs = ["automaticWeight", "inputCurve1", "inputCurve2", "normalizeWeights", "outputCurve", "weight1", "weight2"]
	pass

