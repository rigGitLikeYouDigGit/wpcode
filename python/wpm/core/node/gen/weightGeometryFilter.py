

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	GeometryFilter = Catalogue.GeometryFilter
else:
	from .. import retriever
	GeometryFilter = retriever.getNodeCls("GeometryFilter")
	assert GeometryFilter

# add node doc



# region plug type defs
class WeightsPlug(Plug):
	parent : WeightListPlug = PlugDescriptor("weightList")
	node : WeightGeometryFilter = None
	pass
class WeightListPlug(Plug):
	weights_ : WeightsPlug = PlugDescriptor("weights")
	w_ : WeightsPlug = PlugDescriptor("weights")
	node : WeightGeometryFilter = None
	pass
# endregion


# define node class
class WeightGeometryFilter(GeometryFilter):
	weights_ : WeightsPlug = PlugDescriptor("weights")
	weightList_ : WeightListPlug = PlugDescriptor("weightList")

	# node attributes

	typeName = "weightGeometryFilter"
	typeIdInt = 1146570566
	nodeLeafClassAttrs = ["weights", "weightList"]
	nodeLeafPlugs = ["weightList"]
	pass

