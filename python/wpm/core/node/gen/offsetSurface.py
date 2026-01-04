

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
class DistancePlug(Plug):
	node : OffsetSurface = None
	pass
class InputSurfacePlug(Plug):
	node : OffsetSurface = None
	pass
class MethodPlug(Plug):
	node : OffsetSurface = None
	pass
class OutputSurfacePlug(Plug):
	node : OffsetSurface = None
	pass
# endregion


# define node class
class OffsetSurface(AbstractBaseCreate):
	distance_ : DistancePlug = PlugDescriptor("distance")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	method_ : MethodPlug = PlugDescriptor("method")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")

	# node attributes

	typeName = "offsetSurface"
	apiTypeInt = 644
	apiTypeStr = "kOffsetSurface"
	typeIdInt = 1313821525
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["distance", "inputSurface", "method", "outputSurface"]
	nodeLeafPlugs = ["distance", "inputSurface", "method", "outputSurface"]
	pass

