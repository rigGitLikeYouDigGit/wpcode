

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
DimensionShape = retriever.getNodeCls("DimensionShape")
assert DimensionShape
if T.TYPE_CHECKING:
	from .. import DimensionShape

# add node doc



# region plug type defs
class NurbsGeometryPlug(Plug):
	node : NurbsDimShape = None
	pass
class UParamValuePlug(Plug):
	node : NurbsDimShape = None
	pass
class VParamValuePlug(Plug):
	node : NurbsDimShape = None
	pass
# endregion


# define node class
class NurbsDimShape(DimensionShape):
	nurbsGeometry_ : NurbsGeometryPlug = PlugDescriptor("nurbsGeometry")
	uParamValue_ : UParamValuePlug = PlugDescriptor("uParamValue")
	vParamValue_ : VParamValuePlug = PlugDescriptor("vParamValue")

	# node attributes

	typeName = "nurbsDimShape"
	typeIdInt = 1313100616
	pass

