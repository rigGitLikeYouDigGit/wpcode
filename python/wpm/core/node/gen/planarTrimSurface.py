

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
class DegreePlug(Plug):
	node : PlanarTrimSurface = None
	pass
class InputCurvePlug(Plug):
	node : PlanarTrimSurface = None
	pass
class KeepOutsidePlug(Plug):
	node : PlanarTrimSurface = None
	pass
class OutputSurfacePlug(Plug):
	node : PlanarTrimSurface = None
	pass
class TolerancePlug(Plug):
	node : PlanarTrimSurface = None
	pass
# endregion


# define node class
class PlanarTrimSurface(AbstractBaseCreate):
	degree_ : DegreePlug = PlugDescriptor("degree")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	keepOutside_ : KeepOutsidePlug = PlugDescriptor("keepOutside")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "planarTrimSurface"
	typeIdInt = 1313885268
	pass

