

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
class InputRail1Plug(Plug):
	node : BirailSrf = None
	pass
class InputRail2Plug(Plug):
	node : BirailSrf = None
	pass
class OutputSurfacePlug(Plug):
	node : BirailSrf = None
	pass
class SurfaceCachePlug(Plug):
	node : BirailSrf = None
	pass
class SweepStylePlug(Plug):
	node : BirailSrf = None
	pass
class TransformModePlug(Plug):
	node : BirailSrf = None
	pass
# endregion


# define node class
class BirailSrf(AbstractBaseCreate):
	inputRail1_ : InputRail1Plug = PlugDescriptor("inputRail1")
	inputRail2_ : InputRail2Plug = PlugDescriptor("inputRail2")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	surfaceCache_ : SurfaceCachePlug = PlugDescriptor("surfaceCache")
	sweepStyle_ : SweepStylePlug = PlugDescriptor("sweepStyle")
	transformMode_ : TransformModePlug = PlugDescriptor("transformMode")

	# node attributes

	typeName = "birailSrf"
	typeIdInt = 1312969542
	pass

