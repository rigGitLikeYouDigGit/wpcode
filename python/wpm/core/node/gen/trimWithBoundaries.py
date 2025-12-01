

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
class CreateNewFacePlug(Plug):
	node : TrimWithBoundaries = None
	pass
class FlipNormalPlug(Plug):
	node : TrimWithBoundaries = None
	pass
class InputBoundariesPlug(Plug):
	node : TrimWithBoundaries = None
	pass
class InputSurfacePlug(Plug):
	node : TrimWithBoundaries = None
	pass
class OutputSurfacePlug(Plug):
	node : TrimWithBoundaries = None
	pass
class ToleranceEPlug(Plug):
	node : TrimWithBoundaries = None
	pass
class TolerancePEPlug(Plug):
	node : TrimWithBoundaries = None
	pass
# endregion


# define node class
class TrimWithBoundaries(AbstractBaseCreate):
	createNewFace_ : CreateNewFacePlug = PlugDescriptor("createNewFace")
	flipNormal_ : FlipNormalPlug = PlugDescriptor("flipNormal")
	inputBoundaries_ : InputBoundariesPlug = PlugDescriptor("inputBoundaries")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	toleranceE_ : ToleranceEPlug = PlugDescriptor("toleranceE")
	tolerancePE_ : TolerancePEPlug = PlugDescriptor("tolerancePE")

	# node attributes

	typeName = "trimWithBoundaries"
	apiTypeInt = 933
	apiTypeStr = "kTrimWithBoundaries"
	typeIdInt = 1314150210
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["createNewFace", "flipNormal", "inputBoundaries", "inputSurface", "outputSurface", "toleranceE", "tolerancePE"]
	nodeLeafPlugs = ["createNewFace", "flipNormal", "inputBoundaries", "inputSurface", "outputSurface", "toleranceE", "tolerancePE"]
	pass

