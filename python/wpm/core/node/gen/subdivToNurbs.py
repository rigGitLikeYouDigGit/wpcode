

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
class ApplyMatrixToResultPlug(Plug):
	node : SubdivToNurbs = None
	pass
class InSubdivPlug(Plug):
	node : SubdivToNurbs = None
	pass
class OutputSurfacesPlug(Plug):
	node : SubdivToNurbs = None
	pass
class OutputTypePlug(Plug):
	node : SubdivToNurbs = None
	pass
# endregion


# define node class
class SubdivToNurbs(AbstractBaseCreate):
	applyMatrixToResult_ : ApplyMatrixToResultPlug = PlugDescriptor("applyMatrixToResult")
	inSubdiv_ : InSubdivPlug = PlugDescriptor("inSubdiv")
	outputSurfaces_ : OutputSurfacesPlug = PlugDescriptor("outputSurfaces")
	outputType_ : OutputTypePlug = PlugDescriptor("outputType")

	# node attributes

	typeName = "subdivToNurbs"
	apiTypeInt = 820
	apiTypeStr = "kSubdivToNurbs"
	typeIdInt = 1396986702
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["applyMatrixToResult", "inSubdiv", "outputSurfaces", "outputType"]
	nodeLeafPlugs = ["applyMatrixToResult", "inSubdiv", "outputSurfaces", "outputType"]
	pass

