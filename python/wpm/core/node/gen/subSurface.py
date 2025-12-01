

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
class FaceCountUPlug(Plug):
	node : SubSurface = None
	pass
class FaceCountVPlug(Plug):
	node : SubSurface = None
	pass
class FirstFaceUPlug(Plug):
	node : SubSurface = None
	pass
class FirstFaceVPlug(Plug):
	node : SubSurface = None
	pass
class InputSurfacePlug(Plug):
	node : SubSurface = None
	pass
class OutputSurfacePlug(Plug):
	node : SubSurface = None
	pass
# endregion


# define node class
class SubSurface(AbstractBaseCreate):
	faceCountU_ : FaceCountUPlug = PlugDescriptor("faceCountU")
	faceCountV_ : FaceCountVPlug = PlugDescriptor("faceCountV")
	firstFaceU_ : FirstFaceUPlug = PlugDescriptor("firstFaceU")
	firstFaceV_ : FirstFaceVPlug = PlugDescriptor("firstFaceV")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")

	# node attributes

	typeName = "subSurface"
	apiTypeInt = 781
	apiTypeStr = "kSubSurface"
	typeIdInt = 1314083666
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["faceCountU", "faceCountV", "firstFaceU", "firstFaceV", "inputSurface", "outputSurface"]
	nodeLeafPlugs = ["faceCountU", "faceCountV", "firstFaceU", "firstFaceV", "inputSurface", "outputSurface"]
	pass

