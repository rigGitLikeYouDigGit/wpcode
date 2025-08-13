

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
BoundaryBase = retriever.getNodeCls("BoundaryBase")
assert BoundaryBase
if T.TYPE_CHECKING:
	from .. import BoundaryBase

# add node doc



# region plug type defs
class ContinuityPassed1Plug(Plug):
	node : SquareSrf = None
	pass
class ContinuityPassed2Plug(Plug):
	node : SquareSrf = None
	pass
class ContinuityPassed3Plug(Plug):
	node : SquareSrf = None
	pass
class ContinuityPassed4Plug(Plug):
	node : SquareSrf = None
	pass
class ContinuityType1Plug(Plug):
	node : SquareSrf = None
	pass
class ContinuityType2Plug(Plug):
	node : SquareSrf = None
	pass
class ContinuityType3Plug(Plug):
	node : SquareSrf = None
	pass
class ContinuityType4Plug(Plug):
	node : SquareSrf = None
	pass
class CurveFitCheckpointsPlug(Plug):
	node : SquareSrf = None
	pass
class RebuildCurve1Plug(Plug):
	node : SquareSrf = None
	pass
class RebuildCurve2Plug(Plug):
	node : SquareSrf = None
	pass
class RebuildCurve3Plug(Plug):
	node : SquareSrf = None
	pass
class RebuildCurve4Plug(Plug):
	node : SquareSrf = None
	pass
# endregion


# define node class
class SquareSrf(BoundaryBase):
	continuityPassed1_ : ContinuityPassed1Plug = PlugDescriptor("continuityPassed1")
	continuityPassed2_ : ContinuityPassed2Plug = PlugDescriptor("continuityPassed2")
	continuityPassed3_ : ContinuityPassed3Plug = PlugDescriptor("continuityPassed3")
	continuityPassed4_ : ContinuityPassed4Plug = PlugDescriptor("continuityPassed4")
	continuityType1_ : ContinuityType1Plug = PlugDescriptor("continuityType1")
	continuityType2_ : ContinuityType2Plug = PlugDescriptor("continuityType2")
	continuityType3_ : ContinuityType3Plug = PlugDescriptor("continuityType3")
	continuityType4_ : ContinuityType4Plug = PlugDescriptor("continuityType4")
	curveFitCheckpoints_ : CurveFitCheckpointsPlug = PlugDescriptor("curveFitCheckpoints")
	rebuildCurve1_ : RebuildCurve1Plug = PlugDescriptor("rebuildCurve1")
	rebuildCurve2_ : RebuildCurve2Plug = PlugDescriptor("rebuildCurve2")
	rebuildCurve3_ : RebuildCurve3Plug = PlugDescriptor("rebuildCurve3")
	rebuildCurve4_ : RebuildCurve4Plug = PlugDescriptor("rebuildCurve4")

	# node attributes

	typeName = "squareSrf"
	apiTypeInt = 717
	apiTypeStr = "kSquareSrf"
	typeIdInt = 1314083155
	MFnCls = om.MFnDependencyNode
	pass

