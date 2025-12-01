

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
assert PolyModifierWorld
if T.TYPE_CHECKING:
	from .. import PolyModifierWorld

# add node doc



# region plug type defs
class AngleTolerancePlug(Plug):
	node : PolyBevel = None
	pass
class AutoFitPlug(Plug):
	node : PolyBevel = None
	pass
class FillNgonsPlug(Plug):
	node : PolyBevel = None
	pass
class FractionPlug(Plug):
	node : PolyBevel = None
	pass
class Maya2015Plug(Plug):
	node : PolyBevel = None
	pass
class MergeVertexTolerancePlug(Plug):
	node : PolyBevel = None
	pass
class MergeVerticesPlug(Plug):
	node : PolyBevel = None
	pass
class MiteringAnglePlug(Plug):
	node : PolyBevel = None
	pass
class OffsetPlug(Plug):
	node : PolyBevel = None
	pass
class OffsetAsFractionPlug(Plug):
	node : PolyBevel = None
	pass
class RoundnessPlug(Plug):
	node : PolyBevel = None
	pass
class SegmentsPlug(Plug):
	node : PolyBevel = None
	pass
class SmoothingAnglePlug(Plug):
	node : PolyBevel = None
	pass
class UvAssignmentPlug(Plug):
	node : PolyBevel = None
	pass
# endregion


# define node class
class PolyBevel(PolyModifierWorld):
	angleTolerance_ : AngleTolerancePlug = PlugDescriptor("angleTolerance")
	autoFit_ : AutoFitPlug = PlugDescriptor("autoFit")
	fillNgons_ : FillNgonsPlug = PlugDescriptor("fillNgons")
	fraction_ : FractionPlug = PlugDescriptor("fraction")
	maya2015_ : Maya2015Plug = PlugDescriptor("maya2015")
	mergeVertexTolerance_ : MergeVertexTolerancePlug = PlugDescriptor("mergeVertexTolerance")
	mergeVertices_ : MergeVerticesPlug = PlugDescriptor("mergeVertices")
	miteringAngle_ : MiteringAnglePlug = PlugDescriptor("miteringAngle")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	offsetAsFraction_ : OffsetAsFractionPlug = PlugDescriptor("offsetAsFraction")
	roundness_ : RoundnessPlug = PlugDescriptor("roundness")
	segments_ : SegmentsPlug = PlugDescriptor("segments")
	smoothingAngle_ : SmoothingAnglePlug = PlugDescriptor("smoothingAngle")
	uvAssignment_ : UvAssignmentPlug = PlugDescriptor("uvAssignment")

	# node attributes

	typeName = "polyBevel"
	apiTypeInt = 401
	apiTypeStr = "kPolyBevel"
	typeIdInt = 1346524748
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["angleTolerance", "autoFit", "fillNgons", "fraction", "maya2015", "mergeVertexTolerance", "mergeVertices", "miteringAngle", "offset", "offsetAsFraction", "roundness", "segments", "smoothingAngle", "uvAssignment"]
	nodeLeafPlugs = ["angleTolerance", "autoFit", "fillNgons", "fraction", "maya2015", "mergeVertexTolerance", "mergeVertices", "miteringAngle", "offset", "offsetAsFraction", "roundness", "segments", "smoothingAngle", "uvAssignment"]
	pass

