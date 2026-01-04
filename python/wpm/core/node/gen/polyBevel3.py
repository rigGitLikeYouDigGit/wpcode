

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierWorld = Catalogue.PolyModifierWorld
else:
	from .. import retriever
	PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
	assert PolyModifierWorld

# add node doc



# region plug type defs
class AngleTolerancePlug(Plug):
	node : PolyBevel3 = None
	pass
class AutoFitPlug(Plug):
	node : PolyBevel3 = None
	pass
class ChamferPlug(Plug):
	node : PolyBevel3 = None
	pass
class DepthPlug(Plug):
	node : PolyBevel3 = None
	pass
class ForceParallelPlug(Plug):
	node : PolyBevel3 = None
	pass
class FractionPlug(Plug):
	node : PolyBevel3 = None
	pass
class Maya2015Plug(Plug):
	node : PolyBevel3 = None
	pass
class Maya2016SP3Plug(Plug):
	node : PolyBevel3 = None
	pass
class Maya2017Update1Plug(Plug):
	node : PolyBevel3 = None
	pass
class MergeVertexTolerancePlug(Plug):
	node : PolyBevel3 = None
	pass
class MergeVerticesPlug(Plug):
	node : PolyBevel3 = None
	pass
class MiterAlongPlug(Plug):
	node : PolyBevel3 = None
	pass
class MiteringPlug(Plug):
	node : PolyBevel3 = None
	pass
class MiteringAnglePlug(Plug):
	node : PolyBevel3 = None
	pass
class OffsetPlug(Plug):
	node : PolyBevel3 = None
	pass
class OffsetAsFractionPlug(Plug):
	node : PolyBevel3 = None
	pass
class RoundnessPlug(Plug):
	node : PolyBevel3 = None
	pass
class SegmentsPlug(Plug):
	node : PolyBevel3 = None
	pass
class SmoothingAnglePlug(Plug):
	node : PolyBevel3 = None
	pass
class SubdivideNgonsPlug(Plug):
	node : PolyBevel3 = None
	pass
# endregion


# define node class
class PolyBevel3(PolyModifierWorld):
	angleTolerance_ : AngleTolerancePlug = PlugDescriptor("angleTolerance")
	autoFit_ : AutoFitPlug = PlugDescriptor("autoFit")
	chamfer_ : ChamferPlug = PlugDescriptor("chamfer")
	depth_ : DepthPlug = PlugDescriptor("depth")
	forceParallel_ : ForceParallelPlug = PlugDescriptor("forceParallel")
	fraction_ : FractionPlug = PlugDescriptor("fraction")
	maya2015_ : Maya2015Plug = PlugDescriptor("maya2015")
	maya2016SP3_ : Maya2016SP3Plug = PlugDescriptor("maya2016SP3")
	maya2017Update1_ : Maya2017Update1Plug = PlugDescriptor("maya2017Update1")
	mergeVertexTolerance_ : MergeVertexTolerancePlug = PlugDescriptor("mergeVertexTolerance")
	mergeVertices_ : MergeVerticesPlug = PlugDescriptor("mergeVertices")
	miterAlong_ : MiterAlongPlug = PlugDescriptor("miterAlong")
	mitering_ : MiteringPlug = PlugDescriptor("mitering")
	miteringAngle_ : MiteringAnglePlug = PlugDescriptor("miteringAngle")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	offsetAsFraction_ : OffsetAsFractionPlug = PlugDescriptor("offsetAsFraction")
	roundness_ : RoundnessPlug = PlugDescriptor("roundness")
	segments_ : SegmentsPlug = PlugDescriptor("segments")
	smoothingAngle_ : SmoothingAnglePlug = PlugDescriptor("smoothingAngle")
	subdivideNgons_ : SubdivideNgonsPlug = PlugDescriptor("subdivideNgons")

	# node attributes

	typeName = "polyBevel3"
	apiTypeInt = 1102
	apiTypeStr = "kPolyBevel3"
	typeIdInt = 1346524723
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["angleTolerance", "autoFit", "chamfer", "depth", "forceParallel", "fraction", "maya2015", "maya2016SP3", "maya2017Update1", "mergeVertexTolerance", "mergeVertices", "miterAlong", "mitering", "miteringAngle", "offset", "offsetAsFraction", "roundness", "segments", "smoothingAngle", "subdivideNgons"]
	nodeLeafPlugs = ["angleTolerance", "autoFit", "chamfer", "depth", "forceParallel", "fraction", "maya2015", "maya2016SP3", "maya2017Update1", "mergeVertexTolerance", "mergeVertices", "miterAlong", "mitering", "miteringAngle", "offset", "offsetAsFraction", "roundness", "segments", "smoothingAngle", "subdivideNgons"]
	pass

