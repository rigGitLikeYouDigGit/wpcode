

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseNurbsConversion = retriever.getNodeCls("AbstractBaseNurbsConversion")
assert AbstractBaseNurbsConversion
if T.TYPE_CHECKING:
	from .. import AbstractBaseNurbsConversion

# add node doc



# region plug type defs
class DegreeUPlug(Plug):
	node : RebuildSurface = None
	pass
class DegreeVPlug(Plug):
	node : RebuildSurface = None
	pass
class DirectionPlug(Plug):
	node : RebuildSurface = None
	pass
class EndKnotsPlug(Plug):
	node : RebuildSurface = None
	pass
class FitRebuildPlug(Plug):
	node : RebuildSurface = None
	pass
class InputSurfacePlug(Plug):
	node : RebuildSurface = None
	pass
class KeepControlPointsPlug(Plug):
	node : RebuildSurface = None
	pass
class KeepCornersPlug(Plug):
	node : RebuildSurface = None
	pass
class KeepRangePlug(Plug):
	node : RebuildSurface = None
	pass
class MatchSurfacePlug(Plug):
	node : RebuildSurface = None
	pass
class OldRebuildRationalPlug(Plug):
	node : RebuildSurface = None
	pass
class OutputSurfacePlug(Plug):
	node : RebuildSurface = None
	pass
class RebuildTypePlug(Plug):
	node : RebuildSurface = None
	pass
class SpansUPlug(Plug):
	node : RebuildSurface = None
	pass
class SpansVPlug(Plug):
	node : RebuildSurface = None
	pass
class TolerancePlug(Plug):
	node : RebuildSurface = None
	pass
# endregion


# define node class
class RebuildSurface(AbstractBaseNurbsConversion):
	degreeU_ : DegreeUPlug = PlugDescriptor("degreeU")
	degreeV_ : DegreeVPlug = PlugDescriptor("degreeV")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	endKnots_ : EndKnotsPlug = PlugDescriptor("endKnots")
	fitRebuild_ : FitRebuildPlug = PlugDescriptor("fitRebuild")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	keepControlPoints_ : KeepControlPointsPlug = PlugDescriptor("keepControlPoints")
	keepCorners_ : KeepCornersPlug = PlugDescriptor("keepCorners")
	keepRange_ : KeepRangePlug = PlugDescriptor("keepRange")
	matchSurface_ : MatchSurfacePlug = PlugDescriptor("matchSurface")
	oldRebuildRational_ : OldRebuildRationalPlug = PlugDescriptor("oldRebuildRational")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	rebuildType_ : RebuildTypePlug = PlugDescriptor("rebuildType")
	spansU_ : SpansUPlug = PlugDescriptor("spansU")
	spansV_ : SpansVPlug = PlugDescriptor("spansV")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "rebuildSurface"
	apiTypeInt = 91
	apiTypeStr = "kRebuildSurface"
	typeIdInt = 1314013779
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["degreeU", "degreeV", "direction", "endKnots", "fitRebuild", "inputSurface", "keepControlPoints", "keepCorners", "keepRange", "matchSurface", "oldRebuildRational", "outputSurface", "rebuildType", "spansU", "spansV", "tolerance"]
	nodeLeafPlugs = ["degreeU", "degreeV", "direction", "endKnots", "fitRebuild", "inputSurface", "keepControlPoints", "keepCorners", "keepRange", "matchSurface", "oldRebuildRational", "outputSurface", "rebuildType", "spansU", "spansV", "tolerance"]
	pass

