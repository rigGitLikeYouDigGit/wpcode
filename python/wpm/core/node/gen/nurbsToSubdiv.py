

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
class CollapsePolesPlug(Plug):
	node : NurbsToSubdiv = None
	pass
class InputSurfacePlug(Plug):
	node : NurbsToSubdiv = None
	pass
class MatchPeriodicPlug(Plug):
	node : NurbsToSubdiv = None
	pass
class MaxPolyCountPlug(Plug):
	node : NurbsToSubdiv = None
	pass
class OutputSubdPlug(Plug):
	node : NurbsToSubdiv = None
	pass
class ReverseNormalPlug(Plug):
	node : NurbsToSubdiv = None
	pass
# endregion


# define node class
class NurbsToSubdiv(AbstractBaseNurbsConversion):
	collapsePoles_ : CollapsePolesPlug = PlugDescriptor("collapsePoles")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	matchPeriodic_ : MatchPeriodicPlug = PlugDescriptor("matchPeriodic")
	maxPolyCount_ : MaxPolyCountPlug = PlugDescriptor("maxPolyCount")
	outputSubd_ : OutputSubdPlug = PlugDescriptor("outputSubd")
	reverseNormal_ : ReverseNormalPlug = PlugDescriptor("reverseNormal")

	# node attributes

	typeName = "nurbsToSubdiv"
	apiTypeInt = 760
	apiTypeStr = "kNurbsToSubdiv"
	typeIdInt = 1397642323
	MFnCls = om.MFnDependencyNode
	pass

