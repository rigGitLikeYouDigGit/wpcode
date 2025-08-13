

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : SubsetFalloff = None
	pass
class EndPlug(Plug):
	node : SubsetFalloff = None
	pass
class FalloffTagsPlug(Plug):
	node : SubsetFalloff = None
	pass
class ModePlug(Plug):
	node : SubsetFalloff = None
	pass
class OutputWeightFunctionPlug(Plug):
	node : SubsetFalloff = None
	pass
class Ramp_FloatValuePlug(Plug):
	parent : RampPlug = PlugDescriptor("ramp")
	node : SubsetFalloff = None
	pass
class Ramp_InterpPlug(Plug):
	parent : RampPlug = PlugDescriptor("ramp")
	node : SubsetFalloff = None
	pass
class Ramp_PositionPlug(Plug):
	parent : RampPlug = PlugDescriptor("ramp")
	node : SubsetFalloff = None
	pass
class RampPlug(Plug):
	ramp_FloatValue_ : Ramp_FloatValuePlug = PlugDescriptor("ramp_FloatValue")
	rmpfv_ : Ramp_FloatValuePlug = PlugDescriptor("ramp_FloatValue")
	ramp_Interp_ : Ramp_InterpPlug = PlugDescriptor("ramp_Interp")
	rmpi_ : Ramp_InterpPlug = PlugDescriptor("ramp_Interp")
	ramp_Position_ : Ramp_PositionPlug = PlugDescriptor("ramp_Position")
	rmpp_ : Ramp_PositionPlug = PlugDescriptor("ramp_Position")
	node : SubsetFalloff = None
	pass
class ScalePlug(Plug):
	node : SubsetFalloff = None
	pass
class StartPlug(Plug):
	node : SubsetFalloff = None
	pass
class UseFalloffTagsPlug(Plug):
	node : SubsetFalloff = None
	pass
class UseOriginalGeometryPlug(Plug):
	node : SubsetFalloff = None
	pass
class WithinBoundaryPlug(Plug):
	node : SubsetFalloff = None
	pass
# endregion


# define node class
class SubsetFalloff(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	end_ : EndPlug = PlugDescriptor("end")
	falloffTags_ : FalloffTagsPlug = PlugDescriptor("falloffTags")
	mode_ : ModePlug = PlugDescriptor("mode")
	outputWeightFunction_ : OutputWeightFunctionPlug = PlugDescriptor("outputWeightFunction")
	ramp_FloatValue_ : Ramp_FloatValuePlug = PlugDescriptor("ramp_FloatValue")
	ramp_Interp_ : Ramp_InterpPlug = PlugDescriptor("ramp_Interp")
	ramp_Position_ : Ramp_PositionPlug = PlugDescriptor("ramp_Position")
	ramp_ : RampPlug = PlugDescriptor("ramp")
	scale_ : ScalePlug = PlugDescriptor("scale")
	start_ : StartPlug = PlugDescriptor("start")
	useFalloffTags_ : UseFalloffTagsPlug = PlugDescriptor("useFalloffTags")
	useOriginalGeometry_ : UseOriginalGeometryPlug = PlugDescriptor("useOriginalGeometry")
	withinBoundary_ : WithinBoundaryPlug = PlugDescriptor("withinBoundary")

	# node attributes

	typeName = "subsetFalloff"
	apiTypeInt = 1146
	apiTypeStr = "kSubsetFalloff"
	typeIdInt = 1397966415
	MFnCls = om.MFnDependencyNode
	pass

