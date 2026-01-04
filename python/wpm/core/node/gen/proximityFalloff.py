

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : ProximityFalloff = None
	pass
class BindTagsFilterPlug(Plug):
	node : ProximityFalloff = None
	pass
class EndPlug(Plug):
	node : ProximityFalloff = None
	pass
class OutputWeightFunctionPlug(Plug):
	node : ProximityFalloff = None
	pass
class ProximityGeometryPlug(Plug):
	node : ProximityFalloff = None
	pass
class ProximitySubsetPlug(Plug):
	node : ProximityFalloff = None
	pass
class Ramp_FloatValuePlug(Plug):
	parent : RampPlug = PlugDescriptor("ramp")
	node : ProximityFalloff = None
	pass
class Ramp_InterpPlug(Plug):
	parent : RampPlug = PlugDescriptor("ramp")
	node : ProximityFalloff = None
	pass
class Ramp_PositionPlug(Plug):
	parent : RampPlug = PlugDescriptor("ramp")
	node : ProximityFalloff = None
	pass
class RampPlug(Plug):
	ramp_FloatValue_ : Ramp_FloatValuePlug = PlugDescriptor("ramp_FloatValue")
	rmpfv_ : Ramp_FloatValuePlug = PlugDescriptor("ramp_FloatValue")
	ramp_Interp_ : Ramp_InterpPlug = PlugDescriptor("ramp_Interp")
	rmpi_ : Ramp_InterpPlug = PlugDescriptor("ramp_Interp")
	ramp_Position_ : Ramp_PositionPlug = PlugDescriptor("ramp_Position")
	rmpp_ : Ramp_PositionPlug = PlugDescriptor("ramp_Position")
	node : ProximityFalloff = None
	pass
class StartPlug(Plug):
	node : ProximityFalloff = None
	pass
class UseBindTagsPlug(Plug):
	node : ProximityFalloff = None
	pass
class UseOriginalGeometryPlug(Plug):
	node : ProximityFalloff = None
	pass
class VertexSpacePlug(Plug):
	node : ProximityFalloff = None
	pass
class VolumePlug(Plug):
	node : ProximityFalloff = None
	pass
# endregion


# define node class
class ProximityFalloff(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	bindTagsFilter_ : BindTagsFilterPlug = PlugDescriptor("bindTagsFilter")
	end_ : EndPlug = PlugDescriptor("end")
	outputWeightFunction_ : OutputWeightFunctionPlug = PlugDescriptor("outputWeightFunction")
	proximityGeometry_ : ProximityGeometryPlug = PlugDescriptor("proximityGeometry")
	proximitySubset_ : ProximitySubsetPlug = PlugDescriptor("proximitySubset")
	ramp_FloatValue_ : Ramp_FloatValuePlug = PlugDescriptor("ramp_FloatValue")
	ramp_Interp_ : Ramp_InterpPlug = PlugDescriptor("ramp_Interp")
	ramp_Position_ : Ramp_PositionPlug = PlugDescriptor("ramp_Position")
	ramp_ : RampPlug = PlugDescriptor("ramp")
	start_ : StartPlug = PlugDescriptor("start")
	useBindTags_ : UseBindTagsPlug = PlugDescriptor("useBindTags")
	useOriginalGeometry_ : UseOriginalGeometryPlug = PlugDescriptor("useOriginalGeometry")
	vertexSpace_ : VertexSpacePlug = PlugDescriptor("vertexSpace")
	volume_ : VolumePlug = PlugDescriptor("volume")

	# node attributes

	typeName = "proximityFalloff"
	apiTypeInt = 1145
	apiTypeStr = "kProximityFalloff"
	typeIdInt = 1347962447
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "bindTagsFilter", "end", "outputWeightFunction", "proximityGeometry", "proximitySubset", "ramp_FloatValue", "ramp_Interp", "ramp_Position", "ramp", "start", "useBindTags", "useOriginalGeometry", "vertexSpace", "volume"]
	nodeLeafPlugs = ["binMembership", "bindTagsFilter", "end", "outputWeightFunction", "proximityGeometry", "proximitySubset", "ramp", "start", "useBindTags", "useOriginalGeometry", "vertexSpace", "volume"]
	pass

