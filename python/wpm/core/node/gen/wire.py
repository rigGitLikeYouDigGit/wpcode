

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
WeightGeometryFilter = retriever.getNodeCls("WeightGeometryFilter")
assert WeightGeometryFilter
if T.TYPE_CHECKING:
	from .. import WeightGeometryFilter

# add node doc



# region plug type defs
class BaseWirePlug(Plug):
	node : Wire = None
	pass
class BindToOriginalGeometryPlug(Plug):
	node : Wire = None
	pass
class CacheSetupPlug(Plug):
	node : Wire = None
	pass
class CrossingEffectPlug(Plug):
	node : Wire = None
	pass
class DeformedWirePlug(Plug):
	node : Wire = None
	pass
class DropoffDistancePlug(Plug):
	node : Wire = None
	pass
class FreezeGeometryPlug(Plug):
	node : Wire = None
	pass
class HolderPlug(Plug):
	node : Wire = None
	pass
class LocalInfluencePlug(Plug):
	node : Wire = None
	pass
class RotationPlug(Plug):
	node : Wire = None
	pass
class ScalePlug(Plug):
	node : Wire = None
	pass
class TensionPlug(Plug):
	node : Wire = None
	pass
class WireLocatorEnvelopePlug(Plug):
	node : Wire = None
	pass
class WireLocatorParameterPlug(Plug):
	node : Wire = None
	pass
class WireLocatorPercentagePlug(Plug):
	node : Wire = None
	pass
class WireLocatorTwistPlug(Plug):
	node : Wire = None
	pass
# endregion


# define node class
class Wire(WeightGeometryFilter):
	baseWire_ : BaseWirePlug = PlugDescriptor("baseWire")
	bindToOriginalGeometry_ : BindToOriginalGeometryPlug = PlugDescriptor("bindToOriginalGeometry")
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	crossingEffect_ : CrossingEffectPlug = PlugDescriptor("crossingEffect")
	deformedWire_ : DeformedWirePlug = PlugDescriptor("deformedWire")
	dropoffDistance_ : DropoffDistancePlug = PlugDescriptor("dropoffDistance")
	freezeGeometry_ : FreezeGeometryPlug = PlugDescriptor("freezeGeometry")
	holder_ : HolderPlug = PlugDescriptor("holder")
	localInfluence_ : LocalInfluencePlug = PlugDescriptor("localInfluence")
	rotation_ : RotationPlug = PlugDescriptor("rotation")
	scale_ : ScalePlug = PlugDescriptor("scale")
	tension_ : TensionPlug = PlugDescriptor("tension")
	wireLocatorEnvelope_ : WireLocatorEnvelopePlug = PlugDescriptor("wireLocatorEnvelope")
	wireLocatorParameter_ : WireLocatorParameterPlug = PlugDescriptor("wireLocatorParameter")
	wireLocatorPercentage_ : WireLocatorPercentagePlug = PlugDescriptor("wireLocatorPercentage")
	wireLocatorTwist_ : WireLocatorTwistPlug = PlugDescriptor("wireLocatorTwist")

	# node attributes

	typeName = "wire"
	apiTypeInt = 355
	apiTypeStr = "kWire"
	typeIdInt = 1180125522
	MFnCls = om.MFnGeometryFilter
	nodeLeafClassAttrs = ["baseWire", "bindToOriginalGeometry", "cacheSetup", "crossingEffect", "deformedWire", "dropoffDistance", "freezeGeometry", "holder", "localInfluence", "rotation", "scale", "tension", "wireLocatorEnvelope", "wireLocatorParameter", "wireLocatorPercentage", "wireLocatorTwist"]
	nodeLeafPlugs = ["baseWire", "bindToOriginalGeometry", "cacheSetup", "crossingEffect", "deformedWire", "dropoffDistance", "freezeGeometry", "holder", "localInfluence", "rotation", "scale", "tension", "wireLocatorEnvelope", "wireLocatorParameter", "wireLocatorPercentage", "wireLocatorTwist"]
	pass

