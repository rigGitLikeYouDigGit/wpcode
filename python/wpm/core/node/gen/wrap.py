

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
GeometryFilter = retriever.getNodeCls("GeometryFilter")
assert GeometryFilter
if T.TYPE_CHECKING:
	from .. import GeometryFilter

# add node doc



# region plug type defs
class AutoWeightThresholdPlug(Plug):
	node : Wrap = None
	pass
class AutoWeightThresholdValuePlug(Plug):
	node : Wrap = None
	pass
class BaseDrtPlug(Plug):
	node : Wrap = None
	pass
class BasePointsPlug(Plug):
	node : Wrap = None
	pass
class DriverPointsPlug(Plug):
	node : Wrap = None
	pass
class DropoffPlug(Plug):
	node : Wrap = None
	pass
class ExclusiveBindPlug(Plug):
	node : Wrap = None
	pass
class FalloffModePlug(Plug):
	node : Wrap = None
	pass
class GeomMatrixPlug(Plug):
	node : Wrap = None
	pass
class InflTypePlug(Plug):
	node : Wrap = None
	pass
class MaxDistancePlug(Plug):
	node : Wrap = None
	pass
class NurbsSamplesPlug(Plug):
	node : Wrap = None
	pass
class SmoothnessPlug(Plug):
	node : Wrap = None
	pass
class WeightThresholdPlug(Plug):
	node : Wrap = None
	pass
class WrapBuffersPlug(Plug):
	node : Wrap = None
	pass
class WtDrtyPlug(Plug):
	node : Wrap = None
	pass
# endregion


# define node class
class Wrap(GeometryFilter):
	autoWeightThreshold_ : AutoWeightThresholdPlug = PlugDescriptor("autoWeightThreshold")
	autoWeightThresholdValue_ : AutoWeightThresholdValuePlug = PlugDescriptor("autoWeightThresholdValue")
	baseDrt_ : BaseDrtPlug = PlugDescriptor("baseDrt")
	basePoints_ : BasePointsPlug = PlugDescriptor("basePoints")
	driverPoints_ : DriverPointsPlug = PlugDescriptor("driverPoints")
	dropoff_ : DropoffPlug = PlugDescriptor("dropoff")
	exclusiveBind_ : ExclusiveBindPlug = PlugDescriptor("exclusiveBind")
	falloffMode_ : FalloffModePlug = PlugDescriptor("falloffMode")
	geomMatrix_ : GeomMatrixPlug = PlugDescriptor("geomMatrix")
	inflType_ : InflTypePlug = PlugDescriptor("inflType")
	maxDistance_ : MaxDistancePlug = PlugDescriptor("maxDistance")
	nurbsSamples_ : NurbsSamplesPlug = PlugDescriptor("nurbsSamples")
	smoothness_ : SmoothnessPlug = PlugDescriptor("smoothness")
	weightThreshold_ : WeightThresholdPlug = PlugDescriptor("weightThreshold")
	wrapBuffers_ : WrapBuffersPlug = PlugDescriptor("wrapBuffers")
	wtDrty_ : WtDrtyPlug = PlugDescriptor("wtDrty")

	# node attributes

	typeName = "wrap"
	typeIdInt = 1180127824
	pass

