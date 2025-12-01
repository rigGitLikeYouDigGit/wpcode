

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
	node : ProximityPin = None
	pass
class CacheSetupPlug(Plug):
	node : ProximityPin = None
	pass
class CoordModePlug(Plug):
	node : ProximityPin = None
	pass
class DeformedGeometryPlug(Plug):
	node : ProximityPin = None
	pass
class EnvelopePlug(Plug):
	node : ProximityPin = None
	pass
class InputMatrixPlug(Plug):
	node : ProximityPin = None
	pass
class NormalAxisPlug(Plug):
	node : ProximityPin = None
	pass
class NormalOverridePlug(Plug):
	node : ProximityPin = None
	pass
class OffsetOrientationPlug(Plug):
	node : ProximityPin = None
	pass
class OffsetTranslationPlug(Plug):
	node : ProximityPin = None
	pass
class OriginalGeometryPlug(Plug):
	node : ProximityPin = None
	pass
class OriginalRailCurvePlug(Plug):
	node : ProximityPin = None
	pass
class OutputMatrixPlug(Plug):
	node : ProximityPin = None
	pass
class RailCurvePlug(Plug):
	node : ProximityPin = None
	pass
class RelativeSpaceMatrixPlug(Plug):
	node : ProximityPin = None
	pass
class RelativeSpaceModePlug(Plug):
	node : ProximityPin = None
	pass
class TangentAxisPlug(Plug):
	node : ProximityPin = None
	pass
class UvSetNamePlug(Plug):
	node : ProximityPin = None
	pass
# endregion


# define node class
class ProximityPin(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	coordMode_ : CoordModePlug = PlugDescriptor("coordMode")
	deformedGeometry_ : DeformedGeometryPlug = PlugDescriptor("deformedGeometry")
	envelope_ : EnvelopePlug = PlugDescriptor("envelope")
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	normalAxis_ : NormalAxisPlug = PlugDescriptor("normalAxis")
	normalOverride_ : NormalOverridePlug = PlugDescriptor("normalOverride")
	offsetOrientation_ : OffsetOrientationPlug = PlugDescriptor("offsetOrientation")
	offsetTranslation_ : OffsetTranslationPlug = PlugDescriptor("offsetTranslation")
	originalGeometry_ : OriginalGeometryPlug = PlugDescriptor("originalGeometry")
	originalRailCurve_ : OriginalRailCurvePlug = PlugDescriptor("originalRailCurve")
	outputMatrix_ : OutputMatrixPlug = PlugDescriptor("outputMatrix")
	railCurve_ : RailCurvePlug = PlugDescriptor("railCurve")
	relativeSpaceMatrix_ : RelativeSpaceMatrixPlug = PlugDescriptor("relativeSpaceMatrix")
	relativeSpaceMode_ : RelativeSpaceModePlug = PlugDescriptor("relativeSpaceMode")
	tangentAxis_ : TangentAxisPlug = PlugDescriptor("tangentAxis")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "proximityPin"
	apiTypeInt = 991
	apiTypeStr = "kProximityPin"
	typeIdInt = 1347573840
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "cacheSetup", "coordMode", "deformedGeometry", "envelope", "inputMatrix", "normalAxis", "normalOverride", "offsetOrientation", "offsetTranslation", "originalGeometry", "originalRailCurve", "outputMatrix", "railCurve", "relativeSpaceMatrix", "relativeSpaceMode", "tangentAxis", "uvSetName"]
	nodeLeafPlugs = ["binMembership", "cacheSetup", "coordMode", "deformedGeometry", "envelope", "inputMatrix", "normalAxis", "normalOverride", "offsetOrientation", "offsetTranslation", "originalGeometry", "originalRailCurve", "outputMatrix", "railCurve", "relativeSpaceMatrix", "relativeSpaceMode", "tangentAxis", "uvSetName"]
	pass

