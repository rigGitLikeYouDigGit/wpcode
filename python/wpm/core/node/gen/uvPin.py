

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
	node : UvPin = None
	pass
class CacheSetupPlug(Plug):
	node : UvPin = None
	pass
class CoordinateUPlug(Plug):
	parent : CoordinatePlug = PlugDescriptor("coordinate")
	node : UvPin = None
	pass
class CoordinateVPlug(Plug):
	parent : CoordinatePlug = PlugDescriptor("coordinate")
	node : UvPin = None
	pass
class CoordinatePlug(Plug):
	coordinateU_ : CoordinateUPlug = PlugDescriptor("coordinateU")
	cu_ : CoordinateUPlug = PlugDescriptor("coordinateU")
	coordinateV_ : CoordinateVPlug = PlugDescriptor("coordinateV")
	cv_ : CoordinateVPlug = PlugDescriptor("coordinateV")
	node : UvPin = None
	pass
class DeformedGeometryPlug(Plug):
	node : UvPin = None
	pass
class NormalAxisPlug(Plug):
	node : UvPin = None
	pass
class NormalOverridePlug(Plug):
	node : UvPin = None
	pass
class NormalizedIsoParmsPlug(Plug):
	node : UvPin = None
	pass
class OriginalGeometryPlug(Plug):
	node : UvPin = None
	pass
class OutputMatrixPlug(Plug):
	node : UvPin = None
	pass
class OutputTranslateXPlug(Plug):
	parent : OutputTranslatePlug = PlugDescriptor("outputTranslate")
	node : UvPin = None
	pass
class OutputTranslateYPlug(Plug):
	parent : OutputTranslatePlug = PlugDescriptor("outputTranslate")
	node : UvPin = None
	pass
class OutputTranslateZPlug(Plug):
	parent : OutputTranslatePlug = PlugDescriptor("outputTranslate")
	node : UvPin = None
	pass
class OutputTranslatePlug(Plug):
	outputTranslateX_ : OutputTranslateXPlug = PlugDescriptor("outputTranslateX")
	otx_ : OutputTranslateXPlug = PlugDescriptor("outputTranslateX")
	outputTranslateY_ : OutputTranslateYPlug = PlugDescriptor("outputTranslateY")
	oty_ : OutputTranslateYPlug = PlugDescriptor("outputTranslateY")
	outputTranslateZ_ : OutputTranslateZPlug = PlugDescriptor("outputTranslateZ")
	otz_ : OutputTranslateZPlug = PlugDescriptor("outputTranslateZ")
	node : UvPin = None
	pass
class RailCurvePlug(Plug):
	node : UvPin = None
	pass
class RelativeSpaceMatrixPlug(Plug):
	node : UvPin = None
	pass
class RelativeSpaceModePlug(Plug):
	node : UvPin = None
	pass
class TangentAxisPlug(Plug):
	node : UvPin = None
	pass
class UvSetNamePlug(Plug):
	node : UvPin = None
	pass
# endregion


# define node class
class UvPin(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	coordinateU_ : CoordinateUPlug = PlugDescriptor("coordinateU")
	coordinateV_ : CoordinateVPlug = PlugDescriptor("coordinateV")
	coordinate_ : CoordinatePlug = PlugDescriptor("coordinate")
	deformedGeometry_ : DeformedGeometryPlug = PlugDescriptor("deformedGeometry")
	normalAxis_ : NormalAxisPlug = PlugDescriptor("normalAxis")
	normalOverride_ : NormalOverridePlug = PlugDescriptor("normalOverride")
	normalizedIsoParms_ : NormalizedIsoParmsPlug = PlugDescriptor("normalizedIsoParms")
	originalGeometry_ : OriginalGeometryPlug = PlugDescriptor("originalGeometry")
	outputMatrix_ : OutputMatrixPlug = PlugDescriptor("outputMatrix")
	outputTranslateX_ : OutputTranslateXPlug = PlugDescriptor("outputTranslateX")
	outputTranslateY_ : OutputTranslateYPlug = PlugDescriptor("outputTranslateY")
	outputTranslateZ_ : OutputTranslateZPlug = PlugDescriptor("outputTranslateZ")
	outputTranslate_ : OutputTranslatePlug = PlugDescriptor("outputTranslate")
	railCurve_ : RailCurvePlug = PlugDescriptor("railCurve")
	relativeSpaceMatrix_ : RelativeSpaceMatrixPlug = PlugDescriptor("relativeSpaceMatrix")
	relativeSpaceMode_ : RelativeSpaceModePlug = PlugDescriptor("relativeSpaceMode")
	tangentAxis_ : TangentAxisPlug = PlugDescriptor("tangentAxis")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "uvPin"
	typeIdInt = 1431720534
	pass

