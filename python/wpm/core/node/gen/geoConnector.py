

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
	node : GeoConnector = None
	pass
class ComponentCentroidXPlug(Plug):
	parent : ComponentCentroidPlug = PlugDescriptor("componentCentroid")
	node : GeoConnector = None
	pass
class ComponentCentroidYPlug(Plug):
	parent : ComponentCentroidPlug = PlugDescriptor("componentCentroid")
	node : GeoConnector = None
	pass
class ComponentCentroidZPlug(Plug):
	parent : ComponentCentroidPlug = PlugDescriptor("componentCentroid")
	node : GeoConnector = None
	pass
class ComponentCentroidPlug(Plug):
	componentCentroidX_ : ComponentCentroidXPlug = PlugDescriptor("componentCentroidX")
	ccx_ : ComponentCentroidXPlug = PlugDescriptor("componentCentroidX")
	componentCentroidY_ : ComponentCentroidYPlug = PlugDescriptor("componentCentroidY")
	ccy_ : ComponentCentroidYPlug = PlugDescriptor("componentCentroidY")
	componentCentroidZ_ : ComponentCentroidZPlug = PlugDescriptor("componentCentroidZ")
	ccz_ : ComponentCentroidZPlug = PlugDescriptor("componentCentroidZ")
	node : GeoConnector = None
	pass
class ComponentCentroidLocalXPlug(Plug):
	parent : ComponentCentroidLocalPlug = PlugDescriptor("componentCentroidLocal")
	node : GeoConnector = None
	pass
class ComponentCentroidLocalYPlug(Plug):
	parent : ComponentCentroidLocalPlug = PlugDescriptor("componentCentroidLocal")
	node : GeoConnector = None
	pass
class ComponentCentroidLocalZPlug(Plug):
	parent : ComponentCentroidLocalPlug = PlugDescriptor("componentCentroidLocal")
	node : GeoConnector = None
	pass
class ComponentCentroidLocalPlug(Plug):
	componentCentroidLocalX_ : ComponentCentroidLocalXPlug = PlugDescriptor("componentCentroidLocalX")
	cclx_ : ComponentCentroidLocalXPlug = PlugDescriptor("componentCentroidLocalX")
	componentCentroidLocalY_ : ComponentCentroidLocalYPlug = PlugDescriptor("componentCentroidLocalY")
	clcy_ : ComponentCentroidLocalYPlug = PlugDescriptor("componentCentroidLocalY")
	componentCentroidLocalZ_ : ComponentCentroidLocalZPlug = PlugDescriptor("componentCentroidLocalZ")
	clcz_ : ComponentCentroidLocalZPlug = PlugDescriptor("componentCentroidLocalZ")
	node : GeoConnector = None
	pass
class ComponentPositionsPlug(Plug):
	node : GeoConnector = None
	pass
class ComponentVelocitiesPlug(Plug):
	node : GeoConnector = None
	pass
class CurrentTimePlug(Plug):
	node : GeoConnector = None
	pass
class DeltaTimePlug(Plug):
	node : GeoConnector = None
	pass
class FrictionPlug(Plug):
	node : GeoConnector = None
	pass
class GeometryModifiedPlug(Plug):
	node : GeoConnector = None
	pass
class GroupIdPlug(Plug):
	node : GeoConnector = None
	pass
class IdIndexPlug(Plug):
	parent : IdMappingPlug = PlugDescriptor("idMapping")
	node : GeoConnector = None
	pass
class SortedIdPlug(Plug):
	parent : IdMappingPlug = PlugDescriptor("idMapping")
	node : GeoConnector = None
	pass
class IdMappingPlug(Plug):
	idIndex_ : IdIndexPlug = PlugDescriptor("idIndex")
	idix_ : IdIndexPlug = PlugDescriptor("idIndex")
	sortedId_ : SortedIdPlug = PlugDescriptor("sortedId")
	sid_ : SortedIdPlug = PlugDescriptor("sortedId")
	node : GeoConnector = None
	pass
class InputForcePlug(Plug):
	node : GeoConnector = None
	pass
class InputGeometryMsgPlug(Plug):
	node : GeoConnector = None
	pass
class LocalGeometryPlug(Plug):
	node : GeoConnector = None
	pass
class LocalSweptGeometryPlug(Plug):
	node : GeoConnector = None
	pass
class MatrixModifiedPlug(Plug):
	node : GeoConnector = None
	pass
class OffsetPlug(Plug):
	node : GeoConnector = None
	pass
class OwnerPlug(Plug):
	node : GeoConnector = None
	pass
class OwnerCentroidXPlug(Plug):
	parent : OwnerCentroidPlug = PlugDescriptor("ownerCentroid")
	node : GeoConnector = None
	pass
class OwnerCentroidYPlug(Plug):
	parent : OwnerCentroidPlug = PlugDescriptor("ownerCentroid")
	node : GeoConnector = None
	pass
class OwnerCentroidZPlug(Plug):
	parent : OwnerCentroidPlug = PlugDescriptor("ownerCentroid")
	node : GeoConnector = None
	pass
class OwnerCentroidPlug(Plug):
	ownerCentroidX_ : OwnerCentroidXPlug = PlugDescriptor("ownerCentroidX")
	ocx_ : OwnerCentroidXPlug = PlugDescriptor("ownerCentroidX")
	ownerCentroidY_ : OwnerCentroidYPlug = PlugDescriptor("ownerCentroidY")
	ocy_ : OwnerCentroidYPlug = PlugDescriptor("ownerCentroidY")
	ownerCentroidZ_ : OwnerCentroidZPlug = PlugDescriptor("ownerCentroidZ")
	ocz_ : OwnerCentroidZPlug = PlugDescriptor("ownerCentroidZ")
	node : GeoConnector = None
	pass
class OwnerCentroidLocalXPlug(Plug):
	parent : OwnerCentroidLocalPlug = PlugDescriptor("ownerCentroidLocal")
	node : GeoConnector = None
	pass
class OwnerCentroidLocalYPlug(Plug):
	parent : OwnerCentroidLocalPlug = PlugDescriptor("ownerCentroidLocal")
	node : GeoConnector = None
	pass
class OwnerCentroidLocalZPlug(Plug):
	parent : OwnerCentroidLocalPlug = PlugDescriptor("ownerCentroidLocal")
	node : GeoConnector = None
	pass
class OwnerCentroidLocalPlug(Plug):
	ownerCentroidLocalX_ : OwnerCentroidLocalXPlug = PlugDescriptor("ownerCentroidLocalX")
	olcx_ : OwnerCentroidLocalXPlug = PlugDescriptor("ownerCentroidLocalX")
	ownerCentroidLocalY_ : OwnerCentroidLocalYPlug = PlugDescriptor("ownerCentroidLocalY")
	ocly_ : OwnerCentroidLocalYPlug = PlugDescriptor("ownerCentroidLocalY")
	ownerCentroidLocalZ_ : OwnerCentroidLocalZPlug = PlugDescriptor("ownerCentroidLocalZ")
	oclz_ : OwnerCentroidLocalZPlug = PlugDescriptor("ownerCentroidLocalZ")
	node : GeoConnector = None
	pass
class OwnerMassesPlug(Plug):
	node : GeoConnector = None
	pass
class OwnerPositionsPlug(Plug):
	node : GeoConnector = None
	pass
class OwnerVelocitiesPlug(Plug):
	node : GeoConnector = None
	pass
class PreComponentPositionsPlug(Plug):
	node : GeoConnector = None
	pass
class PreOwnerPositionsPlug(Plug):
	node : GeoConnector = None
	pass
class PrevTimePlug(Plug):
	node : GeoConnector = None
	pass
class RatePPInPlug(Plug):
	node : GeoConnector = None
	pass
class RatePPOutPlug(Plug):
	node : GeoConnector = None
	pass
class ResiliencePlug(Plug):
	node : GeoConnector = None
	pass
class SweptGeometryPlug(Plug):
	node : GeoConnector = None
	pass
class TessellationFactorPlug(Plug):
	node : GeoConnector = None
	pass
class UvSetNamePlug(Plug):
	node : GeoConnector = None
	pass
class WorldMatrixPlug(Plug):
	node : GeoConnector = None
	pass
# endregion


# define node class
class GeoConnector(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	componentCentroidX_ : ComponentCentroidXPlug = PlugDescriptor("componentCentroidX")
	componentCentroidY_ : ComponentCentroidYPlug = PlugDescriptor("componentCentroidY")
	componentCentroidZ_ : ComponentCentroidZPlug = PlugDescriptor("componentCentroidZ")
	componentCentroid_ : ComponentCentroidPlug = PlugDescriptor("componentCentroid")
	componentCentroidLocalX_ : ComponentCentroidLocalXPlug = PlugDescriptor("componentCentroidLocalX")
	componentCentroidLocalY_ : ComponentCentroidLocalYPlug = PlugDescriptor("componentCentroidLocalY")
	componentCentroidLocalZ_ : ComponentCentroidLocalZPlug = PlugDescriptor("componentCentroidLocalZ")
	componentCentroidLocal_ : ComponentCentroidLocalPlug = PlugDescriptor("componentCentroidLocal")
	componentPositions_ : ComponentPositionsPlug = PlugDescriptor("componentPositions")
	componentVelocities_ : ComponentVelocitiesPlug = PlugDescriptor("componentVelocities")
	currentTime_ : CurrentTimePlug = PlugDescriptor("currentTime")
	deltaTime_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	friction_ : FrictionPlug = PlugDescriptor("friction")
	geometryModified_ : GeometryModifiedPlug = PlugDescriptor("geometryModified")
	groupId_ : GroupIdPlug = PlugDescriptor("groupId")
	idIndex_ : IdIndexPlug = PlugDescriptor("idIndex")
	sortedId_ : SortedIdPlug = PlugDescriptor("sortedId")
	idMapping_ : IdMappingPlug = PlugDescriptor("idMapping")
	inputForce_ : InputForcePlug = PlugDescriptor("inputForce")
	inputGeometryMsg_ : InputGeometryMsgPlug = PlugDescriptor("inputGeometryMsg")
	localGeometry_ : LocalGeometryPlug = PlugDescriptor("localGeometry")
	localSweptGeometry_ : LocalSweptGeometryPlug = PlugDescriptor("localSweptGeometry")
	matrixModified_ : MatrixModifiedPlug = PlugDescriptor("matrixModified")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	owner_ : OwnerPlug = PlugDescriptor("owner")
	ownerCentroidX_ : OwnerCentroidXPlug = PlugDescriptor("ownerCentroidX")
	ownerCentroidY_ : OwnerCentroidYPlug = PlugDescriptor("ownerCentroidY")
	ownerCentroidZ_ : OwnerCentroidZPlug = PlugDescriptor("ownerCentroidZ")
	ownerCentroid_ : OwnerCentroidPlug = PlugDescriptor("ownerCentroid")
	ownerCentroidLocalX_ : OwnerCentroidLocalXPlug = PlugDescriptor("ownerCentroidLocalX")
	ownerCentroidLocalY_ : OwnerCentroidLocalYPlug = PlugDescriptor("ownerCentroidLocalY")
	ownerCentroidLocalZ_ : OwnerCentroidLocalZPlug = PlugDescriptor("ownerCentroidLocalZ")
	ownerCentroidLocal_ : OwnerCentroidLocalPlug = PlugDescriptor("ownerCentroidLocal")
	ownerMasses_ : OwnerMassesPlug = PlugDescriptor("ownerMasses")
	ownerPositions_ : OwnerPositionsPlug = PlugDescriptor("ownerPositions")
	ownerVelocities_ : OwnerVelocitiesPlug = PlugDescriptor("ownerVelocities")
	preComponentPositions_ : PreComponentPositionsPlug = PlugDescriptor("preComponentPositions")
	preOwnerPositions_ : PreOwnerPositionsPlug = PlugDescriptor("preOwnerPositions")
	prevTime_ : PrevTimePlug = PlugDescriptor("prevTime")
	ratePPIn_ : RatePPInPlug = PlugDescriptor("ratePPIn")
	ratePPOut_ : RatePPOutPlug = PlugDescriptor("ratePPOut")
	resilience_ : ResiliencePlug = PlugDescriptor("resilience")
	sweptGeometry_ : SweptGeometryPlug = PlugDescriptor("sweptGeometry")
	tessellationFactor_ : TessellationFactorPlug = PlugDescriptor("tessellationFactor")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")
	worldMatrix_ : WorldMatrixPlug = PlugDescriptor("worldMatrix")

	# node attributes

	typeName = "geoConnector"
	apiTypeInt = 922
	apiTypeStr = "kGeoConnector"
	typeIdInt = 1497842516
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "componentCentroidX", "componentCentroidY", "componentCentroidZ", "componentCentroid", "componentCentroidLocalX", "componentCentroidLocalY", "componentCentroidLocalZ", "componentCentroidLocal", "componentPositions", "componentVelocities", "currentTime", "deltaTime", "friction", "geometryModified", "groupId", "idIndex", "sortedId", "idMapping", "inputForce", "inputGeometryMsg", "localGeometry", "localSweptGeometry", "matrixModified", "offset", "owner", "ownerCentroidX", "ownerCentroidY", "ownerCentroidZ", "ownerCentroid", "ownerCentroidLocalX", "ownerCentroidLocalY", "ownerCentroidLocalZ", "ownerCentroidLocal", "ownerMasses", "ownerPositions", "ownerVelocities", "preComponentPositions", "preOwnerPositions", "prevTime", "ratePPIn", "ratePPOut", "resilience", "sweptGeometry", "tessellationFactor", "uvSetName", "worldMatrix"]
	nodeLeafPlugs = ["binMembership", "componentCentroid", "componentCentroidLocal", "componentPositions", "componentVelocities", "currentTime", "deltaTime", "friction", "geometryModified", "groupId", "idMapping", "inputForce", "inputGeometryMsg", "localGeometry", "localSweptGeometry", "matrixModified", "offset", "owner", "ownerCentroid", "ownerCentroidLocal", "ownerMasses", "ownerPositions", "ownerVelocities", "preComponentPositions", "preOwnerPositions", "prevTime", "ratePPIn", "ratePPOut", "resilience", "sweptGeometry", "tessellationFactor", "uvSetName", "worldMatrix"]
	pass

