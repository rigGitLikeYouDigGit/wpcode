

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Shape = Catalogue.Shape
else:
	from .. import retriever
	Shape = retriever.getNodeCls("Shape")
	assert Shape

# add node doc



# region plug type defs
class AuxiliariesOwnedPlug(Plug):
	node : GeoConnectable = None
	pass
class CachedPositionsPlug(Plug):
	node : GeoConnectable = None
	pass
class CachedVelocitiesPlug(Plug):
	node : GeoConnectable = None
	pass
class ComponentPositionsPlug(Plug):
	node : GeoConnectable = None
	pass
class ConnectionsToMePlug(Plug):
	node : GeoConnectable = None
	pass
class DoVelocityPlug(Plug):
	node : GeoConnectable = None
	pass
class GroupIdPlug(Plug):
	node : GeoConnectable = None
	pass
class InputGeometryMsgPlug(Plug):
	node : GeoConnectable = None
	pass
class LocalSurfaceGeometryPlug(Plug):
	node : GeoConnectable = None
	pass
class PrevTimePlug(Plug):
	node : GeoConnectable = None
	pass
class SurfaceGeometryPlug(Plug):
	node : GeoConnectable = None
	pass
class VelocityValidPlug(Plug):
	node : GeoConnectable = None
	pass
# endregion


# define node class
class GeoConnectable(Shape):
	auxiliariesOwned_ : AuxiliariesOwnedPlug = PlugDescriptor("auxiliariesOwned")
	cachedPositions_ : CachedPositionsPlug = PlugDescriptor("cachedPositions")
	cachedVelocities_ : CachedVelocitiesPlug = PlugDescriptor("cachedVelocities")
	componentPositions_ : ComponentPositionsPlug = PlugDescriptor("componentPositions")
	connectionsToMe_ : ConnectionsToMePlug = PlugDescriptor("connectionsToMe")
	doVelocity_ : DoVelocityPlug = PlugDescriptor("doVelocity")
	groupId_ : GroupIdPlug = PlugDescriptor("groupId")
	inputGeometryMsg_ : InputGeometryMsgPlug = PlugDescriptor("inputGeometryMsg")
	localSurfaceGeometry_ : LocalSurfaceGeometryPlug = PlugDescriptor("localSurfaceGeometry")
	prevTime_ : PrevTimePlug = PlugDescriptor("prevTime")
	surfaceGeometry_ : SurfaceGeometryPlug = PlugDescriptor("surfaceGeometry")
	velocityValid_ : VelocityValidPlug = PlugDescriptor("velocityValid")

	# node attributes

	typeName = "geoConnectable"
	apiTypeInt = 326
	apiTypeStr = "kGeoConnectable"
	typeIdInt = 1497842511
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["auxiliariesOwned", "cachedPositions", "cachedVelocities", "componentPositions", "connectionsToMe", "doVelocity", "groupId", "inputGeometryMsg", "localSurfaceGeometry", "prevTime", "surfaceGeometry", "velocityValid"]
	nodeLeafPlugs = ["auxiliariesOwned", "cachedPositions", "cachedVelocities", "componentPositions", "connectionsToMe", "doVelocity", "groupId", "inputGeometryMsg", "localSurfaceGeometry", "prevTime", "surfaceGeometry", "velocityValid"]
	pass

