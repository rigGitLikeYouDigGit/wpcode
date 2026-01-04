

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
	node : ClosestPointOnMesh = None
	pass
class InMeshPlug(Plug):
	node : ClosestPointOnMesh = None
	pass
class InPositionXPlug(Plug):
	parent : InPositionPlug = PlugDescriptor("inPosition")
	node : ClosestPointOnMesh = None
	pass
class InPositionYPlug(Plug):
	parent : InPositionPlug = PlugDescriptor("inPosition")
	node : ClosestPointOnMesh = None
	pass
class InPositionZPlug(Plug):
	parent : InPositionPlug = PlugDescriptor("inPosition")
	node : ClosestPointOnMesh = None
	pass
class InPositionPlug(Plug):
	inPositionX_ : InPositionXPlug = PlugDescriptor("inPositionX")
	ipx_ : InPositionXPlug = PlugDescriptor("inPositionX")
	inPositionY_ : InPositionYPlug = PlugDescriptor("inPositionY")
	ipy_ : InPositionYPlug = PlugDescriptor("inPositionY")
	inPositionZ_ : InPositionZPlug = PlugDescriptor("inPositionZ")
	ipz_ : InPositionZPlug = PlugDescriptor("inPositionZ")
	node : ClosestPointOnMesh = None
	pass
class InputMatrixPlug(Plug):
	node : ClosestPointOnMesh = None
	pass
class ClosestFaceIndexPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	node : ClosestPointOnMesh = None
	pass
class ClosestVertexIndexPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	node : ClosestPointOnMesh = None
	pass
class NormalXPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : ClosestPointOnMesh = None
	pass
class NormalYPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : ClosestPointOnMesh = None
	pass
class NormalZPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : ClosestPointOnMesh = None
	pass
class NormalPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	nx_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	ny_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	nz_ : NormalZPlug = PlugDescriptor("normalZ")
	node : ClosestPointOnMesh = None
	pass
class ParameterUPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	node : ClosestPointOnMesh = None
	pass
class ParameterVPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	node : ClosestPointOnMesh = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : ClosestPointOnMesh = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : ClosestPointOnMesh = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : ClosestPointOnMesh = None
	pass
class PositionPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	pz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : ClosestPointOnMesh = None
	pass
class ResultPlug(Plug):
	closestFaceIndex_ : ClosestFaceIndexPlug = PlugDescriptor("closestFaceIndex")
	f_ : ClosestFaceIndexPlug = PlugDescriptor("closestFaceIndex")
	closestVertexIndex_ : ClosestVertexIndexPlug = PlugDescriptor("closestVertexIndex")
	vt_ : ClosestVertexIndexPlug = PlugDescriptor("closestVertexIndex")
	normal_ : NormalPlug = PlugDescriptor("normal")
	n_ : NormalPlug = PlugDescriptor("normal")
	parameterU_ : ParameterUPlug = PlugDescriptor("parameterU")
	u_ : ParameterUPlug = PlugDescriptor("parameterU")
	parameterV_ : ParameterVPlug = PlugDescriptor("parameterV")
	v_ : ParameterVPlug = PlugDescriptor("parameterV")
	position_ : PositionPlug = PlugDescriptor("position")
	p_ : PositionPlug = PlugDescriptor("position")
	node : ClosestPointOnMesh = None
	pass
# endregion


# define node class
class ClosestPointOnMesh(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inMesh_ : InMeshPlug = PlugDescriptor("inMesh")
	inPositionX_ : InPositionXPlug = PlugDescriptor("inPositionX")
	inPositionY_ : InPositionYPlug = PlugDescriptor("inPositionY")
	inPositionZ_ : InPositionZPlug = PlugDescriptor("inPositionZ")
	inPosition_ : InPositionPlug = PlugDescriptor("inPosition")
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	closestFaceIndex_ : ClosestFaceIndexPlug = PlugDescriptor("closestFaceIndex")
	closestVertexIndex_ : ClosestVertexIndexPlug = PlugDescriptor("closestVertexIndex")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	normal_ : NormalPlug = PlugDescriptor("normal")
	parameterU_ : ParameterUPlug = PlugDescriptor("parameterU")
	parameterV_ : ParameterVPlug = PlugDescriptor("parameterV")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	result_ : ResultPlug = PlugDescriptor("result")

	# node attributes

	typeName = "closestPointOnMesh"
	apiTypeInt = 989
	apiTypeStr = "kClosestPointOnMesh"
	typeIdInt = 1129336653
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "inMesh", "inPositionX", "inPositionY", "inPositionZ", "inPosition", "inputMatrix", "closestFaceIndex", "closestVertexIndex", "normalX", "normalY", "normalZ", "normal", "parameterU", "parameterV", "positionX", "positionY", "positionZ", "position", "result"]
	nodeLeafPlugs = ["binMembership", "inMesh", "inPosition", "inputMatrix", "result"]
	pass

