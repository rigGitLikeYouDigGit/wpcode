

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
	node : SkinBinding = None
	pass
class BindPreMatrixPlug(Plug):
	node : SkinBinding = None
	pass
class CurrentInfluencePlug(Plug):
	node : SkinBinding = None
	pass
class FalloffCurve_FloatValuePlug(Plug):
	parent : FalloffCurvePlug = PlugDescriptor("falloffCurve")
	node : SkinBinding = None
	pass
class FalloffCurve_InterpPlug(Plug):
	parent : FalloffCurvePlug = PlugDescriptor("falloffCurve")
	node : SkinBinding = None
	pass
class FalloffCurve_PositionPlug(Plug):
	parent : FalloffCurvePlug = PlugDescriptor("falloffCurve")
	node : SkinBinding = None
	pass
class FalloffCurvePlug(Plug):
	falloffCurve_FloatValue_ : FalloffCurve_FloatValuePlug = PlugDescriptor("falloffCurve_FloatValue")
	fcfv_ : FalloffCurve_FloatValuePlug = PlugDescriptor("falloffCurve_FloatValue")
	falloffCurve_Interp_ : FalloffCurve_InterpPlug = PlugDescriptor("falloffCurve_Interp")
	fci_ : FalloffCurve_InterpPlug = PlugDescriptor("falloffCurve_Interp")
	falloffCurve_Position_ : FalloffCurve_PositionPlug = PlugDescriptor("falloffCurve_Position")
	fcp_ : FalloffCurve_PositionPlug = PlugDescriptor("falloffCurve_Position")
	node : SkinBinding = None
	pass
class GeomMatrixPlug(Plug):
	node : SkinBinding = None
	pass
class InputGeometryPlug(Plug):
	node : SkinBinding = None
	pass
class LeftCapPlug(Plug):
	node : SkinBinding = None
	pass
class LeftRadiusPlug(Plug):
	node : SkinBinding = None
	pass
class LengthPlug(Plug):
	node : SkinBinding = None
	pass
class LocalMatrixPlug(Plug):
	node : SkinBinding = None
	pass
class OutWeightsPlug(Plug):
	node : SkinBinding = None
	pass
class ParentMatrixPlug(Plug):
	node : SkinBinding = None
	pass
class RightCapPlug(Plug):
	node : SkinBinding = None
	pass
class RightRadiusPlug(Plug):
	node : SkinBinding = None
	pass
class UpdateWeightsPlug(Plug):
	node : SkinBinding = None
	pass
# endregion


# define node class
class SkinBinding(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	bindPreMatrix_ : BindPreMatrixPlug = PlugDescriptor("bindPreMatrix")
	currentInfluence_ : CurrentInfluencePlug = PlugDescriptor("currentInfluence")
	falloffCurve_FloatValue_ : FalloffCurve_FloatValuePlug = PlugDescriptor("falloffCurve_FloatValue")
	falloffCurve_Interp_ : FalloffCurve_InterpPlug = PlugDescriptor("falloffCurve_Interp")
	falloffCurve_Position_ : FalloffCurve_PositionPlug = PlugDescriptor("falloffCurve_Position")
	falloffCurve_ : FalloffCurvePlug = PlugDescriptor("falloffCurve")
	geomMatrix_ : GeomMatrixPlug = PlugDescriptor("geomMatrix")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	leftCap_ : LeftCapPlug = PlugDescriptor("leftCap")
	leftRadius_ : LeftRadiusPlug = PlugDescriptor("leftRadius")
	length_ : LengthPlug = PlugDescriptor("length")
	localMatrix_ : LocalMatrixPlug = PlugDescriptor("localMatrix")
	outWeights_ : OutWeightsPlug = PlugDescriptor("outWeights")
	parentMatrix_ : ParentMatrixPlug = PlugDescriptor("parentMatrix")
	rightCap_ : RightCapPlug = PlugDescriptor("rightCap")
	rightRadius_ : RightRadiusPlug = PlugDescriptor("rightRadius")
	updateWeights_ : UpdateWeightsPlug = PlugDescriptor("updateWeights")

	# node attributes

	typeName = "skinBinding"
	apiTypeInt = 1062
	apiTypeStr = "kSkinBinding"
	typeIdInt = 1397441092
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "bindPreMatrix", "currentInfluence", "falloffCurve_FloatValue", "falloffCurve_Interp", "falloffCurve_Position", "falloffCurve", "geomMatrix", "inputGeometry", "leftCap", "leftRadius", "length", "localMatrix", "outWeights", "parentMatrix", "rightCap", "rightRadius", "updateWeights"]
	nodeLeafPlugs = ["binMembership", "bindPreMatrix", "currentInfluence", "falloffCurve", "geomMatrix", "inputGeometry", "leftCap", "leftRadius", "length", "localMatrix", "outWeights", "parentMatrix", "rightCap", "rightRadius", "updateWeights"]
	pass

