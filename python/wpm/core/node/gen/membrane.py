

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
class BendAngleDropoffPlug(Plug):
	node : Membrane = None
	pass
class BendAngleScalePlug(Plug):
	node : Membrane = None
	pass
class BendResistancePlug(Plug):
	node : Membrane = None
	pass
class BinMembershipPlug(Plug):
	node : Membrane = None
	pass
class CollidePlug(Plug):
	node : Membrane = None
	pass
class CollideMeshPlug(Plug):
	node : Membrane = None
	pass
class CompressionResistancePlug(Plug):
	node : Membrane = None
	pass
class DragPlug(Plug):
	node : Membrane = None
	pass
class EnablePlug(Plug):
	node : Membrane = None
	pass
class FrictionPlug(Plug):
	node : Membrane = None
	pass
class GravityPlug(Plug):
	node : Membrane = None
	pass
class GravityDirectionXPlug(Plug):
	parent : GravityDirectionPlug = PlugDescriptor("gravityDirection")
	node : Membrane = None
	pass
class GravityDirectionYPlug(Plug):
	parent : GravityDirectionPlug = PlugDescriptor("gravityDirection")
	node : Membrane = None
	pass
class GravityDirectionZPlug(Plug):
	parent : GravityDirectionPlug = PlugDescriptor("gravityDirection")
	node : Membrane = None
	pass
class GravityDirectionPlug(Plug):
	gravityDirectionX_ : GravityDirectionXPlug = PlugDescriptor("gravityDirectionX")
	grdx_ : GravityDirectionXPlug = PlugDescriptor("gravityDirectionX")
	gravityDirectionY_ : GravityDirectionYPlug = PlugDescriptor("gravityDirectionY")
	grdy_ : GravityDirectionYPlug = PlugDescriptor("gravityDirectionY")
	gravityDirectionZ_ : GravityDirectionZPlug = PlugDescriptor("gravityDirectionZ")
	grdz_ : GravityDirectionZPlug = PlugDescriptor("gravityDirectionZ")
	node : Membrane = None
	pass
class InputMatrixPlug(Plug):
	node : Membrane = None
	pass
class InputMeshPlug(Plug):
	node : Membrane = None
	pass
class LiftPlug(Plug):
	node : Membrane = None
	pass
class OutputMeshPlug(Plug):
	node : Membrane = None
	pass
class PressurePlug(Plug):
	node : Membrane = None
	pass
class PressureMethodPlug(Plug):
	node : Membrane = None
	pass
class PushOutPlug(Plug):
	node : Membrane = None
	pass
class PushOutRadiusPlug(Plug):
	node : Membrane = None
	pass
class RestLengthScalePlug(Plug):
	node : Membrane = None
	pass
class RestShapeMeshPlug(Plug):
	node : Membrane = None
	pass
class RigidityPlug(Plug):
	node : Membrane = None
	pass
class SelfCollidePlug(Plug):
	node : Membrane = None
	pass
class SelfCollideWidthScalePlug(Plug):
	node : Membrane = None
	pass
class SelfCollisionFlagPlug(Plug):
	node : Membrane = None
	pass
class ShearResistancePlug(Plug):
	node : Membrane = None
	pass
class SpaceScalePlug(Plug):
	node : Membrane = None
	pass
class StepSizePlug(Plug):
	node : Membrane = None
	pass
class StepsPlug(Plug):
	node : Membrane = None
	pass
class StretchResistancePlug(Plug):
	node : Membrane = None
	pass
class SubStepsPlug(Plug):
	node : Membrane = None
	pass
class TangentialDragPlug(Plug):
	node : Membrane = None
	pass
class ThicknessPlug(Plug):
	node : Membrane = None
	pass
class ThicknessPerVertexPlug(Plug):
	node : Membrane = None
	pass
class TurbulencePlug(Plug):
	node : Membrane = None
	pass
class TurbulenceFrequencyPlug(Plug):
	node : Membrane = None
	pass
class TurbulenceOffsetXPlug(Plug):
	parent : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	node : Membrane = None
	pass
class TurbulenceOffsetYPlug(Plug):
	parent : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	node : Membrane = None
	pass
class TurbulenceOffsetZPlug(Plug):
	parent : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	node : Membrane = None
	pass
class TurbulenceOffsetPlug(Plug):
	turbulenceOffsetX_ : TurbulenceOffsetXPlug = PlugDescriptor("turbulenceOffsetX")
	tox_ : TurbulenceOffsetXPlug = PlugDescriptor("turbulenceOffsetX")
	turbulenceOffsetY_ : TurbulenceOffsetYPlug = PlugDescriptor("turbulenceOffsetY")
	toy_ : TurbulenceOffsetYPlug = PlugDescriptor("turbulenceOffsetY")
	turbulenceOffsetZ_ : TurbulenceOffsetZPlug = PlugDescriptor("turbulenceOffsetZ")
	toz_ : TurbulenceOffsetZPlug = PlugDescriptor("turbulenceOffsetZ")
	node : Membrane = None
	pass
class TurbulencePerVertexPlug(Plug):
	node : Membrane = None
	pass
class TurbulenceTimePlug(Plug):
	node : Membrane = None
	pass
class WeightPerVertexPlug(Plug):
	node : Membrane = None
	pass
class WindDirectionXPlug(Plug):
	parent : WindDirectionPlug = PlugDescriptor("windDirection")
	node : Membrane = None
	pass
class WindDirectionYPlug(Plug):
	parent : WindDirectionPlug = PlugDescriptor("windDirection")
	node : Membrane = None
	pass
class WindDirectionZPlug(Plug):
	parent : WindDirectionPlug = PlugDescriptor("windDirection")
	node : Membrane = None
	pass
class WindDirectionPlug(Plug):
	windDirectionX_ : WindDirectionXPlug = PlugDescriptor("windDirectionX")
	widx_ : WindDirectionXPlug = PlugDescriptor("windDirectionX")
	windDirectionY_ : WindDirectionYPlug = PlugDescriptor("windDirectionY")
	widy_ : WindDirectionYPlug = PlugDescriptor("windDirectionY")
	windDirectionZ_ : WindDirectionZPlug = PlugDescriptor("windDirectionZ")
	widz_ : WindDirectionZPlug = PlugDescriptor("windDirectionZ")
	node : Membrane = None
	pass
class WindSpeedPlug(Plug):
	node : Membrane = None
	pass
# endregion


# define node class
class Membrane(_BASE_):
	bendAngleDropoff_ : BendAngleDropoffPlug = PlugDescriptor("bendAngleDropoff")
	bendAngleScale_ : BendAngleScalePlug = PlugDescriptor("bendAngleScale")
	bendResistance_ : BendResistancePlug = PlugDescriptor("bendResistance")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	collide_ : CollidePlug = PlugDescriptor("collide")
	collideMesh_ : CollideMeshPlug = PlugDescriptor("collideMesh")
	compressionResistance_ : CompressionResistancePlug = PlugDescriptor("compressionResistance")
	drag_ : DragPlug = PlugDescriptor("drag")
	enable_ : EnablePlug = PlugDescriptor("enable")
	friction_ : FrictionPlug = PlugDescriptor("friction")
	gravity_ : GravityPlug = PlugDescriptor("gravity")
	gravityDirectionX_ : GravityDirectionXPlug = PlugDescriptor("gravityDirectionX")
	gravityDirectionY_ : GravityDirectionYPlug = PlugDescriptor("gravityDirectionY")
	gravityDirectionZ_ : GravityDirectionZPlug = PlugDescriptor("gravityDirectionZ")
	gravityDirection_ : GravityDirectionPlug = PlugDescriptor("gravityDirection")
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	inputMesh_ : InputMeshPlug = PlugDescriptor("inputMesh")
	lift_ : LiftPlug = PlugDescriptor("lift")
	outputMesh_ : OutputMeshPlug = PlugDescriptor("outputMesh")
	pressure_ : PressurePlug = PlugDescriptor("pressure")
	pressureMethod_ : PressureMethodPlug = PlugDescriptor("pressureMethod")
	pushOut_ : PushOutPlug = PlugDescriptor("pushOut")
	pushOutRadius_ : PushOutRadiusPlug = PlugDescriptor("pushOutRadius")
	restLengthScale_ : RestLengthScalePlug = PlugDescriptor("restLengthScale")
	restShapeMesh_ : RestShapeMeshPlug = PlugDescriptor("restShapeMesh")
	rigidity_ : RigidityPlug = PlugDescriptor("rigidity")
	selfCollide_ : SelfCollidePlug = PlugDescriptor("selfCollide")
	selfCollideWidthScale_ : SelfCollideWidthScalePlug = PlugDescriptor("selfCollideWidthScale")
	selfCollisionFlag_ : SelfCollisionFlagPlug = PlugDescriptor("selfCollisionFlag")
	shearResistance_ : ShearResistancePlug = PlugDescriptor("shearResistance")
	spaceScale_ : SpaceScalePlug = PlugDescriptor("spaceScale")
	stepSize_ : StepSizePlug = PlugDescriptor("stepSize")
	steps_ : StepsPlug = PlugDescriptor("steps")
	stretchResistance_ : StretchResistancePlug = PlugDescriptor("stretchResistance")
	subSteps_ : SubStepsPlug = PlugDescriptor("subSteps")
	tangentialDrag_ : TangentialDragPlug = PlugDescriptor("tangentialDrag")
	thickness_ : ThicknessPlug = PlugDescriptor("thickness")
	thicknessPerVertex_ : ThicknessPerVertexPlug = PlugDescriptor("thicknessPerVertex")
	turbulence_ : TurbulencePlug = PlugDescriptor("turbulence")
	turbulenceFrequency_ : TurbulenceFrequencyPlug = PlugDescriptor("turbulenceFrequency")
	turbulenceOffsetX_ : TurbulenceOffsetXPlug = PlugDescriptor("turbulenceOffsetX")
	turbulenceOffsetY_ : TurbulenceOffsetYPlug = PlugDescriptor("turbulenceOffsetY")
	turbulenceOffsetZ_ : TurbulenceOffsetZPlug = PlugDescriptor("turbulenceOffsetZ")
	turbulenceOffset_ : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	turbulencePerVertex_ : TurbulencePerVertexPlug = PlugDescriptor("turbulencePerVertex")
	turbulenceTime_ : TurbulenceTimePlug = PlugDescriptor("turbulenceTime")
	weightPerVertex_ : WeightPerVertexPlug = PlugDescriptor("weightPerVertex")
	windDirectionX_ : WindDirectionXPlug = PlugDescriptor("windDirectionX")
	windDirectionY_ : WindDirectionYPlug = PlugDescriptor("windDirectionY")
	windDirectionZ_ : WindDirectionZPlug = PlugDescriptor("windDirectionZ")
	windDirection_ : WindDirectionPlug = PlugDescriptor("windDirection")
	windSpeed_ : WindSpeedPlug = PlugDescriptor("windSpeed")

	# node attributes

	typeName = "membrane"
	apiTypeInt = 1038
	apiTypeStr = "kMembrane"
	typeIdInt = 1296387394
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["bendAngleDropoff", "bendAngleScale", "bendResistance", "binMembership", "collide", "collideMesh", "compressionResistance", "drag", "enable", "friction", "gravity", "gravityDirectionX", "gravityDirectionY", "gravityDirectionZ", "gravityDirection", "inputMatrix", "inputMesh", "lift", "outputMesh", "pressure", "pressureMethod", "pushOut", "pushOutRadius", "restLengthScale", "restShapeMesh", "rigidity", "selfCollide", "selfCollideWidthScale", "selfCollisionFlag", "shearResistance", "spaceScale", "stepSize", "steps", "stretchResistance", "subSteps", "tangentialDrag", "thickness", "thicknessPerVertex", "turbulence", "turbulenceFrequency", "turbulenceOffsetX", "turbulenceOffsetY", "turbulenceOffsetZ", "turbulenceOffset", "turbulencePerVertex", "turbulenceTime", "weightPerVertex", "windDirectionX", "windDirectionY", "windDirectionZ", "windDirection", "windSpeed"]
	nodeLeafPlugs = ["bendAngleDropoff", "bendAngleScale", "bendResistance", "binMembership", "collide", "collideMesh", "compressionResistance", "drag", "enable", "friction", "gravity", "gravityDirection", "inputMatrix", "inputMesh", "lift", "outputMesh", "pressure", "pressureMethod", "pushOut", "pushOutRadius", "restLengthScale", "restShapeMesh", "rigidity", "selfCollide", "selfCollideWidthScale", "selfCollisionFlag", "shearResistance", "spaceScale", "stepSize", "steps", "stretchResistance", "subSteps", "tangentialDrag", "thickness", "thicknessPerVertex", "turbulence", "turbulenceFrequency", "turbulenceOffset", "turbulencePerVertex", "turbulenceTime", "weightPerVertex", "windDirection", "windSpeed"]
	pass

