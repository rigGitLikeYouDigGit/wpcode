

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Shape = retriever.getNodeCls("Shape")
assert Shape
if T.TYPE_CHECKING:
	from .. import Shape

# add node doc



# region plug type defs
class AttractionDampPlug(Plug):
	node : Follicle = None
	pass
class AttractionScale_FloatValuePlug(Plug):
	parent : AttractionScalePlug = PlugDescriptor("attractionScale")
	node : Follicle = None
	pass
class AttractionScale_InterpPlug(Plug):
	parent : AttractionScalePlug = PlugDescriptor("attractionScale")
	node : Follicle = None
	pass
class AttractionScale_PositionPlug(Plug):
	parent : AttractionScalePlug = PlugDescriptor("attractionScale")
	node : Follicle = None
	pass
class AttractionScalePlug(Plug):
	attractionScale_FloatValue_ : AttractionScale_FloatValuePlug = PlugDescriptor("attractionScale_FloatValue")
	atsfv_ : AttractionScale_FloatValuePlug = PlugDescriptor("attractionScale_FloatValue")
	attractionScale_Interp_ : AttractionScale_InterpPlug = PlugDescriptor("attractionScale_Interp")
	atsi_ : AttractionScale_InterpPlug = PlugDescriptor("attractionScale_Interp")
	attractionScale_Position_ : AttractionScale_PositionPlug = PlugDescriptor("attractionScale_Position")
	atsp_ : AttractionScale_PositionPlug = PlugDescriptor("attractionScale_Position")
	node : Follicle = None
	pass
class BraidPlug(Plug):
	node : Follicle = None
	pass
class ClumpTwistOffsetPlug(Plug):
	node : Follicle = None
	pass
class ClumpWidthPlug(Plug):
	node : Follicle = None
	pass
class ClumpWidthMultPlug(Plug):
	node : Follicle = None
	pass
class ClumpWidthScale_FloatValuePlug(Plug):
	parent : ClumpWidthScalePlug = PlugDescriptor("clumpWidthScale")
	node : Follicle = None
	pass
class ClumpWidthScale_InterpPlug(Plug):
	parent : ClumpWidthScalePlug = PlugDescriptor("clumpWidthScale")
	node : Follicle = None
	pass
class ClumpWidthScale_PositionPlug(Plug):
	parent : ClumpWidthScalePlug = PlugDescriptor("clumpWidthScale")
	node : Follicle = None
	pass
class ClumpWidthScalePlug(Plug):
	clumpWidthScale_FloatValue_ : ClumpWidthScale_FloatValuePlug = PlugDescriptor("clumpWidthScale_FloatValue")
	cwsfv_ : ClumpWidthScale_FloatValuePlug = PlugDescriptor("clumpWidthScale_FloatValue")
	clumpWidthScale_Interp_ : ClumpWidthScale_InterpPlug = PlugDescriptor("clumpWidthScale_Interp")
	cwsi_ : ClumpWidthScale_InterpPlug = PlugDescriptor("clumpWidthScale_Interp")
	clumpWidthScale_Position_ : ClumpWidthScale_PositionPlug = PlugDescriptor("clumpWidthScale_Position")
	cwsp_ : ClumpWidthScale_PositionPlug = PlugDescriptor("clumpWidthScale_Position")
	node : Follicle = None
	pass
class CollidePlug(Plug):
	node : Follicle = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Follicle = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Follicle = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Follicle = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : Follicle = None
	pass
class ColorBlendPlug(Plug):
	node : Follicle = None
	pass
class CurlMultPlug(Plug):
	node : Follicle = None
	pass
class CurrentPositionPlug(Plug):
	node : Follicle = None
	pass
class DampPlug(Plug):
	node : Follicle = None
	pass
class DegreePlug(Plug):
	node : Follicle = None
	pass
class DensityMultPlug(Plug):
	node : Follicle = None
	pass
class FixedSegmentLengthPlug(Plug):
	node : Follicle = None
	pass
class FlipDirectionPlug(Plug):
	node : Follicle = None
	pass
class HairSysGravityPlug(Plug):
	node : Follicle = None
	pass
class HairSysStiffnessPlug(Plug):
	node : Follicle = None
	pass
class InputMeshPlug(Plug):
	node : Follicle = None
	pass
class InputSurfacePlug(Plug):
	node : Follicle = None
	pass
class InputWorldMatrixPlug(Plug):
	node : Follicle = None
	pass
class LengthFlexPlug(Plug):
	node : Follicle = None
	pass
class MapSetNamePlug(Plug):
	node : Follicle = None
	pass
class OutCurvePlug(Plug):
	node : Follicle = None
	pass
class OutHairPlug(Plug):
	node : Follicle = None
	pass
class OutNormalXPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Follicle = None
	pass
class OutNormalYPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Follicle = None
	pass
class OutNormalZPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Follicle = None
	pass
class OutNormalPlug(Plug):
	outNormalX_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	onx_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	outNormalY_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	ony_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	outNormalZ_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	onz_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	node : Follicle = None
	pass
class OutRotateXPlug(Plug):
	parent : OutRotatePlug = PlugDescriptor("outRotate")
	node : Follicle = None
	pass
class OutRotateYPlug(Plug):
	parent : OutRotatePlug = PlugDescriptor("outRotate")
	node : Follicle = None
	pass
class OutRotateZPlug(Plug):
	parent : OutRotatePlug = PlugDescriptor("outRotate")
	node : Follicle = None
	pass
class OutRotatePlug(Plug):
	outRotateX_ : OutRotateXPlug = PlugDescriptor("outRotateX")
	orx_ : OutRotateXPlug = PlugDescriptor("outRotateX")
	outRotateY_ : OutRotateYPlug = PlugDescriptor("outRotateY")
	ory_ : OutRotateYPlug = PlugDescriptor("outRotateY")
	outRotateZ_ : OutRotateZPlug = PlugDescriptor("outRotateZ")
	orz_ : OutRotateZPlug = PlugDescriptor("outRotateZ")
	node : Follicle = None
	pass
class OutTangentXPlug(Plug):
	parent : OutTangentPlug = PlugDescriptor("outTangent")
	node : Follicle = None
	pass
class OutTangentYPlug(Plug):
	parent : OutTangentPlug = PlugDescriptor("outTangent")
	node : Follicle = None
	pass
class OutTangentZPlug(Plug):
	parent : OutTangentPlug = PlugDescriptor("outTangent")
	node : Follicle = None
	pass
class OutTangentPlug(Plug):
	outTangentX_ : OutTangentXPlug = PlugDescriptor("outTangentX")
	otnx_ : OutTangentXPlug = PlugDescriptor("outTangentX")
	outTangentY_ : OutTangentYPlug = PlugDescriptor("outTangentY")
	otny_ : OutTangentYPlug = PlugDescriptor("outTangentY")
	outTangentZ_ : OutTangentZPlug = PlugDescriptor("outTangentZ")
	otnz_ : OutTangentZPlug = PlugDescriptor("outTangentZ")
	node : Follicle = None
	pass
class OutTranslateXPlug(Plug):
	parent : OutTranslatePlug = PlugDescriptor("outTranslate")
	node : Follicle = None
	pass
class OutTranslateYPlug(Plug):
	parent : OutTranslatePlug = PlugDescriptor("outTranslate")
	node : Follicle = None
	pass
class OutTranslateZPlug(Plug):
	parent : OutTranslatePlug = PlugDescriptor("outTranslate")
	node : Follicle = None
	pass
class OutTranslatePlug(Plug):
	outTranslateX_ : OutTranslateXPlug = PlugDescriptor("outTranslateX")
	otx_ : OutTranslateXPlug = PlugDescriptor("outTranslateX")
	outTranslateY_ : OutTranslateYPlug = PlugDescriptor("outTranslateY")
	oty_ : OutTranslateYPlug = PlugDescriptor("outTranslateY")
	outTranslateZ_ : OutTranslateZPlug = PlugDescriptor("outTranslateZ")
	otz_ : OutTranslateZPlug = PlugDescriptor("outTranslateZ")
	node : Follicle = None
	pass
class OverrideDynamicsPlug(Plug):
	node : Follicle = None
	pass
class ParameterUPlug(Plug):
	node : Follicle = None
	pass
class ParameterVPlug(Plug):
	node : Follicle = None
	pass
class PointLockPlug(Plug):
	node : Follicle = None
	pass
class RestPosePlug(Plug):
	node : Follicle = None
	pass
class RestPositionPlug(Plug):
	node : Follicle = None
	pass
class SampleDensityPlug(Plug):
	node : Follicle = None
	pass
class SegmentLengthPlug(Plug):
	node : Follicle = None
	pass
class SimulationMethodPlug(Plug):
	node : Follicle = None
	pass
class StartCurveAttractPlug(Plug):
	node : Follicle = None
	pass
class StartDirectionPlug(Plug):
	node : Follicle = None
	pass
class StartPositionPlug(Plug):
	node : Follicle = None
	pass
class StartPositionMatrixPlug(Plug):
	node : Follicle = None
	pass
class StiffnessPlug(Plug):
	node : Follicle = None
	pass
class StiffnessScale_FloatValuePlug(Plug):
	parent : StiffnessScalePlug = PlugDescriptor("stiffnessScale")
	node : Follicle = None
	pass
class StiffnessScale_InterpPlug(Plug):
	parent : StiffnessScalePlug = PlugDescriptor("stiffnessScale")
	node : Follicle = None
	pass
class StiffnessScale_PositionPlug(Plug):
	parent : StiffnessScalePlug = PlugDescriptor("stiffnessScale")
	node : Follicle = None
	pass
class StiffnessScalePlug(Plug):
	stiffnessScale_FloatValue_ : StiffnessScale_FloatValuePlug = PlugDescriptor("stiffnessScale_FloatValue")
	stsfv_ : StiffnessScale_FloatValuePlug = PlugDescriptor("stiffnessScale_FloatValue")
	stiffnessScale_Interp_ : StiffnessScale_InterpPlug = PlugDescriptor("stiffnessScale_Interp")
	stsi_ : StiffnessScale_InterpPlug = PlugDescriptor("stiffnessScale_Interp")
	stiffnessScale_Position_ : StiffnessScale_PositionPlug = PlugDescriptor("stiffnessScale_Position")
	stsp_ : StiffnessScale_PositionPlug = PlugDescriptor("stiffnessScale_Position")
	node : Follicle = None
	pass
class ValidUvPlug(Plug):
	node : Follicle = None
	pass
# endregion


# define node class
class Follicle(Shape):
	attractionDamp_ : AttractionDampPlug = PlugDescriptor("attractionDamp")
	attractionScale_FloatValue_ : AttractionScale_FloatValuePlug = PlugDescriptor("attractionScale_FloatValue")
	attractionScale_Interp_ : AttractionScale_InterpPlug = PlugDescriptor("attractionScale_Interp")
	attractionScale_Position_ : AttractionScale_PositionPlug = PlugDescriptor("attractionScale_Position")
	attractionScale_ : AttractionScalePlug = PlugDescriptor("attractionScale")
	braid_ : BraidPlug = PlugDescriptor("braid")
	clumpTwistOffset_ : ClumpTwistOffsetPlug = PlugDescriptor("clumpTwistOffset")
	clumpWidth_ : ClumpWidthPlug = PlugDescriptor("clumpWidth")
	clumpWidthMult_ : ClumpWidthMultPlug = PlugDescriptor("clumpWidthMult")
	clumpWidthScale_FloatValue_ : ClumpWidthScale_FloatValuePlug = PlugDescriptor("clumpWidthScale_FloatValue")
	clumpWidthScale_Interp_ : ClumpWidthScale_InterpPlug = PlugDescriptor("clumpWidthScale_Interp")
	clumpWidthScale_Position_ : ClumpWidthScale_PositionPlug = PlugDescriptor("clumpWidthScale_Position")
	clumpWidthScale_ : ClumpWidthScalePlug = PlugDescriptor("clumpWidthScale")
	collide_ : CollidePlug = PlugDescriptor("collide")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	colorBlend_ : ColorBlendPlug = PlugDescriptor("colorBlend")
	curlMult_ : CurlMultPlug = PlugDescriptor("curlMult")
	currentPosition_ : CurrentPositionPlug = PlugDescriptor("currentPosition")
	damp_ : DampPlug = PlugDescriptor("damp")
	degree_ : DegreePlug = PlugDescriptor("degree")
	densityMult_ : DensityMultPlug = PlugDescriptor("densityMult")
	fixedSegmentLength_ : FixedSegmentLengthPlug = PlugDescriptor("fixedSegmentLength")
	flipDirection_ : FlipDirectionPlug = PlugDescriptor("flipDirection")
	hairSysGravity_ : HairSysGravityPlug = PlugDescriptor("hairSysGravity")
	hairSysStiffness_ : HairSysStiffnessPlug = PlugDescriptor("hairSysStiffness")
	inputMesh_ : InputMeshPlug = PlugDescriptor("inputMesh")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	inputWorldMatrix_ : InputWorldMatrixPlug = PlugDescriptor("inputWorldMatrix")
	lengthFlex_ : LengthFlexPlug = PlugDescriptor("lengthFlex")
	mapSetName_ : MapSetNamePlug = PlugDescriptor("mapSetName")
	outCurve_ : OutCurvePlug = PlugDescriptor("outCurve")
	outHair_ : OutHairPlug = PlugDescriptor("outHair")
	outNormalX_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	outNormalY_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	outNormalZ_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	outNormal_ : OutNormalPlug = PlugDescriptor("outNormal")
	outRotateX_ : OutRotateXPlug = PlugDescriptor("outRotateX")
	outRotateY_ : OutRotateYPlug = PlugDescriptor("outRotateY")
	outRotateZ_ : OutRotateZPlug = PlugDescriptor("outRotateZ")
	outRotate_ : OutRotatePlug = PlugDescriptor("outRotate")
	outTangentX_ : OutTangentXPlug = PlugDescriptor("outTangentX")
	outTangentY_ : OutTangentYPlug = PlugDescriptor("outTangentY")
	outTangentZ_ : OutTangentZPlug = PlugDescriptor("outTangentZ")
	outTangent_ : OutTangentPlug = PlugDescriptor("outTangent")
	outTranslateX_ : OutTranslateXPlug = PlugDescriptor("outTranslateX")
	outTranslateY_ : OutTranslateYPlug = PlugDescriptor("outTranslateY")
	outTranslateZ_ : OutTranslateZPlug = PlugDescriptor("outTranslateZ")
	outTranslate_ : OutTranslatePlug = PlugDescriptor("outTranslate")
	overrideDynamics_ : OverrideDynamicsPlug = PlugDescriptor("overrideDynamics")
	parameterU_ : ParameterUPlug = PlugDescriptor("parameterU")
	parameterV_ : ParameterVPlug = PlugDescriptor("parameterV")
	pointLock_ : PointLockPlug = PlugDescriptor("pointLock")
	restPose_ : RestPosePlug = PlugDescriptor("restPose")
	restPosition_ : RestPositionPlug = PlugDescriptor("restPosition")
	sampleDensity_ : SampleDensityPlug = PlugDescriptor("sampleDensity")
	segmentLength_ : SegmentLengthPlug = PlugDescriptor("segmentLength")
	simulationMethod_ : SimulationMethodPlug = PlugDescriptor("simulationMethod")
	startCurveAttract_ : StartCurveAttractPlug = PlugDescriptor("startCurveAttract")
	startDirection_ : StartDirectionPlug = PlugDescriptor("startDirection")
	startPosition_ : StartPositionPlug = PlugDescriptor("startPosition")
	startPositionMatrix_ : StartPositionMatrixPlug = PlugDescriptor("startPositionMatrix")
	stiffness_ : StiffnessPlug = PlugDescriptor("stiffness")
	stiffnessScale_FloatValue_ : StiffnessScale_FloatValuePlug = PlugDescriptor("stiffnessScale_FloatValue")
	stiffnessScale_Interp_ : StiffnessScale_InterpPlug = PlugDescriptor("stiffnessScale_Interp")
	stiffnessScale_Position_ : StiffnessScale_PositionPlug = PlugDescriptor("stiffnessScale_Position")
	stiffnessScale_ : StiffnessScalePlug = PlugDescriptor("stiffnessScale")
	validUv_ : ValidUvPlug = PlugDescriptor("validUv")

	# node attributes

	typeName = "follicle"
	apiTypeInt = 935
	apiTypeStr = "kFollicle"
	typeIdInt = 1212371542
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["attractionDamp", "attractionScale_FloatValue", "attractionScale_Interp", "attractionScale_Position", "attractionScale", "braid", "clumpTwistOffset", "clumpWidth", "clumpWidthMult", "clumpWidthScale_FloatValue", "clumpWidthScale_Interp", "clumpWidthScale_Position", "clumpWidthScale", "collide", "colorB", "colorG", "colorR", "color", "colorBlend", "curlMult", "currentPosition", "damp", "degree", "densityMult", "fixedSegmentLength", "flipDirection", "hairSysGravity", "hairSysStiffness", "inputMesh", "inputSurface", "inputWorldMatrix", "lengthFlex", "mapSetName", "outCurve", "outHair", "outNormalX", "outNormalY", "outNormalZ", "outNormal", "outRotateX", "outRotateY", "outRotateZ", "outRotate", "outTangentX", "outTangentY", "outTangentZ", "outTangent", "outTranslateX", "outTranslateY", "outTranslateZ", "outTranslate", "overrideDynamics", "parameterU", "parameterV", "pointLock", "restPose", "restPosition", "sampleDensity", "segmentLength", "simulationMethod", "startCurveAttract", "startDirection", "startPosition", "startPositionMatrix", "stiffness", "stiffnessScale_FloatValue", "stiffnessScale_Interp", "stiffnessScale_Position", "stiffnessScale", "validUv"]
	nodeLeafPlugs = ["attractionDamp", "attractionScale", "braid", "clumpTwistOffset", "clumpWidth", "clumpWidthMult", "clumpWidthScale", "collide", "color", "colorBlend", "curlMult", "currentPosition", "damp", "degree", "densityMult", "fixedSegmentLength", "flipDirection", "hairSysGravity", "hairSysStiffness", "inputMesh", "inputSurface", "inputWorldMatrix", "lengthFlex", "mapSetName", "outCurve", "outHair", "outNormal", "outRotate", "outTangent", "outTranslate", "overrideDynamics", "parameterU", "parameterV", "pointLock", "restPose", "restPosition", "sampleDensity", "segmentLength", "simulationMethod", "startCurveAttract", "startDirection", "startPosition", "startPositionMatrix", "stiffness", "stiffnessScale", "validUv"]
	pass

