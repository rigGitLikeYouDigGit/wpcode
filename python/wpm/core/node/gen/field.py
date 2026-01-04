

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	DynBase = Catalogue.DynBase
else:
	from .. import retriever
	DynBase = retriever.getNodeCls("DynBase")
	assert DynBase

# add node doc



# region plug type defs
class ApplyPerVertexPlug(Plug):
	node : Field = None
	pass
class AttenuationPlug(Plug):
	node : Field = None
	pass
class AxialMagnitude_FloatValuePlug(Plug):
	parent : AxialMagnitudePlug = PlugDescriptor("axialMagnitude")
	node : Field = None
	pass
class AxialMagnitude_InterpPlug(Plug):
	parent : AxialMagnitudePlug = PlugDescriptor("axialMagnitude")
	node : Field = None
	pass
class AxialMagnitude_PositionPlug(Plug):
	parent : AxialMagnitudePlug = PlugDescriptor("axialMagnitude")
	node : Field = None
	pass
class AxialMagnitudePlug(Plug):
	axialMagnitude_FloatValue_ : AxialMagnitude_FloatValuePlug = PlugDescriptor("axialMagnitude_FloatValue")
	amagfv_ : AxialMagnitude_FloatValuePlug = PlugDescriptor("axialMagnitude_FloatValue")
	axialMagnitude_Interp_ : AxialMagnitude_InterpPlug = PlugDescriptor("axialMagnitude_Interp")
	amagi_ : AxialMagnitude_InterpPlug = PlugDescriptor("axialMagnitude_Interp")
	axialMagnitude_Position_ : AxialMagnitude_PositionPlug = PlugDescriptor("axialMagnitude_Position")
	amagp_ : AxialMagnitude_PositionPlug = PlugDescriptor("axialMagnitude_Position")
	node : Field = None
	pass
class CurveRadius_FloatValuePlug(Plug):
	parent : CurveRadiusPlug = PlugDescriptor("curveRadius")
	node : Field = None
	pass
class CurveRadius_InterpPlug(Plug):
	parent : CurveRadiusPlug = PlugDescriptor("curveRadius")
	node : Field = None
	pass
class CurveRadius_PositionPlug(Plug):
	parent : CurveRadiusPlug = PlugDescriptor("curveRadius")
	node : Field = None
	pass
class CurveRadiusPlug(Plug):
	curveRadius_FloatValue_ : CurveRadius_FloatValuePlug = PlugDescriptor("curveRadius_FloatValue")
	cradfv_ : CurveRadius_FloatValuePlug = PlugDescriptor("curveRadius_FloatValue")
	curveRadius_Interp_ : CurveRadius_InterpPlug = PlugDescriptor("curveRadius_Interp")
	cradi_ : CurveRadius_InterpPlug = PlugDescriptor("curveRadius_Interp")
	curveRadius_Position_ : CurveRadius_PositionPlug = PlugDescriptor("curveRadius_Position")
	cradp_ : CurveRadius_PositionPlug = PlugDescriptor("curveRadius_Position")
	node : Field = None
	pass
class FalloffCurve_FloatValuePlug(Plug):
	parent : FalloffCurvePlug = PlugDescriptor("falloffCurve")
	node : Field = None
	pass
class FalloffCurve_InterpPlug(Plug):
	parent : FalloffCurvePlug = PlugDescriptor("falloffCurve")
	node : Field = None
	pass
class FalloffCurve_PositionPlug(Plug):
	parent : FalloffCurvePlug = PlugDescriptor("falloffCurve")
	node : Field = None
	pass
class FalloffCurvePlug(Plug):
	falloffCurve_FloatValue_ : FalloffCurve_FloatValuePlug = PlugDescriptor("falloffCurve_FloatValue")
	fcfv_ : FalloffCurve_FloatValuePlug = PlugDescriptor("falloffCurve_FloatValue")
	falloffCurve_Interp_ : FalloffCurve_InterpPlug = PlugDescriptor("falloffCurve_Interp")
	fci_ : FalloffCurve_InterpPlug = PlugDescriptor("falloffCurve_Interp")
	falloffCurve_Position_ : FalloffCurve_PositionPlug = PlugDescriptor("falloffCurve_Position")
	fcp_ : FalloffCurve_PositionPlug = PlugDescriptor("falloffCurve_Position")
	node : Field = None
	pass
class InputCurvePlug(Plug):
	node : Field = None
	pass
class DeltaTimePlug(Plug):
	parent : InputDataPlug = PlugDescriptor("inputData")
	node : Field = None
	pass
class InputMassPlug(Plug):
	parent : InputDataPlug = PlugDescriptor("inputData")
	node : Field = None
	pass
class InputPositionsPlug(Plug):
	parent : InputDataPlug = PlugDescriptor("inputData")
	node : Field = None
	pass
class InputVelocitiesPlug(Plug):
	parent : InputDataPlug = PlugDescriptor("inputData")
	node : Field = None
	pass
class InputDataPlug(Plug):
	deltaTime_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	dt_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	inputMass_ : InputMassPlug = PlugDescriptor("inputMass")
	inm_ : InputMassPlug = PlugDescriptor("inputMass")
	inputPositions_ : InputPositionsPlug = PlugDescriptor("inputPositions")
	inp_ : InputPositionsPlug = PlugDescriptor("inputPositions")
	inputVelocities_ : InputVelocitiesPlug = PlugDescriptor("inputVelocities")
	inv_ : InputVelocitiesPlug = PlugDescriptor("inputVelocities")
	node : Field = None
	pass
class InputForcePlug(Plug):
	node : Field = None
	pass
class InputPPDataPlug(Plug):
	node : Field = None
	pass
class MagnitudePlug(Plug):
	node : Field = None
	pass
class MaxDistancePlug(Plug):
	node : Field = None
	pass
class OutputForcePlug(Plug):
	node : Field = None
	pass
class OwnerPPDataPlug(Plug):
	node : Field = None
	pass
class SectionRadiusPlug(Plug):
	node : Field = None
	pass
class TrapEndsPlug(Plug):
	node : Field = None
	pass
class TrapInsidePlug(Plug):
	node : Field = None
	pass
class TrapRadiusPlug(Plug):
	node : Field = None
	pass
class UseMaxDistancePlug(Plug):
	node : Field = None
	pass
class VolumeExclusionPlug(Plug):
	node : Field = None
	pass
class VolumeOffsetXPlug(Plug):
	parent : VolumeOffsetPlug = PlugDescriptor("volumeOffset")
	node : Field = None
	pass
class VolumeOffsetYPlug(Plug):
	parent : VolumeOffsetPlug = PlugDescriptor("volumeOffset")
	node : Field = None
	pass
class VolumeOffsetZPlug(Plug):
	parent : VolumeOffsetPlug = PlugDescriptor("volumeOffset")
	node : Field = None
	pass
class VolumeOffsetPlug(Plug):
	volumeOffsetX_ : VolumeOffsetXPlug = PlugDescriptor("volumeOffsetX")
	vox_ : VolumeOffsetXPlug = PlugDescriptor("volumeOffsetX")
	volumeOffsetY_ : VolumeOffsetYPlug = PlugDescriptor("volumeOffsetY")
	voy_ : VolumeOffsetYPlug = PlugDescriptor("volumeOffsetY")
	volumeOffsetZ_ : VolumeOffsetZPlug = PlugDescriptor("volumeOffsetZ")
	voz_ : VolumeOffsetZPlug = PlugDescriptor("volumeOffsetZ")
	node : Field = None
	pass
class VolumeShapePlug(Plug):
	node : Field = None
	pass
class VolumeSweepPlug(Plug):
	node : Field = None
	pass
# endregion


# define node class
class Field(DynBase):
	applyPerVertex_ : ApplyPerVertexPlug = PlugDescriptor("applyPerVertex")
	attenuation_ : AttenuationPlug = PlugDescriptor("attenuation")
	axialMagnitude_FloatValue_ : AxialMagnitude_FloatValuePlug = PlugDescriptor("axialMagnitude_FloatValue")
	axialMagnitude_Interp_ : AxialMagnitude_InterpPlug = PlugDescriptor("axialMagnitude_Interp")
	axialMagnitude_Position_ : AxialMagnitude_PositionPlug = PlugDescriptor("axialMagnitude_Position")
	axialMagnitude_ : AxialMagnitudePlug = PlugDescriptor("axialMagnitude")
	curveRadius_FloatValue_ : CurveRadius_FloatValuePlug = PlugDescriptor("curveRadius_FloatValue")
	curveRadius_Interp_ : CurveRadius_InterpPlug = PlugDescriptor("curveRadius_Interp")
	curveRadius_Position_ : CurveRadius_PositionPlug = PlugDescriptor("curveRadius_Position")
	curveRadius_ : CurveRadiusPlug = PlugDescriptor("curveRadius")
	falloffCurve_FloatValue_ : FalloffCurve_FloatValuePlug = PlugDescriptor("falloffCurve_FloatValue")
	falloffCurve_Interp_ : FalloffCurve_InterpPlug = PlugDescriptor("falloffCurve_Interp")
	falloffCurve_Position_ : FalloffCurve_PositionPlug = PlugDescriptor("falloffCurve_Position")
	falloffCurve_ : FalloffCurvePlug = PlugDescriptor("falloffCurve")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	deltaTime_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	inputMass_ : InputMassPlug = PlugDescriptor("inputMass")
	inputPositions_ : InputPositionsPlug = PlugDescriptor("inputPositions")
	inputVelocities_ : InputVelocitiesPlug = PlugDescriptor("inputVelocities")
	inputData_ : InputDataPlug = PlugDescriptor("inputData")
	inputForce_ : InputForcePlug = PlugDescriptor("inputForce")
	inputPPData_ : InputPPDataPlug = PlugDescriptor("inputPPData")
	magnitude_ : MagnitudePlug = PlugDescriptor("magnitude")
	maxDistance_ : MaxDistancePlug = PlugDescriptor("maxDistance")
	outputForce_ : OutputForcePlug = PlugDescriptor("outputForce")
	ownerPPData_ : OwnerPPDataPlug = PlugDescriptor("ownerPPData")
	sectionRadius_ : SectionRadiusPlug = PlugDescriptor("sectionRadius")
	trapEnds_ : TrapEndsPlug = PlugDescriptor("trapEnds")
	trapInside_ : TrapInsidePlug = PlugDescriptor("trapInside")
	trapRadius_ : TrapRadiusPlug = PlugDescriptor("trapRadius")
	useMaxDistance_ : UseMaxDistancePlug = PlugDescriptor("useMaxDistance")
	volumeExclusion_ : VolumeExclusionPlug = PlugDescriptor("volumeExclusion")
	volumeOffsetX_ : VolumeOffsetXPlug = PlugDescriptor("volumeOffsetX")
	volumeOffsetY_ : VolumeOffsetYPlug = PlugDescriptor("volumeOffsetY")
	volumeOffsetZ_ : VolumeOffsetZPlug = PlugDescriptor("volumeOffsetZ")
	volumeOffset_ : VolumeOffsetPlug = PlugDescriptor("volumeOffset")
	volumeShape_ : VolumeShapePlug = PlugDescriptor("volumeShape")
	volumeSweep_ : VolumeSweepPlug = PlugDescriptor("volumeSweep")

	# node attributes

	typeName = "field"
	typeIdInt = 1497779268
	nodeLeafClassAttrs = ["applyPerVertex", "attenuation", "axialMagnitude_FloatValue", "axialMagnitude_Interp", "axialMagnitude_Position", "axialMagnitude", "curveRadius_FloatValue", "curveRadius_Interp", "curveRadius_Position", "curveRadius", "falloffCurve_FloatValue", "falloffCurve_Interp", "falloffCurve_Position", "falloffCurve", "inputCurve", "deltaTime", "inputMass", "inputPositions", "inputVelocities", "inputData", "inputForce", "inputPPData", "magnitude", "maxDistance", "outputForce", "ownerPPData", "sectionRadius", "trapEnds", "trapInside", "trapRadius", "useMaxDistance", "volumeExclusion", "volumeOffsetX", "volumeOffsetY", "volumeOffsetZ", "volumeOffset", "volumeShape", "volumeSweep"]
	nodeLeafPlugs = ["applyPerVertex", "attenuation", "axialMagnitude", "curveRadius", "falloffCurve", "inputCurve", "inputData", "inputForce", "inputPPData", "magnitude", "maxDistance", "outputForce", "ownerPPData", "sectionRadius", "trapEnds", "trapInside", "trapRadius", "useMaxDistance", "volumeExclusion", "volumeOffset", "volumeShape", "volumeSweep"]
	pass

