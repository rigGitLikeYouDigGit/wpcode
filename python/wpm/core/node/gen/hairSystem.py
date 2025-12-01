

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
class ActivePlug(Plug):
	node : HairSystem = None
	pass
class AttachObjectIdPlug(Plug):
	node : HairSystem = None
	pass
class AttractionDampPlug(Plug):
	node : HairSystem = None
	pass
class AttractionScale_FloatValuePlug(Plug):
	parent : AttractionScalePlug = PlugDescriptor("attractionScale")
	node : HairSystem = None
	pass
class AttractionScale_InterpPlug(Plug):
	parent : AttractionScalePlug = PlugDescriptor("attractionScale")
	node : HairSystem = None
	pass
class AttractionScale_PositionPlug(Plug):
	parent : AttractionScalePlug = PlugDescriptor("attractionScale")
	node : HairSystem = None
	pass
class AttractionScalePlug(Plug):
	attractionScale_FloatValue_ : AttractionScale_FloatValuePlug = PlugDescriptor("attractionScale_FloatValue")
	atsfv_ : AttractionScale_FloatValuePlug = PlugDescriptor("attractionScale_FloatValue")
	attractionScale_Interp_ : AttractionScale_InterpPlug = PlugDescriptor("attractionScale_Interp")
	atsi_ : AttractionScale_InterpPlug = PlugDescriptor("attractionScale_Interp")
	attractionScale_Position_ : AttractionScale_PositionPlug = PlugDescriptor("attractionScale_Position")
	atsp_ : AttractionScale_PositionPlug = PlugDescriptor("attractionScale_Position")
	node : HairSystem = None
	pass
class BaldnessMapPlug(Plug):
	node : HairSystem = None
	pass
class BendAnisotropyPlug(Plug):
	node : HairSystem = None
	pass
class BendFollowPlug(Plug):
	node : HairSystem = None
	pass
class BendModelPlug(Plug):
	node : HairSystem = None
	pass
class BendResistancePlug(Plug):
	node : HairSystem = None
	pass
class BouncePlug(Plug):
	node : HairSystem = None
	pass
class CacheableAttributesPlug(Plug):
	node : HairSystem = None
	pass
class CastShadowsPlug(Plug):
	node : HairSystem = None
	pass
class ClumpCurl_FloatValuePlug(Plug):
	parent : ClumpCurlPlug = PlugDescriptor("clumpCurl")
	node : HairSystem = None
	pass
class ClumpCurl_InterpPlug(Plug):
	parent : ClumpCurlPlug = PlugDescriptor("clumpCurl")
	node : HairSystem = None
	pass
class ClumpCurl_PositionPlug(Plug):
	parent : ClumpCurlPlug = PlugDescriptor("clumpCurl")
	node : HairSystem = None
	pass
class ClumpCurlPlug(Plug):
	clumpCurl_FloatValue_ : ClumpCurl_FloatValuePlug = PlugDescriptor("clumpCurl_FloatValue")
	clcfv_ : ClumpCurl_FloatValuePlug = PlugDescriptor("clumpCurl_FloatValue")
	clumpCurl_Interp_ : ClumpCurl_InterpPlug = PlugDescriptor("clumpCurl_Interp")
	clci_ : ClumpCurl_InterpPlug = PlugDescriptor("clumpCurl_Interp")
	clumpCurl_Position_ : ClumpCurl_PositionPlug = PlugDescriptor("clumpCurl_Position")
	clcp_ : ClumpCurl_PositionPlug = PlugDescriptor("clumpCurl_Position")
	node : HairSystem = None
	pass
class ClumpFlatness_FloatValuePlug(Plug):
	parent : ClumpFlatnessPlug = PlugDescriptor("clumpFlatness")
	node : HairSystem = None
	pass
class ClumpFlatness_InterpPlug(Plug):
	parent : ClumpFlatnessPlug = PlugDescriptor("clumpFlatness")
	node : HairSystem = None
	pass
class ClumpFlatness_PositionPlug(Plug):
	parent : ClumpFlatnessPlug = PlugDescriptor("clumpFlatness")
	node : HairSystem = None
	pass
class ClumpFlatnessPlug(Plug):
	clumpFlatness_FloatValue_ : ClumpFlatness_FloatValuePlug = PlugDescriptor("clumpFlatness_FloatValue")
	cflfv_ : ClumpFlatness_FloatValuePlug = PlugDescriptor("clumpFlatness_FloatValue")
	clumpFlatness_Interp_ : ClumpFlatness_InterpPlug = PlugDescriptor("clumpFlatness_Interp")
	cfli_ : ClumpFlatness_InterpPlug = PlugDescriptor("clumpFlatness_Interp")
	clumpFlatness_Position_ : ClumpFlatness_PositionPlug = PlugDescriptor("clumpFlatness_Position")
	cflp_ : ClumpFlatness_PositionPlug = PlugDescriptor("clumpFlatness_Position")
	node : HairSystem = None
	pass
class ClumpInterpolationPlug(Plug):
	node : HairSystem = None
	pass
class ClumpTwistPlug(Plug):
	node : HairSystem = None
	pass
class ClumpWidthPlug(Plug):
	node : HairSystem = None
	pass
class ClumpWidthScale_FloatValuePlug(Plug):
	parent : ClumpWidthScalePlug = PlugDescriptor("clumpWidthScale")
	node : HairSystem = None
	pass
class ClumpWidthScale_InterpPlug(Plug):
	parent : ClumpWidthScalePlug = PlugDescriptor("clumpWidthScale")
	node : HairSystem = None
	pass
class ClumpWidthScale_PositionPlug(Plug):
	parent : ClumpWidthScalePlug = PlugDescriptor("clumpWidthScale")
	node : HairSystem = None
	pass
class ClumpWidthScalePlug(Plug):
	clumpWidthScale_FloatValue_ : ClumpWidthScale_FloatValuePlug = PlugDescriptor("clumpWidthScale_FloatValue")
	cwsfv_ : ClumpWidthScale_FloatValuePlug = PlugDescriptor("clumpWidthScale_FloatValue")
	clumpWidthScale_Interp_ : ClumpWidthScale_InterpPlug = PlugDescriptor("clumpWidthScale_Interp")
	cwsi_ : ClumpWidthScale_InterpPlug = PlugDescriptor("clumpWidthScale_Interp")
	clumpWidthScale_Position_ : ClumpWidthScale_PositionPlug = PlugDescriptor("clumpWidthScale_Position")
	cwsp_ : ClumpWidthScale_PositionPlug = PlugDescriptor("clumpWidthScale_Position")
	node : HairSystem = None
	pass
class CollidePlug(Plug):
	node : HairSystem = None
	pass
class CollideGroundPlug(Plug):
	node : HairSystem = None
	pass
class CollideOverSamplePlug(Plug):
	node : HairSystem = None
	pass
class CollideStrengthPlug(Plug):
	node : HairSystem = None
	pass
class CollideWidthOffsetPlug(Plug):
	node : HairSystem = None
	pass
class CollisionFrictionPlug(Plug):
	parent : CollisionDataPlug = PlugDescriptor("collisionData")
	node : HairSystem = None
	pass
class CollisionGeometryPlug(Plug):
	parent : CollisionDataPlug = PlugDescriptor("collisionData")
	node : HairSystem = None
	pass
class CollisionResiliencePlug(Plug):
	parent : CollisionDataPlug = PlugDescriptor("collisionData")
	node : HairSystem = None
	pass
class CollisionDataPlug(Plug):
	collisionFriction_ : CollisionFrictionPlug = PlugDescriptor("collisionFriction")
	cfr_ : CollisionFrictionPlug = PlugDescriptor("collisionFriction")
	collisionGeometry_ : CollisionGeometryPlug = PlugDescriptor("collisionGeometry")
	cge_ : CollisionGeometryPlug = PlugDescriptor("collisionGeometry")
	collisionResilience_ : CollisionResiliencePlug = PlugDescriptor("collisionResilience")
	crs_ : CollisionResiliencePlug = PlugDescriptor("collisionResilience")
	node : HairSystem = None
	pass
class CollisionFlagPlug(Plug):
	node : HairSystem = None
	pass
class CollisionLayerPlug(Plug):
	node : HairSystem = None
	pass
class CompressionResistancePlug(Plug):
	node : HairSystem = None
	pass
class CurlPlug(Plug):
	node : HairSystem = None
	pass
class CurlFrequencyPlug(Plug):
	node : HairSystem = None
	pass
class CurrentStatePlug(Plug):
	node : HairSystem = None
	pass
class CurrentTimePlug(Plug):
	node : HairSystem = None
	pass
class DampPlug(Plug):
	node : HairSystem = None
	pass
class DetailNoisePlug(Plug):
	node : HairSystem = None
	pass
class DiffuseRandPlug(Plug):
	node : HairSystem = None
	pass
class DisableFollicleAnimPlug(Plug):
	node : HairSystem = None
	pass
class DiskCachePlug(Plug):
	node : HairSystem = None
	pass
class DisplacementScale_FloatValuePlug(Plug):
	parent : DisplacementScalePlug = PlugDescriptor("displacementScale")
	node : HairSystem = None
	pass
class DisplacementScale_InterpPlug(Plug):
	parent : DisplacementScalePlug = PlugDescriptor("displacementScale")
	node : HairSystem = None
	pass
class DisplacementScale_PositionPlug(Plug):
	parent : DisplacementScalePlug = PlugDescriptor("displacementScale")
	node : HairSystem = None
	pass
class DisplacementScalePlug(Plug):
	displacementScale_FloatValue_ : DisplacementScale_FloatValuePlug = PlugDescriptor("displacementScale_FloatValue")
	dscfv_ : DisplacementScale_FloatValuePlug = PlugDescriptor("displacementScale_FloatValue")
	displacementScale_Interp_ : DisplacementScale_InterpPlug = PlugDescriptor("displacementScale_Interp")
	dsci_ : DisplacementScale_InterpPlug = PlugDescriptor("displacementScale_Interp")
	displacementScale_Position_ : DisplacementScale_PositionPlug = PlugDescriptor("displacementScale_Position")
	dscp_ : DisplacementScale_PositionPlug = PlugDescriptor("displacementScale_Position")
	node : HairSystem = None
	pass
class DisplayColorBPlug(Plug):
	parent : DisplayColorPlug = PlugDescriptor("displayColor")
	node : HairSystem = None
	pass
class DisplayColorGPlug(Plug):
	parent : DisplayColorPlug = PlugDescriptor("displayColor")
	node : HairSystem = None
	pass
class DisplayColorRPlug(Plug):
	parent : DisplayColorPlug = PlugDescriptor("displayColor")
	node : HairSystem = None
	pass
class DisplayColorPlug(Plug):
	displayColorB_ : DisplayColorBPlug = PlugDescriptor("displayColorB")
	dcb_ : DisplayColorBPlug = PlugDescriptor("displayColorB")
	displayColorG_ : DisplayColorGPlug = PlugDescriptor("displayColorG")
	dcg_ : DisplayColorGPlug = PlugDescriptor("displayColorG")
	displayColorR_ : DisplayColorRPlug = PlugDescriptor("displayColorR")
	dcr_ : DisplayColorRPlug = PlugDescriptor("displayColorR")
	node : HairSystem = None
	pass
class DisplayQualityPlug(Plug):
	node : HairSystem = None
	pass
class DragPlug(Plug):
	node : HairSystem = None
	pass
class DrawCollideWidthPlug(Plug):
	node : HairSystem = None
	pass
class DynamicsWeightPlug(Plug):
	node : HairSystem = None
	pass
class EvaluationOrderPlug(Plug):
	node : HairSystem = None
	pass
class ExtraBendLinksPlug(Plug):
	node : HairSystem = None
	pass
class FieldDataDeltaTimePlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : HairSystem = None
	pass
class FieldDataMassPlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : HairSystem = None
	pass
class FieldDataPositionPlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : HairSystem = None
	pass
class FieldDataVelocityPlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : HairSystem = None
	pass
class FieldDataPlug(Plug):
	fieldDataDeltaTime_ : FieldDataDeltaTimePlug = PlugDescriptor("fieldDataDeltaTime")
	fdt_ : FieldDataDeltaTimePlug = PlugDescriptor("fieldDataDeltaTime")
	fieldDataMass_ : FieldDataMassPlug = PlugDescriptor("fieldDataMass")
	fdm_ : FieldDataMassPlug = PlugDescriptor("fieldDataMass")
	fieldDataPosition_ : FieldDataPositionPlug = PlugDescriptor("fieldDataPosition")
	fdp_ : FieldDataPositionPlug = PlugDescriptor("fieldDataPosition")
	fieldDataVelocity_ : FieldDataVelocityPlug = PlugDescriptor("fieldDataVelocity")
	fdv_ : FieldDataVelocityPlug = PlugDescriptor("fieldDataVelocity")
	node : HairSystem = None
	pass
class FrictionPlug(Plug):
	node : HairSystem = None
	pass
class GravityPlug(Plug):
	node : HairSystem = None
	pass
class GroundHeightPlug(Plug):
	node : HairSystem = None
	pass
class HairColorBPlug(Plug):
	parent : HairColorPlug = PlugDescriptor("hairColor")
	node : HairSystem = None
	pass
class HairColorGPlug(Plug):
	parent : HairColorPlug = PlugDescriptor("hairColor")
	node : HairSystem = None
	pass
class HairColorRPlug(Plug):
	parent : HairColorPlug = PlugDescriptor("hairColor")
	node : HairSystem = None
	pass
class HairColorPlug(Plug):
	hairColorB_ : HairColorBPlug = PlugDescriptor("hairColorB")
	hcb_ : HairColorBPlug = PlugDescriptor("hairColorB")
	hairColorG_ : HairColorGPlug = PlugDescriptor("hairColorG")
	hcg_ : HairColorGPlug = PlugDescriptor("hairColorG")
	hairColorR_ : HairColorRPlug = PlugDescriptor("hairColorR")
	hcr_ : HairColorRPlug = PlugDescriptor("hairColorR")
	node : HairSystem = None
	pass
class HairColorScale_ColorBPlug(Plug):
	parent : HairColorScale_ColorPlug = PlugDescriptor("hairColorScale_Color")
	node : HairSystem = None
	pass
class HairColorScale_ColorGPlug(Plug):
	parent : HairColorScale_ColorPlug = PlugDescriptor("hairColorScale_Color")
	node : HairSystem = None
	pass
class HairColorScale_ColorRPlug(Plug):
	parent : HairColorScale_ColorPlug = PlugDescriptor("hairColorScale_Color")
	node : HairSystem = None
	pass
class HairColorScale_ColorPlug(Plug):
	parent : HairColorScalePlug = PlugDescriptor("hairColorScale")
	hairColorScale_ColorB_ : HairColorScale_ColorBPlug = PlugDescriptor("hairColorScale_ColorB")
	hcscb_ : HairColorScale_ColorBPlug = PlugDescriptor("hairColorScale_ColorB")
	hairColorScale_ColorG_ : HairColorScale_ColorGPlug = PlugDescriptor("hairColorScale_ColorG")
	hcscg_ : HairColorScale_ColorGPlug = PlugDescriptor("hairColorScale_ColorG")
	hairColorScale_ColorR_ : HairColorScale_ColorRPlug = PlugDescriptor("hairColorScale_ColorR")
	hcscr_ : HairColorScale_ColorRPlug = PlugDescriptor("hairColorScale_ColorR")
	node : HairSystem = None
	pass
class HairColorScale_InterpPlug(Plug):
	parent : HairColorScalePlug = PlugDescriptor("hairColorScale")
	node : HairSystem = None
	pass
class HairColorScale_PositionPlug(Plug):
	parent : HairColorScalePlug = PlugDescriptor("hairColorScale")
	node : HairSystem = None
	pass
class HairColorScalePlug(Plug):
	hairColorScale_Color_ : HairColorScale_ColorPlug = PlugDescriptor("hairColorScale_Color")
	hcsc_ : HairColorScale_ColorPlug = PlugDescriptor("hairColorScale_Color")
	hairColorScale_Interp_ : HairColorScale_InterpPlug = PlugDescriptor("hairColorScale_Interp")
	hcsi_ : HairColorScale_InterpPlug = PlugDescriptor("hairColorScale_Interp")
	hairColorScale_Position_ : HairColorScale_PositionPlug = PlugDescriptor("hairColorScale_Position")
	hcsp_ : HairColorScale_PositionPlug = PlugDescriptor("hairColorScale_Position")
	node : HairSystem = None
	pass
class HairCountsPlug(Plug):
	node : HairSystem = None
	pass
class HairWidthPlug(Plug):
	node : HairSystem = None
	pass
class HairWidthScale_FloatValuePlug(Plug):
	parent : HairWidthScalePlug = PlugDescriptor("hairWidthScale")
	node : HairSystem = None
	pass
class HairWidthScale_InterpPlug(Plug):
	parent : HairWidthScalePlug = PlugDescriptor("hairWidthScale")
	node : HairSystem = None
	pass
class HairWidthScale_PositionPlug(Plug):
	parent : HairWidthScalePlug = PlugDescriptor("hairWidthScale")
	node : HairSystem = None
	pass
class HairWidthScalePlug(Plug):
	hairWidthScale_FloatValue_ : HairWidthScale_FloatValuePlug = PlugDescriptor("hairWidthScale_FloatValue")
	hwsfv_ : HairWidthScale_FloatValuePlug = PlugDescriptor("hairWidthScale_FloatValue")
	hairWidthScale_Interp_ : HairWidthScale_InterpPlug = PlugDescriptor("hairWidthScale_Interp")
	hwsi_ : HairWidthScale_InterpPlug = PlugDescriptor("hairWidthScale_Interp")
	hairWidthScale_Position_ : HairWidthScale_PositionPlug = PlugDescriptor("hairWidthScale_Position")
	hwsp_ : HairWidthScale_PositionPlug = PlugDescriptor("hairWidthScale_Position")
	node : HairSystem = None
	pass
class HairsPerClumpPlug(Plug):
	node : HairSystem = None
	pass
class HueRandPlug(Plug):
	node : HairSystem = None
	pass
class IgnoreSolverGravityPlug(Plug):
	node : HairSystem = None
	pass
class IgnoreSolverWindPlug(Plug):
	node : HairSystem = None
	pass
class InputForcePlug(Plug):
	node : HairSystem = None
	pass
class InputHairPlug(Plug):
	node : HairSystem = None
	pass
class InputHairPinPlug(Plug):
	node : HairSystem = None
	pass
class InternalStatePlug(Plug):
	node : HairSystem = None
	pass
class InterpolationRangePlug(Plug):
	node : HairSystem = None
	pass
class IterationsPlug(Plug):
	node : HairSystem = None
	pass
class LastEvalTimePlug(Plug):
	node : HairSystem = None
	pass
class LengthFlexPlug(Plug):
	node : HairSystem = None
	pass
class LightEachHairPlug(Plug):
	node : HairSystem = None
	pass
class MassPlug(Plug):
	node : HairSystem = None
	pass
class MaxSelfCollisionIterationsPlug(Plug):
	node : HairSystem = None
	pass
class MotionDragPlug(Plug):
	node : HairSystem = None
	pass
class MultiStreakSpread1Plug(Plug):
	node : HairSystem = None
	pass
class MultiStreakSpread2Plug(Plug):
	node : HairSystem = None
	pass
class MultiStreaksPlug(Plug):
	node : HairSystem = None
	pass
class NextStatePlug(Plug):
	node : HairSystem = None
	pass
class NoStretchPlug(Plug):
	node : HairSystem = None
	pass
class NoisePlug(Plug):
	node : HairSystem = None
	pass
class NoiseFrequencyPlug(Plug):
	node : HairSystem = None
	pass
class NoiseFrequencyUPlug(Plug):
	node : HairSystem = None
	pass
class NoiseFrequencyVPlug(Plug):
	node : HairSystem = None
	pass
class NoiseFrequencyWPlug(Plug):
	node : HairSystem = None
	pass
class NoiseMethodPlug(Plug):
	node : HairSystem = None
	pass
class NucleusIdPlug(Plug):
	node : HairSystem = None
	pass
class NumCollideNeighborsPlug(Plug):
	node : HairSystem = None
	pass
class NumUClumpsPlug(Plug):
	node : HairSystem = None
	pass
class NumVClumpsPlug(Plug):
	node : HairSystem = None
	pass
class OpacityPlug(Plug):
	node : HairSystem = None
	pass
class OutputHairPlug(Plug):
	node : HairSystem = None
	pass
class OutputRenderHairsPlug(Plug):
	node : HairSystem = None
	pass
class PlayFromCachePlug(Plug):
	node : HairSystem = None
	pass
class PositionsPlug(Plug):
	node : HairSystem = None
	pass
class ReceiveShadowsPlug(Plug):
	node : HairSystem = None
	pass
class RepulsionPlug(Plug):
	node : HairSystem = None
	pass
class RestLengthScalePlug(Plug):
	node : HairSystem = None
	pass
class SatRandPlug(Plug):
	node : HairSystem = None
	pass
class SelfCollidePlug(Plug):
	node : HairSystem = None
	pass
class SelfCollideWidthScalePlug(Plug):
	node : HairSystem = None
	pass
class SelfCollisionFlagPlug(Plug):
	node : HairSystem = None
	pass
class SimulationMethodPlug(Plug):
	node : HairSystem = None
	pass
class SolverDisplayPlug(Plug):
	node : HairSystem = None
	pass
class SpecularColorBPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : HairSystem = None
	pass
class SpecularColorGPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : HairSystem = None
	pass
class SpecularColorRPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : HairSystem = None
	pass
class SpecularColorPlug(Plug):
	specularColorB_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	spb_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	specularColorG_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	spg_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	specularColorR_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	spr_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	node : HairSystem = None
	pass
class SpecularPowerPlug(Plug):
	node : HairSystem = None
	pass
class SpecularRandPlug(Plug):
	node : HairSystem = None
	pass
class StartCurveAttractPlug(Plug):
	node : HairSystem = None
	pass
class StartFramePlug(Plug):
	node : HairSystem = None
	pass
class StartStatePlug(Plug):
	node : HairSystem = None
	pass
class StartTimePlug(Plug):
	node : HairSystem = None
	pass
class StaticClingPlug(Plug):
	node : HairSystem = None
	pass
class StickinessPlug(Plug):
	node : HairSystem = None
	pass
class StiffnessPlug(Plug):
	node : HairSystem = None
	pass
class StiffnessScale_FloatValuePlug(Plug):
	parent : StiffnessScalePlug = PlugDescriptor("stiffnessScale")
	node : HairSystem = None
	pass
class StiffnessScale_InterpPlug(Plug):
	parent : StiffnessScalePlug = PlugDescriptor("stiffnessScale")
	node : HairSystem = None
	pass
class StiffnessScale_PositionPlug(Plug):
	parent : StiffnessScalePlug = PlugDescriptor("stiffnessScale")
	node : HairSystem = None
	pass
class StiffnessScalePlug(Plug):
	stiffnessScale_FloatValue_ : StiffnessScale_FloatValuePlug = PlugDescriptor("stiffnessScale_FloatValue")
	stsfv_ : StiffnessScale_FloatValuePlug = PlugDescriptor("stiffnessScale_FloatValue")
	stiffnessScale_Interp_ : StiffnessScale_InterpPlug = PlugDescriptor("stiffnessScale_Interp")
	stsi_ : StiffnessScale_InterpPlug = PlugDescriptor("stiffnessScale_Interp")
	stiffnessScale_Position_ : StiffnessScale_PositionPlug = PlugDescriptor("stiffnessScale_Position")
	stsp_ : StiffnessScale_PositionPlug = PlugDescriptor("stiffnessScale_Position")
	node : HairSystem = None
	pass
class StretchDampPlug(Plug):
	node : HairSystem = None
	pass
class StretchResistancePlug(Plug):
	node : HairSystem = None
	pass
class SubClumpMethodPlug(Plug):
	node : HairSystem = None
	pass
class SubClumpRandPlug(Plug):
	node : HairSystem = None
	pass
class SubClumpingPlug(Plug):
	node : HairSystem = None
	pass
class SubSegmentsPlug(Plug):
	node : HairSystem = None
	pass
class TangentialDragPlug(Plug):
	node : HairSystem = None
	pass
class ThinningPlug(Plug):
	node : HairSystem = None
	pass
class TranslucencePlug(Plug):
	node : HairSystem = None
	pass
class TurbulenceFrequencyPlug(Plug):
	node : HairSystem = None
	pass
class TurbulenceSpeedPlug(Plug):
	node : HairSystem = None
	pass
class TurbulenceStrengthPlug(Plug):
	node : HairSystem = None
	pass
class TwistResistancePlug(Plug):
	node : HairSystem = None
	pass
class UsePre70ForceIntensityPlug(Plug):
	node : HairSystem = None
	pass
class ValRandPlug(Plug):
	node : HairSystem = None
	pass
class VelocitiesPlug(Plug):
	node : HairSystem = None
	pass
class VertexCountsPlug(Plug):
	node : HairSystem = None
	pass
class VisibleInReflectionsPlug(Plug):
	node : HairSystem = None
	pass
class VisibleInRefractionsPlug(Plug):
	node : HairSystem = None
	pass
class WidthDrawSkipPlug(Plug):
	node : HairSystem = None
	pass
# endregion


# define node class
class HairSystem(Shape):
	active_ : ActivePlug = PlugDescriptor("active")
	attachObjectId_ : AttachObjectIdPlug = PlugDescriptor("attachObjectId")
	attractionDamp_ : AttractionDampPlug = PlugDescriptor("attractionDamp")
	attractionScale_FloatValue_ : AttractionScale_FloatValuePlug = PlugDescriptor("attractionScale_FloatValue")
	attractionScale_Interp_ : AttractionScale_InterpPlug = PlugDescriptor("attractionScale_Interp")
	attractionScale_Position_ : AttractionScale_PositionPlug = PlugDescriptor("attractionScale_Position")
	attractionScale_ : AttractionScalePlug = PlugDescriptor("attractionScale")
	baldnessMap_ : BaldnessMapPlug = PlugDescriptor("baldnessMap")
	bendAnisotropy_ : BendAnisotropyPlug = PlugDescriptor("bendAnisotropy")
	bendFollow_ : BendFollowPlug = PlugDescriptor("bendFollow")
	bendModel_ : BendModelPlug = PlugDescriptor("bendModel")
	bendResistance_ : BendResistancePlug = PlugDescriptor("bendResistance")
	bounce_ : BouncePlug = PlugDescriptor("bounce")
	cacheableAttributes_ : CacheableAttributesPlug = PlugDescriptor("cacheableAttributes")
	castShadows_ : CastShadowsPlug = PlugDescriptor("castShadows")
	clumpCurl_FloatValue_ : ClumpCurl_FloatValuePlug = PlugDescriptor("clumpCurl_FloatValue")
	clumpCurl_Interp_ : ClumpCurl_InterpPlug = PlugDescriptor("clumpCurl_Interp")
	clumpCurl_Position_ : ClumpCurl_PositionPlug = PlugDescriptor("clumpCurl_Position")
	clumpCurl_ : ClumpCurlPlug = PlugDescriptor("clumpCurl")
	clumpFlatness_FloatValue_ : ClumpFlatness_FloatValuePlug = PlugDescriptor("clumpFlatness_FloatValue")
	clumpFlatness_Interp_ : ClumpFlatness_InterpPlug = PlugDescriptor("clumpFlatness_Interp")
	clumpFlatness_Position_ : ClumpFlatness_PositionPlug = PlugDescriptor("clumpFlatness_Position")
	clumpFlatness_ : ClumpFlatnessPlug = PlugDescriptor("clumpFlatness")
	clumpInterpolation_ : ClumpInterpolationPlug = PlugDescriptor("clumpInterpolation")
	clumpTwist_ : ClumpTwistPlug = PlugDescriptor("clumpTwist")
	clumpWidth_ : ClumpWidthPlug = PlugDescriptor("clumpWidth")
	clumpWidthScale_FloatValue_ : ClumpWidthScale_FloatValuePlug = PlugDescriptor("clumpWidthScale_FloatValue")
	clumpWidthScale_Interp_ : ClumpWidthScale_InterpPlug = PlugDescriptor("clumpWidthScale_Interp")
	clumpWidthScale_Position_ : ClumpWidthScale_PositionPlug = PlugDescriptor("clumpWidthScale_Position")
	clumpWidthScale_ : ClumpWidthScalePlug = PlugDescriptor("clumpWidthScale")
	collide_ : CollidePlug = PlugDescriptor("collide")
	collideGround_ : CollideGroundPlug = PlugDescriptor("collideGround")
	collideOverSample_ : CollideOverSamplePlug = PlugDescriptor("collideOverSample")
	collideStrength_ : CollideStrengthPlug = PlugDescriptor("collideStrength")
	collideWidthOffset_ : CollideWidthOffsetPlug = PlugDescriptor("collideWidthOffset")
	collisionFriction_ : CollisionFrictionPlug = PlugDescriptor("collisionFriction")
	collisionGeometry_ : CollisionGeometryPlug = PlugDescriptor("collisionGeometry")
	collisionResilience_ : CollisionResiliencePlug = PlugDescriptor("collisionResilience")
	collisionData_ : CollisionDataPlug = PlugDescriptor("collisionData")
	collisionFlag_ : CollisionFlagPlug = PlugDescriptor("collisionFlag")
	collisionLayer_ : CollisionLayerPlug = PlugDescriptor("collisionLayer")
	compressionResistance_ : CompressionResistancePlug = PlugDescriptor("compressionResistance")
	curl_ : CurlPlug = PlugDescriptor("curl")
	curlFrequency_ : CurlFrequencyPlug = PlugDescriptor("curlFrequency")
	currentState_ : CurrentStatePlug = PlugDescriptor("currentState")
	currentTime_ : CurrentTimePlug = PlugDescriptor("currentTime")
	damp_ : DampPlug = PlugDescriptor("damp")
	detailNoise_ : DetailNoisePlug = PlugDescriptor("detailNoise")
	diffuseRand_ : DiffuseRandPlug = PlugDescriptor("diffuseRand")
	disableFollicleAnim_ : DisableFollicleAnimPlug = PlugDescriptor("disableFollicleAnim")
	diskCache_ : DiskCachePlug = PlugDescriptor("diskCache")
	displacementScale_FloatValue_ : DisplacementScale_FloatValuePlug = PlugDescriptor("displacementScale_FloatValue")
	displacementScale_Interp_ : DisplacementScale_InterpPlug = PlugDescriptor("displacementScale_Interp")
	displacementScale_Position_ : DisplacementScale_PositionPlug = PlugDescriptor("displacementScale_Position")
	displacementScale_ : DisplacementScalePlug = PlugDescriptor("displacementScale")
	displayColorB_ : DisplayColorBPlug = PlugDescriptor("displayColorB")
	displayColorG_ : DisplayColorGPlug = PlugDescriptor("displayColorG")
	displayColorR_ : DisplayColorRPlug = PlugDescriptor("displayColorR")
	displayColor_ : DisplayColorPlug = PlugDescriptor("displayColor")
	displayQuality_ : DisplayQualityPlug = PlugDescriptor("displayQuality")
	drag_ : DragPlug = PlugDescriptor("drag")
	drawCollideWidth_ : DrawCollideWidthPlug = PlugDescriptor("drawCollideWidth")
	dynamicsWeight_ : DynamicsWeightPlug = PlugDescriptor("dynamicsWeight")
	evaluationOrder_ : EvaluationOrderPlug = PlugDescriptor("evaluationOrder")
	extraBendLinks_ : ExtraBendLinksPlug = PlugDescriptor("extraBendLinks")
	fieldDataDeltaTime_ : FieldDataDeltaTimePlug = PlugDescriptor("fieldDataDeltaTime")
	fieldDataMass_ : FieldDataMassPlug = PlugDescriptor("fieldDataMass")
	fieldDataPosition_ : FieldDataPositionPlug = PlugDescriptor("fieldDataPosition")
	fieldDataVelocity_ : FieldDataVelocityPlug = PlugDescriptor("fieldDataVelocity")
	fieldData_ : FieldDataPlug = PlugDescriptor("fieldData")
	friction_ : FrictionPlug = PlugDescriptor("friction")
	gravity_ : GravityPlug = PlugDescriptor("gravity")
	groundHeight_ : GroundHeightPlug = PlugDescriptor("groundHeight")
	hairColorB_ : HairColorBPlug = PlugDescriptor("hairColorB")
	hairColorG_ : HairColorGPlug = PlugDescriptor("hairColorG")
	hairColorR_ : HairColorRPlug = PlugDescriptor("hairColorR")
	hairColor_ : HairColorPlug = PlugDescriptor("hairColor")
	hairColorScale_ColorB_ : HairColorScale_ColorBPlug = PlugDescriptor("hairColorScale_ColorB")
	hairColorScale_ColorG_ : HairColorScale_ColorGPlug = PlugDescriptor("hairColorScale_ColorG")
	hairColorScale_ColorR_ : HairColorScale_ColorRPlug = PlugDescriptor("hairColorScale_ColorR")
	hairColorScale_Color_ : HairColorScale_ColorPlug = PlugDescriptor("hairColorScale_Color")
	hairColorScale_Interp_ : HairColorScale_InterpPlug = PlugDescriptor("hairColorScale_Interp")
	hairColorScale_Position_ : HairColorScale_PositionPlug = PlugDescriptor("hairColorScale_Position")
	hairColorScale_ : HairColorScalePlug = PlugDescriptor("hairColorScale")
	hairCounts_ : HairCountsPlug = PlugDescriptor("hairCounts")
	hairWidth_ : HairWidthPlug = PlugDescriptor("hairWidth")
	hairWidthScale_FloatValue_ : HairWidthScale_FloatValuePlug = PlugDescriptor("hairWidthScale_FloatValue")
	hairWidthScale_Interp_ : HairWidthScale_InterpPlug = PlugDescriptor("hairWidthScale_Interp")
	hairWidthScale_Position_ : HairWidthScale_PositionPlug = PlugDescriptor("hairWidthScale_Position")
	hairWidthScale_ : HairWidthScalePlug = PlugDescriptor("hairWidthScale")
	hairsPerClump_ : HairsPerClumpPlug = PlugDescriptor("hairsPerClump")
	hueRand_ : HueRandPlug = PlugDescriptor("hueRand")
	ignoreSolverGravity_ : IgnoreSolverGravityPlug = PlugDescriptor("ignoreSolverGravity")
	ignoreSolverWind_ : IgnoreSolverWindPlug = PlugDescriptor("ignoreSolverWind")
	inputForce_ : InputForcePlug = PlugDescriptor("inputForce")
	inputHair_ : InputHairPlug = PlugDescriptor("inputHair")
	inputHairPin_ : InputHairPinPlug = PlugDescriptor("inputHairPin")
	internalState_ : InternalStatePlug = PlugDescriptor("internalState")
	interpolationRange_ : InterpolationRangePlug = PlugDescriptor("interpolationRange")
	iterations_ : IterationsPlug = PlugDescriptor("iterations")
	lastEvalTime_ : LastEvalTimePlug = PlugDescriptor("lastEvalTime")
	lengthFlex_ : LengthFlexPlug = PlugDescriptor("lengthFlex")
	lightEachHair_ : LightEachHairPlug = PlugDescriptor("lightEachHair")
	mass_ : MassPlug = PlugDescriptor("mass")
	maxSelfCollisionIterations_ : MaxSelfCollisionIterationsPlug = PlugDescriptor("maxSelfCollisionIterations")
	motionDrag_ : MotionDragPlug = PlugDescriptor("motionDrag")
	multiStreakSpread1_ : MultiStreakSpread1Plug = PlugDescriptor("multiStreakSpread1")
	multiStreakSpread2_ : MultiStreakSpread2Plug = PlugDescriptor("multiStreakSpread2")
	multiStreaks_ : MultiStreaksPlug = PlugDescriptor("multiStreaks")
	nextState_ : NextStatePlug = PlugDescriptor("nextState")
	noStretch_ : NoStretchPlug = PlugDescriptor("noStretch")
	noise_ : NoisePlug = PlugDescriptor("noise")
	noiseFrequency_ : NoiseFrequencyPlug = PlugDescriptor("noiseFrequency")
	noiseFrequencyU_ : NoiseFrequencyUPlug = PlugDescriptor("noiseFrequencyU")
	noiseFrequencyV_ : NoiseFrequencyVPlug = PlugDescriptor("noiseFrequencyV")
	noiseFrequencyW_ : NoiseFrequencyWPlug = PlugDescriptor("noiseFrequencyW")
	noiseMethod_ : NoiseMethodPlug = PlugDescriptor("noiseMethod")
	nucleusId_ : NucleusIdPlug = PlugDescriptor("nucleusId")
	numCollideNeighbors_ : NumCollideNeighborsPlug = PlugDescriptor("numCollideNeighbors")
	numUClumps_ : NumUClumpsPlug = PlugDescriptor("numUClumps")
	numVClumps_ : NumVClumpsPlug = PlugDescriptor("numVClumps")
	opacity_ : OpacityPlug = PlugDescriptor("opacity")
	outputHair_ : OutputHairPlug = PlugDescriptor("outputHair")
	outputRenderHairs_ : OutputRenderHairsPlug = PlugDescriptor("outputRenderHairs")
	playFromCache_ : PlayFromCachePlug = PlugDescriptor("playFromCache")
	positions_ : PositionsPlug = PlugDescriptor("positions")
	receiveShadows_ : ReceiveShadowsPlug = PlugDescriptor("receiveShadows")
	repulsion_ : RepulsionPlug = PlugDescriptor("repulsion")
	restLengthScale_ : RestLengthScalePlug = PlugDescriptor("restLengthScale")
	satRand_ : SatRandPlug = PlugDescriptor("satRand")
	selfCollide_ : SelfCollidePlug = PlugDescriptor("selfCollide")
	selfCollideWidthScale_ : SelfCollideWidthScalePlug = PlugDescriptor("selfCollideWidthScale")
	selfCollisionFlag_ : SelfCollisionFlagPlug = PlugDescriptor("selfCollisionFlag")
	simulationMethod_ : SimulationMethodPlug = PlugDescriptor("simulationMethod")
	solverDisplay_ : SolverDisplayPlug = PlugDescriptor("solverDisplay")
	specularColorB_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	specularColorG_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	specularColorR_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	specularColor_ : SpecularColorPlug = PlugDescriptor("specularColor")
	specularPower_ : SpecularPowerPlug = PlugDescriptor("specularPower")
	specularRand_ : SpecularRandPlug = PlugDescriptor("specularRand")
	startCurveAttract_ : StartCurveAttractPlug = PlugDescriptor("startCurveAttract")
	startFrame_ : StartFramePlug = PlugDescriptor("startFrame")
	startState_ : StartStatePlug = PlugDescriptor("startState")
	startTime_ : StartTimePlug = PlugDescriptor("startTime")
	staticCling_ : StaticClingPlug = PlugDescriptor("staticCling")
	stickiness_ : StickinessPlug = PlugDescriptor("stickiness")
	stiffness_ : StiffnessPlug = PlugDescriptor("stiffness")
	stiffnessScale_FloatValue_ : StiffnessScale_FloatValuePlug = PlugDescriptor("stiffnessScale_FloatValue")
	stiffnessScale_Interp_ : StiffnessScale_InterpPlug = PlugDescriptor("stiffnessScale_Interp")
	stiffnessScale_Position_ : StiffnessScale_PositionPlug = PlugDescriptor("stiffnessScale_Position")
	stiffnessScale_ : StiffnessScalePlug = PlugDescriptor("stiffnessScale")
	stretchDamp_ : StretchDampPlug = PlugDescriptor("stretchDamp")
	stretchResistance_ : StretchResistancePlug = PlugDescriptor("stretchResistance")
	subClumpMethod_ : SubClumpMethodPlug = PlugDescriptor("subClumpMethod")
	subClumpRand_ : SubClumpRandPlug = PlugDescriptor("subClumpRand")
	subClumping_ : SubClumpingPlug = PlugDescriptor("subClumping")
	subSegments_ : SubSegmentsPlug = PlugDescriptor("subSegments")
	tangentialDrag_ : TangentialDragPlug = PlugDescriptor("tangentialDrag")
	thinning_ : ThinningPlug = PlugDescriptor("thinning")
	translucence_ : TranslucencePlug = PlugDescriptor("translucence")
	turbulenceFrequency_ : TurbulenceFrequencyPlug = PlugDescriptor("turbulenceFrequency")
	turbulenceSpeed_ : TurbulenceSpeedPlug = PlugDescriptor("turbulenceSpeed")
	turbulenceStrength_ : TurbulenceStrengthPlug = PlugDescriptor("turbulenceStrength")
	twistResistance_ : TwistResistancePlug = PlugDescriptor("twistResistance")
	usePre70ForceIntensity_ : UsePre70ForceIntensityPlug = PlugDescriptor("usePre70ForceIntensity")
	valRand_ : ValRandPlug = PlugDescriptor("valRand")
	velocities_ : VelocitiesPlug = PlugDescriptor("velocities")
	vertexCounts_ : VertexCountsPlug = PlugDescriptor("vertexCounts")
	visibleInReflections_ : VisibleInReflectionsPlug = PlugDescriptor("visibleInReflections")
	visibleInRefractions_ : VisibleInRefractionsPlug = PlugDescriptor("visibleInRefractions")
	widthDrawSkip_ : WidthDrawSkipPlug = PlugDescriptor("widthDrawSkip")

	# node attributes

	typeName = "hairSystem"
	apiTypeInt = 936
	apiTypeStr = "kHairSystem"
	typeIdInt = 1213421907
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["active", "attachObjectId", "attractionDamp", "attractionScale_FloatValue", "attractionScale_Interp", "attractionScale_Position", "attractionScale", "baldnessMap", "bendAnisotropy", "bendFollow", "bendModel", "bendResistance", "bounce", "cacheableAttributes", "castShadows", "clumpCurl_FloatValue", "clumpCurl_Interp", "clumpCurl_Position", "clumpCurl", "clumpFlatness_FloatValue", "clumpFlatness_Interp", "clumpFlatness_Position", "clumpFlatness", "clumpInterpolation", "clumpTwist", "clumpWidth", "clumpWidthScale_FloatValue", "clumpWidthScale_Interp", "clumpWidthScale_Position", "clumpWidthScale", "collide", "collideGround", "collideOverSample", "collideStrength", "collideWidthOffset", "collisionFriction", "collisionGeometry", "collisionResilience", "collisionData", "collisionFlag", "collisionLayer", "compressionResistance", "curl", "curlFrequency", "currentState", "currentTime", "damp", "detailNoise", "diffuseRand", "disableFollicleAnim", "diskCache", "displacementScale_FloatValue", "displacementScale_Interp", "displacementScale_Position", "displacementScale", "displayColorB", "displayColorG", "displayColorR", "displayColor", "displayQuality", "drag", "drawCollideWidth", "dynamicsWeight", "evaluationOrder", "extraBendLinks", "fieldDataDeltaTime", "fieldDataMass", "fieldDataPosition", "fieldDataVelocity", "fieldData", "friction", "gravity", "groundHeight", "hairColorB", "hairColorG", "hairColorR", "hairColor", "hairColorScale_ColorB", "hairColorScale_ColorG", "hairColorScale_ColorR", "hairColorScale_Color", "hairColorScale_Interp", "hairColorScale_Position", "hairColorScale", "hairCounts", "hairWidth", "hairWidthScale_FloatValue", "hairWidthScale_Interp", "hairWidthScale_Position", "hairWidthScale", "hairsPerClump", "hueRand", "ignoreSolverGravity", "ignoreSolverWind", "inputForce", "inputHair", "inputHairPin", "internalState", "interpolationRange", "iterations", "lastEvalTime", "lengthFlex", "lightEachHair", "mass", "maxSelfCollisionIterations", "motionDrag", "multiStreakSpread1", "multiStreakSpread2", "multiStreaks", "nextState", "noStretch", "noise", "noiseFrequency", "noiseFrequencyU", "noiseFrequencyV", "noiseFrequencyW", "noiseMethod", "nucleusId", "numCollideNeighbors", "numUClumps", "numVClumps", "opacity", "outputHair", "outputRenderHairs", "playFromCache", "positions", "receiveShadows", "repulsion", "restLengthScale", "satRand", "selfCollide", "selfCollideWidthScale", "selfCollisionFlag", "simulationMethod", "solverDisplay", "specularColorB", "specularColorG", "specularColorR", "specularColor", "specularPower", "specularRand", "startCurveAttract", "startFrame", "startState", "startTime", "staticCling", "stickiness", "stiffness", "stiffnessScale_FloatValue", "stiffnessScale_Interp", "stiffnessScale_Position", "stiffnessScale", "stretchDamp", "stretchResistance", "subClumpMethod", "subClumpRand", "subClumping", "subSegments", "tangentialDrag", "thinning", "translucence", "turbulenceFrequency", "turbulenceSpeed", "turbulenceStrength", "twistResistance", "usePre70ForceIntensity", "valRand", "velocities", "vertexCounts", "visibleInReflections", "visibleInRefractions", "widthDrawSkip"]
	nodeLeafPlugs = ["active", "attachObjectId", "attractionDamp", "attractionScale", "baldnessMap", "bendAnisotropy", "bendFollow", "bendModel", "bendResistance", "bounce", "cacheableAttributes", "castShadows", "clumpCurl", "clumpFlatness", "clumpInterpolation", "clumpTwist", "clumpWidth", "clumpWidthScale", "collide", "collideGround", "collideOverSample", "collideStrength", "collideWidthOffset", "collisionData", "collisionFlag", "collisionLayer", "compressionResistance", "curl", "curlFrequency", "currentState", "currentTime", "damp", "detailNoise", "diffuseRand", "disableFollicleAnim", "diskCache", "displacementScale", "displayColor", "displayQuality", "drag", "drawCollideWidth", "dynamicsWeight", "evaluationOrder", "extraBendLinks", "fieldData", "friction", "gravity", "groundHeight", "hairColor", "hairColorScale", "hairCounts", "hairWidth", "hairWidthScale", "hairsPerClump", "hueRand", "ignoreSolverGravity", "ignoreSolverWind", "inputForce", "inputHair", "inputHairPin", "internalState", "interpolationRange", "iterations", "lastEvalTime", "lengthFlex", "lightEachHair", "mass", "maxSelfCollisionIterations", "motionDrag", "multiStreakSpread1", "multiStreakSpread2", "multiStreaks", "nextState", "noStretch", "noise", "noiseFrequency", "noiseFrequencyU", "noiseFrequencyV", "noiseFrequencyW", "noiseMethod", "nucleusId", "numCollideNeighbors", "numUClumps", "numVClumps", "opacity", "outputHair", "outputRenderHairs", "playFromCache", "positions", "receiveShadows", "repulsion", "restLengthScale", "satRand", "selfCollide", "selfCollideWidthScale", "selfCollisionFlag", "simulationMethod", "solverDisplay", "specularColor", "specularPower", "specularRand", "startCurveAttract", "startFrame", "startState", "startTime", "staticCling", "stickiness", "stiffness", "stiffnessScale", "stretchDamp", "stretchResistance", "subClumpMethod", "subClumpRand", "subClumping", "subSegments", "tangentialDrag", "thinning", "translucence", "turbulenceFrequency", "turbulenceSpeed", "turbulenceStrength", "twistResistance", "usePre70ForceIntensity", "valRand", "velocities", "vertexCounts", "visibleInReflections", "visibleInRefractions", "widthDrawSkip"]
	pass

