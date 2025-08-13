

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
NBase = retriever.getNodeCls("NBase")
assert NBase
if T.TYPE_CHECKING:
	from .. import NBase

# add node doc



# region plug type defs
class BlobbyRadiusScalePlug(Plug):
	node : NParticle = None
	pass
class BounceRandomizePlug(Plug):
	node : NParticle = None
	pass
class BounceScale_FloatValuePlug(Plug):
	parent : BounceScalePlug = PlugDescriptor("bounceScale")
	node : NParticle = None
	pass
class BounceScale_InterpPlug(Plug):
	parent : BounceScalePlug = PlugDescriptor("bounceScale")
	node : NParticle = None
	pass
class BounceScale_PositionPlug(Plug):
	parent : BounceScalePlug = PlugDescriptor("bounceScale")
	node : NParticle = None
	pass
class BounceScalePlug(Plug):
	bounceScale_FloatValue_ : BounceScale_FloatValuePlug = PlugDescriptor("bounceScale_FloatValue")
	boscfv_ : BounceScale_FloatValuePlug = PlugDescriptor("bounceScale_FloatValue")
	bounceScale_Interp_ : BounceScale_InterpPlug = PlugDescriptor("bounceScale_Interp")
	bosci_ : BounceScale_InterpPlug = PlugDescriptor("bounceScale_Interp")
	bounceScale_Position_ : BounceScale_PositionPlug = PlugDescriptor("bounceScale_Position")
	boscp_ : BounceScale_PositionPlug = PlugDescriptor("bounceScale_Position")
	node : NParticle = None
	pass
class BounceScaleInputPlug(Plug):
	node : NParticle = None
	pass
class BounceScaleInputMaxPlug(Plug):
	node : NParticle = None
	pass
class CacheableAttributesPlug(Plug):
	node : NParticle = None
	pass
class CollideStrengthScale_FloatValuePlug(Plug):
	parent : CollideStrengthScalePlug = PlugDescriptor("collideStrengthScale")
	node : NParticle = None
	pass
class CollideStrengthScale_InterpPlug(Plug):
	parent : CollideStrengthScalePlug = PlugDescriptor("collideStrengthScale")
	node : NParticle = None
	pass
class CollideStrengthScale_PositionPlug(Plug):
	parent : CollideStrengthScalePlug = PlugDescriptor("collideStrengthScale")
	node : NParticle = None
	pass
class CollideStrengthScalePlug(Plug):
	collideStrengthScale_FloatValue_ : CollideStrengthScale_FloatValuePlug = PlugDescriptor("collideStrengthScale_FloatValue")
	clscfv_ : CollideStrengthScale_FloatValuePlug = PlugDescriptor("collideStrengthScale_FloatValue")
	collideStrengthScale_Interp_ : CollideStrengthScale_InterpPlug = PlugDescriptor("collideStrengthScale_Interp")
	clsci_ : CollideStrengthScale_InterpPlug = PlugDescriptor("collideStrengthScale_Interp")
	collideStrengthScale_Position_ : CollideStrengthScale_PositionPlug = PlugDescriptor("collideStrengthScale_Position")
	clscp_ : CollideStrengthScale_PositionPlug = PlugDescriptor("collideStrengthScale_Position")
	node : NParticle = None
	pass
class CollideStrengthScaleInputPlug(Plug):
	node : NParticle = None
	pass
class CollideStrengthScaleInputMaxPlug(Plug):
	node : NParticle = None
	pass
class CollideWidthScalePlug(Plug):
	node : NParticle = None
	pass
class Color_ColorBPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : NParticle = None
	pass
class Color_ColorGPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : NParticle = None
	pass
class Color_ColorRPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : NParticle = None
	pass
class Color_ColorPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	color_ColorB_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	clcb_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	color_ColorG_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	clcg_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	color_ColorR_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	clcr_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	node : NParticle = None
	pass
class Color_InterpPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : NParticle = None
	pass
class Color_PositionPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : NParticle = None
	pass
class ColorPlug(Plug):
	color_Color_ : Color_ColorPlug = PlugDescriptor("color_Color")
	clc_ : Color_ColorPlug = PlugDescriptor("color_Color")
	color_Interp_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	cli_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	color_Position_ : Color_PositionPlug = PlugDescriptor("color_Position")
	clp_ : Color_PositionPlug = PlugDescriptor("color_Position")
	node : NParticle = None
	pass
class ColorBluePlug(Plug):
	node : NParticle = None
	pass
class ColorGreenPlug(Plug):
	node : NParticle = None
	pass
class ColorInputPlug(Plug):
	node : NParticle = None
	pass
class ColorInputMaxPlug(Plug):
	node : NParticle = None
	pass
class ColorPerVertexPlug(Plug):
	node : NParticle = None
	pass
class ColorRandomizePlug(Plug):
	node : NParticle = None
	pass
class ColorRedPlug(Plug):
	node : NParticle = None
	pass
class ComputeRotationPlug(Plug):
	node : NParticle = None
	pass
class DragPlug(Plug):
	node : NParticle = None
	pass
class EmissionOverlapPruningPlug(Plug):
	node : NParticle = None
	pass
class EnableSPHPlug(Plug):
	node : NParticle = None
	pass
class FrictionRandomizePlug(Plug):
	node : NParticle = None
	pass
class FrictionScale_FloatValuePlug(Plug):
	parent : FrictionScalePlug = PlugDescriptor("frictionScale")
	node : NParticle = None
	pass
class FrictionScale_InterpPlug(Plug):
	parent : FrictionScalePlug = PlugDescriptor("frictionScale")
	node : NParticle = None
	pass
class FrictionScale_PositionPlug(Plug):
	parent : FrictionScalePlug = PlugDescriptor("frictionScale")
	node : NParticle = None
	pass
class FrictionScalePlug(Plug):
	frictionScale_FloatValue_ : FrictionScale_FloatValuePlug = PlugDescriptor("frictionScale_FloatValue")
	frscfv_ : FrictionScale_FloatValuePlug = PlugDescriptor("frictionScale_FloatValue")
	frictionScale_Interp_ : FrictionScale_InterpPlug = PlugDescriptor("frictionScale_Interp")
	frsci_ : FrictionScale_InterpPlug = PlugDescriptor("frictionScale_Interp")
	frictionScale_Position_ : FrictionScale_PositionPlug = PlugDescriptor("frictionScale_Position")
	frscp_ : FrictionScale_PositionPlug = PlugDescriptor("frictionScale_Position")
	node : NParticle = None
	pass
class FrictionScaleInputPlug(Plug):
	node : NParticle = None
	pass
class FrictionScaleInputMaxPlug(Plug):
	node : NParticle = None
	pass
class IgnoreSolverGravityPlug(Plug):
	node : NParticle = None
	pass
class IgnoreSolverWindPlug(Plug):
	node : NParticle = None
	pass
class Incandescence_ColorBPlug(Plug):
	parent : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	node : NParticle = None
	pass
class Incandescence_ColorGPlug(Plug):
	parent : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	node : NParticle = None
	pass
class Incandescence_ColorRPlug(Plug):
	parent : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	node : NParticle = None
	pass
class Incandescence_ColorPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	incandescence_ColorB_ : Incandescence_ColorBPlug = PlugDescriptor("incandescence_ColorB")
	incacb_ : Incandescence_ColorBPlug = PlugDescriptor("incandescence_ColorB")
	incandescence_ColorG_ : Incandescence_ColorGPlug = PlugDescriptor("incandescence_ColorG")
	incacg_ : Incandescence_ColorGPlug = PlugDescriptor("incandescence_ColorG")
	incandescence_ColorR_ : Incandescence_ColorRPlug = PlugDescriptor("incandescence_ColorR")
	incacr_ : Incandescence_ColorRPlug = PlugDescriptor("incandescence_ColorR")
	node : NParticle = None
	pass
class Incandescence_InterpPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : NParticle = None
	pass
class Incandescence_PositionPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : NParticle = None
	pass
class IncandescencePlug(Plug):
	incandescence_Color_ : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	incac_ : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	incandescence_Interp_ : Incandescence_InterpPlug = PlugDescriptor("incandescence_Interp")
	incai_ : Incandescence_InterpPlug = PlugDescriptor("incandescence_Interp")
	incandescence_Position_ : Incandescence_PositionPlug = PlugDescriptor("incandescence_Position")
	incap_ : Incandescence_PositionPlug = PlugDescriptor("incandescence_Position")
	node : NParticle = None
	pass
class IncandescenceInputPlug(Plug):
	node : NParticle = None
	pass
class IncandescenceInputMaxPlug(Plug):
	node : NParticle = None
	pass
class IncandescencePerVertexPlug(Plug):
	node : NParticle = None
	pass
class IncandescenceRandomizePlug(Plug):
	node : NParticle = None
	pass
class IncompressibilityPlug(Plug):
	node : NParticle = None
	pass
class InputAttractPlug(Plug):
	node : NParticle = None
	pass
class InputAttractDampPlug(Plug):
	node : NParticle = None
	pass
class InternalBounceRampPlug(Plug):
	node : NParticle = None
	pass
class InternalCollideStrengthRampPlug(Plug):
	node : NParticle = None
	pass
class InternalColorRampPlug(Plug):
	node : NParticle = None
	pass
class InternalFieldScaleRampPlug(Plug):
	node : NParticle = None
	pass
class InternalFrictionRampPlug(Plug):
	node : NParticle = None
	pass
class InternalIncandescenceRampPlug(Plug):
	node : NParticle = None
	pass
class InternalMassRampPlug(Plug):
	node : NParticle = None
	pass
class InternalOpacityRampPlug(Plug):
	node : NParticle = None
	pass
class InternalRadiusRampPlug(Plug):
	node : NParticle = None
	pass
class InternalStickinessRampPlug(Plug):
	node : NParticle = None
	pass
class InternalSurfaceTensionRampPlug(Plug):
	node : NParticle = None
	pass
class InternalViscosityRampPlug(Plug):
	node : NParticle = None
	pass
class MassScale_FloatValuePlug(Plug):
	parent : MassScalePlug = PlugDescriptor("massScale")
	node : NParticle = None
	pass
class MassScale_InterpPlug(Plug):
	parent : MassScalePlug = PlugDescriptor("massScale")
	node : NParticle = None
	pass
class MassScale_PositionPlug(Plug):
	parent : MassScalePlug = PlugDescriptor("massScale")
	node : NParticle = None
	pass
class MassScalePlug(Plug):
	massScale_FloatValue_ : MassScale_FloatValuePlug = PlugDescriptor("massScale_FloatValue")
	msscfv_ : MassScale_FloatValuePlug = PlugDescriptor("massScale_FloatValue")
	massScale_Interp_ : MassScale_InterpPlug = PlugDescriptor("massScale_Interp")
	mssci_ : MassScale_InterpPlug = PlugDescriptor("massScale_Interp")
	massScale_Position_ : MassScale_PositionPlug = PlugDescriptor("massScale_Position")
	msscp_ : MassScale_PositionPlug = PlugDescriptor("massScale_Position")
	node : NParticle = None
	pass
class MassScaleInputPlug(Plug):
	node : NParticle = None
	pass
class MassScaleInputMaxPlug(Plug):
	node : NParticle = None
	pass
class MassScaleRandomizePlug(Plug):
	node : NParticle = None
	pass
class MaxTriangleResolutionPlug(Plug):
	node : NParticle = None
	pass
class MeshMethodPlug(Plug):
	node : NParticle = None
	pass
class MeshSmoothingIterationsPlug(Plug):
	node : NParticle = None
	pass
class MeshTriangleSizePlug(Plug):
	node : NParticle = None
	pass
class MotionStreakPlug(Plug):
	node : NParticle = None
	pass
class NumSubdivisionsPlug(Plug):
	node : NParticle = None
	pass
class OpacityPlug(Plug):
	node : NParticle = None
	pass
class OpacityPerVertexPlug(Plug):
	node : NParticle = None
	pass
class OpacityScale_FloatValuePlug(Plug):
	parent : OpacityScalePlug = PlugDescriptor("opacityScale")
	node : NParticle = None
	pass
class OpacityScale_InterpPlug(Plug):
	parent : OpacityScalePlug = PlugDescriptor("opacityScale")
	node : NParticle = None
	pass
class OpacityScale_PositionPlug(Plug):
	parent : OpacityScalePlug = PlugDescriptor("opacityScale")
	node : NParticle = None
	pass
class OpacityScalePlug(Plug):
	opacityScale_FloatValue_ : OpacityScale_FloatValuePlug = PlugDescriptor("opacityScale_FloatValue")
	opcfv_ : OpacityScale_FloatValuePlug = PlugDescriptor("opacityScale_FloatValue")
	opacityScale_Interp_ : OpacityScale_InterpPlug = PlugDescriptor("opacityScale_Interp")
	opci_ : OpacityScale_InterpPlug = PlugDescriptor("opacityScale_Interp")
	opacityScale_Position_ : OpacityScale_PositionPlug = PlugDescriptor("opacityScale_Position")
	opcp_ : OpacityScale_PositionPlug = PlugDescriptor("opacityScale_Position")
	node : NParticle = None
	pass
class OpacityScaleInputPlug(Plug):
	node : NParticle = None
	pass
class OpacityScaleInputMaxPlug(Plug):
	node : NParticle = None
	pass
class OpacityScaleRandomizePlug(Plug):
	node : NParticle = None
	pass
class OutMeshPlug(Plug):
	node : NParticle = None
	pass
class ParallelSPHPlug(Plug):
	node : NParticle = None
	pass
class PointFieldScale_FloatValuePlug(Plug):
	parent : PointFieldScalePlug = PlugDescriptor("pointFieldScale")
	node : NParticle = None
	pass
class PointFieldScale_InterpPlug(Plug):
	parent : PointFieldScalePlug = PlugDescriptor("pointFieldScale")
	node : NParticle = None
	pass
class PointFieldScale_PositionPlug(Plug):
	parent : PointFieldScalePlug = PlugDescriptor("pointFieldScale")
	node : NParticle = None
	pass
class PointFieldScalePlug(Plug):
	pointFieldScale_FloatValue_ : PointFieldScale_FloatValuePlug = PlugDescriptor("pointFieldScale_FloatValue")
	pfscfv_ : PointFieldScale_FloatValuePlug = PlugDescriptor("pointFieldScale_FloatValue")
	pointFieldScale_Interp_ : PointFieldScale_InterpPlug = PlugDescriptor("pointFieldScale_Interp")
	pfsci_ : PointFieldScale_InterpPlug = PlugDescriptor("pointFieldScale_Interp")
	pointFieldScale_Position_ : PointFieldScale_PositionPlug = PlugDescriptor("pointFieldScale_Position")
	pfscp_ : PointFieldScale_PositionPlug = PlugDescriptor("pointFieldScale_Position")
	node : NParticle = None
	pass
class PointFieldScaleInputPlug(Plug):
	node : NParticle = None
	pass
class PointFieldScaleInputMaxPlug(Plug):
	node : NParticle = None
	pass
class PostCacheRampEvalPlug(Plug):
	node : NParticle = None
	pass
class RadiusPlug(Plug):
	node : NParticle = None
	pass
class RadiusScale_FloatValuePlug(Plug):
	parent : RadiusScalePlug = PlugDescriptor("radiusScale")
	node : NParticle = None
	pass
class RadiusScale_InterpPlug(Plug):
	parent : RadiusScalePlug = PlugDescriptor("radiusScale")
	node : NParticle = None
	pass
class RadiusScale_PositionPlug(Plug):
	parent : RadiusScalePlug = PlugDescriptor("radiusScale")
	node : NParticle = None
	pass
class RadiusScalePlug(Plug):
	radiusScale_FloatValue_ : RadiusScale_FloatValuePlug = PlugDescriptor("radiusScale_FloatValue")
	rdcfv_ : RadiusScale_FloatValuePlug = PlugDescriptor("radiusScale_FloatValue")
	radiusScale_Interp_ : RadiusScale_InterpPlug = PlugDescriptor("radiusScale_Interp")
	rdci_ : RadiusScale_InterpPlug = PlugDescriptor("radiusScale_Interp")
	radiusScale_Position_ : RadiusScale_PositionPlug = PlugDescriptor("radiusScale_Position")
	rdcp_ : RadiusScale_PositionPlug = PlugDescriptor("radiusScale_Position")
	node : NParticle = None
	pass
class RadiusScaleInputPlug(Plug):
	node : NParticle = None
	pass
class RadiusScaleInputMaxPlug(Plug):
	node : NParticle = None
	pass
class RadiusScaleRandomizePlug(Plug):
	node : NParticle = None
	pass
class RadiusScaleSPHPlug(Plug):
	node : NParticle = None
	pass
class RecomputePairsSPHPlug(Plug):
	node : NParticle = None
	pass
class RestDensityPlug(Plug):
	node : NParticle = None
	pass
class RotationDampPlug(Plug):
	node : NParticle = None
	pass
class RotationFrictionPlug(Plug):
	node : NParticle = None
	pass
class ScalingRelationPlug(Plug):
	node : NParticle = None
	pass
class SelfCollideWidthScalePlug(Plug):
	node : NParticle = None
	pass
class SolverDisplayPlug(Plug):
	node : NParticle = None
	pass
class StickinessRandomizePlug(Plug):
	node : NParticle = None
	pass
class StickinessScale_FloatValuePlug(Plug):
	parent : StickinessScalePlug = PlugDescriptor("stickinessScale")
	node : NParticle = None
	pass
class StickinessScale_InterpPlug(Plug):
	parent : StickinessScalePlug = PlugDescriptor("stickinessScale")
	node : NParticle = None
	pass
class StickinessScale_PositionPlug(Plug):
	parent : StickinessScalePlug = PlugDescriptor("stickinessScale")
	node : NParticle = None
	pass
class StickinessScalePlug(Plug):
	stickinessScale_FloatValue_ : StickinessScale_FloatValuePlug = PlugDescriptor("stickinessScale_FloatValue")
	stscfv_ : StickinessScale_FloatValuePlug = PlugDescriptor("stickinessScale_FloatValue")
	stickinessScale_Interp_ : StickinessScale_InterpPlug = PlugDescriptor("stickinessScale_Interp")
	stsci_ : StickinessScale_InterpPlug = PlugDescriptor("stickinessScale_Interp")
	stickinessScale_Position_ : StickinessScale_PositionPlug = PlugDescriptor("stickinessScale_Position")
	stscp_ : StickinessScale_PositionPlug = PlugDescriptor("stickinessScale_Position")
	node : NParticle = None
	pass
class StickinessScaleInputPlug(Plug):
	node : NParticle = None
	pass
class StickinessScaleInputMaxPlug(Plug):
	node : NParticle = None
	pass
class SurfaceTensionPlug(Plug):
	node : NParticle = None
	pass
class SurfaceTensionScale_FloatValuePlug(Plug):
	parent : SurfaceTensionScalePlug = PlugDescriptor("surfaceTensionScale")
	node : NParticle = None
	pass
class SurfaceTensionScale_InterpPlug(Plug):
	parent : SurfaceTensionScalePlug = PlugDescriptor("surfaceTensionScale")
	node : NParticle = None
	pass
class SurfaceTensionScale_PositionPlug(Plug):
	parent : SurfaceTensionScalePlug = PlugDescriptor("surfaceTensionScale")
	node : NParticle = None
	pass
class SurfaceTensionScalePlug(Plug):
	surfaceTensionScale_FloatValue_ : SurfaceTensionScale_FloatValuePlug = PlugDescriptor("surfaceTensionScale_FloatValue")
	stnsfv_ : SurfaceTensionScale_FloatValuePlug = PlugDescriptor("surfaceTensionScale_FloatValue")
	surfaceTensionScale_Interp_ : SurfaceTensionScale_InterpPlug = PlugDescriptor("surfaceTensionScale_Interp")
	stnsi_ : SurfaceTensionScale_InterpPlug = PlugDescriptor("surfaceTensionScale_Interp")
	surfaceTensionScale_Position_ : SurfaceTensionScale_PositionPlug = PlugDescriptor("surfaceTensionScale_Position")
	stnsp_ : SurfaceTensionScale_PositionPlug = PlugDescriptor("surfaceTensionScale_Position")
	node : NParticle = None
	pass
class SurfaceTensionScaleInputPlug(Plug):
	node : NParticle = None
	pass
class SurfaceTensionScaleInputMaxPlug(Plug):
	node : NParticle = None
	pass
class ThresholdPlug(Plug):
	node : NParticle = None
	pass
class UseGradientNormalsPlug(Plug):
	node : NParticle = None
	pass
class UvwPerVertexPlug(Plug):
	node : NParticle = None
	pass
class VelocityPerVertexPlug(Plug):
	node : NParticle = None
	pass
class ViscosityPlug(Plug):
	node : NParticle = None
	pass
class ViscosityScale_FloatValuePlug(Plug):
	parent : ViscosityScalePlug = PlugDescriptor("viscosityScale")
	node : NParticle = None
	pass
class ViscosityScale_InterpPlug(Plug):
	parent : ViscosityScalePlug = PlugDescriptor("viscosityScale")
	node : NParticle = None
	pass
class ViscosityScale_PositionPlug(Plug):
	parent : ViscosityScalePlug = PlugDescriptor("viscosityScale")
	node : NParticle = None
	pass
class ViscosityScalePlug(Plug):
	viscosityScale_FloatValue_ : ViscosityScale_FloatValuePlug = PlugDescriptor("viscosityScale_FloatValue")
	vsscfv_ : ViscosityScale_FloatValuePlug = PlugDescriptor("viscosityScale_FloatValue")
	viscosityScale_Interp_ : ViscosityScale_InterpPlug = PlugDescriptor("viscosityScale_Interp")
	vssci_ : ViscosityScale_InterpPlug = PlugDescriptor("viscosityScale_Interp")
	viscosityScale_Position_ : ViscosityScale_PositionPlug = PlugDescriptor("viscosityScale_Position")
	vsscp_ : ViscosityScale_PositionPlug = PlugDescriptor("viscosityScale_Position")
	node : NParticle = None
	pass
class ViscosityScaleInputPlug(Plug):
	node : NParticle = None
	pass
class ViscosityScaleInputMaxPlug(Plug):
	node : NParticle = None
	pass
class WindSelfShadowPlug(Plug):
	node : NParticle = None
	pass
# endregion


# define node class
class NParticle(NBase):
	blobbyRadiusScale_ : BlobbyRadiusScalePlug = PlugDescriptor("blobbyRadiusScale")
	bounceRandomize_ : BounceRandomizePlug = PlugDescriptor("bounceRandomize")
	bounceScale_FloatValue_ : BounceScale_FloatValuePlug = PlugDescriptor("bounceScale_FloatValue")
	bounceScale_Interp_ : BounceScale_InterpPlug = PlugDescriptor("bounceScale_Interp")
	bounceScale_Position_ : BounceScale_PositionPlug = PlugDescriptor("bounceScale_Position")
	bounceScale_ : BounceScalePlug = PlugDescriptor("bounceScale")
	bounceScaleInput_ : BounceScaleInputPlug = PlugDescriptor("bounceScaleInput")
	bounceScaleInputMax_ : BounceScaleInputMaxPlug = PlugDescriptor("bounceScaleInputMax")
	cacheableAttributes_ : CacheableAttributesPlug = PlugDescriptor("cacheableAttributes")
	collideStrengthScale_FloatValue_ : CollideStrengthScale_FloatValuePlug = PlugDescriptor("collideStrengthScale_FloatValue")
	collideStrengthScale_Interp_ : CollideStrengthScale_InterpPlug = PlugDescriptor("collideStrengthScale_Interp")
	collideStrengthScale_Position_ : CollideStrengthScale_PositionPlug = PlugDescriptor("collideStrengthScale_Position")
	collideStrengthScale_ : CollideStrengthScalePlug = PlugDescriptor("collideStrengthScale")
	collideStrengthScaleInput_ : CollideStrengthScaleInputPlug = PlugDescriptor("collideStrengthScaleInput")
	collideStrengthScaleInputMax_ : CollideStrengthScaleInputMaxPlug = PlugDescriptor("collideStrengthScaleInputMax")
	collideWidthScale_ : CollideWidthScalePlug = PlugDescriptor("collideWidthScale")
	color_ColorB_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	color_ColorG_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	color_ColorR_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	color_Color_ : Color_ColorPlug = PlugDescriptor("color_Color")
	color_Interp_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	color_Position_ : Color_PositionPlug = PlugDescriptor("color_Position")
	color_ : ColorPlug = PlugDescriptor("color")
	colorBlue_ : ColorBluePlug = PlugDescriptor("colorBlue")
	colorGreen_ : ColorGreenPlug = PlugDescriptor("colorGreen")
	colorInput_ : ColorInputPlug = PlugDescriptor("colorInput")
	colorInputMax_ : ColorInputMaxPlug = PlugDescriptor("colorInputMax")
	colorPerVertex_ : ColorPerVertexPlug = PlugDescriptor("colorPerVertex")
	colorRandomize_ : ColorRandomizePlug = PlugDescriptor("colorRandomize")
	colorRed_ : ColorRedPlug = PlugDescriptor("colorRed")
	computeRotation_ : ComputeRotationPlug = PlugDescriptor("computeRotation")
	drag_ : DragPlug = PlugDescriptor("drag")
	emissionOverlapPruning_ : EmissionOverlapPruningPlug = PlugDescriptor("emissionOverlapPruning")
	enableSPH_ : EnableSPHPlug = PlugDescriptor("enableSPH")
	frictionRandomize_ : FrictionRandomizePlug = PlugDescriptor("frictionRandomize")
	frictionScale_FloatValue_ : FrictionScale_FloatValuePlug = PlugDescriptor("frictionScale_FloatValue")
	frictionScale_Interp_ : FrictionScale_InterpPlug = PlugDescriptor("frictionScale_Interp")
	frictionScale_Position_ : FrictionScale_PositionPlug = PlugDescriptor("frictionScale_Position")
	frictionScale_ : FrictionScalePlug = PlugDescriptor("frictionScale")
	frictionScaleInput_ : FrictionScaleInputPlug = PlugDescriptor("frictionScaleInput")
	frictionScaleInputMax_ : FrictionScaleInputMaxPlug = PlugDescriptor("frictionScaleInputMax")
	ignoreSolverGravity_ : IgnoreSolverGravityPlug = PlugDescriptor("ignoreSolverGravity")
	ignoreSolverWind_ : IgnoreSolverWindPlug = PlugDescriptor("ignoreSolverWind")
	incandescence_ColorB_ : Incandescence_ColorBPlug = PlugDescriptor("incandescence_ColorB")
	incandescence_ColorG_ : Incandescence_ColorGPlug = PlugDescriptor("incandescence_ColorG")
	incandescence_ColorR_ : Incandescence_ColorRPlug = PlugDescriptor("incandescence_ColorR")
	incandescence_Color_ : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	incandescence_Interp_ : Incandescence_InterpPlug = PlugDescriptor("incandescence_Interp")
	incandescence_Position_ : Incandescence_PositionPlug = PlugDescriptor("incandescence_Position")
	incandescence_ : IncandescencePlug = PlugDescriptor("incandescence")
	incandescenceInput_ : IncandescenceInputPlug = PlugDescriptor("incandescenceInput")
	incandescenceInputMax_ : IncandescenceInputMaxPlug = PlugDescriptor("incandescenceInputMax")
	incandescencePerVertex_ : IncandescencePerVertexPlug = PlugDescriptor("incandescencePerVertex")
	incandescenceRandomize_ : IncandescenceRandomizePlug = PlugDescriptor("incandescenceRandomize")
	incompressibility_ : IncompressibilityPlug = PlugDescriptor("incompressibility")
	inputAttract_ : InputAttractPlug = PlugDescriptor("inputAttract")
	inputAttractDamp_ : InputAttractDampPlug = PlugDescriptor("inputAttractDamp")
	internalBounceRamp_ : InternalBounceRampPlug = PlugDescriptor("internalBounceRamp")
	internalCollideStrengthRamp_ : InternalCollideStrengthRampPlug = PlugDescriptor("internalCollideStrengthRamp")
	internalColorRamp_ : InternalColorRampPlug = PlugDescriptor("internalColorRamp")
	internalFieldScaleRamp_ : InternalFieldScaleRampPlug = PlugDescriptor("internalFieldScaleRamp")
	internalFrictionRamp_ : InternalFrictionRampPlug = PlugDescriptor("internalFrictionRamp")
	internalIncandescenceRamp_ : InternalIncandescenceRampPlug = PlugDescriptor("internalIncandescenceRamp")
	internalMassRamp_ : InternalMassRampPlug = PlugDescriptor("internalMassRamp")
	internalOpacityRamp_ : InternalOpacityRampPlug = PlugDescriptor("internalOpacityRamp")
	internalRadiusRamp_ : InternalRadiusRampPlug = PlugDescriptor("internalRadiusRamp")
	internalStickinessRamp_ : InternalStickinessRampPlug = PlugDescriptor("internalStickinessRamp")
	internalSurfaceTensionRamp_ : InternalSurfaceTensionRampPlug = PlugDescriptor("internalSurfaceTensionRamp")
	internalViscosityRamp_ : InternalViscosityRampPlug = PlugDescriptor("internalViscosityRamp")
	massScale_FloatValue_ : MassScale_FloatValuePlug = PlugDescriptor("massScale_FloatValue")
	massScale_Interp_ : MassScale_InterpPlug = PlugDescriptor("massScale_Interp")
	massScale_Position_ : MassScale_PositionPlug = PlugDescriptor("massScale_Position")
	massScale_ : MassScalePlug = PlugDescriptor("massScale")
	massScaleInput_ : MassScaleInputPlug = PlugDescriptor("massScaleInput")
	massScaleInputMax_ : MassScaleInputMaxPlug = PlugDescriptor("massScaleInputMax")
	massScaleRandomize_ : MassScaleRandomizePlug = PlugDescriptor("massScaleRandomize")
	maxTriangleResolution_ : MaxTriangleResolutionPlug = PlugDescriptor("maxTriangleResolution")
	meshMethod_ : MeshMethodPlug = PlugDescriptor("meshMethod")
	meshSmoothingIterations_ : MeshSmoothingIterationsPlug = PlugDescriptor("meshSmoothingIterations")
	meshTriangleSize_ : MeshTriangleSizePlug = PlugDescriptor("meshTriangleSize")
	motionStreak_ : MotionStreakPlug = PlugDescriptor("motionStreak")
	numSubdivisions_ : NumSubdivisionsPlug = PlugDescriptor("numSubdivisions")
	opacity_ : OpacityPlug = PlugDescriptor("opacity")
	opacityPerVertex_ : OpacityPerVertexPlug = PlugDescriptor("opacityPerVertex")
	opacityScale_FloatValue_ : OpacityScale_FloatValuePlug = PlugDescriptor("opacityScale_FloatValue")
	opacityScale_Interp_ : OpacityScale_InterpPlug = PlugDescriptor("opacityScale_Interp")
	opacityScale_Position_ : OpacityScale_PositionPlug = PlugDescriptor("opacityScale_Position")
	opacityScale_ : OpacityScalePlug = PlugDescriptor("opacityScale")
	opacityScaleInput_ : OpacityScaleInputPlug = PlugDescriptor("opacityScaleInput")
	opacityScaleInputMax_ : OpacityScaleInputMaxPlug = PlugDescriptor("opacityScaleInputMax")
	opacityScaleRandomize_ : OpacityScaleRandomizePlug = PlugDescriptor("opacityScaleRandomize")
	outMesh_ : OutMeshPlug = PlugDescriptor("outMesh")
	parallelSPH_ : ParallelSPHPlug = PlugDescriptor("parallelSPH")
	pointFieldScale_FloatValue_ : PointFieldScale_FloatValuePlug = PlugDescriptor("pointFieldScale_FloatValue")
	pointFieldScale_Interp_ : PointFieldScale_InterpPlug = PlugDescriptor("pointFieldScale_Interp")
	pointFieldScale_Position_ : PointFieldScale_PositionPlug = PlugDescriptor("pointFieldScale_Position")
	pointFieldScale_ : PointFieldScalePlug = PlugDescriptor("pointFieldScale")
	pointFieldScaleInput_ : PointFieldScaleInputPlug = PlugDescriptor("pointFieldScaleInput")
	pointFieldScaleInputMax_ : PointFieldScaleInputMaxPlug = PlugDescriptor("pointFieldScaleInputMax")
	postCacheRampEval_ : PostCacheRampEvalPlug = PlugDescriptor("postCacheRampEval")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	radiusScale_FloatValue_ : RadiusScale_FloatValuePlug = PlugDescriptor("radiusScale_FloatValue")
	radiusScale_Interp_ : RadiusScale_InterpPlug = PlugDescriptor("radiusScale_Interp")
	radiusScale_Position_ : RadiusScale_PositionPlug = PlugDescriptor("radiusScale_Position")
	radiusScale_ : RadiusScalePlug = PlugDescriptor("radiusScale")
	radiusScaleInput_ : RadiusScaleInputPlug = PlugDescriptor("radiusScaleInput")
	radiusScaleInputMax_ : RadiusScaleInputMaxPlug = PlugDescriptor("radiusScaleInputMax")
	radiusScaleRandomize_ : RadiusScaleRandomizePlug = PlugDescriptor("radiusScaleRandomize")
	radiusScaleSPH_ : RadiusScaleSPHPlug = PlugDescriptor("radiusScaleSPH")
	recomputePairsSPH_ : RecomputePairsSPHPlug = PlugDescriptor("recomputePairsSPH")
	restDensity_ : RestDensityPlug = PlugDescriptor("restDensity")
	rotationDamp_ : RotationDampPlug = PlugDescriptor("rotationDamp")
	rotationFriction_ : RotationFrictionPlug = PlugDescriptor("rotationFriction")
	scalingRelation_ : ScalingRelationPlug = PlugDescriptor("scalingRelation")
	selfCollideWidthScale_ : SelfCollideWidthScalePlug = PlugDescriptor("selfCollideWidthScale")
	solverDisplay_ : SolverDisplayPlug = PlugDescriptor("solverDisplay")
	stickinessRandomize_ : StickinessRandomizePlug = PlugDescriptor("stickinessRandomize")
	stickinessScale_FloatValue_ : StickinessScale_FloatValuePlug = PlugDescriptor("stickinessScale_FloatValue")
	stickinessScale_Interp_ : StickinessScale_InterpPlug = PlugDescriptor("stickinessScale_Interp")
	stickinessScale_Position_ : StickinessScale_PositionPlug = PlugDescriptor("stickinessScale_Position")
	stickinessScale_ : StickinessScalePlug = PlugDescriptor("stickinessScale")
	stickinessScaleInput_ : StickinessScaleInputPlug = PlugDescriptor("stickinessScaleInput")
	stickinessScaleInputMax_ : StickinessScaleInputMaxPlug = PlugDescriptor("stickinessScaleInputMax")
	surfaceTension_ : SurfaceTensionPlug = PlugDescriptor("surfaceTension")
	surfaceTensionScale_FloatValue_ : SurfaceTensionScale_FloatValuePlug = PlugDescriptor("surfaceTensionScale_FloatValue")
	surfaceTensionScale_Interp_ : SurfaceTensionScale_InterpPlug = PlugDescriptor("surfaceTensionScale_Interp")
	surfaceTensionScale_Position_ : SurfaceTensionScale_PositionPlug = PlugDescriptor("surfaceTensionScale_Position")
	surfaceTensionScale_ : SurfaceTensionScalePlug = PlugDescriptor("surfaceTensionScale")
	surfaceTensionScaleInput_ : SurfaceTensionScaleInputPlug = PlugDescriptor("surfaceTensionScaleInput")
	surfaceTensionScaleInputMax_ : SurfaceTensionScaleInputMaxPlug = PlugDescriptor("surfaceTensionScaleInputMax")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")
	useGradientNormals_ : UseGradientNormalsPlug = PlugDescriptor("useGradientNormals")
	uvwPerVertex_ : UvwPerVertexPlug = PlugDescriptor("uvwPerVertex")
	velocityPerVertex_ : VelocityPerVertexPlug = PlugDescriptor("velocityPerVertex")
	viscosity_ : ViscosityPlug = PlugDescriptor("viscosity")
	viscosityScale_FloatValue_ : ViscosityScale_FloatValuePlug = PlugDescriptor("viscosityScale_FloatValue")
	viscosityScale_Interp_ : ViscosityScale_InterpPlug = PlugDescriptor("viscosityScale_Interp")
	viscosityScale_Position_ : ViscosityScale_PositionPlug = PlugDescriptor("viscosityScale_Position")
	viscosityScale_ : ViscosityScalePlug = PlugDescriptor("viscosityScale")
	viscosityScaleInput_ : ViscosityScaleInputPlug = PlugDescriptor("viscosityScaleInput")
	viscosityScaleInputMax_ : ViscosityScaleInputMaxPlug = PlugDescriptor("viscosityScaleInputMax")
	windSelfShadow_ : WindSelfShadowPlug = PlugDescriptor("windSelfShadow")

	# node attributes

	typeName = "nParticle"
	apiTypeInt = 1008
	apiTypeStr = "kNParticle"
	typeIdInt = 1313882450
	MFnCls = om.MFnDagNode
	pass

