

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PointEmitter = retriever.getNodeCls("PointEmitter")
assert PointEmitter
if T.TYPE_CHECKING:
	from .. import PointEmitter

# add node doc



# region plug type defs
class DensityEmissionMapPlug(Plug):
	node : FluidEmitter = None
	pass
class DensityMethodPlug(Plug):
	node : FluidEmitter = None
	pass
class DetailTurbulencePlug(Plug):
	node : FluidEmitter = None
	pass
class EmissionFunctionPlug(Plug):
	node : FluidEmitter = None
	pass
class EmitFluidColorPlug(Plug):
	node : FluidEmitter = None
	pass
class FillObjectPlug(Plug):
	node : FluidEmitter = None
	pass
class FluidDensityEmissionPlug(Plug):
	node : FluidEmitter = None
	pass
class FluidDropoffPlug(Plug):
	node : FluidEmitter = None
	pass
class FluidFuelEmissionPlug(Plug):
	node : FluidEmitter = None
	pass
class FluidHeatEmissionPlug(Plug):
	node : FluidEmitter = None
	pass
class FluidJitterPlug(Plug):
	node : FluidEmitter = None
	pass
class FuelEmissionMapPlug(Plug):
	node : FluidEmitter = None
	pass
class FuelMethodPlug(Plug):
	node : FluidEmitter = None
	pass
class HeatEmissionMapPlug(Plug):
	node : FluidEmitter = None
	pass
class HeatMethodPlug(Plug):
	node : FluidEmitter = None
	pass
class InheritVelocityPlug(Plug):
	node : FluidEmitter = None
	pass
class MotionStreakPlug(Plug):
	node : FluidEmitter = None
	pass
class NormalizedDropoffPlug(Plug):
	node : FluidEmitter = None
	pass
class RadiusPPPlug(Plug):
	node : FluidEmitter = None
	pass
class SpeedMethodPlug(Plug):
	node : FluidEmitter = None
	pass
class StartFrameEmissionPlug(Plug):
	node : FluidEmitter = None
	pass
class TurbulencePlug(Plug):
	node : FluidEmitter = None
	pass
class TurbulenceFrequencyXPlug(Plug):
	parent : TurbulenceFrequencyPlug = PlugDescriptor("turbulenceFrequency")
	node : FluidEmitter = None
	pass
class TurbulenceFrequencyYPlug(Plug):
	parent : TurbulenceFrequencyPlug = PlugDescriptor("turbulenceFrequency")
	node : FluidEmitter = None
	pass
class TurbulenceFrequencyZPlug(Plug):
	parent : TurbulenceFrequencyPlug = PlugDescriptor("turbulenceFrequency")
	node : FluidEmitter = None
	pass
class TurbulenceFrequencyPlug(Plug):
	turbulenceFrequencyX_ : TurbulenceFrequencyXPlug = PlugDescriptor("turbulenceFrequencyX")
	tfx_ : TurbulenceFrequencyXPlug = PlugDescriptor("turbulenceFrequencyX")
	turbulenceFrequencyY_ : TurbulenceFrequencyYPlug = PlugDescriptor("turbulenceFrequencyY")
	tfy_ : TurbulenceFrequencyYPlug = PlugDescriptor("turbulenceFrequencyY")
	turbulenceFrequencyZ_ : TurbulenceFrequencyZPlug = PlugDescriptor("turbulenceFrequencyZ")
	tfz_ : TurbulenceFrequencyZPlug = PlugDescriptor("turbulenceFrequencyZ")
	node : FluidEmitter = None
	pass
class TurbulenceOffsetXPlug(Plug):
	parent : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	node : FluidEmitter = None
	pass
class TurbulenceOffsetYPlug(Plug):
	parent : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	node : FluidEmitter = None
	pass
class TurbulenceOffsetZPlug(Plug):
	parent : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	node : FluidEmitter = None
	pass
class TurbulenceOffsetPlug(Plug):
	turbulenceOffsetX_ : TurbulenceOffsetXPlug = PlugDescriptor("turbulenceOffsetX")
	tox_ : TurbulenceOffsetXPlug = PlugDescriptor("turbulenceOffsetX")
	turbulenceOffsetY_ : TurbulenceOffsetYPlug = PlugDescriptor("turbulenceOffsetY")
	toy_ : TurbulenceOffsetYPlug = PlugDescriptor("turbulenceOffsetY")
	turbulenceOffsetZ_ : TurbulenceOffsetZPlug = PlugDescriptor("turbulenceOffsetZ")
	toz_ : TurbulenceOffsetZPlug = PlugDescriptor("turbulenceOffsetZ")
	node : FluidEmitter = None
	pass
class TurbulenceSpeedPlug(Plug):
	node : FluidEmitter = None
	pass
class TurbulenceTypePlug(Plug):
	node : FluidEmitter = None
	pass
class UseDistancePlug(Plug):
	node : FluidEmitter = None
	pass
class UseParticleRadiusPlug(Plug):
	node : FluidEmitter = None
	pass
# endregion


# define node class
class FluidEmitter(PointEmitter):
	densityEmissionMap_ : DensityEmissionMapPlug = PlugDescriptor("densityEmissionMap")
	densityMethod_ : DensityMethodPlug = PlugDescriptor("densityMethod")
	detailTurbulence_ : DetailTurbulencePlug = PlugDescriptor("detailTurbulence")
	emissionFunction_ : EmissionFunctionPlug = PlugDescriptor("emissionFunction")
	emitFluidColor_ : EmitFluidColorPlug = PlugDescriptor("emitFluidColor")
	fillObject_ : FillObjectPlug = PlugDescriptor("fillObject")
	fluidDensityEmission_ : FluidDensityEmissionPlug = PlugDescriptor("fluidDensityEmission")
	fluidDropoff_ : FluidDropoffPlug = PlugDescriptor("fluidDropoff")
	fluidFuelEmission_ : FluidFuelEmissionPlug = PlugDescriptor("fluidFuelEmission")
	fluidHeatEmission_ : FluidHeatEmissionPlug = PlugDescriptor("fluidHeatEmission")
	fluidJitter_ : FluidJitterPlug = PlugDescriptor("fluidJitter")
	fuelEmissionMap_ : FuelEmissionMapPlug = PlugDescriptor("fuelEmissionMap")
	fuelMethod_ : FuelMethodPlug = PlugDescriptor("fuelMethod")
	heatEmissionMap_ : HeatEmissionMapPlug = PlugDescriptor("heatEmissionMap")
	heatMethod_ : HeatMethodPlug = PlugDescriptor("heatMethod")
	inheritVelocity_ : InheritVelocityPlug = PlugDescriptor("inheritVelocity")
	motionStreak_ : MotionStreakPlug = PlugDescriptor("motionStreak")
	normalizedDropoff_ : NormalizedDropoffPlug = PlugDescriptor("normalizedDropoff")
	radiusPP_ : RadiusPPPlug = PlugDescriptor("radiusPP")
	speedMethod_ : SpeedMethodPlug = PlugDescriptor("speedMethod")
	startFrameEmission_ : StartFrameEmissionPlug = PlugDescriptor("startFrameEmission")
	turbulence_ : TurbulencePlug = PlugDescriptor("turbulence")
	turbulenceFrequencyX_ : TurbulenceFrequencyXPlug = PlugDescriptor("turbulenceFrequencyX")
	turbulenceFrequencyY_ : TurbulenceFrequencyYPlug = PlugDescriptor("turbulenceFrequencyY")
	turbulenceFrequencyZ_ : TurbulenceFrequencyZPlug = PlugDescriptor("turbulenceFrequencyZ")
	turbulenceFrequency_ : TurbulenceFrequencyPlug = PlugDescriptor("turbulenceFrequency")
	turbulenceOffsetX_ : TurbulenceOffsetXPlug = PlugDescriptor("turbulenceOffsetX")
	turbulenceOffsetY_ : TurbulenceOffsetYPlug = PlugDescriptor("turbulenceOffsetY")
	turbulenceOffsetZ_ : TurbulenceOffsetZPlug = PlugDescriptor("turbulenceOffsetZ")
	turbulenceOffset_ : TurbulenceOffsetPlug = PlugDescriptor("turbulenceOffset")
	turbulenceSpeed_ : TurbulenceSpeedPlug = PlugDescriptor("turbulenceSpeed")
	turbulenceType_ : TurbulenceTypePlug = PlugDescriptor("turbulenceType")
	useDistance_ : UseDistancePlug = PlugDescriptor("useDistance")
	useParticleRadius_ : UseParticleRadiusPlug = PlugDescriptor("useParticleRadius")

	# node attributes

	typeName = "fluidEmitter"
	apiTypeInt = 920
	apiTypeStr = "kFluidEmitter"
	typeIdInt = 1178946889
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["densityEmissionMap", "densityMethod", "detailTurbulence", "emissionFunction", "emitFluidColor", "fillObject", "fluidDensityEmission", "fluidDropoff", "fluidFuelEmission", "fluidHeatEmission", "fluidJitter", "fuelEmissionMap", "fuelMethod", "heatEmissionMap", "heatMethod", "inheritVelocity", "motionStreak", "normalizedDropoff", "radiusPP", "speedMethod", "startFrameEmission", "turbulence", "turbulenceFrequencyX", "turbulenceFrequencyY", "turbulenceFrequencyZ", "turbulenceFrequency", "turbulenceOffsetX", "turbulenceOffsetY", "turbulenceOffsetZ", "turbulenceOffset", "turbulenceSpeed", "turbulenceType", "useDistance", "useParticleRadius"]
	nodeLeafPlugs = ["densityEmissionMap", "densityMethod", "detailTurbulence", "emissionFunction", "emitFluidColor", "fillObject", "fluidDensityEmission", "fluidDropoff", "fluidFuelEmission", "fluidHeatEmission", "fluidJitter", "fuelEmissionMap", "fuelMethod", "heatEmissionMap", "heatMethod", "inheritVelocity", "motionStreak", "normalizedDropoff", "radiusPP", "speedMethod", "startFrameEmission", "turbulence", "turbulenceFrequency", "turbulenceOffset", "turbulenceSpeed", "turbulenceType", "useDistance", "useParticleRadius"]
	pass

